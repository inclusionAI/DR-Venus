from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from SFT.data_clean.build_sft_arms import (
    build_sft_arms,
    compute_turn_weight,
    count_assistant_loss_tokens_with_tokenizer,
    count_assistant_tokens,
    sample_indices_to_budget,
)
from SFT.data_clean.data_governance import (
    DEFAULT_JUDGE_PROMPT_VERSION,
    CorrectnessEvaluator,
    OpenAICompatibleJudge,
    get_dotted_value,
    load_correctness_cache,
    make_correctness_cache_key,
    normalize_question_key,
    parse_correctness_value,
    parse_judge_response,
    split_records_by_question,
)
from SFT.data_clean.prepare_trajectories import (
    build_argument_parser,
    normalize_row,
    prune_disallowed_tool_turns,
    remove_duplicate_tool_calls,
    run_pipeline,
    validate_messages,
)


def final_messages(question: str, answer: str = "answer") -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": question},
        {"role": "assistant", "content": f"<think>reason</think><answer>{answer}</answer>"},
    ]


def tool_messages(question: str, tool_name: str = "search", argument: str = "q") -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": (
                "<think>look</think><tool_call>"
                + json.dumps({"name": tool_name, "arguments": {"query": [argument]}})
                + "</tool_call>"
            ),
        },
        {"role": "user", "content": "<tool_response>result</tool_response>"},
        {"role": "assistant", "content": "<think>done</think><answer>answer</answer>"},
    ]


class CorrectnessValueTests(unittest.TestCase):
    def test_common_encodings_table(self) -> None:
        cases = [
            (True, 1.0),
            (False, 0.0),
            (1, 1.0),
            (0, 0.0),
            (0.75, 0.75),
            (-1, -1.0),
            ("true", 1.0),
            (" TRUE ", 1.0),
            ("yes", 1.0),
            ("correct", 1.0),
            ("pass", 1.0),
            ("passed", 1.0),
            ("ok", 1.0),
            ("1", 1.0),
            ("false", 0.0),
            ("no", 0.0),
            ("incorrect", 0.0),
            ("wrong", 0.0),
            ("fail", 0.0),
            ("failed", 0.0),
            ("0", 0.0),
            ("0.25", 0.25),
            ({"correct": True}, 1.0),
            ({"is_correct": False}, 0.0),
            ({"passed": "yes"}, 1.0),
            ({"score": 0.8}, 0.8),
            ({"reward": "0.6"}, 0.6),
            ({"judgement": "correct"}, 1.0),
            ({"judgment": "incorrect"}, 0.0),
            ({"label": "pass"}, 1.0),
            (None, None),
            ("", None),
            ("maybe", None),
            (float("nan"), None),
            ({"reason": "only"}, None),
            ([1], None),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(parse_correctness_value(value), expected)

    def test_dotted_field(self) -> None:
        row = {"meta": {"quality": {"correct": True}}}
        self.assertIs(get_dotted_value(row, "meta.quality.correct"), True)
        self.assertIsNone(get_dotted_value(row, "meta.missing.correct"))


class SplitTests(unittest.TestCase):
    def test_exact_90_5_5_question_groups_and_determinism(self) -> None:
        records = [{"question": f"Question {idx}", "messages": [], "gt": "a"} for idx in range(100)]
        first, first_assignments = split_records_by_question(records, seed=42)
        second, second_assignments = split_records_by_question(list(reversed(records)), seed=42)
        self.assertEqual({name: len(rows) for name, rows in first.items()}, {"train": 90, "validation": 5, "test": 5})
        self.assertEqual(first_assignments, second_assignments)

    def test_normalized_duplicates_never_leak(self) -> None:
        records = [
            {"question": "  SAME   Question ", "messages": [], "gt": "a"},
            {"question": "same question", "messages": [], "gt": "b"},
            *[{"question": f"q{idx}", "messages": [], "gt": "a"} for idx in range(30)],
        ]
        splits, _ = split_records_by_question(records, seed=7)
        containing = [name for name, rows in splits.items() if sum("same" in row["question"].casefold() for row in rows)]
        self.assertEqual(containing, [containing[0]])
        self.assertEqual(sum("same" in row["question"].casefold() for rows in splits.values() for row in rows), 2)

    def test_tiny_dataset_populates_all_requested_splits(self) -> None:
        records = [{"question": str(idx), "messages": [], "gt": "a"} for idx in range(3)]
        splits, _ = split_records_by_question(records, seed=1)
        self.assertEqual({name: len(rows) for name, rows in splits.items()}, {"train": 1, "validation": 1, "test": 1})

    def test_seed_changes_assignment(self) -> None:
        records = [{"question": f"q{idx}", "messages": [], "gt": "a"} for idx in range(100)]
        _, first = split_records_by_question(records, seed=1)
        _, second = split_records_by_question(records, seed=2)
        self.assertNotEqual(first, second)

    def test_invalid_ratios_and_question(self) -> None:
        with self.assertRaises(ValueError):
            split_records_by_question([], seed=1, ratios={"train": 0.8, "test": 0.1})
        with self.assertRaises(ValueError):
            split_records_by_question([{"question": "", "messages": [], "gt": "a"}], seed=1)


class CacheAndJudgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clean_row = {"question": "q", "gt": "a", "messages": final_messages("q", "a")}
        self.key = make_correctness_cache_key("q", "a", "a")

    def cache_record(self, **overrides):
        record = {
            "cache_key": self.key,
            "question": "q",
            "reference": "a",
            "candidate": "a",
            "prompt_version": DEFAULT_JUDGE_PROMPT_VERSION,
            "judge_model": "",
            "correct": True,
        }
        record.update(overrides)
        return record

    def test_single_jsonl_record_is_loaded_as_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.jsonl"
            path.write_text(json.dumps(self.cache_record()) + "\n", encoding="utf-8")
            self.assertIn(self.key, load_correctness_cache(path))

    def test_cache_mapping_and_derived_key_formats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mapping_path = Path(directory) / "mapping.json"
            mapping_path.write_text(json.dumps({self.key: {"score": 0.9}}), encoding="utf-8")
            self.assertEqual(load_correctness_cache(mapping_path), {})
            derived_path = Path(directory) / "derived.json"
            derived_path.write_text(
                json.dumps([{"question": "q", "gt": "a", "candidate": "a", "correct": True}]),
                encoding="utf-8",
            )
            self.assertIn(self.key, load_correctness_cache(derived_path))

    def test_field_decisions(self) -> None:
        evaluator = CorrectnessEvaluator(mode="field", field="meta.score", threshold=0.5)
        self.assertTrue(evaluator.evaluate({"meta": {"score": 0.5}}, self.clean_row).keep)
        self.assertEqual(evaluator.evaluate({"meta": {"score": 0.4}}, self.clean_row).reason, "correctness_incorrect")
        self.assertEqual(evaluator.evaluate({"meta": {}}, self.clean_row).reason, "correctness_field_missing")
        self.assertEqual(evaluator.evaluate({"meta": {"score": "maybe"}}, self.clean_row).reason, "correctness_field_invalid")

    def test_offline_cache_hit_miss_and_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.jsonl"
            path.write_text(json.dumps(self.cache_record(score=0.75)) + "\n", encoding="utf-8")
            evaluator = CorrectnessEvaluator(mode="cache", cache_path=path, threshold=0.8)
            self.assertEqual(evaluator.evaluate({}, self.clean_row).reason, "correctness_incorrect")
            missing = dict(self.clean_row, question="other")
            self.assertEqual(evaluator.evaluate({}, missing).reason, "correctness_cache_miss")
            path.write_text(json.dumps(self.cache_record(correct=None, reason="bad")) + "\n", encoding="utf-8")
            invalid = CorrectnessEvaluator(mode="cache", cache_path=path)
            self.assertEqual(invalid.evaluate({}, self.clean_row).reason, "correctness_cache_invalid")

    def test_cache_mode_requires_existing_path(self) -> None:
        with self.assertRaises(ValueError):
            CorrectnessEvaluator(mode="cache")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                CorrectnessEvaluator(mode="cache", cache_path=Path(directory) / "missing")

    def test_judge_response_variants(self) -> None:
        cases = [
            ('{"correct": true, "reason": "same"}', 1.0),
            ("```json\n{\"judgement\": \"incorrect\"}\n```", 0.0),
            ("preface {\"score\": 0.75} suffix", 0.75),
        ]
        for content, expected in cases:
            with self.subTest(content=content):
                score, _ = parse_judge_response({"choices": [{"message": {"content": content}}]})
                self.assertEqual(score, expected)

    def test_judge_retries_and_does_not_require_api_key(self) -> None:
        attempts = []
        sleeps = []

        def transport(url, headers, body, timeout):
            attempts.append((url, headers, json.loads(body), timeout))
            if len(attempts) < 3:
                raise OSError("temporary")
            return {"choices": [{"message": {"content": '{"correct": true}'}}]}

        judge = OpenAICompatibleJudge(
            base_url="https://judge.invalid/v1",
            model="judge-model",
            retries=2,
            transport=transport,
            sleeper=sleeps.append,
        )
        score, _ = judge.evaluate("q", "a", "a")
        self.assertEqual(score, 1.0)
        self.assertEqual(len(attempts), 3)
        self.assertEqual(sleeps, [1, 2])
        self.assertNotIn("Authorization", attempts[0][1])

    def test_judge_cache_is_write_through(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.jsonl"
            judge = OpenAICompatibleJudge(
                base_url="https://judge.invalid/v1",
                model="judge-model",
                retries=0,
                transport=lambda *_: {"choices": [{"message": {"content": '{"correct": true}'}}]},
            )
            evaluator = CorrectnessEvaluator(
                mode="judge", cache_path=path, judge=judge, judge_model="judge-model"
            )
            self.assertTrue(evaluator.evaluate({}, self.clean_row).keep)
            model_key = make_correctness_cache_key("q", "a", "a", judge_model="judge-model")
            self.assertIn(model_key, load_correctness_cache(path))

    def test_cache_rejects_tampered_explicit_key_and_wrong_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.jsonl"
            tampered = self.cache_record(cache_key="0" * 64)
            path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
            self.assertEqual(load_correctness_cache(path), {})

            wrong_model_key = make_correctness_cache_key("q", "a", "a", judge_model="old")
            wrong_model = self.cache_record(
                cache_key=wrong_model_key, judge_model="old"
            )
            path.write_text(json.dumps(wrong_model) + "\n", encoding="utf-8")
            evaluator = CorrectnessEvaluator(
                mode="cache", cache_path=path, judge_model="new"
            )
            self.assertEqual(
                evaluator.evaluate({}, self.clean_row).reason,
                "correctness_cache_miss",
            )


class ExistingCleaningBoundaryTests(unittest.TestCase):
    def test_both_input_schemas(self) -> None:
        clean = {"question": "q", "gt": "a", "messages": final_messages("q", "a")}
        normalized, reason = normalize_row(clean, system_prompt_mode="row", template_system_prompt="fallback")
        self.assertIsNone(reason)
        self.assertEqual(normalized["question"], "q")

        raw = {
            "meta": {"question": "raw q", "answer": "raw a"},
            "messages": [
                {"role": "user", "content": "raw q"},
                {"role": "assistant", "content": "<think>x</think><answer>raw a</answer>"},
            ],
        }
        normalized, reason = normalize_row(raw, system_prompt_mode="template", template_system_prompt="system")
        self.assertIsNone(reason)
        self.assertEqual([message["role"] for message in normalized["messages"]], ["system", "user", "assistant"])

    def test_disallowed_tool_and_paired_response_are_pruned(self) -> None:
        messages = tool_messages("q", tool_name="PythonInterpreter")
        pruned, info = prune_disallowed_tool_turns(messages, {"search", "visit"})
        self.assertEqual(len(info), 1)
        self.assertEqual(len(messages) - len(pruned), 2)
        self.assertTrue(validate_messages(pruned)[0])

    def test_duplicate_tool_and_paired_response_are_removed(self) -> None:
        first = tool_messages("q")
        duplicate_pair = [dict(first[2]), dict(first[3])]
        messages = first[:4] + duplicate_pair + first[4:]
        deduped, info = remove_duplicate_tool_calls(messages)
        self.assertEqual(len(info), 1)
        self.assertEqual(len(messages) - len(deduped), 2)
        self.assertTrue(validate_messages(deduped)[0])

    def test_role_and_tag_errors_table(self) -> None:
        cases = [
            ([{"role": "user", "content": "q"}], "too_short"),
            ([{"role": "user", "content": "s"}, *final_messages("q")[1:]], "bad_role_start"),
            ([*final_messages("q")[:2], {"role": "user", "content": "bad"}], "bad_role_at_2"),
            ([*final_messages("q")[:2], {"role": "assistant", "content": "<answer>a</answer>"}], "assistant_missing_think"),
            ([*final_messages("q")[:2], {"role": "assistant", "content": "<think>x</think>"}], "final_assistant_missing_answer"),
        ]
        for messages, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(validate_messages(messages), (False, expected))


class SftArmTests(unittest.TestCase):
    def test_injected_counter_counts_assistant_only(self) -> None:
        messages = final_messages("long user text", "abc")
        self.assertEqual(count_assistant_tokens(messages, len), len(messages[-1]["content"]))

    def test_tokenizer_counter_matches_assistant_loss_suffixes(self) -> None:
        class FakeTokenizer:
            @staticmethod
            def apply_chat_template(messages, *, tokenize, add_generation_prompt, enable_thinking):
                self = "".join(
                    f"<{message['role']}>{message['content']}</{message['role']}>" for message in messages
                )
                return self + ("<assistant>" if add_generation_prompt else "")

            @staticmethod
            def encode(text, *, add_special_tokens):
                return list(text)

        messages = final_messages("q", "a")
        expected = len(messages[-1]["content"] + "</assistant>")
        self.assertEqual(count_assistant_loss_tokens_with_tokenizer(messages, FakeTokenizer()), expected)
        self.assertEqual(
            count_assistant_loss_tokens_with_tokenizer(
                messages, FakeTokenizer(), max_length=1, truncation="right"
            ),
            0,
        )
        self.assertEqual(
            count_assistant_loss_tokens_with_tokenizer(
                messages, FakeTokenizer(), max_length=expected, truncation="left"
            ),
            expected,
        )

    def test_turn_weights_boundaries(self) -> None:
        def messages_for_turns(turns: int):
            messages = [{"role": "system", "content": "s"}]
            for index in range(turns):
                messages.extend(
                    [
                        {"role": "user", "content": str(index)},
                        {"role": "assistant", "content": "a"},
                    ]
                )
            return messages

        cases = [(50, 1), (51, 2), (100, 2), (101, 5)]
        for turns, expected in cases:
            with self.subTest(turns=turns):
                self.assertEqual(compute_turn_weight(messages_for_turns(turns)), expected)

    def test_sampling_is_deterministic_and_budget_bounded(self) -> None:
        first = sample_indices_to_budget([3, 5, 7], [1, 1, 1], token_budget=20, seed=42)
        second = sample_indices_to_budget([3, 5, 7], [1, 1, 1], token_budget=20, seed=42)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first[1], 20)
        self.assertLess(first[1] - 20, 7)

    def test_sampling_uses_complete_train_batches(self) -> None:
        indices, realized = sample_indices_to_budget(
            [2, 3], [1, 1], token_budget=10, seed=1, batch_multiple=4
        )
        self.assertEqual(len(indices) % 4, 0)
        self.assertGreaterEqual(realized, 10)

    def test_build_arms_same_target_and_reproducible(self) -> None:
        records = [
            {"question": "short", "messages": final_messages("short", "aaa"), "gt": "aaa"},
            {"question": "long", "messages": tool_messages("long"), "gt": "answer"},
        ]
        kwargs = dict(
            token_budget=100,
            seed=9,
            text_counter=len,
            short_max=1,
            medium_max=1,
            short_weight=1,
            medium_weight=2,
            long_weight=5,
            # Keep this unit fixture focused on deterministic arm sampling;
            # real heterogeneous-token matching is fail-closed and covered by
            # the formal verifier.
            assistant_counter=lambda _messages: 10,
        )
        arms, manifest = build_sft_arms(records, **kwargs)
        arms_again, manifest_again = build_sft_arms(records, **kwargs)
        self.assertEqual(arms, arms_again)
        self.assertEqual(manifest, manifest_again)
        self.assertEqual(set(arms), {"s0_uniform", "s1_turn_weighted"})
        for arm in manifest["arms"].values():
            self.assertGreaterEqual(arm["realized_assistant_tokens"], 100)

    def test_sampling_rejects_invalid_inputs(self) -> None:
        cases = [
            ([], [], 10),
            ([0], [1], 10),
            ([1], [0], 10),
            ([1], [1], 0),
            ([1, 2], [1], 10),
            ([1], [1], 10),
        ]
        for case_index, (counts, weights, budget) in enumerate(cases):
            with self.subTest(counts=counts, weights=weights, budget=budget):
                with self.assertRaises(ValueError):
                    sample_indices_to_budget(
                        counts,
                        weights,
                        token_budget=budget,
                        seed=1,
                        batch_multiple=0 if case_index == len(cases) - 1 else 1,
                    )


class PipelineIntegrationTests(unittest.TestCase):
    def test_field_filter_split_train_only_resample_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.parquet"
            output_dir = root / "out"
            records = []
            for index in range(100):
                records.append(
                    {
                        "question": f"Question {index}",
                        "gt": "answer",
                        "messages": tool_messages(f"Question {index}"),
                        "quality": {"correct": True},
                    }
                )
            records.extend(
                [
                    {
                        "question": "incorrect",
                        "gt": "answer",
                        "messages": final_messages("incorrect"),
                        "quality": {"correct": False},
                    },
                    {
                        "question": "missing",
                        "gt": "answer",
                        "messages": final_messages("missing"),
                        "quality": {},
                    },
                ]
            )
            pd.DataFrame(records).to_parquet(input_path, index=False)
            args = build_argument_parser().parse_args(
                [
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                    "--system-prompt-mode",
                    "row",
                    "--correctness-mode",
                    "field",
                    "--correctness-field",
                    "quality.correct",
                    "--short-max-turns",
                    "1",
                    "--medium-max-turns",
                    "2",
                    "--short-multiplier",
                    "1",
                    "--medium-multiplier",
                    "2",
                    "--long-multiplier",
                    "5",
                    "--seed",
                    "42",
                ]
            )
            stats = run_pipeline(args)

            train_clean = pd.read_parquet(output_dir / "train_clean.parquet")
            train = pd.read_parquet(output_dir / "train_resampled.parquet")
            validation = pd.read_parquet(output_dir / "validation.parquet")
            test = pd.read_parquet(output_dir / "test.parquet")
            legacy = pd.read_parquet(output_dir / "cleaned_resampled.parquet")
            manifest = json.loads((output_dir / "data_manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(manifest["counts"]["question_groups_by_split"], {"train": 90, "validation": 5, "test": 5})
            self.assertEqual(stats["drop_reasons"]["correctness_incorrect"], 1)
            self.assertEqual(stats["drop_reasons"]["correctness_field_missing"], 1)
            self.assertEqual(len(train), len(train_clean) * 2)
            self.assertEqual(len(validation), 5)
            self.assertEqual(len(test), 5)
            pd.testing.assert_frame_equal(train, legacy)

            question_sets = [
                {normalize_question_key(question) for question in frame["question"]}
                for frame in (train_clean, validation, test)
            ]
            self.assertFalse(question_sets[0] & question_sets[1])
            self.assertFalse(question_sets[0] & question_sets[2])
            self.assertFalse(question_sets[1] & question_sets[2])
            self.assertEqual(manifest["split"]["resampling_applied_to"], "train_only")
            self.assertIn("logical_sha256", manifest["artifacts"]["train_resampled"])
            self.assertNotIn("api_key", json.dumps(manifest).casefold())

            second_output = root / "out-repeat"
            repeated_args = build_argument_parser().parse_args(
                [
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(second_output),
                    "--system-prompt-mode",
                    "row",
                    "--correctness-mode",
                    "field",
                    "--correctness-field",
                    "quality.correct",
                    "--short-max-turns",
                    "1",
                    "--medium-max-turns",
                    "2",
                    "--short-multiplier",
                    "1",
                    "--medium-multiplier",
                    "2",
                    "--long-multiplier",
                    "5",
                    "--seed",
                    "42",
                ]
            )
            run_pipeline(repeated_args)
            repeated_manifest = json.loads(
                (second_output / "data_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest, repeated_manifest)


if __name__ == "__main__":
    unittest.main()
