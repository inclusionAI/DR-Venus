import json

import pytest

from Inference.eval_inputs import (
    ad_hoc_item,
    build_run_fingerprint,
    completed_keys,
    expand_samples,
    item_from_row,
)
from Inference.token_accounting import count_input_output_tokens


def test_eval_row_is_converted_to_stable_work_item():
    item = item_from_row(
        {
            "data_source": "nq",
            "prompt": [{"role": "user", "content": "Where?"}],
            "reward_model": {"ground_truth": "A<|answer_split|>B"},
            "extra_info": {"eval_sample_id": "nq-1"},
        }
    )

    assert item.sample_id == "nq-1"
    assert item.references == ("A", "B")
    assert [sample.sample_index for sample in expand_samples([item], 4)] == [0, 1, 2, 3]


def test_resume_is_seed_scoped_and_fails_on_corrupt_evidence(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text(
        json.dumps({"sample_id": "a", "sample_index": 1, "seed": 42, "termination": "answered"}) + "\n"
        + json.dumps({"sample_id": "b", "sample_index": 0, "seed": 43, "termination": "exception"}) + "\n",
        encoding="utf-8",
    )
    assert completed_keys(path, seed=42) == {("a", 1)}
    path.write_text("{broken\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid resume"):
        completed_keys(path, seed=42)


def test_resume_is_bound_to_cohort_and_generation_config(tmp_path):
    item = ad_hoc_item("question")
    first = build_run_fingerprint({"model": "m", "temperature": 1.0}, [item])
    second = build_run_fingerprint({"model": "m", "temperature": 0.7}, [item])
    assert first != second

    path = tmp_path / "raw.jsonl"
    path.write_text(
        json.dumps(
            {
                "sample_id": item.sample_id,
                "sample_index": 0,
                "seed": 42,
                "termination": "answered",
                "run_fingerprint": first,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert completed_keys(path, seed=42, run_fingerprint=first) == {
        (item.sample_id, 0)
    }
    with pytest.raises(ValueError, match="different run fingerprint"):
        completed_keys(path, seed=42, run_fingerprint=second)


def test_ad_hoc_ids_are_content_stable():
    assert ad_hoc_item(" question ").sample_id == ad_hoc_item("question").sample_id


class FakeTokenizer:
    @staticmethod
    def apply_chat_template(messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        rendered = "".join(f"<{message['role']}>{message['content']}" for message in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    @staticmethod
    def encode(text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text)


def test_role_aware_token_accounting_excludes_generation_prompt_from_output():
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": "T"},
        {"role": "assistant", "content": "B"},
    ]
    input_tokens, output_tokens = count_input_output_tokens(messages, FakeTokenizer())

    assert output_tokens == 2
    assert input_tokens > output_tokens
