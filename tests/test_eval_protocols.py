import json
from pathlib import Path

import pytest

from drvenus_repro.manifests import sha256_file, write_json, write_predictions
from drvenus_repro.schemas import PredictionRecord
from evaluation.build_eval_web import build_eval_web
from evaluation.build_pass4_subset import build_pass4_subset
from evaluation.judge_calibration import (
    prepare_calibration_template,
    score_calibration_template,
)
from evaluation.judge_predictions import JudgeResult, score_prediction_records


def _write_jsonl(path: Path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _public_rows(prefix, count=4):
    return [
        {"id": f"{prefix}-{index}", "question": f"question {prefix} {index}", "answer": f"a{index}"}
        for index in range(count)
    ]


def _prediction(sample_id="s1", *, score=None, seed=42, sample_index=0):
    return PredictionRecord.from_mapping(
        {
            "schema_version": "1.0",
            "sample_id": sample_id,
            "data_source": "fixture",
            "model": "candidate",
            "seed": seed,
            "sample_index": sample_index,
            "answer": f"answer-{sample_id}",
            "references": [f"answer-{sample_id}"],
            "score": score,
            "completed": True,
            "format_valid": True,
            "turns": 2,
            "tool_calls": 1,
            "browse_calls": 1,
            "timeout_count": 0,
            "input_tokens": 10,
            "output_tokens": 2,
            "latency_seconds": 0.1,
            "cost_cny": 0.0,
            "max_turns": 12,
            "context_length": 32768,
            "trajectory_path": None,
            "metadata": {"question": f"question-{sample_id}"},
        }
    )


def test_eval_web_requires_and_selects_each_real_source(tmp_path):
    en = tmp_path / "en.jsonl"
    zh = tmp_path / "zh.jsonl"
    _write_jsonl(en, _public_rows("en"))
    _write_jsonl(zh, list(reversed(_public_rows("zh"))))
    output = tmp_path / "web.jsonl"
    manifest = build_eval_web(en, zh, output, per_source=2, seed=7)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert manifest["suite"] == "eval-web-4"
    assert manifest["selected_counts"] == {"BrowseComp": 2, "BrowseComp-ZH": 2}
    assert len(rows) == 4
    assert {row["data_source"] for row in rows} == {"BrowseComp", "BrowseComp-ZH"}
    assert all(row["extra_info"]["eval_sample_id"].startswith(row["data_source"] + ":") for row in rows)
    assert sha256_file(output) == manifest["output"]["sha256"]


def test_eval_web_is_order_independent_and_fails_before_writing(tmp_path):
    en_a, en_b, zh = tmp_path / "en-a.jsonl", tmp_path / "en-b.jsonl", tmp_path / "zh.jsonl"
    _write_jsonl(en_a, _public_rows("en"))
    _write_jsonl(en_b, list(reversed(_public_rows("en"))))
    _write_jsonl(zh, _public_rows("zh"))
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    ids_a = build_eval_web(en_a, zh, first, per_source=2, seed=42)["sample_ids"]
    ids_b = build_eval_web(en_b, zh, second, per_source=2, seed=42)["sample_ids"]
    assert ids_a == ids_b

    invalid = tmp_path / "invalid.jsonl"
    _write_jsonl(invalid, [{"id": "missing-answer", "question": "q"}])
    target = tmp_path / "must-not-exist.jsonl"
    with pytest.raises(ValueError, match="answer"):
        build_eval_web(en_a, invalid, target, per_source=1)
    assert not target.exists()


def test_eval_web_supports_official_style_problem_rows_without_explicit_ids(tmp_path):
    en, zh = tmp_path / "en.jsonl", tmp_path / "zh.jsonl"
    _write_jsonl(en, [{"problem": "English problem", "answer": "A"}])
    _write_jsonl(zh, [{"问题": "中文问题", "答案": "乙"}])
    output = tmp_path / "web.jsonl"
    manifest = build_eval_web(en, zh, output, per_source=1)
    assert all("question-sha256:" in sample_id for sample_id in manifest["sample_ids"])


def _parent_cohort(tmp_path):
    rows = []
    for source in ("a", "b"):
        for index in range(3):
            rows.append(
                {
                    "data_source": source,
                    "prompt": [{"role": "user", "content": f"q {source} {index}"}],
                    "reward_model": {"ground_truth": f"answer {index}"},
                    "extra_info": {"eval_sample_id": f"{source}-{index}", "index": f"{source}-{index}"},
                }
            )
    cohort = tmp_path / "core.jsonl"
    _write_jsonl(cohort, rows)
    manifest_path = tmp_path / "core.jsonl.manifest.json"
    write_json(
        manifest_path,
        {
            "suite": "eval-core-6",
            "num_samples": 6,
            "output": {"path": cohort.as_posix(), "sha256": sha256_file(cohort)},
            "sample_ids": [row["extra_info"]["eval_sample_id"] for row in rows],
        },
    )
    return cohort, manifest_path


def test_pass4_subset_validates_parent_and_is_stratified(tmp_path):
    cohort, parent_manifest = _parent_cohort(tmp_path)
    output = tmp_path / "pass4.jsonl"
    manifest = build_pass4_subset(
        cohort, output, total=4, seed=5, parent_manifest_path=parent_manifest
    )
    assert manifest["selected_counts"] == {"a": 2, "b": 2}
    assert len(manifest["sample_ids"]) == 4
    assert manifest["parent"]["cohort_sha256"] == sha256_file(cohort)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert all(row["extra_info"]["pass4_suite"] == "pass4-4" for row in rows)


def test_pass4_subset_rejects_parent_hash_drift(tmp_path):
    cohort, parent_manifest = _parent_cohort(tmp_path)
    raw = json.loads(parent_manifest.read_text(encoding="utf-8"))
    raw["output"]["sha256"] = "f" * 64
    write_json(parent_manifest, raw)
    with pytest.raises(ValueError, match="hash"):
        build_pass4_subset(cohort, tmp_path / "out.jsonl", total=2, parent_manifest_path=parent_manifest)


def test_judge_online_retry_then_offline_cache_replay(tmp_path):
    record = _prediction()
    cache = tmp_path / "judge.jsonl"
    calls = 0

    def fake_call(_):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("fixture timeout")
        return JudgeResult(1.0, "correct", "a" * 64, "2026-01-01T00:00:00+00:00", {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5})

    online, stats = score_prediction_records(
        [record],
        mode="online",
        model="judge-fixture",
        cache_path=cache,
        retries=2,
        judge_call=fake_call,
        sleep=lambda _: None,
    )
    assert calls == 2
    assert stats["online_calls"] == 1
    assert online[0].score == 1.0
    assert online[0].metadata["judge"]["prompt_version"]

    offline, offline_stats = score_prediction_records(
        [record], mode="offline", model="judge-fixture", cache_path=cache
    )
    assert offline[0].score == 1.0
    assert offline[0].metadata["judge"]["mode"] == "offline-cache"
    assert offline_stats["cache_hits"] == 1


def test_judge_offline_miss_and_online_credentials_fail_closed(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="cache miss"):
        score_prediction_records(
            [_prediction()], mode="offline", model="judge-fixture", cache_path=tmp_path / "none"
        )
    monkeypatch.delenv("TEST_JUDGE_KEY", raising=False)
    with pytest.raises(ValueError, match="TEST_JUDGE_KEY"):
        score_prediction_records(
            [_prediction()],
            mode="online",
            model="judge-fixture",
            base_url="https://judge.invalid/v1",
            api_key_env="TEST_JUDGE_KEY",
            cache_path=tmp_path / "cache",
        )
    with pytest.raises(ValueError, match="provenance"):
        score_prediction_records(
            [_prediction(score=1.0)],
            mode="offline",
            model="judge-fixture",
            cache_path=tmp_path / "none",
        )


def _write_scored_predictions(tmp_path):
    records = [_prediction("one", score=1.0), _prediction("two", score=0.0)]
    path = tmp_path / "scored.jsonl"
    write_predictions(path, records)
    return path, records


def test_blinded_calibration_template_and_agreement(tmp_path):
    predictions, records = _write_scored_predictions(tmp_path)
    template = tmp_path / "review.jsonl"
    manifest = prepare_calibration_template(
        records, predictions_path=predictions, output_path=template, size=2, seed=9
    )
    rows = [json.loads(line) for line in template.read_text(encoding="utf-8").splitlines()]
    assert manifest["blinded"] is True
    assert all("judge_score" not in row and row["human_score"] is None for row in rows)
    for row in rows:
        row["human_score"] = 1
        row["notes"] = "manual fixture"
    _write_jsonl(template, rows)

    report = score_calibration_template(
        template, template.with_name(template.name + ".manifest.json"), predictions
    )
    assert report["num_reviewed"] == 2
    assert report["agreement_rate"] == 0.5
    assert report["target_met"] is False
    assert report["confusion_matrix"]["false_negative"] == 1


def test_calibration_rejects_immutable_edits_and_incomplete_labels(tmp_path):
    predictions, records = _write_scored_predictions(tmp_path)
    template = tmp_path / "review.jsonl"
    prepare_calibration_template(records, predictions_path=predictions, output_path=template, size=2)
    manifest = template.with_name(template.name + ".manifest.json")
    rows = [json.loads(line) for line in template.read_text(encoding="utf-8").splitlines()]
    rows[0]["question"] = "tampered"
    rows[0]["human_score"] = 1
    rows[1]["human_score"] = 0
    _write_jsonl(template, rows)
    with pytest.raises(ValueError, match="immutable"):
        score_calibration_template(template, manifest, predictions)

    prepare_calibration_template(records, predictions_path=predictions, output_path=template, size=2)
    with pytest.raises(ValueError, match="human_score"):
        score_calibration_template(template, manifest, predictions)


def test_calibration_requires_requested_number_of_scored_primary_records(tmp_path):
    predictions, records = _write_scored_predictions(tmp_path)
    with pytest.raises(ValueError, match="requires 3"):
        prepare_calibration_template(
            records, predictions_path=predictions, output_path=tmp_path / "review", size=3
        )
