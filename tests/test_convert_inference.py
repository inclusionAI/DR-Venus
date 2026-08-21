import json

import pytest

from evaluation.convert_inference import convert_file, convert_raw_record


def raw_record():
    return {
        "sample_id": "nq-1",
        "sample_index": 0,
        "data_source": "nq",
        "question": "Where?",
        "references": ["Paris"],
        "model": "model",
        "seed": 42,
        "messages": [{"role": "assistant", "content": "<think>x</think><answer>Paris</answer>"}],
        "prediction": "Paris",
        "termination": "answered",
        "turns": 2,
        "parse_error_count": 0,
        "tool_call_count": 1,
        "browse_call_count": 1,
        "tool_error_count": 0,
        "timeout_count": 0,
        "dedup_count": 0,
        "final_token_count": 30,
        "input_tokens": 20,
        "output_tokens": 10,
        "elapsed_seconds": 1.5,
        "max_turns": 12,
        "context_length": 32768,
        "cost_cny": 0.02,
        "final_answer_strict_format_valid": True,
        "strict_format_violation_count": 0,
        "run_fingerprint": "f" * 64,
        "cache_hit_count": 2,
        "cache_miss_count": 1,
        "cached_tool_error_count": 0,
        "checkpoint_revision": "a" * 40,
        "checkpoint_sha256": "b" * 64,
        "tool_cache_sha256": "c" * 64,
        "llm_call_count": 3,
        "llm_sampling_seeds": [42, 43, 44],
    }


def test_raw_record_converts_to_valid_prediction():
    prediction = convert_raw_record(raw_record(), trajectory_path="raw.jsonl", line_number=3)

    assert prediction.completed is True
    assert prediction.format_valid is True
    assert prediction.total_tokens == 30
    assert prediction.trajectory_path == "raw.jsonl#L3"
    assert prediction.metadata["cache_hit_count"] == 2
    assert prediction.metadata["run_fingerprint"] == "f" * 64
    assert prediction.metadata["checkpoint_sha256"] == "b" * 64
    assert prediction.metadata["llm_sampling_seeds"] == [42, 43, 44]


def test_exception_record_is_retained_as_incomplete():
    raw = raw_record()
    raw.update({
        "prediction": None,
        "termination": "exception",
        "messages": None,
        "turns": 0,
        "final_answer_strict_format_valid": False,
    })

    prediction = convert_raw_record(raw, trajectory_path="raw.jsonl", line_number=1)

    assert prediction.answer == ""
    assert prediction.completed is False
    assert prediction.format_valid is False


def test_converter_rejects_missing_strict_format_evidence():
    raw = raw_record()
    del raw["final_answer_strict_format_valid"]
    with pytest.raises(ValueError, match="strict_format"):
        convert_raw_record(raw, trajectory_path="raw.jsonl", line_number=1)


def test_file_conversion_fails_closed_on_bad_line(tmp_path):
    source = tmp_path / "raw.jsonl"
    target = tmp_path / "predictions.jsonl"
    source.write_text(json.dumps(raw_record()) + "\n{broken\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot convert"):
        convert_file(source, target)
    assert not target.exists()


def test_file_conversion_keeps_latest_retry(tmp_path):
    source = tmp_path / "raw.jsonl"
    target = tmp_path / "predictions.jsonl"
    failed = raw_record()
    failed.update({
        "termination": "exception",
        "prediction": None,
        "messages": None,
        "final_answer_strict_format_valid": False,
    })
    succeeded = raw_record()
    source.write_text(
        json.dumps(failed) + "\n" + json.dumps(succeeded) + "\n",
        encoding="utf-8",
    )

    records = convert_file(source, target)

    assert len(records) == 1
    assert records[0].completed is True
    assert records[0].trajectory_path.endswith("#L2")


def test_file_conversion_rejects_mixed_run_fingerprints(tmp_path):
    source = tmp_path / "raw.jsonl"
    target = tmp_path / "predictions.jsonl"
    first = raw_record()
    second = raw_record()
    second["sample_id"] = "nq-2"
    second["run_fingerprint"] = "e" * 64
    source.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mixes multiple run_fingerprint"):
        convert_file(source, target)
