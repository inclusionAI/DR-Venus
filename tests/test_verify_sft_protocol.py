from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from experiments.verify_sft_protocol import verify_sft_protocol


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path):
    s0 = tmp_path / "s0_uniform.parquet"
    s1 = tmp_path / "s1_turn_weighted.parquet"
    val = tmp_path / "validation.parquet"
    test = tmp_path / "test.parquet"
    pd.DataFrame({"question": [f"train a {index}" for index in range(8)]}).to_parquet(s0, index=False)
    pd.DataFrame({"question": [f"train b {index}" for index in range(16)]}).to_parquet(s1, index=False)
    pd.DataFrame({"question": ["held out"]}).to_parquet(val, index=False)
    pd.DataFrame({"question": ["final test"]}).to_parquet(test, index=False)
    data_manifest = {
        "correctness": {"mode": "cache"},
        "split": {
            "unit": "normalized_question",
            "ratios": {"train": 0.9, "validation": 0.05, "test": 0.05},
            "resampling_applied_to": "train_only",
        },
        "artifacts": {
            "train_clean": {"sha256": "a" * 64},
            "validation": {"sha256": _hash(val)},
            "test": {"sha256": _hash(test)},
        },
    }
    data_manifest_path = tmp_path / "data_manifest.json"
    data_manifest_path.write_text(json.dumps(data_manifest), encoding="utf-8")
    manifest = {
        "target_assistant_tokens": 100,
        "seed": 42,
        "sampling_batch_multiple": 8,
        "between_arm_token_gap": 1,
        "between_arm_token_gap_tolerance": 1,
        "max_between_arm_token_gap_ratio": 0.001,
        "max_source_assistant_tokens": 10,
        "input": {"sha256": "a" * 64},
        "data_manifest": {"sha256": _hash(data_manifest_path)},
        "token_counter": {
            "type": "huggingface_multiturn_loss_mask_equivalent",
            "tokenizer": "Qwen/Qwen3-4B-Thinking-2507",
            "max_length": 32768,
            "truncation": "right",
        },
        "arms": {
            "s0_uniform": {
                "rows": 8,
                "realized_assistant_tokens": 101,
                "effective_rows_after_drop_last": 8,
                "effective_assistant_tokens": 101,
                "dropped_rows": 0,
            },
            "s1_turn_weighted": {
                "rows": 16,
                "realized_assistant_tokens": 102,
                "effective_rows_after_drop_last": 16,
                "effective_assistant_tokens": 102,
                "dropped_rows": 0,
            },
        },
        "artifacts": {
            "s0_uniform": {"file": s0.name, "sha256": _hash(s0)},
            "s1_turn_weighted": {"file": s1.name, "sha256": _hash(s1)},
        },
    }
    manifest_path = tmp_path / "sft_arms_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return s0, s1, val, test, manifest_path, data_manifest_path


def test_verify_sft_protocol_accepts_exact_counter_and_disjoint_val(tmp_path):
    s0, s1, val, test, manifest, data_manifest = _fixture(tmp_path)
    result = verify_sft_protocol(
        s0_path=s0,
        s1_path=s1,
        validation_path=val,
        test_path=test,
        manifest_path=manifest,
        data_manifest_path=data_manifest,
        expected_token_budget=100,
    )
    assert result["target_assistant_tokens"] == 100
    assert result["total_optimizer_steps"] == 3


def test_verify_sft_protocol_rejects_validation_leakage(tmp_path):
    s0, s1, val, test, manifest, data_manifest = _fixture(tmp_path)
    pd.DataFrame({"question": ["  TRAIN   A 0 "]}).to_parquet(val, index=False)
    data_payload = json.loads(data_manifest.read_text(encoding="utf-8"))
    data_payload["artifacts"]["validation"]["sha256"] = _hash(val)
    data_manifest.write_text(json.dumps(data_payload), encoding="utf-8")
    arm_payload = json.loads(manifest.read_text(encoding="utf-8"))
    arm_payload["data_manifest"]["sha256"] = _hash(data_manifest)
    manifest.write_text(json.dumps(arm_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="leaks"):
        verify_sft_protocol(
            s0_path=s0,
            s1_path=s1,
            validation_path=val,
            test_path=test,
            manifest_path=manifest,
            data_manifest_path=data_manifest,
            expected_token_budget=100,
        )


def test_verify_sft_protocol_rejects_approximate_counter(tmp_path):
    s0, s1, val, test, manifest, data_manifest = _fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["token_counter"]["type"] = "approximate_assistant_content"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exact tokenizer"):
        verify_sft_protocol(
            s0_path=s0,
            s1_path=s1,
            validation_path=val,
            test_path=test,
            manifest_path=manifest,
            data_manifest_path=data_manifest,
            expected_token_budget=100,
        )
