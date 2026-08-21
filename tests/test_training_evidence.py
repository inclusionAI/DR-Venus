import json
from pathlib import Path

import pytest

from experiments import training_evidence


def _clean_git(monkeypatch):
    monkeypatch.setattr(training_evidence, "_dirty_state", lambda: (False, None, []))
    monkeypatch.setattr(training_evidence, "_git", lambda *args, **kwargs: "personal-commit")
    monkeypatch.setattr(training_evidence, "_upstream_base", lambda: "upstream-commit")


def test_snapshot_and_failed_finalize_preserve_evidence(tmp_path, monkeypatch):
    _clean_git(monkeypatch)
    identity = tmp_path / "checkpoint_identity.json"
    identity.write_text('{"sha256":"abc"}\n', encoding="utf-8")
    cohort = tmp_path / "train.parquet"
    cohort.write_bytes(b"cohort")
    command = tmp_path / "resolved_command.txt"
    command.write_text("torchrun -m trainer seed=42\n", encoding="utf-8")
    snapshot_path = tmp_path / "run/run_snapshot.json"

    snapshot = training_evidence.create_snapshot(
        output=snapshot_path,
        stage="rl",
        run_id="r1-seed42",
        protocol_label="4b-short-rl-ablation",
        checkpoint_identity=identity,
        command_file=command,
        data=[f"train={cohort}"],
        config=["seed=42", "use_info_gain=true"],
        gpu_type="A100-80G",
        gpu_count=4,
    )
    log = tmp_path / "run/training.log"
    log.write_text("failed honestly\n", encoding="utf-8")
    frozen_cache = tmp_path / "run/frozen_tool_cache.jsonl"
    frozen_cache.write_text('{"cache":"evidence"}\n', encoding="utf-8")
    manifest_path = tmp_path / "run/experiment_manifest.yaml"
    manifest = training_evidence.finalize_snapshot(
        snapshot_path=snapshot_path,
        output=manifest_path,
        exit_code=7,
        elapsed_seconds=900,
        api_cost_cny=12.5,
        log=log,
        extra_artifacts=[f"tool_cache={frozen_cache}"],
    )

    assert snapshot["config"]["seed"] == 42
    assert manifest["status"] == "failed"
    assert manifest["exit_code"] == 7
    assert manifest["usage"]["gpu_hours"] == 1.0
    assert manifest["artifacts"]["training_log"]["sha256"]
    assert manifest["artifacts"]["tool_cache"]["sha256"]
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["run_id"] == "r1-seed42"


def test_formal_snapshot_rejects_dirty_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(training_evidence, "_dirty_state", lambda: (True, "d" * 64, [" M"]))
    identity = tmp_path / "identity.json"
    identity.write_text("{}", encoding="utf-8")
    command = tmp_path / "command.txt"
    command.write_text("python train.py", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty"):
        training_evidence.create_snapshot(
            output=tmp_path / "snapshot.json",
            stage="sft",
            run_id="s0",
            protocol_label="formal",
            checkpoint_identity=identity,
            command_file=command,
            data=[],
            config=[],
            gpu_type="A100-80G",
            gpu_count=2,
        )


def test_snapshot_refuses_sensitive_config_keys(tmp_path, monkeypatch):
    _clean_git(monkeypatch)
    identity = tmp_path / "identity.json"
    identity.write_text("{}", encoding="utf-8")
    command = tmp_path / "command.txt"
    command.write_text("python train.py", encoding="utf-8")
    with pytest.raises(ValueError, match="sensitive"):
        training_evidence.create_snapshot(
            output=tmp_path / "snapshot.json",
            stage="rl",
            run_id="r0",
            protocol_label="formal",
            checkpoint_identity=identity,
            command_file=command,
            data=[],
            config=["API_KEY=do-not-store"],
            gpu_type="A100-80G",
            gpu_count=4,
        )


def test_token_budget_is_not_misclassified_as_a_credential():
    assert training_evidence._parse_pairs(
        ["assistant_token_budget=20000000", "max_tokens_per_turn=1024"],
        label="config",
    ) == {
        "assistant_token_budget": "20000000",
        "max_tokens_per_turn": "1024",
    }
