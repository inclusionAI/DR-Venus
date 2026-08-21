from pathlib import Path

import pytest

from Inference.evidence_identity import sha256_artifact
from experiments.verify_checkpoint import verify_checkpoint


def test_remote_checkpoint_requires_exact_model_and_immutable_revision():
    revision = "a" * 40
    result = verify_checkpoint(
        "Qwen/model", expected_model_id="Qwen/model", revision=revision
    )
    assert result["revision"] == revision
    with pytest.raises(ValueError, match="formal remote"):
        verify_checkpoint("other", expected_model_id="Qwen/model", revision=revision)


def test_local_checkpoint_hash_is_verified(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    digest = sha256_artifact(checkpoint)
    result = verify_checkpoint(
        str(checkpoint),
        expected_model_id="Qwen/model",
        revision=digest,
        expected_sha256=digest,
    )
    assert result["sha256"] == digest
    with pytest.raises(ValueError, match="mismatch"):
        verify_checkpoint(
            str(checkpoint),
            expected_model_id="Qwen/model",
            revision=digest,
            expected_sha256="0" * 64,
        )
