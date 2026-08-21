import pytest
from pathlib import Path

from evaluation.verify_fixed_protocol import verify_fixed_protocol


def test_committed_core_and_pass4_contracts_are_locked():
    core = verify_fixed_protocol(
        "evaluation/data/eval-core-100.parquet",
        max_turns=12,
        context_length=32768,
        samples_per_question=1,
        seed=42,
        max_tokens_per_turn=1024,
    )
    pass4 = verify_fixed_protocol(
        "evaluation/data/pass4-20.parquet",
        max_turns=12,
        context_length=32768,
        samples_per_question=4,
        seed=42,
        max_tokens_per_turn=1024,
    )
    assert core["rows"] == 100
    assert pass4["rows"] == 20


def test_contract_rejects_parameter_drift():
    with pytest.raises(ValueError, match="samples_per_question"):
        verify_fixed_protocol(
            "evaluation/data/eval-core-100.parquet",
            max_turns=12,
            context_length=32768,
            samples_per_question=4,
            seed=42,
            max_tokens_per_turn=1024,
        )


def test_contract_rejects_self_asserted_copy(tmp_path):
    source = Path("evaluation/data/eval-core-100.parquet")
    sidecar = source.with_name(source.name + ".manifest.json")
    copied = tmp_path / source.name
    copied.write_bytes(source.read_bytes())
    copied.with_name(copied.name + ".manifest.json").write_bytes(sidecar.read_bytes())

    with pytest.raises(ValueError, match="committed cohort"):
        verify_fixed_protocol(
            copied,
            max_turns=12,
            context_length=32768,
            samples_per_question=1,
            seed=42,
            max_tokens_per_turn=1024,
        )
