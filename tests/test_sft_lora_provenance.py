import importlib.util
import json
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / "SFT/verl/model_merger/lora_provenance.py"
SPEC = importlib.util.spec_from_file_location("sft_lora_provenance", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _complete_config(**overrides):
    config = {
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "target_modules": {"q_proj", "v_proj"},
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "peft_type": "LORA",
        "inference_mode": False,
    }
    config.update(overrides)
    return config


def test_checkpoint_round_trip_preserves_nonzero_alpha_and_full_config(tmp_path: Path):
    checkpoint = tmp_path / "global_step_50"
    provenance_path = MODULE.write_lora_provenance(checkpoint, _complete_config())

    document = json.loads(provenance_path.read_text(encoding="utf-8"))
    generated = MODULE.load_lora_provenance(checkpoint)
    adapter_config_path = MODULE.write_adapter_config(tmp_path / "merged" / "lora_adapter", generated)
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))

    assert provenance_path.name == "lora_adapter_config.json"
    assert document["schema_version"] == 1
    assert generated["r"] == 32
    assert generated["lora_alpha"] == 64
    assert generated["lora_alpha"] > 0
    assert generated["lora_dropout"] == 0.05
    assert generated["target_modules"] == ["q_proj", "v_proj"]
    assert adapter_config_path.name == "adapter_config.json"
    assert adapter_config["lora_alpha"] == 64
    assert adapter_config["lora_alpha"] > 0


def test_lora_checkpoint_without_provenance_fails_closed(tmp_path: Path):
    checkpoint = tmp_path / "legacy" / "global_step_10"
    checkpoint.mkdir(parents=True)

    with pytest.raises(MODULE.LoraProvenanceError, match="cannot be recovered"):
        MODULE.load_lora_provenance(checkpoint)


def test_historical_checkpoint_requires_complete_explicit_config(tmp_path: Path):
    checkpoint = tmp_path / "legacy"
    checkpoint.mkdir()
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({"r": 32, "target_modules": ["q_proj"]}), encoding="utf-8")

    with pytest.raises(MODULE.LoraProvenanceError, match="lora_alpha"):
        MODULE.load_lora_provenance(checkpoint, explicit_config_path=incomplete)


def test_explicit_override_is_applied_but_rank_is_checked_against_weights(tmp_path: Path):
    checkpoint = tmp_path / "global_step_20"
    MODULE.write_lora_provenance(checkpoint, _complete_config())
    generated = MODULE.load_lora_provenance(checkpoint, overrides={"lora_alpha": 96})
    assert generated["lora_alpha"] == 96

    class TensorShape:
        shape = (16, 128)

    state_dict = {"base_model.model.q_proj.lora_A.default.weight": TensorShape()}
    with pytest.raises(MODULE.LoraProvenanceError, match="declares rank 32"):
        MODULE.validate_rank_matches_weights(generated, state_dict)
