import importlib.util
from pathlib import Path

import pytest

from drvenus_repro.igpo_reference import apply_format_penalty


PATH = Path(__file__).resolve().parents[1] / "RL/verl/trainer/ppo/format_penalty.py"
SPEC = importlib.util.spec_from_file_location("production_format_penalty", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_reference_and_production_use_replacement_semantics():
    production_penalty = MODULE.replacement_format_penalty(0.25)
    assert production_penalty == -0.25
    assert apply_format_penalty(0.8, valid=False, penalty=production_penalty) == -0.25
    assert apply_format_penalty(0.8, valid=True, penalty=production_penalty) == 0.8


@pytest.mark.parametrize("scale", [-1, float("inf"), float("nan")])
def test_production_penalty_rejects_invalid_scale(scale):
    with pytest.raises(ValueError):
        MODULE.replacement_format_penalty(scale)
