import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / "RL/scrl/llm_agent/turn_budget.py"
SPEC = importlib.util.spec_from_file_location("turn_budget_under_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_zero_jitter_makes_formal_turn_budget_exact():
    assert MODULE.resolve_turn_limit(12, jitter=0, random_value=lambda: 0.999) == 12


@pytest.mark.parametrize(
    "random_value,expected",
    [(0.0, 7), (0.1, 8), (0.5, 12), (0.999, 16)],
)
def test_default_five_window_preserves_upstream_rule(random_value, expected):
    assert MODULE.resolve_turn_limit(12, jitter=5, random_value=lambda: random_value) == expected


def test_turn_jitter_environment_is_validated(monkeypatch):
    monkeypatch.setenv("ROLLOUT_TURN_JITTER", "0")
    assert MODULE.turn_jitter_from_env() == 0
    monkeypatch.setenv("ROLLOUT_TURN_JITTER", "-1")
    with pytest.raises(ValueError):
        MODULE.turn_jitter_from_env()
    monkeypatch.setenv("ROLLOUT_TURN_JITTER", "not-an-int")
    with pytest.raises(ValueError):
        MODULE.turn_jitter_from_env()
