import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / "RL/verl/trainer/ppo/info_gain_boundaries.py"
SPEC = importlib.util.spec_from_file_location("ig_boundaries_under_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_explicit_boundary_survives_zero_reward():
    rewards = [[0.0, 1.0]]
    mask = MODULE.boundary_rows([[1, 3]], batch_size=1, sequence_length=5)

    assert rewards[0][0] == 0.0
    assert mask == [[False, True, False, True, False]]


def test_duplicate_boundaries_are_idempotent():
    assert MODULE.boundary_rows([[2, 2]], batch_size=1, sequence_length=4) == [
        [False, False, True, False]
    ]


@pytest.mark.parametrize("positions", [[[-1]], [[3]], [[1.5]]])
def test_invalid_boundary_positions_fail_closed(positions):
    with pytest.raises(ValueError):
        MODULE.boundary_rows(positions, batch_size=1, sequence_length=3)
