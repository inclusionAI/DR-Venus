"""Numerical parity check that runs in the full torch/verl training image."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "RL"))

try:
    from verl.trainer.ppo.info_gain_advantage import compute_grpo_info_gain_advantage
except ImportError as exc:  # pragma: no cover - exercised in the GPU image
    pytest.skip(f"full verl dependencies unavailable: {exc}", allow_module_level=True)


def test_zero_ig_explicit_boundary_matches_hand_calculation_within_1e6():
    rewards = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    response_mask = torch.ones_like(rewards)
    config = SimpleNamespace(
        gamma=0.5,
        info_gain_norm_mode="separate",
        norm_adv_by_std_in_grpo=True,
        adaptive_gamma_min_weight=0.0,
        ig_weight=1.0,
        use_curriculum=False,
        _ig_warmup_gate_off=False,
    )

    advantages, _, _ = compute_grpo_info_gain_advantage(
        rewards,
        response_mask,
        np.array(["question", "question"]),
        config=config,
        ig_boundary_positions=np.array([[1], [1]], dtype=object),
    )

    # Terminal GRPO normalization is approximately [+1, -1]. The explicit
    # zero-IG boundary discounts only the first two tokens by gamma=0.5.
    normalized = 0.5 / ((0.25 + 1e-8) ** 0.5 + 1e-6)
    expected = torch.tensor(
        [
            [0.5 * normalized, 0.5 * normalized, normalized, normalized],
            [-0.5 * normalized, -0.5 * normalized, -normalized, -normalized],
        ],
        dtype=advantages.dtype,
    )
    assert torch.max(torch.abs(advantages - expected)).item() <= 1e-6
