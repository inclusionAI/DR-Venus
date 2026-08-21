import math

import pytest

from drvenus_repro.igpo_reference import (
    actor_policy_mask,
    apply_format_penalty,
    broadcast_turn_advantages,
    discount_turn_rewards,
    information_gain,
    normalize_group_rewards,
)


def test_three_turn_information_gain_and_discount_hand_calculation():
    rewards = [
        information_gain(-2.0, -3.0),
        information_gain(-1.25, -2.0),
        0.5,
    ]

    discounted, effective_gamma = discount_turn_rewards(rewards, gamma=0.5)

    assert effective_gamma == 0.5
    assert discounted == pytest.approx([1.5, 1.0, 0.5], abs=1e-6)


def test_turn_advantage_broadcast_keeps_zero_reward_boundary():
    discounted, _ = discount_turn_rewards([1.0, 0.0, 2.0], gamma=0.5)

    tokens = broadcast_turn_advantages([1, 3, 5], discounted, [1, 1, 1, 1, 1, 0])

    assert tokens == pytest.approx([1.5, 1.5, 1.0, 1.0, 2.0, 0.0], abs=1e-6)


def test_adaptive_gamma_preserves_requested_early_turn_weight():
    _, effective_gamma = discount_turn_rewards(
        [0.0, 0.0, 0.0], gamma=0.95, adaptive_min_weight=0.125
    )

    assert effective_gamma == pytest.approx(0.5, abs=1e-6)
    assert effective_gamma**3 == pytest.approx(0.125, abs=1e-6)


@pytest.mark.parametrize("mode", ["joint", "separate", "scaled_separate", "raw_ig"])
def test_all_normalization_modes_are_finite_for_zero_variance_groups(mode):
    result = normalize_group_rewards(
        [1.0, 1.0],
        [[0.25, 0.25], [0.25, 0.25]],
        mode=mode,
    )

    values = [*result.outcomes, *(value for row in result.information_gains for value in row)]
    assert all(math.isfinite(value) for value in values)
    if mode in {"separate", "scaled_separate"}:
        assert values == pytest.approx([0.0] * len(values), abs=1e-6)


def test_separate_normalization_matches_toy_batch():
    result = normalize_group_rewards(
        [0.0, 2.0],
        [[1.0], [3.0]],
        mode="separate",
    )

    assert result.outcomes == pytest.approx([-0.999999, 0.999999], abs=1e-6)
    assert result.information_gains[0] == pytest.approx([-0.999999], abs=1e-6)
    assert result.information_gains[1] == pytest.approx([0.999999], abs=1e-6)


def test_scaled_separate_balances_ig_and_outcome_magnitudes():
    result = normalize_group_rewards(
        [0.0, 2.0],
        [[-100.0], [100.0]],
        mode="scaled_separate",
    )

    outcome_abs_mean = sum(abs(value) for value in result.outcomes) / 2
    ig_values = [row[0] for row in result.information_gains]
    ig_abs_mean = sum(abs(value) for value in ig_values) / 2
    assert result.ig_scale is not None
    assert ig_abs_mean == pytest.approx(outcome_abs_mean, abs=2e-6)


def test_raw_ig_does_not_center_and_forces_zero_gamma():
    result = normalize_group_rewards(
        [0.0, 0.0],
        [[1.0], [3.0]],
        mode="raw_ig",
    )

    assert result.force_gamma_zero is True
    # IG/std remains positive; terminal GRPO credit is then added per sample.
    assert result.information_gains[0][0] > 0.0
    assert result.information_gains[1][0] > result.information_gains[0][0]


def test_response_mask_and_actor_loss_mask_have_different_roles():
    response = [1, 1, 1, 1, 1, 1]
    assistant = [1, 1, 0, 0, 1, 1]

    assert actor_policy_mask(response, assistant) == [1, 1, 0, 0, 1, 1]
    assert broadcast_turn_advantages([3, 5], [2.0, 1.0], response) == [2.0, 2.0, 2.0, 2.0, 1.0, 1.0]


def test_format_penalty_and_invalid_inputs():
    assert apply_format_penalty(0.8, valid=True, penalty=-0.25) == 0.8
    assert apply_format_penalty(0.8, valid=False, penalty=-0.25) == -0.25
    with pytest.raises(ValueError):
        apply_format_penalty(0.8, valid=False, penalty=0.25)
    with pytest.raises(ValueError):
        broadcast_turn_advantages([2, 1], [1.0, 1.0], [1, 1, 1])
