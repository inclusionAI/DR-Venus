"""Small, dependency-free reference equations for studying IGPO.

This module is deliberately separate from the optimized ``verl`` implementation.
It is intended for hand calculations, table-driven tests, and interview demos; the
training path remains ``RL/verl/trainer/ppo/info_gain_advantage.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


VALID_NORMALIZATION_MODES = {"joint", "separate", "scaled_separate", "raw_ig"}


def _finite(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def information_gain(log_prob_after: float, log_prob_before: float) -> float:
    """Return the ground-truth log-probability gain caused by one visit turn."""

    after = _finite(log_prob_after, name="log_prob_after")
    before = _finite(log_prob_before, name="log_prob_before")
    return after - before


def discount_turn_rewards(
    rewards: Sequence[float],
    *,
    gamma: float = 0.95,
    adaptive_min_weight: float = 0.0,
) -> tuple[list[float], float]:
    """Reverse-discount turn rewards exactly as the training implementation does."""

    gamma = _finite(gamma, name="gamma")
    adaptive_min_weight = _finite(adaptive_min_weight, name="adaptive_min_weight")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not 0.0 <= adaptive_min_weight <= 1.0:
        raise ValueError("adaptive_min_weight must be in [0, 1]")

    values = [_finite(value, name="reward") for value in rewards]
    effective_gamma = gamma
    if adaptive_min_weight > 0.0 and len(values) > 1:
        effective_gamma = adaptive_min_weight ** (1.0 / len(values))

    discounted = [0.0] * len(values)
    next_advantage = 0.0
    for index in range(len(values) - 1, -1, -1):
        next_advantage = values[index] + effective_gamma * next_advantage
        discounted[index] = next_advantage
    return discounted, effective_gamma


def broadcast_turn_advantages(
    turn_end_positions: Sequence[int],
    advantages: Sequence[float],
    response_mask: Sequence[int | bool],
) -> list[float]:
    """Broadcast one advantage over each turn, then apply the response mask.

    A zero-valued turn reward still needs an explicit boundary. This is why callers
    pass boundary positions rather than asking this function to infer them from
    non-zero rewards.
    """

    if len(turn_end_positions) != len(advantages):
        raise ValueError("turn_end_positions and advantages must have equal length")
    if any(value not in (0, 1, False, True) for value in response_mask):
        raise ValueError("response_mask must be binary")

    result = [0.0] * len(response_mask)
    previous_end = -1
    for position, advantage in zip(turn_end_positions, advantages):
        if not isinstance(position, int) or position <= previous_end:
            raise ValueError("turn boundaries must be strictly increasing integers")
        if position >= len(response_mask):
            raise ValueError("turn boundary is outside response_mask")
        value = _finite(advantage, name="advantage")
        for token_index in range(previous_end + 1, position + 1):
            result[token_index] = value * int(bool(response_mask[token_index]))
        previous_end = position
    return result


def actor_policy_mask(
    response_mask: Sequence[int | bool], assistant_token_mask: Sequence[int | bool]
) -> list[int]:
    """Return the actor-loss mask; tool tokens may stay in response_mask only."""

    if len(response_mask) != len(assistant_token_mask):
        raise ValueError("response and assistant masks must have equal length")
    if any(value not in (0, 1, False, True) for value in (*response_mask, *assistant_token_mask)):
        raise ValueError("masks must be binary")
    return [int(bool(response) and bool(assistant)) for response, assistant in zip(response_mask, assistant_token_mask)]


def apply_format_penalty(outcome_reward: float, *, valid: bool, penalty: float) -> float:
    """Replace an invalid turn reward with the non-positive penalty value.

    This mirrors the production R2 path: the penalty is not added to IG or to
    the terminal outcome.  Replacement keeps malformed turns at a fixed value
    independent of the reward they would otherwise have received.
    """

    outcome = _finite(outcome_reward, name="outcome_reward")
    penalty = _finite(penalty, name="penalty")
    if penalty > 0.0:
        raise ValueError("format penalty must be non-positive")
    return outcome if valid else penalty


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    mean = sum(values) / len(values)
    if len(values) <= 1:
        return mean, 1.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    # Mirrors the optimized implementation's numerical floor.
    return mean, math.sqrt(variance + 1e-8)


def _standardize(values: Sequence[float], *, epsilon: float, center: bool = True) -> list[float]:
    mean, std = _mean_std(values)
    return [((value - mean) if center else value) / (std + epsilon) for value in values]


@dataclass(frozen=True)
class NormalizedGroup:
    """Normalized terminal and per-turn rewards for one rollout group."""

    outcomes: list[float]
    information_gains: list[list[float]]
    ig_scale: float | None = None
    force_gamma_zero: bool = False


def normalize_group_rewards(
    outcomes: Sequence[float],
    information_gains: Sequence[Sequence[float]],
    *,
    mode: str,
    epsilon: float = 1e-6,
    normalize_by_std: bool = True,
    ig_weight: float = 1.0,
) -> NormalizedGroup:
    """Readable reference for the four normalization modes used by DR-Venus.

    IG values are flattened across visit turns within a rollout group, matching
    ``compute_group_stats`` in the optimized implementation. Ragged trajectories
    are supported and restored to their original shape after normalization.
    """

    if mode not in VALID_NORMALIZATION_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_NORMALIZATION_MODES)}")
    if len(outcomes) != len(information_gains):
        raise ValueError("one IG sequence is required for each outcome")
    epsilon = _finite(epsilon, name="epsilon")
    ig_weight = _finite(ig_weight, name="ig_weight")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    outcome_values = [_finite(value, name="outcome") for value in outcomes]
    row_lengths = [len(row) for row in information_gains]
    flat_ig = [_finite(value, name="information_gain") for row in information_gains for value in row]

    def restore(values: Iterable[float]) -> list[list[float]]:
        flat_values = list(values)
        restored: list[list[float]] = []
        offset = 0
        for length in row_lengths:
            restored.append(flat_values[offset : offset + length])
            offset += length
        return restored

    def maybe_standardize(values: Sequence[float], *, center: bool = True) -> list[float]:
        if not values:
            return []
        if not normalize_by_std:
            mean = sum(values) / len(values) if center else 0.0
            return [value - mean for value in values]
        return _standardize(values, epsilon=epsilon, center=center)

    if mode == "joint":
        joint = [*outcome_values, *flat_ig]
        normalized = maybe_standardize(joint)
        normalized_outcomes = normalized[: len(outcome_values)]
        normalized_ig = normalized[len(outcome_values) :]
        return NormalizedGroup(normalized_outcomes, restore(value * ig_weight for value in normalized_ig))

    normalized_outcomes = maybe_standardize(outcome_values)

    if mode == "raw_ig":
        normalized_ig = [value * ig_weight for value in maybe_standardize(flat_ig, center=False)]
        rows = restore(normalized_ig)
        rows = [
            [turn_value + normalized_outcomes[sample_index] for turn_value in row]
            for sample_index, row in enumerate(rows)
        ]
        return NormalizedGroup(normalized_outcomes, rows, force_gamma_zero=True)

    normalized_ig = maybe_standardize(flat_ig)
    scale: float | None = None
    if mode == "scaled_separate":
        outcome_abs_mean = (
            sum(abs(value) for value in normalized_outcomes) / len(normalized_outcomes)
            if normalized_outcomes
            else 1.0
        )
        outcome_abs_mean = max(outcome_abs_mean, 0.3)
        ig_abs_mean = sum(abs(value) for value in normalized_ig) / len(normalized_ig) if normalized_ig else 1.0
        scale = min(outcome_abs_mean / (ig_abs_mean + epsilon), 10.0)
        normalized_ig = [value * scale for value in normalized_ig]

    return NormalizedGroup(
        normalized_outcomes,
        restore(value * ig_weight for value in normalized_ig),
        ig_scale=scale,
    )
