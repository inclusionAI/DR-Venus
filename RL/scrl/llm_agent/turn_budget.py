"""Turn-limit policy shared by synchronous and asynchronous rollouts."""

from __future__ import annotations

import os
import random
from typing import Callable


def turn_jitter_from_env(*, default: int = 5) -> int:
    raw = os.environ.get("ROLLOUT_TURN_JITTER", str(default))
    try:
        jitter = int(raw)
    except ValueError as exc:
        raise ValueError(f"ROLLOUT_TURN_JITTER must be an integer, got {raw!r}") from exc
    if jitter < 0:
        raise ValueError("ROLLOUT_TURN_JITTER must be non-negative")
    return jitter


def resolve_turn_limit(
    base_turns: int,
    *,
    jitter: int,
    random_value: Callable[[], float] = random.random,
) -> int:
    """Resolve a limit while preserving the upstream random-jitter rule.

    ``jitter=5`` reproduces ``int(random()*10 - 5)``. Budget-matched
    experiments set it to zero so the configured maximum is exact.
    """

    if base_turns < 1:
        raise ValueError("base_turns must be positive")
    if jitter < 0:
        raise ValueError("jitter must be non-negative")
    if jitter == 0:
        return base_turns
    unit = float(random_value())
    if not 0.0 <= unit < 1.0:
        raise ValueError("random_value must return a value in [0, 1)")
    offset = int(unit * (2 * jitter) - jitter)
    return max(1, base_turns + offset)
