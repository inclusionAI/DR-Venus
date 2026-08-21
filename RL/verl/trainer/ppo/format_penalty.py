"""Pure helpers for the production turn-format replacement penalty."""

from __future__ import annotations

import math


def replacement_format_penalty(scale: float) -> float:
    """Return the reward value that replaces an invalid turn's reward."""

    value = float(scale)
    if not math.isfinite(value) or value < 0:
        raise ValueError("format penalty scale must be finite and non-negative")
    return -value
