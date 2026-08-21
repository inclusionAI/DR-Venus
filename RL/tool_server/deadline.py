"""Monotonic end-to-end deadlines for RL external tool requests."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time


class ToolTimeoutError(TimeoutError):
    """Raised when one tool invocation exhausts its shared wall-clock budget."""


def is_timeout_error(exc: BaseException) -> bool:
    """Recognize stdlib, requests, httpx, and OpenAI timeout families."""

    name = type(exc).__name__.casefold()
    message = str(exc).casefold()
    return isinstance(exc, TimeoutError) or "timeout" in name or "timed out" in message


def positive_timeout(value: object, *, name: str = "tool timeout") -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return timeout


@dataclass(frozen=True)
class ToolDeadline:
    """One monotonic deadline shared by every HTTP attempt in a tool call."""

    expires_at: float
    timeout_seconds: float

    @classmethod
    def start(cls, timeout_seconds: object) -> "ToolDeadline":
        timeout = positive_timeout(timeout_seconds)
        return cls(time.monotonic() + timeout, timeout)

    def remaining(self) -> float:
        remaining = self.expires_at - time.monotonic()
        if remaining <= 0:
            raise ToolTimeoutError(
                f"tool execution timed out after {self.timeout_seconds:g} seconds"
            )
        # Some transports reject a zero-ish positive socket timeout.
        return max(remaining, 0.001)

    def sleep(self, seconds: float) -> None:
        remaining = self.remaining()
        time.sleep(min(max(seconds, 0.0), remaining))
