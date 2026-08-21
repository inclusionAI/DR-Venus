"""Shared monotonic deadline helpers for external tool requests."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time


class ToolTimeoutError(TimeoutError):
    """Raised when an external tool exhausts its end-to-end time budget."""


def is_timeout_error(exc: BaseException) -> bool:
    """Recognize stdlib, requests, and OpenAI timeout exception families."""

    name = type(exc).__name__.casefold()
    message = str(exc).casefold()
    return (
        isinstance(exc, TimeoutError)
        or "timeout" in name
        or "timed out" in message
    )


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
    """One deadline shared by all HTTP attempts within a tool invocation."""

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
        # Very small positive socket timeouts can be rejected by transports.
        return min(self.timeout_seconds, max(remaining, 0.001))

    def sleep(self, seconds: float) -> None:
        remaining = self.remaining()
        time.sleep(min(max(seconds, 0.0), remaining))
