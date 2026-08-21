"""Dependency-light tool error classification used by batch inference."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

try:
    from drvenus_repro.tool_cache import CachedToolExecutionError, ToolCacheMissError
except ModuleNotFoundError as exc:
    if exc.name not in {"drvenus_repro", "drvenus_repro.tool_cache"}:
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from drvenus_repro.tool_cache import CachedToolExecutionError, ToolCacheMissError


@dataclass(frozen=True)
class ToolExecutionResult:
    content: str
    failed: bool
    timed_out: bool
    fatal_replay_error: bool = False
    cache_miss: bool = False
    cached_error: bool = False


class ReplayToolEvidenceError(RuntimeError):
    """A frozen replay cache cannot satisfy the requested invocation."""

    def __init__(self, result: ToolExecutionResult):
        super().__init__(result.content)
        self.cache_miss = result.cache_miss
        self.cached_error = result.cached_error


def _looks_like_timeout(exc: BaseException) -> bool:
    return isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.casefold() or "timed out" in str(exc).casefold()


def execute_tool_call(
    tool_call: Mapping[str, Any],
    dispatch: Callable[..., Any],
    *,
    timeout_seconds: float,
) -> ToolExecutionResult:
    """Call a synchronous dispatcher and convert failures into Agent feedback.

    The dispatcher receives the timeout and is responsible for applying it to
    its underlying HTTP clients.  No wrapper thread is created, so a timed-out
    request cannot leave a non-daemon worker that blocks interpreter shutdown.
    """

    try:
        result = dispatch(dict(tool_call), timeout=timeout_seconds)
        if result is None or (isinstance(result, str) and not result.strip()):
            return ToolExecutionResult("ERROR: empty tool result", True, False)
        if isinstance(result, list):
            result = "\n".join(str(item) for item in result)
        return ToolExecutionResult(str(result), False, False)
    except Exception as exc:  # Agent recovery must preserve the trajectory.
        timed_out = _looks_like_timeout(exc)
        cache_miss = isinstance(exc, ToolCacheMissError)
        cached_error = isinstance(exc, CachedToolExecutionError)
        category = "tool timeout" if timed_out else "tool execution failed"
        return ToolExecutionResult(
            f"ERROR: {category}: {exc}",
            True,
            timed_out,
            fatal_replay_error=cache_miss or cached_error,
            cache_miss=cache_miss,
            cached_error=cached_error,
        )
