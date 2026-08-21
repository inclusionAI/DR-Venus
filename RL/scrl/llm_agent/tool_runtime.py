"""Dependency-light synchronous tool execution for the async RL agent loop."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolExecutionResult:
    content: str
    failed: bool
    timed_out: bool


def _looks_like_timeout(exc: BaseException) -> bool:
    return (
        isinstance(exc, TimeoutError)
        or "timeout" in type(exc).__name__.casefold()
        or "timed out" in str(exc).casefold()
    )


def execute_tool_call(
    tool_call: Mapping[str, Any],
    dispatch: Callable[..., Any],
    *,
    timeout_seconds: float,
) -> ToolExecutionResult:
    """Pass the timeout to the provider-aware dispatcher and classify errors."""

    try:
        value = dispatch(dict(tool_call), timeout=timeout_seconds)
        if value is None or (isinstance(value, str) and not value.strip()):
            return ToolExecutionResult("Tool returned empty response", True, False)
        if isinstance(value, list):
            value = "\n".join(str(item) for item in value)
        return ToolExecutionResult(str(value), False, False)
    except Exception as exc:
        timed_out = _looks_like_timeout(exc)
        if timed_out:
            content = (
                f"Tool execution timed out after {timeout_seconds:.0f} seconds: {exc}"
            )
        else:
            content = f"Tool execution error: {exc!r}"
        return ToolExecutionResult(content, True, timed_out)
