"""Strict validation of unsanitized Agent protocol outputs."""

from __future__ import annotations

import re


_STRICT_OUTPUT_RE = re.compile(
    r"\A\s*<think>(?P<think>.+?)</think>\s*"
    r"(?:(?P<answer><answer>.+?</answer>)|"
    r"(?P<tool><tool_call>.+?</tool_call>))\s*\Z",
    re.DOTALL,
)


def strict_raw_format_valid(content: object) -> bool:
    """Accept exactly one think block followed by one non-empty action.

    This deliberately runs before any repair/sanitization.  Mixed actions,
    repeated tags, missing closing tags, and text outside the protocol are
    violations even when the runtime can recover from them.
    """

    if not isinstance(content, str):
        return False
    if content.count("<think>") != 1 or content.count("</think>") != 1:
        return False
    answer_pairs = (content.count("<answer>"), content.count("</answer>"))
    tool_pairs = (content.count("<tool_call>"), content.count("</tool_call>"))
    if not (
        (answer_pairs == (1, 1) and tool_pairs == (0, 0))
        or (answer_pairs == (0, 0) and tool_pairs == (1, 1))
    ):
        return False
    match = _STRICT_OUTPUT_RE.fullmatch(content)
    if match is None or not match.group("think").strip():
        return False
    action = match.group("answer") or match.group("tool") or ""
    inner = re.sub(r"\A<[^>]+>|</[^>]+>\Z", "", action, flags=re.DOTALL)
    return bool(inner.strip())
