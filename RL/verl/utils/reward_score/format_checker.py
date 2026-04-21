"""
Turn-level format penalty for DeepResearcher agent outputs.

Expected formats:
  Non-answer turn: <think>...</think><tool_call>...</tool_call>
  Answer turn:     <think>...</think><answer>...</answer>

Returns a binary result: True (correct) or False (any error detected).
"""

import re

try:
    import json5 as _json_lib
except ImportError:
    import json as _json_lib

_THINK_OPEN = re.compile(r"<think>")
_THINK_CLOSE = re.compile(r"</think>")
_TC_OPEN = re.compile(r"<tool_call>")
_TC_CLOSE = re.compile(r"</tool_call>")
_ANS_OPEN = re.compile(r"<answer>")
_ANS_CLOSE = re.compile(r"</answer>")


def check_turn_format(content: str, is_final_turn: bool) -> bool:
    """Check whether a single assistant turn has correct format.

    Args:
        content: The raw assistant message content for this turn.
        is_final_turn: True if this is the last turn (should contain <answer>).

    Returns:
        True if format is correct, False if any error detected.
    """
    if not content or not content.strip():
        return False

    think_opens = [m.start() for m in _THINK_OPEN.finditer(content)]
    think_closes = [m.start() for m in _THINK_CLOSE.finditer(content)]
    tc_opens = [m.start() for m in _TC_OPEN.finditer(content)]
    tc_closes = [m.start() for m in _TC_CLOSE.finditer(content)]
    ans_opens = [m.start() for m in _ANS_OPEN.finditer(content)]
    ans_closes = [m.start() for m in _ANS_CLOSE.finditer(content)]

    # --- Think tags: must exist, properly paired, strictly sequential ---
    if not think_opens or len(think_opens) != len(think_closes):
        return False
    # No stray text before the first <think>
    if content[:think_opens[0]].strip():
        return False
    for i in range(len(think_opens)):
        if think_opens[i] >= think_closes[i]:
            return False
        if i > 0 and think_closes[i - 1] >= think_opens[i]:
            return False

    last_think_close = think_closes[-1]

    last_think_close_end = last_think_close + len("</think>")

    if is_final_turn:
        # --- Answer turn: <think>...</think><answer>...</answer> ---
        if tc_opens or tc_closes:
            return False
        if len(ans_opens) != 1 or len(ans_closes) != 1:
            return False
        if ans_opens[0] < last_think_close or ans_closes[0] < ans_opens[0]:
            return False

        # No stray text between </think> and <answer>
        if content[last_think_close_end:ans_opens[0]].strip():
            return False
        # No trailing text after </answer>
        if content[ans_closes[0] + len("</answer>"):].strip():
            return False
        # Answer body must not be empty
        if not content[ans_opens[0] + len("<answer>"):ans_closes[0]].strip():
            return False
    else:
        # --- Tool-call turn: <think>...</think><tool_call>...</tool_call> ---
        if ans_opens or ans_closes:
            return False
        if len(tc_opens) != 1 or len(tc_closes) != 1:
            return False
        if tc_opens[0] < last_think_close or tc_closes[0] < tc_opens[0]:
            return False

        # No stray text between </think> and <tool_call>
        if content[last_think_close_end:tc_opens[0]].strip():
            return False
        # No trailing text after </tool_call>
        if content[tc_closes[0] + len("</tool_call>"):].strip():
            return False

        tc_body = content[tc_opens[0] + len("<tool_call>"):tc_closes[0]].strip()
        if not tc_body:
            return False

        # PythonInterpreter special format: body contains <code>...</code>
        if ("pythoninterpreter" in tc_body.lower()
                and "<code>" in tc_body and "</code>" in tc_body):
            return True

        try:
            parsed = _json_lib.loads(tc_body)
            if not isinstance(parsed, dict):
                return False
            if "name" not in parsed or "arguments" not in parsed:
                return False
            if not isinstance(parsed["arguments"], dict):
                return False
        except Exception:
            return False

    return True
