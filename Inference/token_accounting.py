"""Role-aware token accounting compatible with the multi-turn SFT template."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _encode_length(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def count_input_output_tokens(
    messages: Sequence[Mapping[str, Any]], tokenizer: Any
) -> tuple[int, int]:
    """Count condition tokens separately from assistant-generated tokens."""

    input_tokens = 0
    output_tokens = 0
    for index, message in enumerate(messages):
        role = message.get("role")
        previous = (
            tokenizer.apply_chat_template(
                messages[:index], tokenize=False, add_generation_prompt=False
            )
            if index
            else ""
        )
        current = tokenizer.apply_chat_template(
            messages[: index + 1], tokenize=False, add_generation_prompt=False
        )
        if not isinstance(previous, str) or not isinstance(current, str):
            raise ValueError("chat template must return text when tokenize=False")
        if role == "assistant":
            previous_with_prompt = tokenizer.apply_chat_template(
                messages[:index], tokenize=False, add_generation_prompt=True
            )
            if not isinstance(previous_with_prompt, str):
                raise ValueError("chat template must return text when tokenize=False")
            input_tokens += _encode_length(tokenizer, previous_with_prompt[len(previous) :])
            output_tokens += _encode_length(tokenizer, current[len(previous_with_prompt) :])
        else:
            input_tokens += _encode_length(tokenizer, current[len(previous) :])
    return input_tokens, output_tokens
