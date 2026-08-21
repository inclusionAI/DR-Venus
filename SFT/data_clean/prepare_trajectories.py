#!/usr/bin/env python3
"""One-click trajectory cleaning pipeline.

Stages:
1. Raw trajectory normalization (legacy red_clean behavior, with the improved
   search-result parser from the newer version).
2. Duplicate search/visit tool-call removal (legacy post_clean behavior).
3. Conversation validation.
4. Optional correctness filtering from a field, offline cache, or explicit
   OpenAI-compatible judge mode.
5. Deterministic question-level train/validation/test splitting.
6. Length-based resampling of the training split only.

Supported input schemas:
- raw: columns include `meta` and `messages`
- clean: columns include `question`, `messages`, and `gt`
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import json5
import numpy as np
import pandas as pd

try:
    from .data_governance import (
        DEFAULT_JUDGE_PROMPT_VERSION,
        SPLIT_ALGORITHM,
        CorrectnessEvaluator,
        OpenAICompatibleJudge,
        api_key_from_env,
        logical_records_sha256,
        sha256_file,
        split_records_by_question,
    )
except ImportError:  # Support direct execution: python prepare_trajectories.py
    from data_governance import (  # type: ignore[no-redef]
        DEFAULT_JUDGE_PROMPT_VERSION,
        SPLIT_ALGORITHM,
        CorrectnessEvaluator,
        OpenAICompatibleJudge,
        api_key_from_env,
        logical_records_sha256,
        sha256_file,
        split_records_by_question,
    )


DEFAULT_SYSTEM_PROMPT = """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For each user request, you must actively seek out and **cross-check information** from credible and diverse sources, then integrate the findings into a response that is comprehensive, accurate, well-structured, and objective. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer in `<answer></answer>` tags.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "search", "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "visit", "description": "Visit webpage(s) and return the summary of the content.", "parameters": {"type": "object", "properties": {"url": {"type": "array", "items": {"type": "string"}, "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."}, "goal": {"type": "string", "description": "The specific information goal for visiting webpage(s)."}}, "required": ["url", "goal"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
"""

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
TOOL_DEDUP_NAMES = {"search", "visit"}
DEFAULT_ALLOWED_TOOLS = ("search", "visit")

SEARCH_HEADER_RE = re.compile(r"^\s*A Google search for (.+?) found \d+ results:\s*$")
RESULT_START_RE = re.compile(r"^\s*\d+\.\s+\[")
DATE_METADATA_RE = re.compile(r"^\s*Date\s+(published|updated)\s*:\s*", re.IGNORECASE)


def fix_surrogates(text: Any) -> Any:
    if isinstance(text, str):
        try:
            return text.encode("utf-16-le", "surrogatepass").decode("utf-16-le")
        except Exception:
            return text.encode("utf-8", "ignore").decode("utf-8")
    return text


def deep_fix(obj: Any) -> Any:
    if isinstance(obj, str):
        return fix_surrogates(obj)
    if isinstance(obj, list):
        return [deep_fix(item) for item in obj]
    if isinstance(obj, dict):
        return {key: deep_fix(value) for key, value in obj.items()}
    return obj


def to_list(messages: Any) -> Any:
    if isinstance(messages, np.ndarray):
        messages = messages.tolist()
    if isinstance(messages, list):
        return [
            {key: value for key, value in message.items()} if isinstance(message, dict) else message
            for message in messages
        ]
    return messages


def normalize_str_list(value: Any) -> list[str]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if value is None:
        return []
    raw = value if isinstance(value, list) else [value]
    result = []
    for item in raw:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            result.append(text)
    return result


def build_system_prompt(current_date: str | None) -> str:
    base_prompt = DEFAULT_SYSTEM_PROMPT.rstrip("\n")
    if not current_date:
        return base_prompt
    return f"{base_prompt}\n\nCurrent date: {current_date}\n"


def choose_system_prompt(
    row: dict[str, Any],
    *,
    system_prompt_mode: str,
    template_system_prompt: str,
) -> str:
    if system_prompt_mode == "template":
        return template_system_prompt
    if system_prompt_mode == "row":
        row_prompt = row.get("system_prompt")
        if isinstance(row_prompt, str) and row_prompt.strip():
            return row_prompt
        messages = to_list(row.get("messages"))
        if isinstance(messages, list) and messages:
            first = messages[0]
            if isinstance(first, dict) and first.get("role") == "system":
                content = first.get("content")
                if isinstance(content, str) and content.strip():
                    return content
        return template_system_prompt
    raise ValueError(f"Unsupported system_prompt_mode: {system_prompt_mode}")


def _extract_query_from_header(header_line: str) -> str:
    match = SEARCH_HEADER_RE.match(header_line.strip())
    if not match:
        return ""
    raw_query = match.group(1).strip()
    if len(raw_query) >= 2 and raw_query[0] == raw_query[-1] and raw_query[0] in {"'", '"'}:
        return raw_query[1:-1]
    return raw_query


def _split_search_blocks(text: str) -> list[tuple[str, list[str]]]:
    blocks = []
    current_header = None
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        if SEARCH_HEADER_RE.match(line.strip()):
            if current_header is not None:
                blocks.append((current_header, current_lines))
            current_header = line.strip()
            current_lines = []
        elif current_header is not None:
            current_lines.append(line)

    if current_header is not None:
        blocks.append((current_header, current_lines))
    return blocks


def _parse_markdown_link(line: str) -> tuple[str | None, str | None, str | None]:
    line = line.strip()
    prefix_match = re.match(r"^\s*\d+\.\s+", line)
    if not prefix_match:
        return None, None, None

    idx = prefix_match.end()
    if idx >= len(line) or line[idx] != "[":
        return None, None, None

    idx += 1
    title_chars: list[str] = []
    bracket_depth = 1
    while idx < len(line):
        char = line[idx]
        if char == "[":
            bracket_depth += 1
            title_chars.append(char)
        elif char == "]":
            bracket_depth -= 1
            if bracket_depth == 0:
                if idx + 1 >= len(line) or line[idx + 1] != "(":
                    return None, None, None
                idx += 2
                break
            title_chars.append(char)
        else:
            title_chars.append(char)
        idx += 1
    else:
        return None, None, None

    url_chars: list[str] = []
    paren_depth = 1
    while idx < len(line):
        char = line[idx]
        if char == "(":
            paren_depth += 1
            url_chars.append(char)
        elif char == ")":
            paren_depth -= 1
            if paren_depth == 0:
                idx += 1
                break
            url_chars.append(char)
        else:
            url_chars.append(char)
        idx += 1
    else:
        return None, None, None

    title = "".join(title_chars).strip()
    url = "".join(url_chars).strip()
    trailing = line[idx:].strip()
    return title, url, trailing


def _strip_leading_metadata(lines: list[str]) -> list[str]:
    cleaned = list(lines)
    while cleaned and DATE_METADATA_RE.match(cleaned[0].strip()):
        cleaned.pop(0)
    return cleaned


def _parse_results_from_lines(lines: list[str]) -> list[dict[str, str]]:
    content_lines = list(lines)
    for idx, raw_line in enumerate(content_lines):
        if raw_line.strip() == "## Web Results":
            content_lines = content_lines[idx + 1 :]
            break

    entries = []
    current_entry = None
    for raw_line in content_lines:
        stripped = raw_line.strip()
        if RESULT_START_RE.match(stripped):
            if current_entry is not None:
                entries.append(current_entry)
            current_entry = {"header": stripped, "body_lines": []}
            continue
        if current_entry is not None:
            current_entry["body_lines"].append(raw_line.rstrip())

    if current_entry is not None:
        entries.append(current_entry)

    results = []
    for entry in entries:
        title, url, trailing = _parse_markdown_link(entry["header"])
        if not title or not url:
            continue

        body_lines = []
        if trailing:
            body_lines.append(trailing)
        body_lines.extend(line.strip() for line in entry["body_lines"])

        while body_lines and not body_lines[0]:
            body_lines.pop(0)
        while body_lines and not body_lines[-1]:
            body_lines.pop()

        body_lines = _strip_leading_metadata(body_lines)
        snippet = "\n".join(line for line in body_lines if line).strip()
        results.append({"title": title, "url": url, "snippet": snippet})
    return results


def parse_search_results(text: str) -> str:
    text = text.replace("<tool_response>", "").replace("</tool_response>", "").strip()
    blocks = _split_search_blocks(text)
    if not blocks:
        blocks = [("", text.splitlines())]

    parsed_blocks = []
    for header, lines in blocks:
        parsed_blocks.append(
            {
                "query": _extract_query_from_header(header) if header else "",
                "web_page_info_list": _parse_results_from_lines(lines),
            }
        )
    return json.dumps(parsed_blocks, ensure_ascii=False)


def normalize_tool_call(tool_call: dict[str, Any], raw_content: str) -> dict[str, Any] | str | None:
    name = tool_call.get("name")
    arguments = tool_call.get("arguments")

    if name == "search":
        if not isinstance(arguments, dict):
            return None
        query = arguments.get("query", arguments.get("queries"))
        if isinstance(query, str):
            query = [query]
        elif not isinstance(query, list):
            return None
        return {"name": "search", "arguments": {"query": query}}

    if name == "visit":
        if not isinstance(arguments, dict):
            return None
        goal = arguments.get("goal")
        if isinstance(goal, list):
            goal = goal[0] if goal else ""
        elif not isinstance(goal, str):
            return None

        urls = arguments.get("url", arguments.get("urls"))
        if isinstance(urls, str):
            urls = [urls]
        elif not isinstance(urls, list):
            return None

        return {"name": "visit", "arguments": {"url": urls, "goal": goal}}

    if name == "PythonInterpreter":
        if isinstance(arguments, dict) and not arguments:
            return raw_content
        return None

    if name and isinstance(arguments, dict):
        return {"name": name, "arguments": arguments}

    return None


def clean_toolcall_output(text: Any) -> str | None:
    if not isinstance(text, str):
        return None

    think_match = THINK_RE.search(text)
    if not think_match:
        return None
    think_content = think_match.group(1).strip()
    think_content = think_content.replace("<answer>", "").replace("</answer>", "")
    think_content = think_content.replace("<summary>", "").replace("</summary>", "")
    if not think_content:
        return None

    tool_match = TOOL_RE.search(text)
    if not tool_match:
        return None
    raw_tool_call = tool_match.group(1).strip()

    try:
        tool_call_content = raw_tool_call.split("<code>")[0].strip()
        tool_call = json5.loads(tool_call_content)
    except Exception:
        return None
    if not isinstance(tool_call, dict):
        return None

    normalized = normalize_tool_call(tool_call, raw_tool_call)
    if normalized is None:
        return None
    if isinstance(normalized, str):
        new_content = normalized
    else:
        new_content = json.dumps(normalized, ensure_ascii=False)

    return f"<think>\n{think_content}\n</think>\n<tool_call>\n{new_content}\n</tool_call>"


def clean_answer_output(text: Any) -> str | None:
    if not isinstance(text, str):
        return None

    answer_matches = list(ANSWER_RE.finditer(text))
    if not answer_matches:
        return None
    answer_content = answer_matches[-1].group(1).strip()

    think_match = THINK_RE.search(text)
    if not think_match:
        return None
    think_content = think_match.group(1).strip()
    think_content = re.sub(r"<summary>(.*?)</summary>", r"\1", think_content, flags=re.DOTALL | re.IGNORECASE)
    think_content = re.sub(r"<answer>(.*?)</answer>", r"\1", think_content, flags=re.DOTALL | re.IGNORECASE)
    think_content = think_content.replace("<answer>", "").replace("</answer>", "")
    think_content = think_content.replace("<summary>", "").replace("</summary>", "")

    external_summaries = [
        match.group(1).strip()
        for match in re.finditer(r"<summary>(.*?)</summary>", text.split("</think>")[-1], re.DOTALL | re.IGNORECASE)
    ]

    think_parts = []
    if think_content:
        think_parts.append(think_content)
    if external_summaries:
        think_parts.extend(external_summaries)
    final_think_content = "\n\n".join(think_parts)

    return f"<think>\n{final_think_content}\n</think>\n<answer>\n{answer_content}\n</answer>"


def normalize_raw_messages(messages: Any, system_prompt: str) -> list[dict[str, str]] | None:
    raw_messages = to_list(messages)
    if not isinstance(raw_messages, list) or not raw_messages:
        return None

    new_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for index, turn in enumerate(raw_messages):
        if not isinstance(turn, dict):
            return None
        content = turn.get("content")
        role = turn.get("role")
        if not isinstance(content, str):
            return None

        if index == 0:
            new_messages.append({"role": "user", "content": content})
            continue

        if index == len(raw_messages) - 1:
            cleaned = clean_answer_output(content)
            if cleaned is None:
                return None
            new_messages.append({"role": "assistant", "content": cleaned})
            continue

        if role == "user":
            if new_messages[-1]["role"] != "assistant":
                continue
            content = content.replace("<tool_response>", "").replace("</tool_response>", "").strip()
            if "Web Results" in content:
                new_content = f"<tool_response>\n{parse_search_results(content)}\n</tool_response>"
            else:
                new_content = f"<tool_response>\n{content}\n</tool_response>"
            new_messages.append({"role": "user", "content": new_content})
            continue

        if role == "assistant":
            cleaned = clean_toolcall_output(content)
            if cleaned is None:
                continue
            new_messages.append({"role": "assistant", "content": cleaned})
            continue

        return None

    return deep_fix(new_messages)


def normalize_clean_messages(
    messages: Any,
    *,
    system_prompt_mode: str,
    system_prompt: str,
) -> list[dict[str, str]] | None:
    clean_messages = to_list(messages)
    if not isinstance(clean_messages, list) or not clean_messages:
        return None
    clean_messages = deep_fix(clean_messages)

    if not isinstance(clean_messages[0], dict):
        return None

    if clean_messages[0].get("role") == "system":
        if system_prompt_mode == "template":
            clean_messages[0]["content"] = system_prompt
        return clean_messages

    return [{"role": "system", "content": system_prompt}] + clean_messages


def extract_question_gt(row: dict[str, Any]) -> tuple[str | None, str | None]:
    meta = row.get("meta")
    if isinstance(meta, dict):
        question = meta.get("question")
        answer = meta.get("answer")
        return question if isinstance(question, str) else None, answer if isinstance(answer, str) else None

    question = row.get("question")
    answer = row.get("gt")
    return question if isinstance(question, str) else None, answer if isinstance(answer, str) else None


def normalize_row(
    row: dict[str, Any],
    *,
    system_prompt_mode: str,
    template_system_prompt: str,
) -> tuple[dict[str, Any] | None, str | None]:
    question, gt = extract_question_gt(row)
    if not question or not gt:
        return None, "missing_question_or_gt"

    system_prompt = choose_system_prompt(
        row,
        system_prompt_mode=system_prompt_mode,
        template_system_prompt=template_system_prompt,
    )

    if isinstance(row.get("meta"), dict):
        messages = normalize_raw_messages(row.get("messages"), system_prompt)
        if messages is None:
            return None, "raw_clean_failed"
    else:
        messages = normalize_clean_messages(
            row.get("messages"),
            system_prompt_mode=system_prompt_mode,
            system_prompt=system_prompt,
        )
        if messages is None:
            return None, "clean_input_invalid"

    return {"question": question, "messages": messages, "gt": gt}, None


def parse_tool_calls(content: str) -> list[dict[str, Any]]:
    tool_calls = []
    for match in TOOL_RE.finditer(content or ""):
        raw = match.group(1).strip().split("<code>")[0].strip()
        try:
            tool_call = json.loads(raw)
        except Exception:
            try:
                tool_call = json5.loads(raw)
            except Exception:
                continue
        if isinstance(tool_call, dict):
            tool_calls.append(tool_call)
    return tool_calls


def extract_tool_names(content: str) -> list[str]:
    names = []
    for tool_call in parse_tool_calls(content):
        name = tool_call.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def prune_disallowed_tool_turns(
    messages: list[dict[str, str]],
    allowed_tools: set[str],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    pruned_info = []
    new_messages = []
    idx = 0

    while idx < len(messages):
        message = messages[idx]
        role = message.get("role", "")
        content = message.get("content", "")

        if role != "assistant":
            new_messages.append(message)
            idx += 1
            continue

        tool_names = extract_tool_names(content)
        if not tool_names:
            new_messages.append(message)
            idx += 1
            continue

        disallowed_names = [name for name in tool_names if name.lower() not in allowed_tools]
        if not disallowed_names:
            new_messages.append(message)
            idx += 1
            continue

        info = {
            "assistant_position": idx,
            "dropped_tools": disallowed_names,
            "tool_response_positions": [],
        }

        idx += 1
        while idx < len(messages):
            next_message = messages[idx]
            if next_message.get("role") == "user" and "<tool_response>" in (next_message.get("content", "") or ""):
                info["tool_response_positions"].append(idx)
                idx += 1
            else:
                break
        pruned_info.append(info)

    return new_messages, pruned_info


def canonical_signature(tool_call: dict[str, Any]) -> tuple[str, tuple[str, ...]] | None:
    name = tool_call.get("name")
    arguments = tool_call.get("arguments", {})
    if name not in TOOL_DEDUP_NAMES:
        return None

    if name == "search":
        if isinstance(arguments, dict):
            values = normalize_str_list(arguments.get("query", arguments.get("queries")))
        else:
            values = normalize_str_list(arguments)
        values = tuple(sorted(set(values)))
        return name, values

    if name == "visit":
        collected: list[str] = []
        if isinstance(arguments, dict):
            for key in ("url", "urls", "uri", "uris", "link", "links", "page", "pages", "query", "queries"):
                if key in arguments:
                    collected.extend(normalize_str_list(arguments.get(key)))
        else:
            collected.extend(normalize_str_list(arguments))
        values = tuple(sorted(set(collected)))
        return name, values

    return None


def remove_duplicate_tool_calls(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    duplicate_info = []
    new_messages = []
    seen_signatures = set()
    idx = 0

    while idx < len(messages):
        message = messages[idx]
        role = message.get("role", "")
        content = message.get("content", "")

        if role != "assistant":
            new_messages.append(message)
            idx += 1
            continue

        tool_calls = parse_tool_calls(content)
        signatures = [sig for sig in (canonical_signature(tc) for tc in tool_calls) if sig is not None]
        if not signatures:
            new_messages.append(message)
            idx += 1
            continue

        if all(signature in seen_signatures for signature in signatures):
            info = {
                "assistant_position": idx,
                "duplicate_queries": [{"tool": sig[0], "values": list(sig[1])} for sig in signatures],
                "tool_response_positions": [],
            }
            idx += 1
            while idx < len(messages):
                next_message = messages[idx]
                if next_message.get("role") == "user" and "<tool_response>" in (next_message.get("content", "") or ""):
                    info["tool_response_positions"].append(idx)
                    idx += 1
                else:
                    break
            duplicate_info.append(info)
            continue

        for signature in signatures:
            seen_signatures.add(signature)
        new_messages.append(message)
        idx += 1

    return new_messages, duplicate_info


def validate_assistant_message(content: str, is_final: bool) -> tuple[bool, str]:
    if not THINK_RE.search(content):
        return False, "assistant_missing_think"
    has_answer = ANSWER_RE.search(content) is not None
    has_tool = TOOL_RE.search(content) is not None
    if is_final:
        if not has_answer:
            return False, "final_assistant_missing_answer"
        if has_tool:
            return False, "final_assistant_contains_tool"
        return True, "ok"

    if not has_tool:
        return False, "assistant_missing_tool_call"
    if has_answer:
        return False, "assistant_tool_turn_contains_answer"
    return True, "ok"


def validate_messages(messages: Any) -> tuple[bool, str]:
    messages = to_list(messages)
    if not isinstance(messages, list) or len(messages) < 3:
        return False, "too_short"
    if not isinstance(messages[0], dict) or messages[0].get("role") != "system":
        return False, "bad_role_start"

    for idx, message in enumerate(messages[1:], start=1):
        if not isinstance(message, dict):
            return False, "non_dict_message"
        expected_role = "user" if idx % 2 == 1 else "assistant"
        if message.get("role") != expected_role:
            return False, f"bad_role_at_{idx}"

        content = message.get("content", "")
        if not isinstance(content, str):
            return False, f"non_str_content_at_{idx}"

        if expected_role == "user":
            if idx == 1:
                if "<tool_response>" in content or "</tool_response>" in content:
                    return False, "first_user_contains_tool_response"
            else:
                if "<tool_response>" not in content or "</tool_response>" not in content:
                    return False, f"user_missing_tool_response_at_{idx}"
        else:
            is_final = idx == len(messages) - 1
            ok, reason = validate_assistant_message(content, is_final=is_final)
            if not ok:
                return False, reason

    return True, "ok"


def compute_turns(messages: list[dict[str, str]]) -> int:
    return max(len(messages) - 1, 0) // 2


def compute_multiplier(turns: int, short_max: int, medium_max: int, short_mult: int, medium_mult: int, long_mult: int) -> int:
    if turns <= short_max:
        return short_mult
    if turns <= medium_max:
        return medium_mult
    return long_mult


def list_input_files(input_path: Path, file_pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(path for path in input_path.glob(file_pattern) if path.is_file())


def write_jsonl(df: pd.DataFrame, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for record in df.to_dict("records"):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean, correctness-filter, split, and train-only-resample trajectory data."
    )
    parser.add_argument("--input", "-i", required=True, help="Input parquet file or directory.")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory.")
    parser.add_argument("--file-pattern", default="train-*.parquet", help="Glob used when --input is a directory.")
    parser.add_argument(
        "--system-prompt-mode",
        choices=("template", "row"),
        default="template",
        help="Use the bundled prompt template or preserve the prompt from each input row when available.",
    )
    parser.add_argument("--current-date", default=None, help="Optional date appended to the bundled system prompt.")
    parser.add_argument("--short-max-turns", type=int, default=50, help="Max turn count for the short bucket.")
    parser.add_argument("--medium-max-turns", type=int, default=100, help="Max turn count for the medium bucket.")
    parser.add_argument("--short-multiplier", type=int, default=1, help="Resampling multiplier for turns <= short-max-turns.")
    parser.add_argument("--medium-multiplier", type=int, default=2, help="Resampling multiplier for short-max-turns < turns <= medium-max-turns.")
    parser.add_argument("--long-multiplier", type=int, default=5, help="Resampling multiplier for turns > medium-max-turns.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle each output split deterministically.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for question-level splitting and shuffling.")
    parser.add_argument("--train-ratio", type=float, default=0.90, help="Question-group ratio assigned to train.")
    parser.add_argument(
        "--validation-ratio", type=float, default=0.05, help="Question-group ratio assigned to validation."
    )
    parser.add_argument("--test-ratio", type=float, default=0.05, help="Question-group ratio assigned to test.")
    parser.add_argument("--keep-intermediate", action="store_true", help="Write stage outputs before and after deduplication.")
    parser.add_argument("--keep-metadata", action="store_true", help="Keep debug metadata columns in the final output.")
    parser.add_argument("--write-jsonl", action="store_true", help="Also export the final dataset as JSONL.")
    parser.add_argument(
        "--allowed-tools",
        nargs="+",
        default=list(DEFAULT_ALLOWED_TOOLS),
        help="Only these tool names are kept. Disallowed tool turns are removed together with their tool responses.",
    )
    parser.add_argument(
        "--correctness-mode",
        choices=("none", "field", "cache", "judge"),
        default="none",
        help=(
            "Correctness source. 'none' preserves legacy inclusion and never accesses the network; 'field' reads "
            "an input column; 'cache' is strictly offline; 'judge' explicitly enables an OpenAI-compatible API."
        ),
    )
    parser.add_argument(
        "--correctness-field",
        default="correctness",
        help="Input field used by field mode; dotted paths such as meta.correctness are supported.",
    )
    parser.add_argument("--correctness-threshold", type=float, default=0.5, help="Minimum score retained.")
    parser.add_argument(
        "--judge-cache",
        default=None,
        help="JSON/JSONL correctness cache. Required by cache mode; read-through/write-through in judge mode.",
    )
    parser.add_argument(
        "--judge-base-url",
        default=None,
        help="OpenAI-compatible /v1 base URL or full chat/completions endpoint; used only by judge mode.",
    )
    parser.add_argument("--judge-model", default=None, help="Judge model name; required by judge mode.")
    parser.add_argument(
        "--judge-api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the judge API key. Literal keys are intentionally unsupported.",
    )
    parser.add_argument("--judge-timeout", type=float, default=60.0, help="Per-attempt judge timeout in seconds.")
    parser.add_argument("--judge-retries", type=int, default=2, help="Judge retries after the first failed attempt.")
    return parser


def _records_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    dataframe = pd.DataFrame(records)
    for column in ("question", "messages", "gt"):
        if column not in dataframe.columns:
            dataframe[column] = pd.Series(dtype="object")
    if "messages" in dataframe.columns:
        dataframe["messages"] = dataframe["messages"].apply(deep_fix)
    return dataframe


def _public_output(dataframe: pd.DataFrame, *, keep_metadata: bool) -> pd.DataFrame:
    if keep_metadata:
        return dataframe.copy()
    return dataframe[["question", "messages", "gt"]].copy()


def _resample_training_records(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resampled_rows = []
    original_bucket_counts: defaultdict[str, int] = defaultdict(int)
    resampled_bucket_counts: defaultdict[str, int] = defaultdict(int)
    multiplier_counts: Counter[str] = Counter()

    for record in records:
        turns = int(record["_turns"])
        multiplier = compute_multiplier(
            turns,
            args.short_max_turns,
            args.medium_max_turns,
            args.short_multiplier,
            args.medium_multiplier,
            args.long_multiplier,
        )
        if multiplier < 1:
            raise ValueError("resampling multipliers must all be at least 1")

        if turns <= args.short_max_turns:
            bucket = f"0-{args.short_max_turns}"
        elif turns <= args.medium_max_turns:
            bucket = f"{args.short_max_turns + 1}-{args.medium_max_turns}"
        else:
            bucket = f">{args.medium_max_turns}"

        original_bucket_counts[bucket] += 1
        multiplier_counts[str(multiplier)] += 1
        for sample_copy in range(multiplier):
            new_record = dict(record)
            new_record["_resample_multiplier"] = multiplier
            new_record["_sample_copy"] = sample_copy
            resampled_rows.append(new_record)
            resampled_bucket_counts[bucket] += 1

    return resampled_rows, {
        "turn_buckets": {
            "original": dict(original_bucket_counts),
            "resampled": dict(resampled_bucket_counts),
        },
        "multiplier_distribution": dict(multiplier_counts),
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Run the pipeline from a parsed namespace and return stats for tests/callers."""

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed_tools = {tool.lower() for tool in args.allowed_tools}
    if args.short_max_turns < 0 or args.medium_max_turns < args.short_max_turns:
        raise ValueError("turn thresholds must satisfy 0 <= short-max-turns <= medium-max-turns")
    if min(args.short_multiplier, args.medium_multiplier, args.long_multiplier) < 1:
        raise ValueError("resampling multipliers must all be at least 1")
    ratio_sum = args.train_ratio + args.validation_ratio + args.test_ratio
    if min(args.train_ratio, args.validation_ratio, args.test_ratio) < 0 or abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(f"split ratios must be non-negative and sum to 1.0, got {ratio_sum}")

    files = list_input_files(input_path, args.file_pattern)
    if not files:
        raise FileNotFoundError(f"No input files found under {input_path} with pattern {args.file_pattern!r}")

    cache_path = Path(args.judge_cache).expanduser().resolve() if args.judge_cache else None
    judge = None
    if args.correctness_mode == "judge":
        if not args.judge_base_url or not args.judge_model:
            raise ValueError("judge mode requires both --judge-base-url and --judge-model")
        judge = OpenAICompatibleJudge(
            base_url=args.judge_base_url,
            model=args.judge_model,
            api_key=api_key_from_env(args.judge_api_key_env),
            timeout=args.judge_timeout,
            retries=args.judge_retries,
        )
    correctness_evaluator = CorrectnessEvaluator(
        mode=args.correctness_mode,
        field=args.correctness_field,
        threshold=args.correctness_threshold,
        cache_path=cache_path,
        prompt_version=DEFAULT_JUDGE_PROMPT_VERSION,
        judge=judge,
        judge_model=args.judge_model,
    )

    template_system_prompt = build_system_prompt(args.current_date)

    stage1_rows = []
    stage2_rows = []
    dropped_counter = Counter()
    duplicate_counter = Counter()
    pruned_tool_counter = Counter()
    tool_counter = Counter()
    correctness_counter = Counter()
    structurally_valid_rows = 0
    per_file_stats = {}
    source_descriptors = []

    for file_path in files:
        df = pd.read_parquet(file_path)
        source_descriptors.append(
            {
                "file": file_path.name,
                "rows": int(len(df)),
                "bytes": int(file_path.stat().st_size),
                "sha256": sha256_file(file_path),
            }
        )
        file_stats = {
            "source_rows": len(df),
            "stage1_rows": 0,
            "structurally_valid_rows": 0,
            "stage2_rows": 0,
            "dropped": Counter(),
            "correctness": Counter(),
            "pruned_tool_turns_removed": 0,
            "pruned_tool_response_turns_removed": 0,
            "duplicate_tool_turns_removed": 0,
        }

        for row_idx, row in enumerate(df.to_dict("records")):
            normalized_row, reason = normalize_row(
                row,
                system_prompt_mode=args.system_prompt_mode,
                template_system_prompt=template_system_prompt,
            )
            if normalized_row is None:
                dropped_counter[reason] += 1
                file_stats["dropped"][reason] += 1
                continue

            normalized_row["_source_file"] = file_path.name
            normalized_row["_source_row"] = row_idx
            stage1_rows.append(normalized_row)
            file_stats["stage1_rows"] += 1

            pruned_messages, pruned_info = prune_disallowed_tool_turns(normalized_row["messages"], allowed_tools)
            file_stats["pruned_tool_turns_removed"] += len(pruned_info)
            pruned_tool_counter["pruned_tool_turns_removed"] += len(pruned_info)
            for info in pruned_info:
                file_stats["pruned_tool_response_turns_removed"] += len(info.get("tool_response_positions", []))
                pruned_tool_counter["pruned_tool_response_turns_removed"] += len(info.get("tool_response_positions", []))
                for name in info.get("dropped_tools", []):
                    pruned_tool_counter[f"tool:{name}"] += 1

            deduped_messages, duplicate_info = remove_duplicate_tool_calls(pruned_messages)
            file_stats["duplicate_tool_turns_removed"] += len(duplicate_info)
            duplicate_counter["duplicate_tool_turns_removed"] += len(duplicate_info)

            deduped_row = dict(normalized_row)
            deduped_row["messages"] = deduped_messages

            is_valid, reason = validate_messages(deduped_messages)
            if not is_valid:
                dropped_counter[reason] += 1
                file_stats["dropped"][reason] += 1
                continue

            deduped_row["_turns"] = compute_turns(deduped_messages)
            structurally_valid_rows += 1
            file_stats["structurally_valid_rows"] += 1

            decision = correctness_evaluator.evaluate(row, deduped_row)
            correctness_counter[f"source:{decision.source}"] += 1
            correctness_counter[f"reason:{decision.reason}"] += 1
            file_stats["correctness"][decision.reason] += 1
            if not decision.keep:
                dropped_counter[decision.reason] += 1
                file_stats["dropped"][decision.reason] += 1
                continue
            deduped_row["_correctness_source"] = decision.source
            deduped_row["_correctness_score"] = decision.score
            deduped_row["_correctness_cache_key"] = decision.cache_key

            for message in deduped_messages:
                if message.get("role") != "assistant":
                    continue
                for tool_call in parse_tool_calls(message.get("content", "")):
                    name = tool_call.get("name")
                    if isinstance(name, str):
                        tool_counter[name] += 1

            stage2_rows.append(deduped_row)
            file_stats["stage2_rows"] += 1

        file_stats["dropped"] = dict(file_stats["dropped"])
        file_stats["correctness"] = dict(file_stats["correctness"])
        per_file_stats[file_path.name] = file_stats
        print(
            f"{file_path.name}: source={file_stats['source_rows']} "
            f"stage1={file_stats['stage1_rows']} stage2={file_stats['stage2_rows']} "
            f"pruned_tools_removed={file_stats['pruned_tool_turns_removed']} "
            f"duplicates_removed={file_stats['duplicate_tool_turns_removed']}"
        )

    stage1_df = _records_dataframe(stage1_rows)
    stage2_df = _records_dataframe(stage2_rows)

    if args.keep_intermediate:
        if not stage1_df.empty:
            stage1_df.to_parquet(output_dir / "stage1_normalized.parquet", index=False)
        if not stage2_df.empty:
            stage2_df.to_parquet(output_dir / "stage2_deduped.parquet", index=False)

    ratios = {
        "train": args.train_ratio,
        "validation": args.validation_ratio,
        "test": args.test_ratio,
    }
    split_records, question_assignments = split_records_by_question(stage2_rows, seed=args.seed, ratios=ratios)
    train_clean_records = split_records["train"]
    train_resampled_records, resample_stats = _resample_training_records(train_clean_records, args)

    split_dataframes = {
        "train_clean": _records_dataframe(train_clean_records),
        "train_resampled": _records_dataframe(train_resampled_records),
        "validation": _records_dataframe(split_records["validation"]),
        "test": _records_dataframe(split_records["test"]),
    }
    if args.shuffle:
        for offset, name in enumerate(split_dataframes):
            dataframe = split_dataframes[name]
            if not dataframe.empty:
                split_dataframes[name] = dataframe.sample(
                    frac=1.0, random_state=args.seed + offset
                ).reset_index(drop=True)

    output_dataframes = {
        name: _public_output(dataframe, keep_metadata=args.keep_metadata)
        for name, dataframe in split_dataframes.items()
    }
    artifact_paths = {
        "train_clean": output_dir / "train_clean.parquet",
        "train_resampled": output_dir / "train_resampled.parquet",
        "validation": output_dir / "validation.parquet",
        "test": output_dir / "test.parquet",
        # Preserve the historical path while making its role unambiguous: it is
        # now an alias of the leakage-free, resampled training split.
        "legacy_train_resampled": output_dir / "cleaned_resampled.parquet",
    }
    for name in ("train_clean", "train_resampled", "validation", "test"):
        output_dataframes[name].to_parquet(artifact_paths[name], index=False)
    output_dataframes["train_resampled"].to_parquet(artifact_paths["legacy_train_resampled"], index=False)

    if args.write_jsonl:
        for name in ("train_clean", "train_resampled", "validation", "test"):
            jsonl_path = output_dir / f"{name}.jsonl"
            write_jsonl(output_dataframes[name], jsonl_path)
        legacy_jsonl = output_dir / "cleaned_resampled.jsonl"
        write_jsonl(output_dataframes["train_resampled"], legacy_jsonl)

    question_split_counts = Counter(question_assignments.values())
    final_output_df = output_dataframes["train_resampled"]
    final_parquet = artifact_paths["legacy_train_resampled"]

    stats = {
        "input": str(input_path),
        "files": [str(path) for path in files],
        "output_dir": str(output_dir),
        "system_prompt_mode": args.system_prompt_mode,
        "allowed_tools": sorted(allowed_tools),
        "turn_formula": "(len(messages) - 1) / 2",
        "resample_policy": {
            "short_max_turns": args.short_max_turns,
            "medium_max_turns": args.medium_max_turns,
            "short_multiplier": args.short_multiplier,
            "medium_multiplier": args.medium_multiplier,
            "long_multiplier": args.long_multiplier,
        },
        "rows": {
            "source": int(sum(item["source_rows"] for item in per_file_stats.values())),
            "stage1_normalized": int(len(stage1_df)),
            "structurally_valid": int(structurally_valid_rows),
            "stage2_correctness_kept": int(len(stage2_df)),
            "train_clean": int(len(output_dataframes["train_clean"])),
            "train_resampled": int(len(output_dataframes["train_resampled"])),
            "validation": int(len(output_dataframes["validation"])),
            "test": int(len(output_dataframes["test"])),
            "final_resampled": int(len(final_output_df)),
        },
        "pruned_tool_turns_removed": int(pruned_tool_counter["pruned_tool_turns_removed"]),
        "pruned_tool_response_turns_removed": int(pruned_tool_counter["pruned_tool_response_turns_removed"]),
        "pruned_tool_name_counts": {
            key.removeprefix("tool:"): value
            for key, value in pruned_tool_counter.items()
            if key.startswith("tool:")
        },
        "duplicate_tool_turns_removed": int(duplicate_counter["duplicate_tool_turns_removed"]),
        "drop_reasons": dict(dropped_counter),
        "correctness": {
            "mode": args.correctness_mode,
            "field": args.correctness_field if args.correctness_mode == "field" else None,
            "threshold": args.correctness_threshold,
            "prompt_version": DEFAULT_JUDGE_PROMPT_VERSION,
            "source_counts": {
                key.removeprefix("source:"): value
                for key, value in correctness_counter.items()
                if key.startswith("source:")
            },
            "decision_counts": {
                key.removeprefix("reason:"): value
                for key, value in correctness_counter.items()
                if key.startswith("reason:")
            },
        },
        "tools_after_cleaning": dict(tool_counter),
        "split": {
            "unit": "normalized_question",
            "algorithm": SPLIT_ALGORITHM,
            "seed": args.seed,
            "ratios": ratios,
            "question_groups": dict(question_split_counts),
        },
        **resample_stats,
        "per_file": per_file_stats,
        "artifacts": {
            "final_parquet": str(final_parquet),
            "final_jsonl": str(output_dir / "cleaned_resampled.jsonl") if args.write_jsonl else None,
            "train_clean_parquet": str(artifact_paths["train_clean"]),
            "train_resampled_parquet": str(artifact_paths["train_resampled"]),
            "validation_parquet": str(artifact_paths["validation"]),
            "test_parquet": str(artifact_paths["test"]),
            "stage1_parquet": str(output_dir / "stage1_normalized.parquet") if args.keep_intermediate else None,
            "stage2_parquet": str(output_dir / "stage2_deduped.parquet") if args.keep_intermediate else None,
        },
    }

    stats_path = output_dir / "stats.json"

    artifact_manifest = {}
    for name, path in artifact_paths.items():
        logical_name = "train_resampled" if name == "legacy_train_resampled" else name
        artifact_manifest[name] = {
            "file": path.name,
            "rows": int(len(output_dataframes[logical_name])),
            "sha256": sha256_file(path),
            "logical_sha256": logical_records_sha256(output_dataframes[logical_name].to_dict("records")),
        }
    cache_manifest = None
    if cache_path is not None and cache_path.exists():
        cache_manifest = {
            "file": cache_path.name,
            "sha256": sha256_file(cache_path),
            "used_judge_models": sorted(correctness_evaluator.used_cache_judge_models),
            "used_prompt_versions": sorted(correctness_evaluator.used_cache_prompt_versions),
        }
    data_manifest = {
        "schema_version": 1,
        "sources": source_descriptors,
        "counts": {
            "source_rows": int(sum(item["source_rows"] for item in per_file_stats.values())),
            "normalized_rows": int(len(stage1_df)),
            "structurally_valid_rows": int(structurally_valid_rows),
            "correctness_kept_rows": int(len(stage2_df)),
            "correctness_dropped_rows": int(structurally_valid_rows - len(stage2_df)),
            "unique_questions": int(len(question_assignments)),
            "question_groups_by_split": {name: int(question_split_counts.get(name, 0)) for name in ratios},
            "rows_by_split": {
                "train_clean": int(len(output_dataframes["train_clean"])),
                "train_resampled": int(len(output_dataframes["train_resampled"])),
                "validation": int(len(output_dataframes["validation"])),
                "test": int(len(output_dataframes["test"])),
            },
        },
        "split": {
            "unit": "normalized_question",
            "normalization": "NFKC + whitespace collapse + casefold",
            "algorithm": SPLIT_ALGORITHM,
            "seed": args.seed,
            "ratios": ratios,
            "resampling_applied_to": "train_only",
        },
        "correctness": {
            "mode": args.correctness_mode,
            "field": args.correctness_field if args.correctness_mode == "field" else None,
            "threshold": args.correctness_threshold,
            "cache": cache_manifest,
            "judge": (
                {"protocol": "openai-compatible", "model": args.judge_model}
                if args.correctness_mode == "judge"
                else None
            ),
            "prompt_version": DEFAULT_JUDGE_PROMPT_VERSION,
            "decision_counts": stats["correctness"]["decision_counts"],
        },
        "resample_policy": stats["resample_policy"],
        "artifacts": artifact_manifest,
    }
    manifest_path = output_dir / "data_manifest.json"
    manifest_path.write_text(json.dumps(data_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    stats["artifacts"]["data_manifest"] = str(manifest_path)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved final parquet to: {final_parquet}")
    if args.write_jsonl:
        print(f"Saved final jsonl to: {output_dir / 'cleaned_resampled.jsonl'}")
    print(f"Saved validation parquet to: {artifact_paths['validation']}")
    print(f"Saved test parquet to: {artifact_paths['test']}")
    print(f"Saved stats to: {stats_path}")
    print(f"Saved data manifest to: {manifest_path}")
    return stats


def main(argv: list[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    run_pipeline(args)


if __name__ == "__main__":
    main()
