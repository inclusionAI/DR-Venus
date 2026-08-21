"""Small, deterministic helpers shared by evaluation-cohort builders."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import unicodedata


ANSWER_SPLIT_TOKEN = "<|answer_split|>"


def plain(value: Any) -> Any:
    """Convert pandas/numpy containers to JSON-compatible Python values."""

    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load JSON, JSONL, or parquet rows without guessing malformed schemas."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"evaluation input does not exist: {source}")
    suffix = source.suffix.casefold()
    if suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("parquet input requires pandas and pyarrow") from exc
        rows = pd.read_parquet(source).to_dict(orient="records")
    elif suffix == ".jsonl":
        rows = []
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{source}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(row, Mapping):
                    raise ValueError(f"{source}:{line_number}: row must be an object")
                rows.append(row)
    elif suffix == ".json":
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}: invalid JSON: {exc}") from exc
        if isinstance(payload, Mapping) and isinstance(payload.get("data"), list):
            payload = payload["data"]
        if not isinstance(payload, list):
            raise ValueError(f"{source}: JSON input must be an array or {{\"data\": [...]}}")
        rows = payload
    else:
        raise ValueError("evaluation input must end in .json, .jsonl, or .parquet")

    normalized: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"{source}: row {row_number} must be an object")
        normalized.append(plain(dict(row)))
    if not normalized:
        raise ValueError(f"evaluation input contains no rows: {source}")
    return normalized


def write_rows(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write an evaluation cohort in a stable JSONL or parquet representation."""

    target = Path(path)
    normalized = [plain(dict(row)) for row in rows]
    if not normalized:
        raise ValueError("refusing to write an empty evaluation cohort")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.casefold() == ".jsonl":
        content = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in normalized
        )
        target.write_text(content, encoding="utf-8", newline="\n")
    elif target.suffix.casefold() == ".parquet":
        try:
            import pandas as pd  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("parquet output requires pandas and pyarrow") from exc
        pd.DataFrame(normalized).to_parquet(target, index=False)
    else:
        raise ValueError("evaluation output must end in .jsonl or .parquet")


def extract_question(row: Mapping[str, Any]) -> str:
    """Extract one question from common public-dataset or DR-Venus schemas."""

    for field in ("question", "query", "problem", "Question", "问题"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    if isinstance(prompt, Sequence) and not isinstance(prompt, (str, bytes)):
        parts = [
            str(message["content"]).strip()
            for message in prompt
            if isinstance(message, Mapping)
            and message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and str(message["content"]).strip()
        ]
        if parts:
            return "\n".join(parts)
    raise ValueError("row has no non-empty question/query/user prompt")


def extract_references(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract non-empty accepted answers from common dataset schemas."""

    raw: Any = None
    for field in (
        "references",
        "answers",
        "answer",
        "reference",
        "ground_truth",
        "答案",
    ):
        if row.get(field) is not None:
            raw = row[field]
            break
    if raw is None and isinstance(row.get("reward_model"), Mapping):
        raw = row["reward_model"].get("ground_truth")
    if isinstance(raw, str):
        values = tuple(part.strip() for part in raw.split(ANSWER_SPLIT_TOKEN) if part.strip())
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = tuple(str(item).strip() for item in raw if str(item).strip())
    else:
        raise ValueError("row has no string/array answer or ground_truth")
    if not values:
        raise ValueError("row has no non-empty accepted answer")
    return values


def extract_raw_id(row: Mapping[str, Any], question: str) -> str:
    """Use a dataset ID when present and a declared question hash otherwise."""

    for field in ("id", "sample_id", "index", "question_id", "_id"):
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    extra = row.get("extra_info")
    if isinstance(extra, Mapping):
        for field in ("eval_sample_id", "index"):
            value = extra.get(field)
            if value is not None and str(value).strip():
                return str(value).strip()
    normalized = " ".join(unicodedata.normalize("NFKC", question).strip().split())
    return "question-sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()
