"""Dependency-light input/resume helpers for batch Agent evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


@dataclass(frozen=True)
class EvaluationItem:
    sample_id: str
    data_source: str
    question: str
    references: tuple[str, ...]
    sample_index: int = 0

    @property
    def key(self) -> tuple[str, int]:
        return self.sample_id, self.sample_index


def _question(prompt: Any) -> str:
    prompt = _plain(prompt)
    if not isinstance(prompt, list):
        raise ValueError("prompt must be a message list")
    parts = [
        str(message["content"])
        for message in prompt
        if isinstance(message, Mapping)
        and message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and message["content"].strip()
    ]
    if not parts:
        raise ValueError("prompt must contain a non-empty user message")
    return "\n".join(parts)


def _references(reward_model: Any) -> tuple[str, ...]:
    reward_model = _plain(reward_model)
    if not isinstance(reward_model, Mapping):
        raise ValueError("reward_model must be an object")
    ground_truth = reward_model.get("ground_truth")
    if isinstance(ground_truth, str):
        text = ground_truth.strip()
        if not text:
            raise ValueError("ground_truth must not be empty")
        # Some upstream data stores accepted aliases around this delimiter.
        values = tuple(part.strip() for part in text.split("<|answer_split|>") if part.strip())
    elif isinstance(ground_truth, Sequence):
        values = tuple(str(part).strip() for part in ground_truth if str(part).strip())
    else:
        raise ValueError("ground_truth must be a string or array")
    if not values:
        raise ValueError("ground_truth must contain at least one reference")
    return values


def item_from_row(row: Mapping[str, Any]) -> EvaluationItem:
    row = _plain(row)
    question = _question(row.get("prompt"))
    extra = row.get("extra_info") or {}
    if not isinstance(extra, Mapping):
        raise ValueError("extra_info must be an object")
    sample_id = extra.get("eval_sample_id", extra.get("index"))
    if sample_id is None or not str(sample_id).strip():
        sample_id = "sha256:" + hashlib.sha256(question.encode("utf-8")).hexdigest()
    source = row.get("data_source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("data_source must be a non-empty string")
    return EvaluationItem(
        sample_id=str(sample_id),
        data_source=source,
        question=question,
        references=_references(row.get("reward_model")),
    )


def load_evaluation_items(path: str | Path, *, limit: int | None = None) -> list[EvaluationItem]:
    path = Path(path)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if path.suffix.lower() == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("parquet input requires pandas and pyarrow") from exc
        rows = pd.read_parquet(path).to_dict(orient="records")
    elif path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        raise ValueError("evaluation input must end in .parquet or .jsonl")
    items = [item_from_row(row) for row in rows]
    if len({item.sample_id for item in items}) != len(items):
        raise ValueError("evaluation input contains duplicate sample IDs")
    return items[:limit]


def expand_samples(items: Sequence[EvaluationItem], samples_per_question: int) -> list[EvaluationItem]:
    if samples_per_question < 1:
        raise ValueError("samples_per_question must be positive")
    return [
        EvaluationItem(
            sample_id=item.sample_id,
            data_source=item.data_source,
            question=item.question,
            references=item.references,
            sample_index=sample_index,
        )
        for item in items
        for sample_index in range(samples_per_question)
    ]


def build_run_fingerprint(
    config: Mapping[str, Any], items: Sequence[EvaluationItem]
) -> str:
    """Bind resume state to generation settings and cohort semantics."""

    payload = {
        "config": _plain(config),
        "items": [
            {
                "sample_id": item.sample_id,
                "data_source": item.data_source,
                "question": item.question,
                "references": list(item.references),
            }
            for item in items
        ],
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def completed_keys(
    path: str | Path, *, seed: int, run_fingerprint: str | None = None
) -> set[tuple[str, int]]:
    path = Path(path)
    if not path.exists():
        return set()
    result: set[tuple[str, int]] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if (
                run_fingerprint is not None
                and record.get("run_fingerprint") != run_fingerprint
            ):
                raise ValueError(
                    f"{path}:{line_number}: output belongs to a different "
                    "run fingerprint; use a new output file"
                )
            if int(record.get("seed", seed)) != seed:
                continue
            termination = str(record.get("termination", ""))
            if termination.startswith("answered"):
                result.add((str(record["sample_id"]), int(record.get("sample_index", 0))))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid resume record: {exc}") from exc
    return result


def ad_hoc_item(question: str) -> EvaluationItem:
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")
    sample_id = "adhoc:" + hashlib.sha256(question.encode("utf-8")).hexdigest()
    return EvaluationItem(sample_id, "adhoc", question, ("not-scored",))
