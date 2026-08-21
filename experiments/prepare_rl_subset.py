"""Build the fixed, source-stratified 96-question RL training subset."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping
import unicodedata

try:
    from drvenus_repro.manifests import sha256_file, write_json
    from evaluation.build_eval_core import allocate_stratified_counts
except ModuleNotFoundError as exc:
    if exc.name not in {"drvenus_repro", "evaluation"}:
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from drvenus_repro.manifests import sha256_file, write_json
    from evaluation.build_eval_core import allocate_stratified_counts


SELECTION_ALGORITHM = "unique-question+proportional-largest-remainder+sha256(seed,question)-v1"


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


def question_from_row(row: Mapping[str, Any]) -> str:
    prompt = _plain(row.get("prompt"))
    if not isinstance(prompt, list):
        raise ValueError("every row must contain a prompt message list")
    user_parts = []
    for message in prompt:
        if isinstance(message, Mapping) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                user_parts.append(content)
    if not user_parts:
        raise ValueError("every row must contain a non-empty user question")
    return "\n".join(user_parts)


def normalize_question(question: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", question).split()).casefold()


def _stable_row_id(row: Mapping[str, Any]) -> str:
    extra = _plain(row.get("extra_info") or {})
    if isinstance(extra, Mapping) and extra.get("index") is not None:
        return str(extra["index"])
    payload = json.dumps(_plain(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_training_questions(
    rows: Iterable[Mapping[str, Any]], *, total: int = 96, seed: int = 42
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deduplicate questions globally, stratify by source, and select stably."""

    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    input_rows = 0
    for raw in rows:
        input_rows += 1
        row = _plain(dict(raw))
        source = row.get("data_source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("every row must contain a non-empty data_source")
        candidates[normalize_question(question_from_row(row))].append(row)
    if not candidates:
        raise ValueError("input dataset is empty")

    # Stable representative selection makes the result independent of parquet
    # row order while ensuring a duplicated question can only be selected once.
    unique_rows: list[tuple[str, dict[str, Any]]] = []
    cross_source_duplicates = 0
    for question_key, duplicate_rows in candidates.items():
        sources = {str(row["data_source"]) for row in duplicate_rows}
        if len(sources) > 1:
            cross_source_duplicates += 1
        representative = min(duplicate_rows, key=_stable_row_id)
        unique_rows.append((question_key, representative))

    by_source: dict[str, list[tuple[str, str, dict[str, Any]]]] = defaultdict(list)
    for question_key, row in unique_rows:
        selection_hash = hashlib.sha256(f"{seed}\0{question_key}".encode("utf-8")).hexdigest()
        question_id = "sha256:" + hashlib.sha256(question_key.encode("utf-8")).hexdigest()
        by_source[str(row["data_source"])].append((selection_hash, question_id, row))

    unique_source_counts = dict(sorted((source, len(items)) for source, items in by_source.items()))
    selected_counts = allocate_stratified_counts(unique_source_counts, total)
    selected: list[dict[str, Any]] = []
    question_ids: list[str] = []
    for source in sorted(by_source):
        ranked = sorted(by_source[source], key=lambda item: (item[0], item[1]))
        for source_rank, (selection_hash, question_id, original) in enumerate(
            ranked[: selected_counts[source]], start=1
        ):
            row = copy.deepcopy(original)
            extra = row.get("extra_info") or {}
            if not isinstance(extra, Mapping):
                raise ValueError(f"extra_info for {question_id} must be an object")
            extra = dict(extra)
            extra.update(
                {
                    "rl_subset": f"rl-train-{total}",
                    "rl_question_id": question_id,
                    "rl_selection_seed": seed,
                    "rl_source_rank": source_rank,
                    "rl_selection_hash": selection_hash,
                }
            )
            row["extra_info"] = extra
            selected.append(row)
            question_ids.append(question_id)

    metadata = {
        "input_rows": input_rows,
        "unique_questions": len(unique_rows),
        "duplicate_rows_removed": input_rows - len(unique_rows),
        "cross_source_duplicate_questions": cross_source_duplicates,
        "unique_source_counts": unique_source_counts,
        "selected_counts": selected_counts,
        "question_ids": question_ids,
    }
    return selected, metadata


def build_rl_subset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    total: int = 96,
    seed: int = 42,
    expected_input_rows: int | None = 80_000,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("building the RL subset requires pandas and pyarrow") from exc

    input_path = Path(input_path)
    output_path = Path(output_path)
    dataframe = pd.read_parquet(input_path)
    required = {"data_source", "prompt", "reward_model", "extra_info"}
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(f"input parquet is missing columns: {', '.join(missing)}")
    if expected_input_rows is not None and len(dataframe) != expected_input_rows:
        raise ValueError(f"expected {expected_input_rows} source rows, found {len(dataframe)}")

    selected, selection = select_training_questions(
        dataframe.to_dict(orient="records"), total=total, seed=seed
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        pd.DataFrame(selected, columns=list(dataframe.columns)).to_parquet(output_path, index=False)
    elif output_path.suffix.lower() == ".jsonl":
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in selected:
                handle.write(json.dumps(_plain(row), ensure_ascii=False, sort_keys=True) + "\n")
    else:
        raise ValueError("output must end in .parquet or .jsonl")

    manifest = {
        "schema_version": "1.0",
        "suite": f"rl-train-{total}",
        "selection_algorithm": SELECTION_ALGORITHM,
        "selection_seed": seed,
        "num_samples": len(selected),
        "input": {
            "path": input_path.as_posix(),
            "sha256": sha256_file(input_path),
            "num_samples": len(dataframe),
        },
        "output": {"path": output_path.as_posix(), "sha256": sha256_file(output_path)},
        **selection,
    }
    target_manifest = (
        Path(manifest_path)
        if manifest_path is not None
        else output_path.with_name(output_path.name + ".manifest.json")
    )
    write_json(target_manifest, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="RL/data/train.parquet")
    parser.add_argument("--output", default="experiments/data/rl-train-96.parquet")
    parser.add_argument("--manifest")
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--expected-input-rows",
        type=int,
        default=80_000,
        help="fail on source drift; use 0 to disable",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_rl_subset(
        args.input,
        args.output,
        total=args.size,
        seed=args.seed,
        expected_input_rows=args.expected_input_rows or None,
        manifest_path=args.manifest,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
