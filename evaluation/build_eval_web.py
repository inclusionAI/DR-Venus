"""Build the frozen 30 BrowseComp + 30 BrowseComp-ZH web cohort.

This command never downloads data.  Both source files are required and every
selected row must expose a stable ID, question, and accepted answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any, Mapping

from drvenus_repro.manifests import sha256_file, write_json
from evaluation.cohort_io import (
    ANSWER_SPLIT_TOKEN,
    extract_question,
    extract_raw_id,
    extract_references,
    load_rows,
    write_rows,
)


SELECTION_ALGORITHM = "per-source-sha256(seed,source,stable-id)-v1"
SOURCE_SPECS = (("BrowseComp", "browsecomp"), ("BrowseComp-ZH", "browsecomp_zh"))


def _normalized_question(question: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", question).casefold().split())


def _normalize_source_rows(
    rows: list[Mapping[str, Any]], *, source: str, seed: int
) -> list[tuple[str, str, dict[str, Any]]]:
    candidates: list[tuple[str, str, dict[str, Any]]] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        try:
            question = extract_question(row)
            references = extract_references(row)
            raw_id = extract_raw_id(row, question)
        except ValueError as exc:
            raise ValueError(f"{source} row {row_number}: {exc}") from exc
        sample_id = f"{source}:{raw_id}"
        if sample_id in seen_ids:
            raise ValueError(f"{source} contains duplicate stable ID: {raw_id!r}")
        normalized_question = _normalized_question(question)
        if normalized_question in seen_questions:
            raise ValueError(f"{source} contains a duplicate normalized question")
        seen_ids.add(sample_id)
        seen_questions.add(normalized_question)
        selection_hash = hashlib.sha256(
            f"{seed}\0{source}\0{raw_id}".encode("utf-8")
        ).hexdigest()
        canonical = {
            "data_source": source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "search & question answering",
            "reward_model": {
                "ground_truth": ANSWER_SPLIT_TOKEN.join(references),
                "style": "unknown",
            },
            "extra_info": {
                "index": sample_id,
                "eval_sample_id": sample_id,
                "raw_sample_id": raw_id,
                "eval_selection_seed": seed,
                "eval_selection_hash": selection_hash,
            },
        }
        candidates.append((selection_hash, sample_id, canonical))
    return sorted(candidates, key=lambda item: (item[0], item[1]))


def build_eval_web(
    browsecomp_path: str | Path,
    browsecomp_zh_path: str | Path,
    output_path: str | Path,
    *,
    seed: int = 42,
    per_source: int = 30,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate both inputs, choose equal source quotas, and write cohort evidence."""

    if per_source < 1:
        raise ValueError("per_source must be positive")
    input_paths = {
        "BrowseComp": Path(browsecomp_path),
        "BrowseComp-ZH": Path(browsecomp_zh_path),
    }
    input_rows: dict[str, list[dict[str, Any]]] = {}
    selected_rows: list[dict[str, Any]] = []
    selected_ids: list[str] = []
    for source, _ in SOURCE_SPECS:
        rows = load_rows(input_paths[source])
        candidates = _normalize_source_rows(rows, source=source, seed=seed)
        if len(candidates) < per_source:
            raise ValueError(
                f"{source} requires at least {per_source} valid unique rows; found {len(candidates)}"
            )
        input_rows[source] = rows
        for rank, (_, sample_id, row) in enumerate(candidates[:per_source], start=1):
            row["extra_info"].update(
                {
                    "eval_suite": f"eval-web-{per_source * len(SOURCE_SPECS)}",
                    "eval_source_rank": rank,
                }
            )
            selected_rows.append(row)
            selected_ids.append(sample_id)

    output = Path(output_path)
    write_rows(output, selected_rows)
    manifest = {
        "schema_version": "1.0",
        "suite": f"eval-web-{per_source * len(SOURCE_SPECS)}",
        "selection_algorithm": SELECTION_ALGORITHM,
        "selection_seed": seed,
        "inputs": {
            source: {
                "path": input_paths[source].as_posix(),
                "sha256": sha256_file(input_paths[source]),
                "num_samples": len(input_rows[source]),
            }
            for source, _ in SOURCE_SPECS
        },
        "selected_counts": {source: per_source for source, _ in SOURCE_SPECS},
        "num_samples": len(selected_rows),
        "output": {"path": output.as_posix(), "sha256": sha256_file(output)},
        "sample_ids": selected_ids,
    }
    target_manifest = (
        Path(manifest_path)
        if manifest_path is not None
        else output.with_name(output.name + ".manifest.json")
    )
    write_json(target_manifest, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browsecomp", required=True, help="local BrowseComp JSON/JSONL/parquet")
    parser.add_argument(
        "--browsecomp-zh", required=True, help="local BrowseComp-ZH JSON/JSONL/parquet"
    )
    parser.add_argument(
        "--output", default="evaluation/data/eval-web-60.parquet", help="cohort output"
    )
    parser.add_argument("--manifest", help="manifest path (defaults beside output)")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_eval_web(
        args.browsecomp,
        args.browsecomp_zh,
        args.output,
        seed=args.seed,
        per_source=30,
        manifest_path=args.manifest,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
