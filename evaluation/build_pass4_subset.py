"""Freeze the deterministic 20-question subset used only for Pass@4."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from drvenus_repro.manifests import sha256_file, write_json
from evaluation.build_eval_core import allocate_stratified_counts, sample_id_for_row
from evaluation.cohort_io import load_rows, write_rows


SELECTION_ALGORITHM = "stratified-largest-remainder+sha256(pass4-v1,seed,sample-id)-v1"


def _load_parent_manifest(
    path: Path, cohort_path: Path, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"parent cohort manifest does not exist: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid parent cohort manifest: {path}: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("parent cohort manifest must be an object")
    output = manifest.get("output")
    if not isinstance(output, Mapping) or output.get("sha256") != sha256_file(cohort_path):
        raise ValueError("parent manifest output hash does not match input cohort")
    if int(manifest.get("num_samples", -1)) != len(rows):
        raise ValueError("parent manifest sample count does not match input cohort")
    declared_ids = manifest.get("sample_ids")
    actual_ids = [sample_id_for_row(row) for row in rows]
    if not isinstance(declared_ids, list) or set(map(str, declared_ids)) != set(actual_ids):
        raise ValueError("parent manifest sample IDs do not match input cohort")
    return dict(manifest)


def build_pass4_subset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    total: int = 20,
    seed: int = 42,
    parent_manifest_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    if total < 1:
        raise ValueError("total must be positive")
    cohort = Path(input_path)
    rows = load_rows(cohort)
    if total > len(rows):
        raise ValueError(f"cannot select {total} rows from a {len(rows)}-row cohort")
    parent_path = (
        Path(parent_manifest_path)
        if parent_manifest_path is not None
        else cohort.with_name(cohort.name + ".manifest.json")
    )
    parent = _load_parent_manifest(parent_path, cohort, rows)

    grouped: dict[str, list[tuple[str, str, dict[str, Any]]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        source = row.get("data_source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"input row {row_number} has no data_source")
        sample_id = sample_id_for_row(row)
        if sample_id in seen_ids:
            raise ValueError(f"duplicate sample ID in input cohort: {sample_id}")
        seen_ids.add(sample_id)
        # Domain separation avoids merely taking the prefix of eval-core when
        # both builders intentionally use the same public seed.
        selection_hash = hashlib.sha256(
            f"pass4-v1\0{seed}\0{sample_id}".encode("utf-8")
        ).hexdigest()
        grouped[source].append((selection_hash, sample_id, row))

    source_counts = {source: len(items) for source, items in sorted(grouped.items())}
    selected_counts = allocate_stratified_counts(source_counts, total)
    selected: list[dict[str, Any]] = []
    selected_ids: list[str] = []
    for source in sorted(grouped):
        candidates = sorted(grouped[source], key=lambda item: (item[0], item[1]))
        for source_rank, (selection_hash, sample_id, original) in enumerate(
            candidates[: selected_counts[source]], start=1
        ):
            row = copy.deepcopy(original)
            extra = row.get("extra_info") or {}
            if not isinstance(extra, Mapping):
                raise ValueError(f"extra_info for {sample_id} must be an object")
            row["extra_info"] = {
                **dict(extra),
                "pass4_suite": f"pass4-{total}",
                "pass4_selection_seed": seed,
                "pass4_selection_hash": selection_hash,
                "pass4_source_rank": source_rank,
            }
            selected.append(row)
            selected_ids.append(sample_id)

    output = Path(output_path)
    write_rows(output, selected)
    manifest = {
        "schema_version": "1.0",
        "suite": f"pass4-{total}",
        "purpose": "fixed subset for four generations per question; not an overall-score cohort",
        "selection_algorithm": SELECTION_ALGORITHM,
        "selection_seed": seed,
        "num_samples": len(selected),
        "parent": {
            "suite": parent.get("suite"),
            "cohort_path": cohort.as_posix(),
            "cohort_sha256": sha256_file(cohort),
            "manifest_path": parent_path.as_posix(),
            "manifest_sha256": sha256_file(parent_path),
            "num_samples": len(rows),
        },
        "source_counts": source_counts,
        "selected_counts": dict(sorted(selected_counts.items())),
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
    parser.add_argument("--input", default="evaluation/data/eval-core-100.parquet")
    parser.add_argument(
        "--input-manifest",
        default="evaluation/data/eval-core-100.parquet.manifest.json",
    )
    parser.add_argument("--output", default="evaluation/data/pass4-20.parquet")
    parser.add_argument("--manifest", help="output manifest path")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_pass4_subset(
        args.input,
        args.output,
        total=20,
        seed=args.seed,
        parent_manifest_path=args.input_manifest,
        manifest_path=args.manifest,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
