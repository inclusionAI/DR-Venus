"""Build the deterministic, source-stratified ``eval-core-100`` cohort."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from drvenus_repro.manifests import sha256_file, write_json


SELECTION_ALGORITHM = "proportional-largest-remainder+sha256(seed,sample_id)-v1"


def _plain(value: Any) -> Any:
    """Convert numpy/pandas containers to deterministic Python/JSON values."""

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


def _canonical_json(value: Any) -> str:
    return json.dumps(_plain(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sample_id_for_row(row: Mapping[str, Any]) -> str:
    """Use the dataset ID when present and a content hash otherwise."""

    extra_info = _plain(row.get("extra_info") or {})
    if isinstance(extra_info, Mapping):
        candidate = extra_info.get("index")
        if candidate is not None and str(candidate).strip():
            return str(candidate)
    identity = {
        "data_source": _plain(row.get("data_source")),
        "prompt": _plain(row.get("prompt")),
        "reward_model": _plain(row.get("reward_model")),
    }
    return "sha256:" + hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


def allocate_stratified_counts(source_counts: Mapping[str, int], total: int) -> dict[str, int]:
    """Allocate ``total`` proportionally with deterministic largest remainders.

    If the requested size permits, every non-empty source receives at least one
    item.  Ties are broken by source name, making allocation independent from
    dictionary or parquet row order.
    """

    counts = {str(source): int(count) for source, count in source_counts.items() if count > 0}
    if not counts:
        raise ValueError("source_counts must contain at least one non-empty source")
    population = sum(counts.values())
    if total < 1 or total > population:
        raise ValueError(f"total must be in [1, {population}], got {total}")

    quotas = {source: total * count / population for source, count in counts.items()}
    allocation = {source: math.floor(quota) for source, quota in quotas.items()}
    remaining = total - sum(allocation.values())
    order = sorted(counts, key=lambda source: (-(quotas[source] - allocation[source]), source))
    for source in order:
        if remaining == 0:
            break
        if allocation[source] < counts[source]:
            allocation[source] += 1
            remaining -= 1

    if total >= len(counts):
        for source in sorted(counts):
            if allocation[source] > 0:
                continue
            donors = [candidate for candidate in counts if allocation[candidate] > 1]
            if not donors:
                raise ValueError("unable to ensure representation for every source")
            donor = max(
                donors,
                key=lambda candidate: (allocation[candidate] - quotas[candidate], candidate),
            )
            allocation[donor] -= 1
            allocation[source] = 1

    if sum(allocation.values()) != total:
        raise AssertionError("stratified allocation does not sum to requested total")
    if any(allocation[source] > counts[source] for source in counts):
        raise AssertionError("stratified allocation exceeds a source population")
    return dict(sorted(allocation.items()))


def select_stratified_rows(
    rows: Iterable[Mapping[str, Any]], *, total: int = 100, seed: int = 42
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    """Select rows using stable hashes and attach auditable cohort metadata."""

    grouped: dict[str, list[tuple[str, str, dict[str, Any]]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for raw_row in rows:
        row = _plain(dict(raw_row))
        source = row.get("data_source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("every row must have a non-empty data_source")
        sample_id = sample_id_for_row(row)
        if sample_id in seen_ids:
            raise ValueError(f"duplicate stable sample ID in input dataset: {sample_id}")
        seen_ids.add(sample_id)
        selection_key = hashlib.sha256(f"{seed}\0{sample_id}".encode("utf-8")).hexdigest()
        grouped[source].append((selection_key, sample_id, row))

    source_counts = dict(sorted((source, len(items)) for source, items in grouped.items()))
    selected_counts = allocate_stratified_counts(source_counts, total)
    selected: list[dict[str, Any]] = []
    for source in sorted(grouped):
        candidates = sorted(grouped[source], key=lambda item: (item[0], item[1]))
        for source_rank, (selection_key, sample_id, original) in enumerate(
            candidates[: selected_counts[source]], start=1
        ):
            row = copy.deepcopy(original)
            extra_info = row.get("extra_info") or {}
            if not isinstance(extra_info, Mapping):
                raise ValueError(f"extra_info for {sample_id} must be an object")
            extra_info = dict(extra_info)
            extra_info.update(
                {
                    "eval_suite": f"eval-core-{total}",
                    "eval_sample_id": sample_id,
                    "eval_selection_seed": seed,
                    "eval_source_rank": source_rank,
                    "eval_selection_hash": selection_key,
                }
            )
            row["extra_info"] = extra_info
            selected.append(row)
    return selected, source_counts, selected_counts


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(_plain(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    path.write_text(content, encoding="utf-8", newline="\n")


def build_eval_core(
    input_path: str | Path,
    output_path: str | Path,
    *,
    total: int = 100,
    seed: int = 42,
    expected_source_count: int | None = 7,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Read the source parquet, select the cohort, and write data + manifest."""

    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("building eval-core requires pandas and pyarrow") from exc

    input_path = Path(input_path)
    output_path = Path(output_path)
    dataframe = pd.read_parquet(input_path)
    required_columns = {"data_source", "prompt", "reward_model", "extra_info"}
    missing = sorted(required_columns - set(dataframe.columns))
    if missing:
        raise ValueError(f"input parquet is missing columns: {', '.join(missing)}")
    rows = [_plain(row) for row in dataframe.to_dict(orient="records")]
    selected, source_counts, selected_counts = select_stratified_rows(
        rows, total=total, seed=seed
    )
    if expected_source_count is not None and len(source_counts) != expected_source_count:
        raise ValueError(
            f"expected {expected_source_count} data sources, found {len(source_counts)}: "
            f"{', '.join(source_counts)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        pd.DataFrame(selected, columns=list(dataframe.columns)).to_parquet(output_path, index=False)
    elif output_path.suffix.lower() == ".jsonl":
        _write_jsonl(output_path, selected)
    else:
        raise ValueError("output must end in .parquet or .jsonl")

    manifest = {
        "schema_version": "1.0",
        "suite": f"eval-core-{total}",
        "selection_algorithm": SELECTION_ALGORITHM,
        "selection_seed": seed,
        "num_samples": len(selected),
        "input": {
            "path": input_path.as_posix(),
            "sha256": sha256_file(input_path),
            "num_samples": len(rows),
        },
        "output": {
            "path": output_path.as_posix(),
            "sha256": sha256_file(output_path),
        },
        "source_counts": source_counts,
        "selected_counts": selected_counts,
        "sample_ids": [row["extra_info"]["eval_sample_id"] for row in selected],
    }
    target_manifest = (
        Path(manifest_path)
        if manifest_path is not None
        else output_path.with_name(output_path.name + ".manifest.json")
    )
    write_json(target_manifest, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deterministic eval-core-100 from RL/data/test.parquet."
    )
    parser.add_argument("--input", default="RL/data/test.parquet", help="source parquet")
    parser.add_argument(
        "--output", default="evaluation/data/eval-core-100.parquet", help="output parquet/jsonl"
    )
    parser.add_argument("--manifest", help="dataset manifest path (defaults beside output)")
    parser.add_argument("--size", type=int, default=100, help="cohort size")
    parser.add_argument("--seed", type=int, default=42, help="selection seed")
    parser.add_argument(
        "--expected-source-count",
        type=int,
        default=7,
        help="fail if source count changes; use 0 to disable",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_eval_core(
        args.input,
        args.output,
        total=args.size,
        seed=args.seed,
        expected_source_count=args.expected_source_count or None,
        manifest_path=args.manifest,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
