#!/usr/bin/env python3
"""Build a frozen RL validation cohort excluding the formal train questions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drvenus_repro.manifests import sha256_file, write_json
from experiments.prepare_rl_subset import (
    SELECTION_ALGORITHM,
    normalize_question,
    question_from_row,
    select_training_questions,
)


def build_rl_validation(
    source_path: str | Path,
    train_path: str | Path,
    output_path: str | Path,
    *,
    total: int = 32,
    seed: int = 43,
    expected_source_rows: int | None = 80_000,
) -> dict:
    source_path = Path(source_path)
    train_path = Path(train_path)
    output_path = Path(output_path)
    source = pd.read_parquet(source_path)
    train = pd.read_parquet(train_path)
    if expected_source_rows is not None and len(source) != expected_source_rows:
        raise ValueError(
            f"expected {expected_source_rows} source rows, found {len(source)}"
        )
    train_questions = {
        normalize_question(question_from_row(row))
        for row in train.to_dict(orient="records")
    }
    candidates = [
        row
        for row in source.to_dict(orient="records")
        if normalize_question(question_from_row(row)) not in train_questions
    ]
    selected, selection = select_training_questions(candidates, total=total, seed=seed)
    for row in selected:
        extra = dict(row.get("extra_info") or {})
        extra["rl_subset"] = f"rl-val-{total}"
        row["extra_info"] = extra

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected, columns=list(source.columns)).to_parquet(output_path, index=False)
    manifest = {
        "schema_version": "1.0",
        "suite": f"rl-val-{total}",
        "selection_algorithm": "exclude-train+" + SELECTION_ALGORITHM,
        "selection_seed": seed,
        "num_samples": len(selected),
        "input": {
            "path": source_path.as_posix(),
            "sha256": sha256_file(source_path),
            "num_samples": len(source),
        },
        "excluded_train": {
            "path": train_path.as_posix(),
            "sha256": sha256_file(train_path),
            "normalized_questions": len(train_questions),
        },
        "output": {"path": output_path.as_posix(), "sha256": sha256_file(output_path)},
        **selection,
    }
    write_json(output_path.with_name(output_path.name + ".manifest.json"), manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="RL/data/train.parquet")
    parser.add_argument("--train", default="experiments/data/rl-train-96.parquet")
    parser.add_argument("--output", default="experiments/data/rl-val-32.parquet")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--expected-input-rows", type=int, default=80_000)
    args = parser.parse_args(argv)
    manifest = build_rl_validation(
        args.input,
        args.train,
        args.output,
        total=args.size,
        seed=args.seed,
        expected_source_rows=args.expected_input_rows or None,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
