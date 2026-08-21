#!/usr/bin/env python3
"""Print the physical row count of a Parquet or non-empty JSONL dataset."""

from __future__ import annotations

import argparse
from pathlib import Path


def count_rows(path: str | Path) -> int:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"dataset does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).metadata.num_rows
    if path.suffix.lower() == ".jsonl":
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    raise ValueError("dataset must end in .parquet or .jsonl")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    args = parser.parse_args(argv)
    count = count_rows(args.path)
    if count <= 0:
        parser.error("dataset is empty")
    print(count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
