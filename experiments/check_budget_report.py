#!/usr/bin/env python3
"""Fail closed unless a measured-pilot budget report authorizes a run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from experiments.budget_guard import validate_budget_report
except ModuleNotFoundError as exc:
    if exc.name != "experiments":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiments.budget_guard import validate_budget_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--maximum-hard-cap-cny", type=float, default=3000.0)
    args = parser.parse_args(argv)
    try:
        estimate = validate_budget_report(
            args.report,
            expected_gpu_count=args.gpu_count,
            expected_total_steps=args.total_steps,
            maximum_hard_cap_cny=args.maximum_hard_cap_cny,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(
        f"Budget approved: projected CNY {estimate.projected_total_cny:.2f} "
        f"<= cap {estimate.hard_cap_cny:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
