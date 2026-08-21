"""Estimate a scaled experiment cost from a measured pilot.

The command exits with status 2 when the projected total exceeds the hard cap.
It never contacts a cloud provider or starts a training job.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class BudgetEstimate:
    gpu_count: int
    hourly_rate_per_gpu_cny: float
    pilot_seconds: float
    pilot_steps: int
    total_steps: int
    projected_gpu_hours: float
    projected_training_cny: float
    fixed_api_cny: float
    already_spent_cny: float
    projected_total_cny: float
    hard_cap_cny: float
    within_cap: bool


def validate_budget_report(
    report: str | Path | Mapping[str, Any],
    *,
    expected_gpu_count: int | None = None,
    expected_total_steps: int | None = None,
    maximum_hard_cap_cny: float = 3000.0,
) -> BudgetEstimate:
    """Validate the approval artifact required by paid experiment launchers."""

    if isinstance(report, Mapping):
        payload = dict(report)
    else:
        try:
            payload = json.loads(Path(report).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid budget report: {exc}") from exc
    try:
        reported = BudgetEstimate(
            **{field: payload[field] for field in BudgetEstimate.__dataclass_fields__}
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"budget report is missing or has invalid fields: {exc}") from exc

    integer_values = (reported.gpu_count, reported.pilot_steps, reported.total_steps)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in integer_values):
        raise ValueError("gpu_count, pilot_steps, and total_steps must be integers")
    numeric_values = (
        reported.hourly_rate_per_gpu_cny,
        reported.pilot_seconds,
        reported.projected_gpu_hours,
        reported.projected_training_cny,
        reported.fixed_api_cny,
        reported.already_spent_cny,
        reported.projected_total_cny,
        reported.hard_cap_cny,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
        for value in numeric_values
    ):
        raise ValueError("budget report contains an invalid numeric value")
    if not isinstance(reported.within_cap, bool):
        raise ValueError("within_cap must be a boolean")
    if not 3 <= reported.pilot_steps <= 5:
        raise ValueError("budget approval requires a 3-5 step pilot")
    if expected_gpu_count is not None and reported.gpu_count != expected_gpu_count:
        raise ValueError(
            f"budget report covers {reported.gpu_count} GPUs, expected {expected_gpu_count}"
        )
    if expected_total_steps is not None and reported.total_steps != expected_total_steps:
        raise ValueError(
            f"budget report covers {reported.total_steps} total steps, expected {expected_total_steps}"
        )

    # Derived fields are evidence, not authority.  Recompute them from the
    # measured pilot so editing projected_total_cny/within_cap cannot bypass
    # the paid-run gate.
    estimate = estimate_cost(
        gpu_count=reported.gpu_count,
        hourly_rate_per_gpu_cny=float(reported.hourly_rate_per_gpu_cny),
        pilot_seconds=float(reported.pilot_seconds),
        pilot_steps=reported.pilot_steps,
        total_steps=reported.total_steps,
        fixed_api_cny=float(reported.fixed_api_cny),
        already_spent_cny=float(reported.already_spent_cny),
        hard_cap_cny=float(reported.hard_cap_cny),
    )
    derived_fields = (
        "projected_gpu_hours",
        "projected_training_cny",
        "projected_total_cny",
        "within_cap",
    )
    mismatches = [
        field
        for field in derived_fields
        if getattr(reported, field) != getattr(estimate, field)
    ]
    if mismatches:
        raise ValueError(
            "budget report derived fields do not match the measured pilot: "
            + ", ".join(mismatches)
        )

    if estimate.hard_cap_cny > maximum_hard_cap_cny:
        raise ValueError(
            f"report hard cap {estimate.hard_cap_cny} exceeds project cap {maximum_hard_cap_cny}"
        )
    if not estimate.within_cap or estimate.projected_total_cny > estimate.hard_cap_cny:
        raise ValueError("projected cost exceeds the approved hard cap")
    return estimate


def estimate_cost(
    *,
    gpu_count: int,
    hourly_rate_per_gpu_cny: float,
    pilot_seconds: float,
    pilot_steps: int,
    total_steps: int,
    fixed_api_cny: float = 0.0,
    already_spent_cny: float = 0.0,
    hard_cap_cny: float = 3000.0,
) -> BudgetEstimate:
    if gpu_count <= 0:
        raise ValueError("gpu_count must be positive")
    if hourly_rate_per_gpu_cny < 0 or pilot_seconds <= 0:
        raise ValueError("rate must be non-negative and pilot_seconds must be positive")
    if pilot_steps <= 0 or total_steps <= 0:
        raise ValueError("pilot_steps and total_steps must be positive")
    if fixed_api_cny < 0 or already_spent_cny < 0 or hard_cap_cny <= 0:
        raise ValueError("costs must be non-negative and hard_cap_cny must be positive")

    projected_seconds = pilot_seconds * total_steps / pilot_steps
    gpu_hours = projected_seconds * gpu_count / 3600.0
    training_cost = gpu_hours * hourly_rate_per_gpu_cny
    total = training_cost + fixed_api_cny + already_spent_cny
    return BudgetEstimate(
        gpu_count=gpu_count,
        hourly_rate_per_gpu_cny=hourly_rate_per_gpu_cny,
        pilot_seconds=pilot_seconds,
        pilot_steps=pilot_steps,
        total_steps=total_steps,
        projected_gpu_hours=round(gpu_hours, 4),
        projected_training_cny=round(training_cost, 2),
        fixed_api_cny=round(fixed_api_cny, 2),
        already_spent_cny=round(already_spent_cny, 2),
        projected_total_cny=round(total, 2),
        hard_cap_cny=round(hard_cap_cny, 2),
        within_cap=total <= hard_cap_cny,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument("--hourly-rate-per-gpu-cny", type=float, required=True)
    parser.add_argument("--pilot-seconds", type=float, required=True)
    parser.add_argument("--pilot-steps", type=int, required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--fixed-api-cny", type=float, default=0.0)
    parser.add_argument("--already-spent-cny", type=float, default=0.0)
    parser.add_argument("--hard-cap-cny", type=float, default=3000.0)
    parser.add_argument("--output", help="Optional path for the JSON estimate")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        estimate = estimate_cost(
            gpu_count=args.gpu_count,
            hourly_rate_per_gpu_cny=args.hourly_rate_per_gpu_cny,
            pilot_seconds=args.pilot_seconds,
            pilot_steps=args.pilot_steps,
            total_steps=args.total_steps,
            fixed_api_cny=args.fixed_api_cny,
            already_spent_cny=args.already_spent_cny,
            hard_cap_cny=args.hard_cap_cny,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    payload = json.dumps(asdict(estimate), ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    return 0 if estimate.within_cap else 2


if __name__ == "__main__":
    raise SystemExit(main())
