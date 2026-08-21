from __future__ import annotations

import pytest

from dataclasses import asdict

from experiments.budget_guard import estimate_cost, validate_budget_report


def test_estimate_scales_pilot_by_steps_and_gpu_count() -> None:
    estimate = estimate_cost(
        gpu_count=4,
        hourly_rate_per_gpu_cny=10,
        pilot_seconds=1800,
        pilot_steps=3,
        total_steps=12,
        fixed_api_cny=20,
        already_spent_cny=100,
        hard_cap_cny=1000,
    )

    assert estimate.projected_gpu_hours == 8.0
    assert estimate.projected_training_cny == 80.0
    assert estimate.projected_total_cny == 200.0
    assert estimate.within_cap is True


def test_estimate_marks_cap_violation() -> None:
    estimate = estimate_cost(
        gpu_count=4,
        hourly_rate_per_gpu_cny=25,
        pilot_seconds=3600,
        pilot_steps=1,
        total_steps=40,
        hard_cap_cny=3000,
    )

    assert estimate.projected_total_cny == 4000.0
    assert estimate.within_cap is False


def test_valid_report_requires_matching_scope_and_measured_pilot() -> None:
    estimate = estimate_cost(
        gpu_count=4,
        hourly_rate_per_gpu_cny=10,
        pilot_seconds=30,
        pilot_steps=3,
        total_steps=120,
        hard_cap_cny=3000,
    )

    checked = validate_budget_report(
        asdict(estimate), expected_gpu_count=4, expected_total_steps=120
    )

    assert checked == estimate


@pytest.mark.parametrize(
    "change,match",
    [
        ({"pilot_steps": 2}, "3-5 step"),
        ({"hard_cap_cny": 3001}, "project cap"),
        ({"within_cap": False}, "derived fields"),
        ({"projected_total_cny": 3000.01}, "derived fields"),
        ({"gpu_count": 2}, "GPUs"),
        ({"total_steps": 119}, "total steps"),
    ],
)
def test_report_validation_fails_closed(change, match) -> None:
    estimate = estimate_cost(
        gpu_count=4,
        hourly_rate_per_gpu_cny=1,
        pilot_seconds=30,
        pilot_steps=3,
        total_steps=120,
        hard_cap_cny=3000,
    )
    payload = asdict(estimate)
    payload.update(change)

    with pytest.raises(ValueError, match=match):
        validate_budget_report(payload, expected_gpu_count=4, expected_total_steps=120)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gpu_count": 0},
        {"pilot_seconds": 0},
        {"pilot_steps": 0},
        {"total_steps": 0},
        {"hard_cap_cny": 0},
    ],
)
def test_estimate_rejects_invalid_values(kwargs: dict[str, float]) -> None:
    values = {
        "gpu_count": 1,
        "hourly_rate_per_gpu_cny": 1.0,
        "pilot_seconds": 1.0,
        "pilot_steps": 1,
        "total_steps": 1,
        "hard_cap_cny": 1.0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        estimate_cost(**values)


def test_report_cannot_forge_derived_cost_fields() -> None:
    estimate = estimate_cost(
        gpu_count=4,
        hourly_rate_per_gpu_cny=100,
        pilot_seconds=3600,
        pilot_steps=3,
        total_steps=120,
        hard_cap_cny=3000,
    )
    payload = asdict(estimate)
    payload.update(
        projected_gpu_hours=0,
        projected_training_cny=0,
        projected_total_cny=0,
        within_cap=True,
    )

    with pytest.raises(ValueError, match="derived fields"):
        validate_budget_report(payload, expected_gpu_count=4, expected_total_steps=120)
