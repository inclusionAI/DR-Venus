import json
from pathlib import Path

import pytest

from evaluation.evaluate import (
    bootstrap_mean_ci,
    build_report,
    compute_metrics,
    compute_pass_at_k,
    exact_match,
    paired_comparison,
    render_markdown,
    token_f1,
)
from drvenus_repro.manifests import load_experiment_manifest, load_predictions
from drvenus_repro.schemas import PredictionRecord, SchemaValidationError


def record(sample_id="sample-1", sample_index=0, **overrides):
    value = {
        "schema_version": "1.0",
        "sample_id": sample_id,
        "data_source": "nq",
        "model": "candidate",
        "seed": 42,
        "sample_index": sample_index,
        "answer": "Paris",
        "references": ["Paris"],
        "score": None,
        "completed": True,
        "format_valid": True,
        "turns": 2,
        "tool_calls": 2,
        "browse_calls": 1,
        "timeout_count": 0,
        "input_tokens": 10,
        "output_tokens": 5,
        "latency_seconds": 1.0,
        "cost_cny": 0.1,
        "max_turns": 12,
        "context_length": 32768,
        "trajectory_path": f"trajectories/{sample_id}-{sample_index}.json",
        "metadata": {},
    }
    value.update(overrides)
    return PredictionRecord.from_mapping(value)


def test_em_and_f1_support_alternatives_articles_punctuation_and_cjk():
    assert exact_match("The Eiffel Tower!", ["Eiffel tower"]) == 1.0
    assert exact_match("wrong", ["first", "wrong"]) == 1.0
    assert token_f1("北京大学", ["北京"]) == pytest.approx(2 / 3)
    assert token_f1("red blue blue", ["red blue"]) == pytest.approx(0.8)


def test_compute_metrics_covers_quality_agent_and_cost_fields():
    records = [
        record(),
        record(
            "sample-2",
            answer="London",
            completed=False,
            format_valid=False,
            turns=4,
            tool_calls=0,
            browse_calls=0,
            timeout_count=1,
            input_tokens=20,
            output_tokens=10,
            latency_seconds=3.0,
            cost_cny=0.2,
        ),
    ]
    metrics = compute_metrics(records)
    assert metrics["answer_accuracy"] == 0.5
    assert metrics["answer_accuracy_basis"] == "exact_match"
    assert metrics["exact_match"] == 0.5
    assert metrics["token_f1"] == 0.5
    assert metrics["completion_rate"] == 0.5
    assert metrics["format_valid_rate"] == 0.5
    assert metrics["turns_mean"] == 3.0
    assert metrics["turns_p95"] == pytest.approx(3.9)
    assert metrics["browse_ratio"] == 0.5
    assert metrics["timeout_rate"] == 0.5
    assert metrics["timeout_count"] == 1
    assert metrics["tokens_total"] == 45
    assert metrics["tokens_mean"] == 22.5
    assert metrics["latency_seconds_mean"] == 2.0
    assert metrics["cost_cny_total"] == pytest.approx(0.3)


def test_answer_accuracy_prefers_judge_score_when_complete():
    metrics = compute_metrics(
        [record(score=0.75), record("sample-2", answer="wrong", score=0.25)]
    )
    assert metrics["answer_accuracy"] == 0.5
    assert metrics["answer_accuracy_basis"] == "judge_score"
    assert metrics["judge_score_coverage"] == 1.0


def test_bootstrap_is_fixed_seed_and_handles_constant_values():
    first = bootstrap_mean_ci([0.0, 0.5, 1.0], seed=7, samples=200)
    second = bootstrap_mean_ci([0.0, 0.5, 1.0], seed=7, samples=200)
    assert first == second
    constant = bootstrap_mean_ci([0.5, 0.5], seed=42, samples=20)
    assert constant["lower"] == constant["upper"] == constant["point"] == 0.5


def test_pass_at_k_uses_unbiased_estimator_and_reports_coverage():
    records = []
    for sample_id, correct_indices in (("a", {0}), ("b", set())):
        for index in range(4):
            is_correct = index in correct_indices
            records.append(
                record(
                    sample_id,
                    index,
                    answer="yes" if is_correct else "no",
                    references=["yes"],
                )
            )
    result = compute_pass_at_k(records, [1, 4])
    assert result["1"]["estimate"] == pytest.approx(0.125)
    assert result["4"]["estimate"] == pytest.approx(0.5)
    assert result["4"]["coverage"] == 1.0


def test_paired_comparison_is_candidate_minus_baseline():
    baseline = [record(model="baseline"), record("sample-2", model="baseline", answer="wrong")]
    candidate = [record(), record("sample-2")]
    comparison = paired_comparison(baseline, candidate, samples=100, seed=1)
    assert comparison["num_pairs"] == 2
    assert comparison["metrics"]["exact_match"]["point"] == 0.5


def test_paired_comparison_uses_only_sample_index_zero():
    baseline = [record(model="baseline", answer="wrong")]
    candidate = [record(answer="wrong"), record(sample_index=1, answer="Paris")]
    comparison = paired_comparison(baseline, candidate, samples=20, seed=1)
    assert comparison["sample_index"] == 0
    assert comparison["metrics"]["exact_match"]["point"] == 0.0


def test_paired_comparison_rejects_different_cohort_or_limits():
    with pytest.raises(SchemaValidationError, match="identical"):
        paired_comparison([record(model="base")], [record("different")], samples=10)
    with pytest.raises(SchemaValidationError, match="max_turns"):
        paired_comparison(
            [record(model="base")], [record(max_turns=10, turns=2)], samples=10
        )


def test_report_is_machine_readable_and_cannot_look_official():
    records = [record(), record("sample-2", data_source="hotpotqa")]
    report = build_report(records, bootstrap_samples=20, pass_ks=[1, 4])
    serialized = json.dumps(report)
    assert json.loads(serialized)["evaluation_scope"]["official_comparable"] is False
    assert report["evaluation_scope"]["max_turns"] == 12
    assert report["evaluation_scope"]["context_length"] == 32768
    assert set(report["by_data_source"]) == {"hotpotqa", "nq"}
    assert report["pass_at_k"]["4"]["coverage"] == 0.0

    markdown = render_markdown(report)
    assert "NON-OFFICIAL / BUDGET-MATCHED" in markdown
    assert "max_turns=12" in markdown
    assert "context_length=32768" in markdown
    assert "Official comparable: **false**" in markdown


def test_overall_and_ci_use_index_zero_while_pass_k_uses_all_generations():
    records = [
        record("a", 0, answer="wrong", references=["right"]),
        record("a", 1, answer="right", references=["right"]),
        record("b", 0, answer="right", references=["right"]),
        record("b", 1, answer="right", references=["right"]),
    ]
    report = build_report(records, bootstrap_samples=20, pass_ks=[1, 2])
    assert report["metrics"]["num_predictions"] == 2
    assert report["metrics"]["exact_match"] == 0.5
    assert report["bootstrap_95_ci"]["exact_match"]["point"] == 0.5
    assert report["generation_accounting"] == {
        "overall_sample_index": 0,
        "overall_predictions": 2,
        "all_generations": 4,
        "pass_at_k_uses_all_generations": True,
    }
    assert report["pass_at_k"]["2"]["estimate"] == 1.0


def test_report_with_manifest_requires_and_verifies_cohort_path():
    repository = Path(__file__).resolve().parents[1]
    predictions_path = repository / "evaluation/examples/predictions.example.jsonl"
    manifest_path = repository / "evaluation/examples/experiment_manifest.example.yaml"
    records = load_predictions(predictions_path)
    manifest = load_experiment_manifest(manifest_path)
    with pytest.raises(SchemaValidationError, match="manifest_path is required"):
        build_report(records, manifest=manifest, bootstrap_samples=5, pass_ks=[1])
    report = build_report(
        records,
        manifest=manifest,
        manifest_path=manifest_path,
        bootstrap_samples=5,
        pass_ks=[1],
    )
    assert report["evaluation_scope"]["cohort_validated"] is True
