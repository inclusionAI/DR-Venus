"""Evaluate DR-Venus prediction artifacts and emit JSON + Markdown evidence."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from drvenus_repro.manifests import (
    load_experiment_manifest,
    load_predictions,
    sha256_file,
    validate_evaluation_evidence,
    validate_formal_evaluation_evidence,
    write_json,
)
from drvenus_repro.schemas import (
    ExperimentManifest,
    PredictionRecord,
    SchemaValidationError,
    validate_prediction_collection,
    validate_predictions_against_manifest,
)


REPORT_SCHEMA_VERSION = "1.0"
NON_OFFICIAL_LABEL = "budget-matched-non-official"
NON_OFFICIAL_NOTICE = (
    "Budget-matched evaluation with explicitly reduced max_turns/context_length; "
    "these results are not directly comparable to the official DR-Venus scores."
)


def _is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def answer_tokens(text: str) -> list[str]:
    """SQuAD-style tokens, with CJK characters tokenized individually."""

    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith("P")
    )
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    normalized = " ".join(normalized.split())

    tokens: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            tokens.append("".join(buffer))
            buffer.clear()

    for character in normalized:
        if _is_cjk(character):
            flush()
            tokens.append(character)
        elif character.isspace():
            flush()
        else:
            buffer.append(character)
    flush()
    return tokens


def normalize_answer(text: str) -> str:
    return " ".join(answer_tokens(text))


def exact_match(answer: str, references: Sequence[str]) -> float:
    normalized = normalize_answer(answer)
    return float(any(normalized == normalize_answer(reference) for reference in references))


def token_f1(answer: str, references: Sequence[str]) -> float:
    prediction_tokens = answer_tokens(answer)
    best = 0.0
    for reference in references:
        reference_tokens = answer_tokens(reference)
        if not prediction_tokens and not reference_tokens:
            score = 1.0
        elif not prediction_tokens or not reference_tokens:
            score = 0.0
        else:
            overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
            if overlap == 0:
                score = 0.0
            else:
                precision = overlap / len(prediction_tokens)
                recall = overlap / len(reference_tokens)
                score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _metric_vectors(records: Sequence[PredictionRecord]) -> dict[str, list[float]]:
    em = [exact_match(record.answer, record.references) for record in records]
    f1 = [token_f1(record.answer, record.references) for record in records]
    accuracy = [
        record.score if record.score is not None else em_value
        for record, em_value in zip(records, em, strict=True)
    ]
    return {"answer_accuracy": accuracy, "exact_match": em, "token_f1": f1}


def compute_metrics(records: Sequence[PredictionRecord]) -> dict[str, Any]:
    records = validate_prediction_collection(records)
    vectors = _metric_vectors(records)
    count = len(records)
    judge_count = sum(record.score is not None for record in records)
    tool_calls = sum(record.tool_calls for record in records)
    total_tokens = sum(record.total_tokens for record in records)
    accuracy_basis = (
        "judge_score"
        if judge_count == count
        else "exact_match"
        if judge_count == 0
        else "judge_score_with_exact_match_fallback"
    )
    return {
        "num_predictions": count,
        "num_unique_samples": len({record.sample_id for record in records}),
        "answer_accuracy": fmean(vectors["answer_accuracy"]),
        "answer_accuracy_basis": accuracy_basis,
        "judge_score_coverage": judge_count / count,
        "exact_match": fmean(vectors["exact_match"]),
        "token_f1": fmean(vectors["token_f1"]),
        "completion_rate": sum(record.completed for record in records) / count,
        "format_valid_rate": sum(record.format_valid for record in records) / count,
        "turns_mean": fmean(record.turns for record in records),
        "turns_p95": percentile([record.turns for record in records], 0.95),
        "browse_ratio": sum(record.browse_calls for record in records) / tool_calls
        if tool_calls
        else 0.0,
        "samples_with_browse_rate": sum(record.browse_calls > 0 for record in records) / count,
        "tool_calls_total": tool_calls,
        "browse_calls_total": sum(record.browse_calls for record in records),
        "timeout_rate": sum(record.timeout_count > 0 for record in records) / count,
        "timeout_count": sum(record.timeout_count for record in records),
        "input_tokens_total": sum(record.input_tokens for record in records),
        "output_tokens_total": sum(record.output_tokens for record in records),
        "tokens_total": total_tokens,
        "tokens_mean": total_tokens / count,
        "latency_seconds_total": sum(record.latency_seconds for record in records),
        "latency_seconds_mean": fmean(record.latency_seconds for record in records),
        "latency_seconds_p95": percentile(
            [record.latency_seconds for record in records], 0.95
        ),
        "cost_cny_total": sum(record.cost_cny for record in records),
        "cost_cny_mean": fmean(record.cost_cny for record in records),
    }


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int = 42,
    samples: int = 10_000,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Deterministic percentile-bootstrap confidence interval for a mean."""

    if not values:
        raise ValueError("bootstrap requires at least one value")
    if samples < 1:
        raise ValueError("samples must be >= 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    numeric = [float(value) for value in values]
    rng = random.Random(seed)
    length = len(numeric)
    estimates = [
        sum(numeric[rng.randrange(length)] for _ in range(length)) / length
        for _ in range(samples)
    ]
    alpha = (1.0 - confidence) / 2.0
    return {
        "point": fmean(numeric),
        "lower": percentile(estimates, alpha),
        "upper": percentile(estimates, 1.0 - alpha),
        "confidence": confidence,
        "bootstrap_samples": samples,
        "seed": seed,
    }


def paired_bootstrap_difference_ci(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    seed: int = 42,
    samples: int = 10_000,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """CI for candidate-minus-baseline using paired example resampling."""

    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("paired bootstrap requires non-empty, equally sized inputs")
    differences = [
        float(candidate_value) - float(baseline_value)
        for baseline_value, candidate_value in zip(baseline, candidate, strict=True)
    ]
    return bootstrap_mean_ci(
        differences, seed=seed, samples=samples, confidence=confidence
    )


def _cluster_values_by_sample(
    records: Sequence[PredictionRecord], values: Sequence[float]
) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record, value in zip(records, values, strict=True):
        grouped[record.sample_id].append(float(value))
    return [fmean(grouped[sample_id]) for sample_id in sorted(grouped)]


def bootstrap_metric_cis(
    records: Sequence[PredictionRecord], *, seed: int = 42, samples: int = 10_000
) -> dict[str, dict[str, float | int]]:
    vectors = _metric_vectors(records)
    result: dict[str, dict[str, float | int]] = {}
    for offset, metric_name in enumerate(("answer_accuracy", "exact_match", "token_f1")):
        sample_values = _cluster_values_by_sample(records, vectors[metric_name])
        result[metric_name] = bootstrap_mean_ci(
            sample_values, seed=seed + offset, samples=samples
        )
    return result


def _success(record: PredictionRecord, threshold: float) -> bool:
    if record.score is not None:
        return record.score >= threshold
    return exact_match(record.answer, record.references) == 1.0


def compute_pass_at_k(
    records: Sequence[PredictionRecord],
    ks: Iterable[int],
    *,
    success_threshold: float = 0.5,
) -> dict[str, dict[str, float | int]]:
    """Compute the standard unbiased ``pass@k`` estimator by sample ID."""

    if not 0.0 <= success_threshold <= 1.0:
        raise ValueError("success_threshold must be in [0, 1]")
    grouped: dict[str, list[PredictionRecord]] = defaultdict(list)
    for record in records:
        grouped[record.sample_id].append(record)
    total_samples = len(grouped)
    output: dict[str, dict[str, float | int]] = {}
    for k in sorted(set(ks)):
        if k < 1:
            raise ValueError("k must be >= 1")
        estimates: list[float] = []
        for sample_id in sorted(grouped):
            generations = grouped[sample_id]
            n = len(generations)
            if n < k:
                continue
            correct = sum(_success(record, success_threshold) for record in generations)
            estimate = 1.0 - math.comb(n - correct, k) / math.comb(n, k)
            estimates.append(estimate)
        output[str(k)] = {
            "estimate": fmean(estimates) if estimates else 0.0,
            "eligible_samples": len(estimates),
            "total_samples": total_samples,
            "coverage": len(estimates) / total_samples if total_samples else 0.0,
            "success_threshold": success_threshold,
        }
    return output


def _record_key(record: PredictionRecord) -> tuple[str, int, int]:
    return (record.sample_id, record.seed, record.sample_index)


def paired_comparison(
    baseline: Sequence[PredictionRecord],
    candidate: Sequence[PredictionRecord],
    *,
    seed: int = 42,
    samples: int = 10_000,
) -> dict[str, Any]:
    baseline_all = validate_prediction_collection(baseline)
    candidate_all = validate_prediction_collection(candidate)
    baseline_model = _single_value(baseline_all, "model")
    candidate_model = _single_value(candidate_all, "model")
    baseline_limits = (
        _single_value(baseline_all, "max_turns"),
        _single_value(baseline_all, "context_length"),
    )
    candidate_limits = (
        _single_value(candidate_all, "max_turns"),
        _single_value(candidate_all, "context_length"),
    )
    if baseline_limits != candidate_limits:
        raise SchemaValidationError(
            "paired comparison requires identical max_turns and context_length"
        )

    baseline = [record for record in baseline_all if record.sample_index == 0]
    candidate = [record for record in candidate_all if record.sample_index == 0]
    if not baseline or not candidate:
        raise SchemaValidationError(
            "paired comparison requires sample_index=0 predictions in both artifacts"
        )
    baseline_by_key = {_record_key(record): record for record in baseline}
    candidate_by_key = {_record_key(record): record for record in candidate}
    if baseline_by_key.keys() != candidate_by_key.keys():
        missing_candidate = sorted(baseline_by_key.keys() - candidate_by_key.keys())
        missing_baseline = sorted(candidate_by_key.keys() - baseline_by_key.keys())
        raise SchemaValidationError(
            "paired comparison requires identical sample_index=0 (sample_id, seed) keys; "
            f"missing_candidate={len(missing_candidate)}, missing_baseline={len(missing_baseline)}"
        )
    keys = sorted(baseline_by_key)
    baseline_ordered = [baseline_by_key[key] for key in keys]
    candidate_ordered = [candidate_by_key[key] for key in keys]
    for baseline_record, candidate_record in zip(
        baseline_ordered, candidate_ordered, strict=True
    ):
        if (
            baseline_record.data_source != candidate_record.data_source
            or baseline_record.references != candidate_record.references
        ):
            raise SchemaValidationError(
                "paired comparison requires identical data_source and references for every key"
            )
        if (baseline_record.score is None) != (candidate_record.score is None):
            raise SchemaValidationError(
                "paired answer_accuracy requires matching judge-score availability"
            )
    baseline_vectors = _metric_vectors(baseline_ordered)
    candidate_vectors = _metric_vectors(candidate_ordered)
    metrics: dict[str, Any] = {}
    for offset, metric_name in enumerate(("answer_accuracy", "exact_match", "token_f1")):
        baseline_by_sample: dict[str, list[float]] = defaultdict(list)
        candidate_by_sample: dict[str, list[float]] = defaultdict(list)
        for record, baseline_value, candidate_value in zip(
            baseline_ordered,
            baseline_vectors[metric_name],
            candidate_vectors[metric_name],
            strict=True,
        ):
            baseline_by_sample[record.sample_id].append(baseline_value)
            candidate_by_sample[record.sample_id].append(candidate_value)
        sample_ids = sorted(baseline_by_sample)
        metrics[metric_name] = paired_bootstrap_difference_ci(
            [fmean(baseline_by_sample[sample_id]) for sample_id in sample_ids],
            [fmean(candidate_by_sample[sample_id]) for sample_id in sample_ids],
            seed=seed + offset,
            samples=samples,
        )
    return {
        "difference_direction": "candidate_minus_baseline",
        "num_pairs": len({record.sample_id for record in baseline_ordered}),
        "num_primary_pairs": len(keys),
        "bootstrap_unit": "sample_id",
        "sample_index": 0,
        "baseline_model": baseline_model,
        "candidate_model": candidate_model,
        "metrics": metrics,
    }


def _single_value(records: Sequence[PredictionRecord], attribute: str) -> Any:
    values = {getattr(record, attribute) for record in records}
    if len(values) != 1:
        raise SchemaValidationError(
            f"one predictions artifact must have exactly one {attribute}; found {sorted(values)}"
        )
    return next(iter(values))


def build_report(
    records: Sequence[PredictionRecord],
    *,
    manifest: ExperimentManifest | None = None,
    manifest_path: str | Path | None = None,
    baseline: Sequence[PredictionRecord] | None = None,
    bootstrap_seed: int = 42,
    bootstrap_samples: int = 10_000,
    pass_ks: Iterable[int] = (1, 4),
    success_threshold: float = 0.5,
) -> dict[str, Any]:
    all_records = (
        validate_predictions_against_manifest(records, manifest)
        if manifest is not None
        else validate_prediction_collection(records)
    )
    cohort_validation = None
    if manifest is not None:
        if manifest_path is None:
            raise SchemaValidationError(
                "manifest_path is required to verify the frozen evaluation cohort"
            )
        cohort_validation = validate_evaluation_evidence(
            all_records, manifest, manifest_path
        )
    primary_records = [record for record in all_records if record.sample_index == 0]
    if not primary_records:
        raise SchemaValidationError(
            "overall evaluation requires at least one sample_index=0 prediction"
        )

    model = _single_value(all_records, "model")
    max_turns = _single_value(all_records, "max_turns")
    context_length = _single_value(all_records, "context_length")
    grouped: dict[str, list[PredictionRecord]] = defaultdict(list)
    for record in primary_records:
        grouped[record.data_source].append(record)

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "evaluation_scope": {
            "label": manifest.protocol["label"] if manifest else NON_OFFICIAL_LABEL,
            "official_comparable": False,
            "notice": NON_OFFICIAL_NOTICE,
            "model": model,
            "dataset": manifest.evaluation_dataset["name"] if manifest else "unspecified",
            "max_turns": max_turns,
            "context_length": context_length,
            "manifest_validated": manifest is not None,
            "cohort_validated": False,
        },
        "generation_accounting": {
            "overall_sample_index": 0,
            "overall_predictions": len(primary_records),
            "all_generations": len(all_records),
            "pass_at_k_uses_all_generations": True,
        },
        "metrics": compute_metrics(primary_records),
        "bootstrap_95_ci": bootstrap_metric_cis(
            primary_records, seed=bootstrap_seed, samples=bootstrap_samples
        ),
        "pass_at_k": compute_pass_at_k(
            all_records, pass_ks, success_threshold=success_threshold
        ),
        "by_data_source": {
            source: compute_metrics(grouped[source]) for source in sorted(grouped)
        },
    }
    if manifest is not None:
        report["evaluation_scope"]["cohort_validated"] = True
        report["cohort_validation"] = cohort_validation
        report["experiment"] = {
            "experiment_id": manifest.experiment_id,
            "upstream_commit": manifest.upstream_commit,
            "checkpoint": manifest.checkpoint,
            "seeds": list(manifest.seeds),
            "gpu_hours": manifest.usage["gpu_hours"],
            "api_cost_cny": manifest.usage["api_cost_cny"],
        }
    if baseline is not None:
        report["paired_comparison"] = paired_comparison(
            baseline,
            records,
            seed=bootstrap_seed,
            samples=bootstrap_samples,
        )
    return report


def _format_number(value: Any, *, percentage: bool = False) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return f"{100 * value:.2f}%" if percentage else f"{value:.4f}"
    return str(value).replace("|", "\\|")


def render_markdown(report: Mapping[str, Any]) -> str:
    scope = report["evaluation_scope"]
    metrics = report["metrics"]
    lines = [
        "# DR-Venus Evaluation Report",
        "",
        "> **NON-OFFICIAL / BUDGET-MATCHED.** " + scope["notice"],
        "",
        f"- Model: `{scope['model']}`",
        f"- Dataset: `{scope['dataset']}`",
        f"- Protocol: `{scope['label']}`",
        f"- Limits: `max_turns={scope['max_turns']}`, `context_length={scope['context_length']}`",
        f"- Official comparable: **{str(scope['official_comparable']).lower()}**",
        f"- Manifest/cohort validated: **{str(scope['manifest_validated']).lower()} / "
        f"{str(scope['cohort_validated']).lower()}**",
        "- Overall/CI selection: `sample_index=0`; Pass@K selection: all generations",
        "",
        "## Overall metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    metric_rows = (
        ("Predictions", "num_predictions", False),
        ("Unique samples", "num_unique_samples", False),
        ("Answer accuracy", "answer_accuracy", True),
        ("Exact match", "exact_match", True),
        ("Token F1", "token_f1", True),
        ("Completion rate", "completion_rate", True),
        ("Format-valid rate", "format_valid_rate", True),
        ("Turns mean", "turns_mean", False),
        ("Turns p95", "turns_p95", False),
        ("Browse/tool-call ratio", "browse_ratio", True),
        ("Timeout rate", "timeout_rate", True),
        ("Tokens total", "tokens_total", False),
        ("Latency mean (s)", "latency_seconds_mean", False),
        ("Latency p95 (s)", "latency_seconds_p95", False),
        ("Cost total (CNY)", "cost_cny_total", False),
    )
    for label, key, percentage in metric_rows:
        lines.append(f"| {label} | {_format_number(metrics[key], percentage=percentage)} |")

    lines.extend(
        [
            "",
            "Answer accuracy basis: `" + metrics["answer_accuracy_basis"] + "`.",
            "",
            "## Bootstrap 95% confidence intervals",
            "",
            "| Metric | Point | 95% CI | Seed | Resamples |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for metric_name, interval in report["bootstrap_95_ci"].items():
        lines.append(
            f"| {metric_name} | {interval['point']:.4f} | "
            f"[{interval['lower']:.4f}, {interval['upper']:.4f}] | "
            f"{interval['seed']} | {interval['bootstrap_samples']} |"
        )

    lines.extend(
        [
            "",
            "## By data source",
            "",
            "| Source | N | Accuracy | EM | F1 | Completion | Format valid |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for source, source_metrics in report["by_data_source"].items():
        escaped_source = str(source).replace("|", "\\|")
        lines.append(
            f"| {escaped_source} | {source_metrics['num_predictions']} | "
            f"{source_metrics['answer_accuracy']:.4f} | {source_metrics['exact_match']:.4f} | "
            f"{source_metrics['token_f1']:.4f} | {source_metrics['completion_rate']:.4f} | "
            f"{source_metrics['format_valid_rate']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Pass@K",
            "",
            "| K | Estimate | Eligible / total samples | Coverage |",
            "|---:|---:|---:|---:|",
        ]
    )
    for k, value in report["pass_at_k"].items():
        lines.append(
            f"| {k} | {value['estimate']:.4f} | {value['eligible_samples']} / "
            f"{value['total_samples']} | {value['coverage']:.2%} |"
        )

    comparison = report.get("paired_comparison")
    if comparison:
        lines.extend(
            [
                "",
                "## Paired comparison",
                "",
                f"Candidate `{comparison['candidate_model']}` minus baseline "
                f"`{comparison['baseline_model']}` over {comparison['num_pairs']} pairs.",
                "",
                "| Metric | Difference | 95% CI |",
                "|---|---:|---:|",
            ]
        )
        for metric_name, interval in comparison["metrics"].items():
            lines.append(
                f"| {metric_name} | {interval['point']:+.4f} | "
                f"[{interval['lower']:+.4f}, {interval['upper']:+.4f}] |"
            )
    return "\n".join(lines) + "\n"


def _atomic_markdown_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate predictions and emit budget-matched DR-Venus evaluation evidence."
    )
    parser.add_argument("predictions", help="canonical predictions.jsonl")
    parser.add_argument("--manifest", help="experiment_manifest.yaml")
    parser.add_argument("--baseline", help="baseline predictions.jsonl for paired CIs")
    parser.add_argument("--output-json", help="machine-readable report path")
    parser.add_argument("--output-markdown", help="Markdown report path")
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--pass-k", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--success-threshold", type=float, default=0.5)
    parser.add_argument(
        "--allow-legacy-manifest",
        action="store_true",
        help=(
            "Development-only downgrade for old/example manifests without hashed artifacts "
            "and canonical execution identity. Never use for final evidence."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prediction_path = Path(args.predictions)
    records = load_predictions(prediction_path)
    manifest = load_experiment_manifest(args.manifest) if args.manifest else None
    baseline = load_predictions(args.baseline) if args.baseline else None
    formal_evidence = None
    if manifest is not None:
        formal_evidence = validate_formal_evaluation_evidence(
            records,
            manifest,
            args.manifest,
            prediction_path,
            strict=not args.allow_legacy_manifest,
        )
    report = build_report(
        records,
        manifest=manifest,
        manifest_path=args.manifest,
        baseline=baseline,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
        pass_ks=args.pass_k,
        success_threshold=args.success_threshold,
    )
    if formal_evidence is not None:
        report["formal_evidence_validation"] = formal_evidence
        report["evaluation_scope"]["artifact_identity_validated"] = bool(
            formal_evidence["identity_verified"]
        )
        if args.allow_legacy_manifest:
            report["evaluation_scope"]["label"] = "legacy-development-only"
            report["evaluation_scope"]["notice"] = (
                "Legacy manifest downgrade: artifact hashes and canonical execution identity "
                "were not fully authenticated. This report is not final evidence."
            )
    report["artifacts"] = {
        "predictions_path": prediction_path.as_posix(),
        "predictions_sha256": sha256_file(prediction_path),
    }
    if args.manifest:
        report["artifacts"]["experiment_manifest_path"] = Path(args.manifest).as_posix()
        report["artifacts"]["experiment_manifest_sha256"] = sha256_file(args.manifest)
    if args.baseline:
        report["artifacts"]["baseline_predictions_path"] = Path(args.baseline).as_posix()
        report["artifacts"]["baseline_predictions_sha256"] = sha256_file(args.baseline)

    output_json = Path(args.output_json) if args.output_json else prediction_path.with_suffix(
        ".evaluation.json"
    )
    output_markdown = (
        Path(args.output_markdown)
        if args.output_markdown
        else prediction_path.with_suffix(".evaluation.md")
    )
    write_json(output_json, report)
    _atomic_markdown_write(output_markdown, render_markdown(report))
    print(
        json.dumps(
            {
                "output_json": output_json.as_posix(),
                "output_markdown": output_markdown.as_posix(),
                "official_comparable": False,
                "max_turns": report["evaluation_scope"]["max_turns"],
                "context_length": report["evaluation_scope"]["context_length"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
