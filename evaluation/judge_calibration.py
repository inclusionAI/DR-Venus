"""Prepare a blinded 100-item human review and measure Judge agreement."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from drvenus_repro.manifests import (
    canonical_json_sha256,
    load_predictions,
    sha256_file,
    write_json,
)
from drvenus_repro.schemas import PredictionRecord


TEMPLATE_FIELDS = frozenset(
    {
        "schema_version",
        "review_id",
        "sample_id",
        "data_source",
        "model_seed",
        "sample_index",
        "question",
        "candidate_answer",
        "references",
        "human_score",
        "notes",
    }
)


def _prediction_key(record: PredictionRecord) -> dict[str, Any]:
    return {
        "sample_id": record.sample_id,
        "seed": record.seed,
        "sample_index": record.sample_index,
    }


def review_id(record: PredictionRecord) -> str:
    return canonical_json_sha256(_prediction_key(record))


def _question(record: PredictionRecord) -> str:
    question = record.metadata.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"prediction {record.sample_id!r} has no metadata.question")
    return question.strip()


def _template_row(record: PredictionRecord) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "review_id": review_id(record),
        "sample_id": record.sample_id,
        "data_source": record.data_source,
        "model_seed": record.seed,
        "sample_index": record.sample_index,
        "question": _question(record),
        "candidate_answer": record.answer,
        "references": list(record.references),
        "human_score": None,
        "notes": "",
    }


def _immutable_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in sorted(TEMPLATE_FIELDS - {"human_score", "notes"})}


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    path.write_text(content, encoding="utf-8", newline="\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"{path}:{line_number}: review row must be an object")
            unknown = sorted(set(row) - TEMPLATE_FIELDS)
            missing = sorted(TEMPLATE_FIELDS - set(row))
            if unknown or missing:
                raise ValueError(
                    f"{path}:{line_number}: template fields differ; missing={missing}, unknown={unknown}"
                )
            rows.append(dict(row))
    if not rows:
        raise ValueError("human review template contains no rows")
    return rows


def prepare_calibration_template(
    records: Sequence[PredictionRecord],
    *,
    predictions_path: str | Path,
    output_path: str | Path,
    size: int = 100,
    seed: int = 42,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    if size < 1:
        raise ValueError("size must be positive")
    candidates = [
        record
        for record in records
        if record.sample_index == 0 and record.score is not None
    ]
    if len(candidates) < size:
        raise ValueError(
            f"calibration requires {size} scored sample_index=0 predictions; found {len(candidates)}"
        )
    ranked = sorted(
        candidates,
        key=lambda record: (
            hashlib.sha256(
                f"{seed}\0{record.sample_id}\0{record.seed}\0{record.sample_index}".encode(
                    "utf-8"
                )
            ).hexdigest(),
            record.sample_id,
            record.seed,
        ),
    )
    selected = ranked[:size]
    rows = [_template_row(record) for record in selected]
    output = Path(output_path)
    _write_jsonl(output, rows)
    prediction_file = Path(predictions_path)
    manifest = {
        "schema_version": "1.0",
        "suite": f"judge-human-calibration-{size}",
        "selection_algorithm": "sample-index-0-scored+sha256(seed,key)-v1",
        "selection_seed": seed,
        "blinded": True,
        "instructions": "Set human_score to 1 for correct or 0 for incorrect; do not change other fields.",
        "num_samples": size,
        "predictions": {
            "path": prediction_file.as_posix(),
            "sha256": sha256_file(prediction_file),
        },
        "template": {
            "path": output.as_posix(),
            "initial_sha256": sha256_file(output),
            "judge_scores_included": False,
        },
        "review_ids": [row["review_id"] for row in rows],
        "immutable_payload_sha256": {
            row["review_id"]: canonical_json_sha256(_immutable_payload(row)) for row in rows
        },
    }
    target_manifest = (
        Path(manifest_path)
        if manifest_path is not None
        else output.with_name(output.name + ".manifest.json")
    )
    write_json(target_manifest, manifest)
    return manifest


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float]:
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return {"lower": max(0.0, centre - margin), "upper": min(1.0, centre + margin)}


def score_calibration_template(
    template_path: str | Path,
    manifest_path: str | Path,
    predictions_path: str | Path,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    review_path = Path(template_path)
    evidence_path = Path(manifest_path)
    prediction_path = Path(predictions_path)
    try:
        manifest = json.loads(evidence_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid calibration manifest: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("calibration manifest must be an object")
    expected_prediction = manifest.get("predictions")
    if not isinstance(expected_prediction, Mapping) or expected_prediction.get("sha256") != sha256_file(
        prediction_path
    ):
        raise ValueError("prediction file hash does not match calibration manifest")
    rows = _read_jsonl(review_path)
    expected_ids = manifest.get("review_ids")
    immutable_hashes = manifest.get("immutable_payload_sha256")
    if not isinstance(expected_ids, list) or not isinstance(immutable_hashes, Mapping):
        raise ValueError("calibration manifest is missing review evidence")
    if len(rows) != int(manifest.get("num_samples", -1)):
        raise ValueError("completed review row count does not match calibration manifest")

    row_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = row.get("review_id")
        if not isinstance(identifier, str) or identifier in row_by_id:
            raise ValueError("review_id must be a unique string")
        if canonical_json_sha256(_immutable_payload(row)) != immutable_hashes.get(identifier):
            raise ValueError(f"review immutable fields changed for {identifier}")
        human = row.get("human_score")
        if isinstance(human, bool) or human not in (0, 1, 0.0, 1.0):
            raise ValueError(f"human_score must be completed with 0 or 1 for {identifier}")
        if not isinstance(row.get("notes"), str):
            raise ValueError(f"notes must be a string for {identifier}")
        row_by_id[identifier] = row
    if set(row_by_id) != set(map(str, expected_ids)):
        raise ValueError("completed review IDs do not exactly match calibration manifest")

    predictions = load_predictions(prediction_path)
    prediction_by_id = {review_id(record): record for record in predictions}
    counts = {"true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0}
    disagreements: list[dict[str, Any]] = []
    judge_positive = 0
    human_positive = 0
    for identifier in map(str, expected_ids):
        record = prediction_by_id.get(identifier)
        if record is None or record.score is None:
            raise ValueError(f"scored prediction is missing for review_id {identifier}")
        human_label = int(row_by_id[identifier]["human_score"])
        judge_label = int(record.score >= threshold)
        judge_positive += judge_label
        human_positive += human_label
        if human_label == 1 and judge_label == 1:
            counts["true_positive"] += 1
        elif human_label == 0 and judge_label == 0:
            counts["true_negative"] += 1
        elif human_label == 0 and judge_label == 1:
            counts["false_positive"] += 1
        else:
            counts["false_negative"] += 1
        if human_label != judge_label:
            disagreements.append(
                {
                    "review_id": identifier,
                    "sample_id": record.sample_id,
                    "judge_score": record.score,
                    "human_score": human_label,
                    "notes": row_by_id[identifier]["notes"],
                }
            )

    total = len(expected_ids)
    agreements = counts["true_positive"] + counts["true_negative"]
    observed = agreements / total
    judge_rate = judge_positive / total
    human_rate = human_positive / total
    expected = judge_rate * human_rate + (1.0 - judge_rate) * (1.0 - human_rate)
    kappa = None if math.isclose(expected, 1.0) else (observed - expected) / (1.0 - expected)
    return {
        "schema_version": "1.0",
        "suite": manifest.get("suite"),
        "num_reviewed": total,
        "judge_threshold": threshold,
        "agreement_rate": observed,
        "agreement_95_ci_wilson": _wilson_interval(agreements, total),
        "target_agreement_rate": 0.9,
        "target_met": observed >= 0.9,
        "cohen_kappa": kappa,
        "judge_positive_rate": judge_rate,
        "human_positive_rate": human_rate,
        "confusion_matrix": counts,
        "disagreements": disagreements,
        "evidence": {
            "predictions_sha256": sha256_file(prediction_path),
            "completed_template_sha256": sha256_file(review_path),
            "calibration_manifest_sha256": sha256_file(evidence_path),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="create a blinded 100-row JSONL template")
    prepare.add_argument("--predictions", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--manifest")
    prepare.add_argument("--seed", type=int, default=42)
    score = subparsers.add_parser("score", help="validate completed review and compute agreement")
    score.add_argument("--template", required=True)
    score.add_argument("--manifest", required=True)
    score.add_argument("--predictions", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--threshold", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        records = load_predictions(args.predictions)
        report = prepare_calibration_template(
            records,
            predictions_path=args.predictions,
            output_path=args.output,
            size=100,
            seed=args.seed,
            manifest_path=args.manifest,
        )
    else:
        report = score_calibration_template(
            args.template,
            args.manifest,
            args.predictions,
            threshold=args.threshold,
        )
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
