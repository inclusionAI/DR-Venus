"""Validated schemas for reproducible evaluation artifacts.

The module deliberately uses dataclasses instead of a validation framework so
that predictions can be inspected and validated in a minimal Python
environment.  Unknown fields are rejected: experiment evidence should fail
early on misspelled metric names instead of silently dropping them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "1.0"


class SchemaValidationError(ValueError):
    """Raised when a reproducibility artifact does not match its schema."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _required(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise SchemaValidationError(f"missing required field: {key}")
    return mapping[key]


def _string(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise SchemaValidationError(f"{name} must be a string")
    if not allow_empty and not value.strip():
        raise SchemaValidationError(f"{name} must not be empty")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if not _is_int(value) or value < minimum:
        raise SchemaValidationError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise SchemaValidationError(f"{name} must be finite")
    if result < minimum:
        raise SchemaValidationError(f"{name} must be >= {minimum}")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise SchemaValidationError(f"{name} must be a boolean")
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaValidationError(f"{name} must be an object")
    return dict(value)


@dataclass(frozen=True)
class PredictionRecord:
    """One model generation and its resource/trajectory evidence.

    ``sample_index`` distinguishes repeated generations of the same example
    for Pass@K.  ``score`` is a normalized judge score in ``[0, 1]`` and may be
    null; EM is used as the accuracy fallback when a complete file has no judge
    scores.
    """

    sample_id: str
    data_source: str
    model: str
    seed: int
    sample_index: int
    answer: str
    references: tuple[str, ...]
    score: float | None
    completed: bool
    format_valid: bool
    turns: int
    tool_calls: int
    browse_calls: int
    timeout_count: int
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    cost_cny: float
    max_turns: int
    context_length: int
    trajectory_path: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    _FIELDS = frozenset(
        {
            "schema_version",
            "sample_id",
            "data_source",
            "model",
            "seed",
            "sample_index",
            "answer",
            "references",
            "score",
            "completed",
            "format_valid",
            "turns",
            "tool_calls",
            "browse_calls",
            "timeout_count",
            "input_tokens",
            "output_tokens",
            "latency_seconds",
            "cost_cny",
            "max_turns",
            "context_length",
            "trajectory_path",
            "metadata",
        }
    )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PredictionRecord":
        if not isinstance(value, Mapping):
            raise SchemaValidationError("prediction must be an object")
        unknown = sorted(set(value) - cls._FIELDS)
        if unknown:
            raise SchemaValidationError(f"unknown prediction fields: {', '.join(unknown)}")

        version = _string(_required(value, "schema_version"), "schema_version")
        if version != SCHEMA_VERSION:
            raise SchemaValidationError(
                f"unsupported prediction schema_version {version!r}; expected {SCHEMA_VERSION!r}"
            )

        raw_references = _required(value, "references")
        if not isinstance(raw_references, Sequence) or isinstance(raw_references, (str, bytes)):
            raise SchemaValidationError("references must be a non-empty array of strings")
        references = tuple(_string(item, "references[]") for item in raw_references)
        if not references:
            raise SchemaValidationError("references must not be empty")

        raw_score = _required(value, "score")
        score = None if raw_score is None else _number(raw_score, "score")
        if score is not None and score > 1.0:
            raise SchemaValidationError("score must be <= 1")

        trajectory = _required(value, "trajectory_path")
        if trajectory is not None:
            trajectory = _string(trajectory, "trajectory_path")

        record = cls(
            schema_version=version,
            sample_id=_string(_required(value, "sample_id"), "sample_id"),
            data_source=_string(_required(value, "data_source"), "data_source"),
            model=_string(_required(value, "model"), "model"),
            seed=_integer(_required(value, "seed"), "seed"),
            sample_index=_integer(_required(value, "sample_index"), "sample_index"),
            answer=_string(_required(value, "answer"), "answer", allow_empty=True),
            references=references,
            score=score,
            completed=_boolean(_required(value, "completed"), "completed"),
            format_valid=_boolean(_required(value, "format_valid"), "format_valid"),
            turns=_integer(_required(value, "turns"), "turns"),
            tool_calls=_integer(_required(value, "tool_calls"), "tool_calls"),
            browse_calls=_integer(_required(value, "browse_calls"), "browse_calls"),
            timeout_count=_integer(_required(value, "timeout_count"), "timeout_count"),
            input_tokens=_integer(_required(value, "input_tokens"), "input_tokens"),
            output_tokens=_integer(_required(value, "output_tokens"), "output_tokens"),
            latency_seconds=_number(_required(value, "latency_seconds"), "latency_seconds"),
            cost_cny=_number(_required(value, "cost_cny"), "cost_cny"),
            max_turns=_integer(_required(value, "max_turns"), "max_turns", minimum=1),
            context_length=_integer(
                _required(value, "context_length"), "context_length", minimum=1
            ),
            trajectory_path=trajectory,
            metadata=_mapping(value.get("metadata", {}), "metadata"),
        )
        record.validate_relations()
        return record

    def validate_relations(self) -> None:
        if self.browse_calls > self.tool_calls:
            raise SchemaValidationError("browse_calls must not exceed tool_calls")
        if self.turns > self.max_turns:
            raise SchemaValidationError("turns must not exceed max_turns")
        if self.completed and not self.answer.strip():
            raise SchemaValidationError("a completed prediction must contain an answer")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "data_source": self.data_source,
            "model": self.model,
            "seed": self.seed,
            "sample_index": self.sample_index,
            "answer": self.answer,
            "references": list(self.references),
            "score": self.score,
            "completed": self.completed,
            "format_valid": self.format_valid,
            "turns": self.turns,
            "tool_calls": self.tool_calls,
            "browse_calls": self.browse_calls,
            "timeout_count": self.timeout_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_seconds": self.latency_seconds,
            "cost_cny": self.cost_cny,
            "max_turns": self.max_turns,
            "context_length": self.context_length,
            "trajectory_path": self.trajectory_path,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ExperimentManifest:
    """Provenance, budget, and evaluation limits for one experiment."""

    experiment_id: str
    created_at: str
    upstream_commit: str
    checkpoint: str
    model: str
    evaluation_dataset: dict[str, Any]
    seeds: tuple[int, ...]
    hardware: dict[str, Any]
    environment: dict[str, Any]
    limits: dict[str, Any]
    usage: dict[str, Any]
    protocol: dict[str, Any]
    artifacts: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    _FIELDS = frozenset(
        {
            "schema_version",
            "experiment_id",
            "created_at",
            "upstream_commit",
            "checkpoint",
            "model",
            "evaluation_dataset",
            "seeds",
            "hardware",
            "environment",
            "limits",
            "usage",
            "protocol",
            "artifacts",
            "config",
        }
    )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExperimentManifest":
        if not isinstance(value, Mapping):
            raise SchemaValidationError("experiment manifest must be an object")
        unknown = sorted(set(value) - cls._FIELDS)
        if unknown:
            raise SchemaValidationError(f"unknown manifest fields: {', '.join(unknown)}")
        version = _string(_required(value, "schema_version"), "schema_version")
        if version != SCHEMA_VERSION:
            raise SchemaValidationError(
                f"unsupported manifest schema_version {version!r}; expected {SCHEMA_VERSION!r}"
            )

        created_at = _string(_required(value, "created_at"), "created_at")
        try:
            parsed_created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SchemaValidationError("created_at must be an ISO-8601 timestamp") from exc
        if parsed_created_at.tzinfo is None:
            raise SchemaValidationError("created_at must include a timezone offset")

        seeds_raw = _required(value, "seeds")
        if not isinstance(seeds_raw, Sequence) or isinstance(seeds_raw, (str, bytes)):
            raise SchemaValidationError("seeds must be a non-empty array of integers")
        seeds = tuple(_integer(seed, "seeds[]") for seed in seeds_raw)
        if not seeds:
            raise SchemaValidationError("seeds must not be empty")
        if len(set(seeds)) != len(seeds):
            raise SchemaValidationError("seeds must not contain duplicates")

        manifest = cls(
            schema_version=version,
            experiment_id=_string(_required(value, "experiment_id"), "experiment_id"),
            created_at=created_at,
            upstream_commit=_string(_required(value, "upstream_commit"), "upstream_commit"),
            checkpoint=_string(_required(value, "checkpoint"), "checkpoint"),
            model=_string(_required(value, "model"), "model"),
            evaluation_dataset=_mapping(
                _required(value, "evaluation_dataset"), "evaluation_dataset"
            ),
            seeds=seeds,
            hardware=_mapping(_required(value, "hardware"), "hardware"),
            environment=_mapping(_required(value, "environment"), "environment"),
            limits=_mapping(_required(value, "limits"), "limits"),
            usage=_mapping(_required(value, "usage"), "usage"),
            protocol=_mapping(_required(value, "protocol"), "protocol"),
            artifacts=_mapping(value.get("artifacts", {}), "artifacts"),
            config=_mapping(value.get("config", {}), "config"),
        )
        manifest.validate_nested()
        return manifest

    def validate_nested(self) -> None:
        dataset = self.evaluation_dataset
        _string(_required(dataset, "name"), "evaluation_dataset.name")
        _string(_required(dataset, "path"), "evaluation_dataset.path")
        dataset_hash = _string(_required(dataset, "sha256"), "evaluation_dataset.sha256")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", dataset_hash):
            raise SchemaValidationError("evaluation_dataset.sha256 must be 64 hexadecimal characters")
        _integer(_required(dataset, "num_samples"), "evaluation_dataset.num_samples", minimum=1)

        _string(_required(self.hardware, "gpu_type"), "hardware.gpu_type")
        _integer(_required(self.hardware, "gpu_count"), "hardware.gpu_count")

        _string(_required(self.environment, "python"), "environment.python")
        _string(_required(self.environment, "platform"), "environment.platform")

        _integer(_required(self.limits, "max_turns"), "limits.max_turns", minimum=1)
        _integer(
            _required(self.limits, "context_length"), "limits.context_length", minimum=1
        )

        _number(_required(self.usage, "gpu_hours"), "usage.gpu_hours")
        _number(_required(self.usage, "api_cost_cny"), "usage.api_cost_cny")

        official = _required(self.protocol, "official_comparable")
        if official is not False:
            raise SchemaValidationError(
                "protocol.official_comparable must be false for budget-matched DR-Venus results"
            )
        _string(_required(self.protocol, "label"), "protocol.label")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "created_at": self.created_at,
            "upstream_commit": self.upstream_commit,
            "checkpoint": self.checkpoint,
            "model": self.model,
            "evaluation_dataset": self.evaluation_dataset,
            "seeds": list(self.seeds),
            "hardware": self.hardware,
            "environment": self.environment,
            "limits": self.limits,
            "usage": self.usage,
            "protocol": self.protocol,
            "artifacts": self.artifacts,
            "config": self.config,
        }


def validate_prediction_collection(records: Iterable[PredictionRecord]) -> list[PredictionRecord]:
    """Validate collection-wide invariants and return a materialized list."""

    result = list(records)
    if not result:
        raise SchemaValidationError("predictions must not be empty")
    seen: set[tuple[str, int, int]] = set()
    for record in result:
        key = (record.sample_id, record.seed, record.sample_index)
        if key in seen:
            raise SchemaValidationError(
                "duplicate (sample_id, seed, sample_index): "
                f"{record.sample_id!r}, {record.seed}, {record.sample_index}"
            )
        seen.add(key)
    return result


def validate_predictions_against_manifest(
    records: Iterable[PredictionRecord], manifest: ExperimentManifest
) -> list[PredictionRecord]:
    """Cross-check predictions against manifest limits, model, seeds, and cohort size."""

    result = validate_prediction_collection(records)
    unique_samples = {record.sample_id for record in result}
    expected_count = int(manifest.evaluation_dataset["num_samples"])
    if len(unique_samples) != expected_count:
        raise SchemaValidationError(
            f"manifest expects {expected_count} unique samples, predictions contain {len(unique_samples)}"
        )
    for record in result:
        if record.model != manifest.model:
            raise SchemaValidationError(
                f"prediction model {record.model!r} does not match manifest model {manifest.model!r}"
            )
        if record.seed not in manifest.seeds:
            raise SchemaValidationError(
                f"prediction seed {record.seed} is absent from manifest seeds {manifest.seeds}"
            )
        if record.max_turns != manifest.limits["max_turns"]:
            raise SchemaValidationError("prediction max_turns does not match manifest limits")
        if record.context_length != manifest.limits["context_length"]:
            raise SchemaValidationError("prediction context_length does not match manifest limits")
    return result


def ensure_artifact_path(path: str | Path) -> Path:
    """Return a normalized path and reject directory targets."""

    result = Path(path)
    if result.exists() and result.is_dir():
        raise SchemaValidationError(f"artifact path points to a directory: {result}")
    return result
