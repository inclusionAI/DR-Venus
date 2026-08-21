"""Read, write, hash, and validate reproducibility artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .schemas import (
    ExperimentManifest,
    PredictionRecord,
    SchemaValidationError,
    ensure_artifact_path,
    validate_prediction_collection,
    validate_predictions_against_manifest,
)


ANSWER_SPLIT_TOKEN = "<|answer_split|>"


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_text_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_predictions(path: str | Path) -> list[PredictionRecord]:
    result: list[PredictionRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                result.append(PredictionRecord.from_mapping(raw))
            except (json.JSONDecodeError, SchemaValidationError) as exc:
                raise SchemaValidationError(f"{path}:{line_number}: {exc}") from exc
    return validate_prediction_collection(result)


def write_predictions(
    path: str | Path, records: Iterable[PredictionRecord | Mapping[str, Any]]
) -> None:
    normalized = [
        item if isinstance(item, PredictionRecord) else PredictionRecord.from_mapping(item)
        for item in records
    ]
    normalized = validate_prediction_collection(normalized)
    content = "".join(
        json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        for record in normalized
    )
    _atomic_text_write(ensure_artifact_path(path), content)


def load_experiment_manifest(path: str | Path) -> ExperimentManifest:
    """Load JSON-compatible YAML, or ordinary YAML when PyYAML is installed."""

    text = Path(path).read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise SchemaValidationError(
                "non-JSON YAML requires the optional PyYAML package"
            ) from exc
        try:
            raw = yaml.safe_load(text)
        except Exception as exc:  # PyYAML exposes several parser exception types.
            raise SchemaValidationError(f"invalid experiment manifest YAML: {exc}") from exc
    return ExperimentManifest.from_mapping(raw)


def _plain(value: Any) -> Any:
    """Convert pandas/numpy containers from parquet rows to Python values."""

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


def _load_cohort_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.casefold()
    if suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore[import-not-found]
        except ImportError as exc:
            raise SchemaValidationError(
                "validating a parquet evaluation cohort requires pandas and pyarrow"
            ) from exc
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            raise SchemaValidationError(f"cannot read evaluation cohort parquet: {path}: {exc}") from exc
        return [_plain(row) for row in frame.to_dict(orient="records")]
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SchemaValidationError(
                        f"invalid evaluation cohort JSON at {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(row, Mapping):
                    raise SchemaValidationError(
                        f"evaluation cohort row must be an object: {path}:{line_number}"
                    )
                rows.append(_plain(row))
        return rows
    raise SchemaValidationError(
        f"unsupported evaluation cohort format {path.suffix!r}; expected .parquet or .jsonl"
    )


def _cohort_sample_id(row: Mapping[str, Any], row_number: int) -> str:
    extra_info = row.get("extra_info") or {}
    if not isinstance(extra_info, Mapping):
        raise SchemaValidationError(f"cohort row {row_number} extra_info must be an object")
    candidate = row.get("sample_id")
    if candidate is None:
        candidate = extra_info.get("eval_sample_id", extra_info.get("index"))
    if candidate is None or not str(candidate).strip():
        raise SchemaValidationError(
            f"cohort row {row_number} has no sample_id/extra_info.eval_sample_id/index"
        )
    return str(candidate)


def _cohort_references(row: Mapping[str, Any], row_number: int) -> tuple[str, ...]:
    raw_references = row.get("references")
    if raw_references is None:
        reward_model = row.get("reward_model")
        if not isinstance(reward_model, Mapping):
            raise SchemaValidationError(f"cohort row {row_number} reward_model must be an object")
        raw_references = reward_model.get("ground_truth")
    if isinstance(raw_references, str):
        references = tuple(
            part.strip() for part in raw_references.split(ANSWER_SPLIT_TOKEN) if part.strip()
        )
    elif isinstance(raw_references, (list, tuple)):
        references = tuple(
            str(reference).strip()
            for reference in raw_references
            if str(reference).strip()
        )
    else:
        raise SchemaValidationError(
            f"cohort row {row_number} references/ground_truth must be a string or array"
        )
    if not references:
        raise SchemaValidationError(f"cohort row {row_number} has invalid or empty references")
    return references


def validate_evaluation_evidence(
    records: Iterable[PredictionRecord],
    manifest: ExperimentManifest,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Fail-closed validation of predictions against the frozen cohort file.

    The dataset path is resolved relative to the manifest file, not the current
    working directory.  The function verifies the file hash and row count
    before matching primary predictions and all Pass@K generations to cohort
    sample IDs, data sources, and references.
    """

    normalized_records = validate_prediction_collection(records)
    manifest_file = Path(manifest_path).resolve()
    if not manifest_file.is_file():
        raise SchemaValidationError(f"experiment manifest does not exist: {manifest_file}")
    declared_dataset_path = Path(str(manifest.evaluation_dataset["path"]))
    dataset_path = (
        declared_dataset_path
        if declared_dataset_path.is_absolute()
        else manifest_file.parent / declared_dataset_path
    ).resolve()
    if not dataset_path.is_file():
        raise SchemaValidationError(
            "evaluation_dataset.path does not exist relative to the manifest: "
            f"{manifest.evaluation_dataset['path']!r}"
        )

    actual_hash = sha256_file(dataset_path)
    expected_hash = str(manifest.evaluation_dataset["sha256"]).casefold()
    if actual_hash != expected_hash:
        raise SchemaValidationError(
            "evaluation_dataset.sha256 mismatch: "
            f"expected {expected_hash}, computed {actual_hash}"
        )

    rows = _load_cohort_rows(dataset_path)
    expected_count = int(manifest.evaluation_dataset["num_samples"])
    if len(rows) != expected_count:
        raise SchemaValidationError(
            f"evaluation_dataset.num_samples is {expected_count}, cohort contains {len(rows)} rows"
        )

    cohort: dict[str, tuple[str, tuple[str, ...]]] = {}
    for row_number, row in enumerate(rows, start=1):
        sample_id = _cohort_sample_id(row, row_number)
        if sample_id in cohort:
            raise SchemaValidationError(f"duplicate sample_id in evaluation cohort: {sample_id!r}")
        data_source = row.get("data_source")
        if not isinstance(data_source, str) or not data_source.strip():
            raise SchemaValidationError(
                f"cohort row {row_number} has no non-empty data_source"
            )
        cohort[sample_id] = (data_source, _cohort_references(row, row_number))

    primary_ids = {record.sample_id for record in normalized_records if record.sample_index == 0}
    cohort_ids = set(cohort)
    if primary_ids != cohort_ids:
        missing = sorted(cohort_ids - primary_ids)
        unexpected = sorted(primary_ids - cohort_ids)
        raise SchemaValidationError(
            "sample_index=0 predictions must exactly cover the manifest cohort; "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    for record in normalized_records:
        if record.sample_id not in cohort:
            raise SchemaValidationError(
                f"prediction sample_id is absent from evaluation cohort: {record.sample_id!r}"
            )
        expected_source, expected_references = cohort[record.sample_id]
        if record.data_source != expected_source:
            raise SchemaValidationError(
                f"prediction {record.sample_id!r} data_source does not match cohort"
            )
        if record.references != expected_references:
            raise SchemaValidationError(
                f"prediction {record.sample_id!r} references do not match cohort"
            )

    return {
        "declared_path": str(manifest.evaluation_dataset["path"]),
        "sha256": actual_hash,
        "num_samples": len(rows),
        "primary_sample_ids_verified": len(primary_ids),
        "all_generations_verified": len(normalized_records),
    }


def _resolve_manifest_artifact(
    manifest_file: Path,
    name: str,
    entry: Any,
    *,
    strict: bool,
) -> tuple[Path | None, dict[str, Any]]:
    """Resolve and authenticate one manifest artifact.

    Formal evidence uses structured ``{path, sha256, bytes}`` entries.  Plain
    strings predate the evidence contract and are accepted only behind the
    explicit legacy downgrade used for examples/development reports.
    """

    if isinstance(entry, str):
        if strict:
            raise SchemaValidationError(
                f"artifacts.{name} must be an object with path and sha256 for formal evidence"
            )
        return None, {"legacy_unverified_path": entry}
    if not isinstance(entry, Mapping):
        raise SchemaValidationError(f"artifacts.{name} must be an object")
    declared_path = entry.get("path")
    declared_sha = entry.get("sha256")
    if not isinstance(declared_path, str) or not declared_path.strip():
        raise SchemaValidationError(f"artifacts.{name}.path must be a non-empty string")
    if not isinstance(declared_sha, str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", declared_sha
    ):
        raise SchemaValidationError(
            f"artifacts.{name}.sha256 must be 64 hexadecimal characters"
        )
    path = Path(declared_path)
    path = (path if path.is_absolute() else manifest_file.parent / path).resolve()
    if not path.is_file():
        raise SchemaValidationError(f"artifacts.{name}.path does not exist: {declared_path!r}")
    actual_sha = sha256_file(path)
    if actual_sha != declared_sha.casefold():
        raise SchemaValidationError(
            f"artifacts.{name}.sha256 mismatch: expected {declared_sha.casefold()}, "
            f"computed {actual_sha}"
        )
    if "bytes" in entry:
        declared_bytes = entry["bytes"]
        if (
            not isinstance(declared_bytes, int)
            or isinstance(declared_bytes, bool)
            or declared_bytes < 0
            or declared_bytes != path.stat().st_size
        ):
            raise SchemaValidationError(f"artifacts.{name}.bytes does not match the file")
    return path, {
        "declared_path": declared_path,
        "resolved_path": path.as_posix(),
        "sha256": actual_sha,
        "bytes": path.stat().st_size,
    }


def _one_metadata_value(
    records: Iterable[PredictionRecord],
    field: str,
    *,
    required: bool = True,
) -> Any:
    values: list[Any] = []
    for record in records:
        if field not in record.metadata or record.metadata[field] is None:
            if required:
                raise SchemaValidationError(
                    f"prediction {record.sample_id!r} is missing metadata.{field}"
                )
            values.append(None)
        else:
            values.append(record.metadata[field])
    serialized = {
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        for value in values
    }
    if len(serialized) != 1:
        raise SchemaValidationError(f"predictions mix multiple metadata.{field} values")
    return values[0]


def _equal_evidence(actual: Any, expected: Any) -> bool:
    if (
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
    ):
        return math.isfinite(float(actual)) and math.isclose(
            float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12
        )
    return actual == expected


def _load_jsonl_objects(path: Path, *, artifact_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaValidationError(
                    f"artifacts.{artifact_name} contains invalid JSON at line {line_number}"
                ) from exc
            if not isinstance(row, Mapping):
                raise SchemaValidationError(
                    f"artifacts.{artifact_name} line {line_number} must be an object"
                )
            rows.append(dict(row))
    return rows


def _validate_judge_provenance(
    records: list[PredictionRecord],
    manifest: ExperimentManifest,
    judge_cache_path: Path | None,
) -> dict[str, Any]:
    expected_model = manifest.config.get("judge_model")
    judge_config = manifest.config.get("judge", {})
    if not isinstance(judge_config, Mapping):
        raise SchemaValidationError("config.judge must be an object")
    if judge_config.get("model") is not None:
        if expected_model not in (None, judge_config.get("model")):
            raise SchemaValidationError("config.judge.model conflicts with config.judge_model")
        expected_model = judge_config.get("model")
    if any(record.score is not None for record in records):
        if not isinstance(expected_model, str) or expected_model in {"", "not-used"}:
            raise SchemaValidationError("scored formal evidence requires config.judge_model")
    expected_prompt = judge_config.get("prompt_version")
    expected_provider = judge_config.get("provider")

    cache_by_key: dict[str, dict[str, Any]] = {}
    if judge_cache_path is not None:
        for row in _load_jsonl_objects(judge_cache_path, artifact_name="judge_cache"):
            key = row.get("cache_key")
            if not isinstance(key, str) or not key:
                raise SchemaValidationError("judge cache record is missing cache_key")
            if key in cache_by_key and cache_by_key[key] != row:
                raise SchemaValidationError(f"judge cache contains conflicting key {key!r}")
            cache_by_key[key] = row

    verified = 0
    prompt_versions: set[str] = set()
    providers: set[str] = set()
    for record in records:
        if record.score is None:
            continue
        judge = record.metadata.get("judge")
        if not isinstance(judge, Mapping):
            raise SchemaValidationError(
                f"scored prediction {record.sample_id!r} is missing metadata.judge provenance"
            )
        for field in ("model", "provider", "prompt_version", "cache_key", "response_sha256"):
            if not isinstance(judge.get(field), str) or not str(judge[field]).strip():
                raise SchemaValidationError(
                    f"prediction {record.sample_id!r} metadata.judge.{field} is missing"
                )
        if judge["model"] != expected_model:
            raise SchemaValidationError("prediction Judge model does not match manifest config")
        prompt_versions.add(str(judge["prompt_version"]))
        providers.add(str(judge["provider"]))
        if expected_prompt is not None and judge["prompt_version"] != expected_prompt:
            raise SchemaValidationError("prediction Judge prompt_version does not match manifest")
        if expected_provider is not None and judge["provider"] != expected_provider:
            raise SchemaValidationError("prediction Judge provider does not match manifest")
        response_sha = str(judge["response_sha256"])
        if not re.fullmatch(r"[0-9a-fA-F]{64}", response_sha):
            raise SchemaValidationError("metadata.judge.response_sha256 must be hexadecimal SHA-256")
        if judge_cache_path is not None:
            row = cache_by_key.get(str(judge["cache_key"]))
            if row is None:
                raise SchemaValidationError(
                    f"Judge cache lacks prediction cache_key {judge['cache_key']!r}"
                )
            for field in ("model", "provider", "prompt_version", "response_sha256"):
                if row.get(field) != judge.get(field):
                    raise SchemaValidationError(
                        f"Judge cache {field} does not match prediction provenance"
                    )
            if not _equal_evidence(row.get("score"), record.score):
                raise SchemaValidationError("Judge cache score does not match prediction score")
        verified += 1
    if len(prompt_versions) > 1 or len(providers) > 1:
        raise SchemaValidationError("predictions mix Judge prompt/provider provenance")
    if records and any(record.score is not None for record in records):
        if expected_prompt is None or expected_provider is None:
            raise SchemaValidationError(
                "formal scored evidence requires config.judge prompt_version and provider"
            )
    return {
        "scored_predictions_verified": verified,
        "judge_cache_records": len(cache_by_key),
        "model": expected_model,
        "prompt_version": next(iter(prompt_versions), None),
        "provider": next(iter(providers), None),
    }


def validate_formal_evaluation_evidence(
    records: Iterable[PredictionRecord],
    manifest: ExperimentManifest,
    manifest_path: str | Path,
    predictions_path: str | Path,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Authenticate the files and execution identity used by final evaluation.

    This validation deliberately does not trust protocol labels.  Supplying a
    manifest is formal/fail-closed unless the caller explicitly selects the
    legacy-development downgrade.
    """

    normalized = validate_predictions_against_manifest(records, manifest)
    manifest_file = Path(manifest_path).resolve()
    current_predictions = Path(predictions_path).resolve()
    if not current_predictions.is_file():
        raise SchemaValidationError(f"predictions input does not exist: {current_predictions}")
    if load_predictions(current_predictions) != normalized:
        raise SchemaValidationError(
            "in-memory predictions do not match the current predictions input"
        )

    artifact_results: dict[str, Any] = {}
    artifact_paths: dict[str, Path | None] = {}
    if "predictions" not in manifest.artifacts:
        if strict:
            raise SchemaValidationError("formal manifest is missing artifacts.predictions")
    for name, entry in manifest.artifacts.items():
        path, result = _resolve_manifest_artifact(
            manifest_file, str(name), entry, strict=strict
        )
        artifact_paths[str(name)] = path
        artifact_results[str(name)] = result
    declared_predictions = artifact_paths.get("predictions")
    if declared_predictions is not None and declared_predictions != current_predictions:
        raise SchemaValidationError(
            "current predictions input is not the file declared by artifacts.predictions.path"
        )
    if not strict:
        return {
            "mode": "legacy-development-only",
            "identity_verified": False,
            "artifacts": artifact_results,
        }

    config_generation = manifest.config.get("generation")
    if not isinstance(config_generation, Mapping):
        raise SchemaValidationError("formal manifest requires config.generation")
    expected_generation_fields = (
        "temperature",
        "top_p",
        "tool_timeout_seconds",
        "max_tokens_per_turn",
        "sampling_seed_strategy",
    )
    generation_verified: dict[str, Any] = {}
    for field in expected_generation_fields:
        if field not in config_generation:
            raise SchemaValidationError(f"formal manifest is missing config.generation.{field}")
        actual = _one_metadata_value(normalized, field)
        expected = config_generation[field]
        if not _equal_evidence(actual, expected):
            raise SchemaValidationError(
                f"metadata.{field} does not match config.generation.{field}"
            )
        generation_verified[field] = actual
    samples_per_question = config_generation.get("samples_per_question")
    if (
        not isinstance(samples_per_question, int)
        or isinstance(samples_per_question, bool)
        or samples_per_question < 1
    ):
        raise SchemaValidationError(
            "config.generation.samples_per_question must be a positive integer"
        )
    expected_indices = set(range(samples_per_question))
    sample_ids = {record.sample_id for record in normalized if record.sample_index == 0}
    generation_groups: dict[tuple[str, int], set[int]] = {}
    for record in normalized:
        generation_groups.setdefault((record.sample_id, record.seed), set()).add(
            record.sample_index
        )
    expected_groups = {
        (sample_id, seed) for sample_id in sample_ids for seed in manifest.seeds
    }
    if set(generation_groups) != expected_groups or any(
        indices != expected_indices for indices in generation_groups.values()
    ):
        raise SchemaValidationError(
            "predictions do not match config.generation.samples_per_question for every sample/seed"
        )
    seed_strategy = generation_verified["sampling_seed_strategy"]
    seed_match = re.fullmatch(
        r"base\+sample_index\*([1-9][0-9]*)\+llm_call_index", str(seed_strategy)
    )
    if seed_match is None:
        raise SchemaValidationError("unsupported metadata.sampling_seed_strategy")
    sample_seed_stride = int(seed_match.group(1))
    for record in normalized:
        expected_sampling_seed = record.seed + record.sample_index * sample_seed_stride
        if record.metadata.get("sampling_seed") != expected_sampling_seed:
            raise SchemaValidationError(
                f"prediction {record.sample_id!r} metadata.sampling_seed is inconsistent"
            )
        call_count = record.metadata.get("llm_call_count")
        call_seeds = record.metadata.get("llm_sampling_seeds")
        if (
            not isinstance(call_count, int)
            or isinstance(call_count, bool)
            or call_count < 0
            or not isinstance(call_seeds, list)
            or call_seeds
            != [expected_sampling_seed + index for index in range(call_count)]
        ):
            raise SchemaValidationError(
                f"prediction {record.sample_id!r} LLM sampling-seed evidence is inconsistent"
            )
    generation_verified["samples_per_question"] = samples_per_question

    revision = manifest.config.get("checkpoint_revision")
    if not isinstance(revision, str) or not revision.strip():
        raise SchemaValidationError("formal manifest requires config.checkpoint_revision")
    actual_revision = _one_metadata_value(normalized, "checkpoint_revision")
    if actual_revision != revision:
        raise SchemaValidationError(
            "metadata.checkpoint_revision does not match config.checkpoint_revision"
        )
    model_revision = _one_metadata_value(normalized, "model_revision", required=False)
    if model_revision is not None and model_revision != revision:
        raise SchemaValidationError("metadata.model_revision conflicts with checkpoint revision")
    actual_checkpoint_sha = _one_metadata_value(
        normalized, "checkpoint_sha256", required=False
    )
    declared_checkpoint_sha = manifest.config.get("checkpoint_sha256")
    if actual_checkpoint_sha is not None:
        if not isinstance(actual_checkpoint_sha, str) or not re.fullmatch(
            r"[0-9a-fA-F]{64}", actual_checkpoint_sha
        ):
            raise SchemaValidationError("metadata.checkpoint_sha256 must be hexadecimal SHA-256")
        if actual_checkpoint_sha != declared_checkpoint_sha:
            raise SchemaValidationError(
                "metadata.checkpoint_sha256 does not match config.checkpoint_sha256"
            )
    elif declared_checkpoint_sha not in (None, ""):
        raise SchemaValidationError("manifest declares checkpoint SHA absent from predictions")

    fingerprints_by_seed: dict[str, str] = {}
    for seed in manifest.seeds:
        seed_records = [record for record in normalized if record.seed == seed]
        if not seed_records:
            raise SchemaValidationError(f"manifest seed {seed} has no predictions")
        fingerprint = _one_metadata_value(seed_records, "run_fingerprint")
        if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", fingerprint):
            raise SchemaValidationError("metadata.run_fingerprint must be hexadecimal SHA-256")
        fingerprints_by_seed[str(seed)] = fingerprint.casefold()
    declared_fingerprints = manifest.config.get("run_fingerprints")
    if declared_fingerprints != fingerprints_by_seed:
        raise SchemaValidationError(
            "prediction run_fingerprint values do not match config.run_fingerprints"
        )

    cache_mode = _one_metadata_value(normalized, "tool_cache_mode")
    if cache_mode != manifest.protocol.get("tool_cache_mode"):
        raise SchemaValidationError("metadata.tool_cache_mode does not match manifest protocol")
    tool_cache_path = artifact_paths.get("tool_cache")
    tool_cache_sha = _one_metadata_value(normalized, "tool_cache_sha256", required=False)
    if cache_mode == "replay":
        if tool_cache_path is None:
            raise SchemaValidationError("replay evidence requires authenticated artifacts.tool_cache")
        expected_tool_sha = artifact_results["tool_cache"]["sha256"]
        if tool_cache_sha != expected_tool_sha:
            raise SchemaValidationError(
                "metadata.tool_cache_sha256 does not match artifacts.tool_cache.sha256"
            )

    raw_path = artifact_paths.get("raw_trajectories")
    if raw_path is not None:
        raw_lines = raw_path.read_text(encoding="utf-8").splitlines()
        for record in normalized:
            if not record.trajectory_path or "#L" not in record.trajectory_path:
                raise SchemaValidationError(
                    f"prediction {record.sample_id!r} lacks a raw trajectory line reference"
                )
            source, line_text = record.trajectory_path.rsplit("#L", 1)
            try:
                line_number = int(line_text)
            except ValueError as exc:
                raise SchemaValidationError("trajectory_path has an invalid line reference") from exc
            source_path = Path(source)
            if not source_path.is_absolute():
                source_path = (Path.cwd() / source_path).resolve()
            else:
                source_path = source_path.resolve()
            if source_path != raw_path or not 1 <= line_number <= len(raw_lines):
                raise SchemaValidationError(
                    "prediction trajectory_path does not reference artifacts.raw_trajectories"
                )
            try:
                raw_record = json.loads(raw_lines[line_number - 1])
            except (json.JSONDecodeError, TypeError) as exc:
                raise SchemaValidationError(
                    "prediction trajectory_path references an invalid raw JSON record"
                ) from exc
            if not isinstance(raw_record, Mapping):
                raise SchemaValidationError("raw trajectory record must be an object")
            raw_answer = raw_record.get("prediction")
            raw_answer = raw_answer if isinstance(raw_answer, str) else ""
            direct_fields = {
                "sample_id": record.sample_id,
                "data_source": record.data_source,
                "model": record.model,
                "seed": record.seed,
                "sample_index": record.sample_index,
                "references": list(record.references),
                "max_turns": record.max_turns,
                "context_length": record.context_length,
                "prediction": record.answer,
                "final_answer_strict_format_valid": record.format_valid,
            }
            direct_fields["prediction"] = raw_answer
            for field, expected in direct_fields.items():
                actual = raw_answer if field == "prediction" else raw_record.get(field)
                canonical = record.answer if field == "prediction" else expected
                if actual != canonical:
                    raise SchemaValidationError(
                        f"raw trajectory {field} does not match canonical prediction"
                    )
            for field in (
                "run_fingerprint",
                "model_revision",
                "checkpoint_revision",
                "checkpoint_sha256",
                "tool_cache_sha256",
                "tool_timeout_seconds",
                "tool_cache_mode",
                "temperature",
                "top_p",
                "max_tokens_per_turn",
                "sampling_seed",
                "sampling_seed_strategy",
                "llm_sampling_seeds",
                "llm_call_count",
            ):
                if raw_record.get(field) != record.metadata.get(field):
                    raise SchemaValidationError(
                        f"raw trajectory {field} does not match canonical metadata"
                    )

    judge = _validate_judge_provenance(
        normalized, manifest, artifact_paths.get("judge_cache")
    )
    return {
        "mode": "formal-v1",
        "identity_verified": True,
        "artifacts": artifact_results,
        "run_fingerprints": fingerprints_by_seed,
        "checkpoint_revision": revision,
        "checkpoint_sha256": actual_checkpoint_sha,
        "tool_cache_sha256": tool_cache_sha,
        "generation": generation_verified,
        "judge": judge,
    }


def write_experiment_manifest(
    path: str | Path, manifest: ExperimentManifest | Mapping[str, Any]
) -> None:
    """Write deterministic JSON syntax, which is valid YAML 1.2.

    Keeping the on-disk subset JSON-compatible makes validation work without a
    YAML dependency while retaining the requested ``.yaml`` artifact name.
    """

    normalized = (
        manifest if isinstance(manifest, ExperimentManifest) else ExperimentManifest.from_mapping(manifest)
    )
    content = json.dumps(normalized.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_text_write(ensure_artifact_path(path), content)


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_text_write(ensure_artifact_path(path), content)
