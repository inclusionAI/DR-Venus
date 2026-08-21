import json
from pathlib import Path

import pytest

from evaluation.evaluate import main as evaluate_main

from drvenus_repro.manifests import (
    load_experiment_manifest,
    load_predictions,
    sha256_file,
    validate_evaluation_evidence,
    validate_formal_evaluation_evidence,
    write_experiment_manifest,
    write_predictions,
)
from drvenus_repro.schemas import (
    ExperimentManifest,
    PredictionRecord,
    SchemaValidationError,
    validate_prediction_collection,
    validate_predictions_against_manifest,
)


def prediction_dict(**overrides):
    value = {
        "schema_version": "1.0",
        "sample_id": "nq_1",
        "data_source": "nq",
        "model": "candidate",
        "seed": 42,
        "sample_index": 0,
        "answer": "Paris",
        "references": ["Paris"],
        "score": 1.0,
        "completed": True,
        "format_valid": True,
        "turns": 2,
        "tool_calls": 2,
        "browse_calls": 1,
        "timeout_count": 0,
        "input_tokens": 100,
        "output_tokens": 20,
        "latency_seconds": 1.25,
        "cost_cny": 0.01,
        "max_turns": 12,
        "context_length": 32768,
        "trajectory_path": "trajectories/nq_1.json",
        "metadata": {"judge": "fixture"},
    }
    value.update(overrides)
    return value


def manifest_dict(**overrides):
    value = {
        "schema_version": "1.0",
        "experiment_id": "eval-candidate-seed42",
        "created_at": "2026-08-14T10:00:00+08:00",
        "upstream_commit": "a" * 40,
        "checkpoint": "checkpoints/candidate",
        "model": "candidate",
        "evaluation_dataset": {
            "name": "eval-core-1",
            "path": "evaluation/data/eval-core-1.parquet",
            "sha256": "b" * 64,
            "num_samples": 1,
        },
        "seeds": [42],
        "hardware": {"gpu_type": "A100-80G", "gpu_count": 2},
        "environment": {"python": "3.11", "platform": "linux"},
        "limits": {"max_turns": 12, "context_length": 32768},
        "usage": {"gpu_hours": 1.5, "api_cost_cny": 12.3},
        "protocol": {
            "label": "budget-matched-non-official",
            "official_comparable": False,
        },
        "artifacts": {"predictions": "predictions.jsonl"},
        "config": {"temperature": 0.7},
    }
    value.update(overrides)
    return value


def test_prediction_round_trip_and_total_tokens(tmp_path):
    record = PredictionRecord.from_mapping(prediction_dict())
    assert record.total_tokens == 120
    target = tmp_path / "predictions.jsonl"
    write_predictions(target, [record])
    assert load_predictions(target) == [record]
    assert json.loads(target.read_text(encoding="utf-8"))["max_turns"] == 12


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"schema_version": "2"}, "unsupported"),
        ({"unknown_metric": 1}, "unknown prediction fields"),
        ({"references": []}, "references must not be empty"),
        ({"score": 1.1}, "score must be <= 1"),
        ({"latency_seconds": float("nan")}, "latency_seconds must be finite"),
        ({"browse_calls": 3}, "browse_calls must not exceed tool_calls"),
        ({"turns": 13}, "turns must not exceed max_turns"),
        ({"answer": ""}, "completed prediction must contain an answer"),
        ({"seed": True}, "seed must be an integer"),
    ],
)
def test_prediction_rejects_invalid_values(update, message):
    with pytest.raises(SchemaValidationError, match=message):
        PredictionRecord.from_mapping(prediction_dict(**update))


def test_collection_key_includes_seed_but_rejects_true_duplicates():
    first = PredictionRecord.from_mapping(prediction_dict())
    other_seed = PredictionRecord.from_mapping(prediction_dict(seed=43))
    assert len(validate_prediction_collection([first, other_seed])) == 2
    with pytest.raises(SchemaValidationError, match="duplicate"):
        validate_prediction_collection([first, first])


def test_manifest_json_compatible_yaml_round_trip(tmp_path):
    manifest = ExperimentManifest.from_mapping(manifest_dict())
    target = tmp_path / "experiment_manifest.yaml"
    write_experiment_manifest(target, manifest)
    assert load_experiment_manifest(target) == manifest
    assert target.read_text(encoding="utf-8").lstrip().startswith("{")


def test_manifest_must_mark_results_non_official():
    raw = manifest_dict()
    raw["protocol"] = {
        "label": "official",
        "official_comparable": True,
    }
    with pytest.raises(SchemaValidationError, match="must be false"):
        ExperimentManifest.from_mapping(raw)


@pytest.mark.parametrize(
    "prediction_update",
    [
        {"model": "other"},
        {"seed": 43},
        {"max_turns": 10},
        {"context_length": 4096},
    ],
)
def test_prediction_manifest_cross_validation(prediction_update):
    manifest = ExperimentManifest.from_mapping(manifest_dict())
    record = PredictionRecord.from_mapping(prediction_dict(**prediction_update))
    with pytest.raises(SchemaValidationError):
        validate_predictions_against_manifest([record], manifest)


def test_prediction_loader_reports_line_number(tmp_path):
    target = tmp_path / "bad.jsonl"
    target.write_text("\n{not-json}\n", encoding="utf-8")
    with pytest.raises(SchemaValidationError, match=r":2:"):
        load_predictions(target)


def test_repository_examples_match_the_schemas():
    repository = Path(__file__).resolve().parents[1]
    records = load_predictions(repository / "evaluation/examples/predictions.example.jsonl")
    manifest_path = repository / "evaluation/examples/experiment_manifest.example.yaml"
    manifest = load_experiment_manifest(manifest_path)
    assert len(validate_predictions_against_manifest(records, manifest)) == 1
    evidence = validate_evaluation_evidence(records, manifest, manifest_path)
    assert evidence["primary_sample_ids_verified"] == 1


def _write_cohort_and_manifest(tmp_path, *, row=None, dataset_overrides=None):
    cohort_row = row or {
        "data_source": "nq",
        "prompt": [{"role": "user", "content": "Where?"}],
        "reward_model": {"ground_truth": "Paris"},
        "extra_info": {"eval_sample_id": "nq_1"},
    }
    cohort_path = tmp_path / "cohort.jsonl"
    cohort_path.write_text(json.dumps(cohort_row) + "\n", encoding="utf-8")
    raw_manifest = manifest_dict()
    dataset = {
        "name": "fixture-cohort",
        "path": "cohort.jsonl",
        "sha256": sha256_file(cohort_path),
        "num_samples": 1,
    }
    dataset.update(dataset_overrides or {})
    raw_manifest["evaluation_dataset"] = dataset
    manifest_path = tmp_path / "experiment_manifest.yaml"
    write_experiment_manifest(manifest_path, raw_manifest)
    return cohort_path, manifest_path, load_experiment_manifest(manifest_path)


def test_evidence_validation_resolves_dataset_relative_to_manifest(tmp_path):
    _, manifest_path, manifest = _write_cohort_and_manifest(tmp_path)
    records = [
        PredictionRecord.from_mapping(prediction_dict()),
        PredictionRecord.from_mapping(prediction_dict(sample_index=1)),
    ]
    evidence = validate_evaluation_evidence(records, manifest, manifest_path)
    assert evidence["sha256"] == manifest.evaluation_dataset["sha256"]
    assert evidence["primary_sample_ids_verified"] == 1
    assert evidence["all_generations_verified"] == 2


@pytest.mark.parametrize(
    ("dataset_overrides", "message"),
    [
        ({"path": "missing.jsonl"}, "does not exist"),
        ({"sha256": "f" * 64}, "sha256 mismatch"),
        ({"num_samples": 2}, "num_samples"),
    ],
)
def test_evidence_validation_fails_closed_on_dataset_provenance(
    tmp_path, dataset_overrides, message
):
    _, manifest_path, manifest = _write_cohort_and_manifest(
        tmp_path, dataset_overrides=dataset_overrides
    )
    with pytest.raises(SchemaValidationError, match=message):
        validate_evaluation_evidence(
            [PredictionRecord.from_mapping(prediction_dict())], manifest, manifest_path
        )


@pytest.mark.parametrize(
    ("prediction_update", "message"),
    [
        ({"sample_id": "unexpected"}, "exactly cover"),
        ({"data_source": "hotpotqa"}, "data_source"),
        ({"references": ["London"]}, "references"),
        ({"sample_index": 1}, "exactly cover"),
    ],
)
def test_evidence_validation_fails_closed_on_cohort_semantics(
    tmp_path, prediction_update, message
):
    _, manifest_path, manifest = _write_cohort_and_manifest(tmp_path)
    record = PredictionRecord.from_mapping(prediction_dict(**prediction_update))
    with pytest.raises(SchemaValidationError, match=message):
        validate_evaluation_evidence([record], manifest, manifest_path)


def _write_formal_evidence(tmp_path):
    cohort = tmp_path / "cohort.jsonl"
    cohort.write_text(
        json.dumps(
            {
                "sample_id": "nq_1",
                "data_source": "nq",
                "references": ["Paris"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tool_cache = tmp_path / "tool-cache.jsonl"
    tool_cache.write_text("{}\n", encoding="utf-8")
    judge_cache = tmp_path / "judge-cache.jsonl"
    judge_cache.write_text(
        json.dumps(
            {
                "cache_key": "judge-key",
                "model": "judge-model",
                "provider": "openai-compatible",
                "prompt_version": "answer-correctness-v1",
                "response_sha256": "d" * 64,
                "score": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    metadata = {
        "run_fingerprint": "f" * 64,
        "checkpoint_revision": "c" * 40,
        "checkpoint_sha256": None,
        "model_revision": "c" * 40,
        "tool_cache_mode": "replay",
        "tool_cache_sha256": sha256_file(tool_cache),
        "temperature": 1.0,
        "top_p": 0.95,
        "tool_timeout_seconds": 150.0,
        "max_tokens_per_turn": 1024,
        "sampling_seed_strategy": "base+sample_index*1000003+llm_call_index",
        "sampling_seed": 42,
        "llm_call_count": 1,
        "llm_sampling_seeds": [42],
        "judge": {
            "cache_key": "judge-key",
            "model": "judge-model",
            "provider": "openai-compatible",
            "prompt_version": "answer-correctness-v1",
            "response_sha256": "d" * 64,
        },
    }
    predictions = tmp_path / "predictions.jsonl"
    write_predictions(predictions, [prediction_dict(metadata=metadata)])
    raw = manifest_dict(
        checkpoint="remote/checkpoint",
        evaluation_dataset={
            "name": "cohort",
            "path": "cohort.jsonl",
            "sha256": sha256_file(cohort),
            "num_samples": 1,
        },
        artifacts={
            "predictions": {
                "path": "predictions.jsonl",
                "sha256": sha256_file(predictions),
                "bytes": predictions.stat().st_size,
            },
            "tool_cache": {
                "path": "tool-cache.jsonl",
                "sha256": sha256_file(tool_cache),
                "bytes": tool_cache.stat().st_size,
            },
            "judge_cache": {
                "path": "judge-cache.jsonl",
                "sha256": sha256_file(judge_cache),
                "bytes": judge_cache.stat().st_size,
            },
        },
        protocol={
            "label": "budget-matched-non-official",
            "official_comparable": False,
            "tool_cache_mode": "replay",
            "evidence_mode": "formal-v1",
        },
        config={
            "checkpoint_revision": "c" * 40,
            "checkpoint_sha256": None,
            "run_fingerprints": {"42": "f" * 64},
            "judge_model": "judge-model",
            "judge": {
                "model": "judge-model",
                "provider": "openai-compatible",
                "prompt_version": "answer-correctness-v1",
            },
            "generation": {
                "temperature": 1.0,
                "top_p": 0.95,
                "tool_timeout_seconds": 150.0,
                "max_tokens_per_turn": 1024,
                "sampling_seed_strategy": "base+sample_index*1000003+llm_call_index",
                "samples_per_question": 1,
            },
        },
    )
    manifest_path = tmp_path / "experiment_manifest.yaml"
    write_experiment_manifest(manifest_path, raw)
    return {
        "cohort": cohort,
        "predictions": predictions,
        "tool_cache": tool_cache,
        "judge_cache": judge_cache,
        "manifest_path": manifest_path,
    }


def _rewrite_artifact_hash(manifest_path, artifact_name, artifact_path):
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["artifacts"][artifact_name]["sha256"] = sha256_file(artifact_path)
    raw["artifacts"][artifact_name]["bytes"] = artifact_path.stat().st_size
    write_experiment_manifest(manifest_path, raw)


def test_formal_evidence_authenticates_artifacts_and_identity(tmp_path):
    paths = _write_formal_evidence(tmp_path)
    records = load_predictions(paths["predictions"])
    manifest = load_experiment_manifest(paths["manifest_path"])
    result = validate_formal_evaluation_evidence(
        records, manifest, paths["manifest_path"], paths["predictions"]
    )
    assert result["identity_verified"] is True
    assert result["artifacts"]["predictions"]["sha256"] == sha256_file(
        paths["predictions"]
    )
    assert result["judge"]["scored_predictions_verified"] == 1


def test_evaluate_cli_embeds_formal_authentication_result(tmp_path):
    paths = _write_formal_evidence(tmp_path)
    report_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    assert (
        evaluate_main(
            [
                str(paths["predictions"]),
                "--manifest",
                str(paths["manifest_path"]),
                "--output-json",
                str(report_path),
                "--output-markdown",
                str(markdown_path),
                "--bootstrap-samples",
                "5",
                "--pass-k",
                "1",
            ]
        )
        == 0
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["evaluation_scope"]["artifact_identity_validated"] is True
    assert report["formal_evidence_validation"]["mode"] == "formal-v1"


def test_formal_evidence_rejects_tampered_or_substituted_prediction_file(tmp_path):
    paths = _write_formal_evidence(tmp_path)
    manifest = load_experiment_manifest(paths["manifest_path"])
    records = load_predictions(paths["predictions"])
    substitute = tmp_path / "same-bytes.jsonl"
    substitute.write_bytes(paths["predictions"].read_bytes())
    with pytest.raises(SchemaValidationError, match="not the file declared"):
        validate_formal_evaluation_evidence(
            records, manifest, paths["manifest_path"], substitute
        )

    paths["predictions"].write_text(
        paths["predictions"].read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(SchemaValidationError, match="sha256 mismatch"):
        validate_formal_evaluation_evidence(
            load_predictions(paths["predictions"]),
            manifest,
            paths["manifest_path"],
            paths["predictions"],
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("checkpoint_revision", "e" * 40, "checkpoint_revision"),
        ("run_fingerprint", "e" * 64, "run_fingerprint"),
        ("temperature", 0.7, "temperature"),
        ("tool_cache_sha256", "e" * 64, "tool_cache_sha256"),
    ],
)
def test_formal_evidence_rejects_canonical_identity_mismatch(
    tmp_path, field, value, message
):
    paths = _write_formal_evidence(tmp_path)
    row = json.loads(paths["predictions"].read_text(encoding="utf-8"))
    row["metadata"][field] = value
    paths["predictions"].write_text(json.dumps(row) + "\n", encoding="utf-8")
    _rewrite_artifact_hash(
        paths["manifest_path"], "predictions", paths["predictions"]
    )
    with pytest.raises(SchemaValidationError, match=message):
        validate_formal_evaluation_evidence(
            load_predictions(paths["predictions"]),
            load_experiment_manifest(paths["manifest_path"]),
            paths["manifest_path"],
            paths["predictions"],
        )


def test_formal_evidence_rejects_judge_cache_provenance_tampering(tmp_path):
    paths = _write_formal_evidence(tmp_path)
    row = json.loads(paths["judge_cache"].read_text(encoding="utf-8"))
    row["score"] = 0.0
    paths["judge_cache"].write_text(json.dumps(row) + "\n", encoding="utf-8")
    _rewrite_artifact_hash(paths["manifest_path"], "judge_cache", paths["judge_cache"])
    with pytest.raises(SchemaValidationError, match="cache score"):
        validate_formal_evaluation_evidence(
            load_predictions(paths["predictions"]),
            load_experiment_manifest(paths["manifest_path"]),
            paths["manifest_path"],
            paths["predictions"],
        )
