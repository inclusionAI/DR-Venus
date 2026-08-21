#!/usr/bin/env python3
"""Create a validated experiment_manifest.yaml from measured run artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

try:
    from drvenus_repro.manifests import (
        load_predictions,
        sha256_file,
        validate_formal_evaluation_evidence,
        write_experiment_manifest,
    )
    from drvenus_repro.schemas import ExperimentManifest
except ModuleNotFoundError as exc:
    if exc.name != "drvenus_repro":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from drvenus_repro.manifests import (
        load_predictions,
        sha256_file,
        validate_formal_evaluation_evidence,
        write_experiment_manifest,
    )
    from drvenus_repro.schemas import ExperimentManifest


ROOT = Path(__file__).resolve().parents[1]


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _try_git(*arguments: str) -> str | None:
    try:
        return _git(*arguments)
    except (OSError, subprocess.CalledProcessError):
        return None


def _default_upstream_commit() -> str:
    """Resolve the fork base, never silently relabel personal HEAD as upstream."""

    for reference in ("upstream/master", "upstream/main", "origin/master", "origin/main"):
        if _try_git("rev-parse", "--verify", reference):
            merge_base = _try_git("merge-base", "HEAD", reference)
            if merge_base:
                return merge_base
    raise RuntimeError(
        "cannot resolve an upstream base commit; pass --upstream-commit explicitly"
    )


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("torch", "vllm", "ray", "transformers", "pyarrow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    try:
        import torch

        versions["cuda_runtime"] = str(torch.version.cuda or "not-available")
    except (ImportError, OSError):
        versions["cuda_runtime"] = "not-available"
    return versions


def _row_count(path: Path) -> int:
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("parquet manifests require pyarrow") from exc
        return pq.ParquetFile(path).metadata.num_rows
    if path.suffix.lower() == ".jsonl":
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    raise ValueError("evaluation dataset must end in .parquet or .jsonl")


def _relative_artifact(path: Path, manifest_path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), manifest_path.parent.resolve())).as_posix()


def _artifact(path: Path, manifest_path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"artifact does not exist: {path}")
    return {
        "path": _relative_artifact(path, manifest_path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _uniform_metadata(records, field: str, *, required: bool = True) -> Any:
    values: list[Any] = []
    for record in records:
        value = record.metadata.get(field)
        if value is None and required:
            raise ValueError(
                f"formal evidence requires metadata.{field} on every prediction"
            )
        values.append(value)
    encoded = {
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        for value in values
    }
    if len(encoded) != 1:
        raise ValueError(f"predictions mix multiple metadata.{field} values")
    return values[0]


def build_manifest(
    *,
    output_path: str | Path,
    experiment_id: str,
    checkpoint: str,
    model: str,
    evaluation_dataset: str | Path,
    predictions: str | Path,
    seeds: list[int],
    gpu_type: str,
    gpu_count: int,
    gpu_hours: float,
    api_cost_cny: float,
    gpu_hours_source: str = "declared-cli",
    api_cost_source: str = "declared-cli",
    max_turns: int,
    context_length: int,
    tool_cache_mode: str,
    raw_trajectories: str | Path | None = None,
    tool_cache: str | Path | None = None,
    judge_cache: str | Path | None = None,
    upstream_commit: str | None = None,
    checkpoint_revision: str,
    temperature: float = 1.0,
    top_p: float = 0.95,
    samples_per_question: int = 1,
    tool_timeout_seconds: float = 150.0,
    max_tokens_per_turn: int = 1024,
    allow_dirty: bool = False,
    judge_model: str | None = None,
    protocol_label: str = "budget-matched-non-official",
    allow_unscored: bool = False,
    allow_legacy_evidence: bool = False,
) -> ExperimentManifest:
    output_path = Path(output_path)
    dataset_path = Path(evaluation_dataset)
    predictions_path = Path(predictions)
    if tool_cache_mode not in {"online", "record", "replay"}:
        raise ValueError("tool_cache_mode must be online, record, or replay")
    if not checkpoint_revision.strip():
        raise ValueError("checkpoint_revision must be a non-empty immutable revision or hash")
    if tool_cache_mode == "replay" and tool_cache is None:
        raise ValueError("replay evidence requires a tool_cache artifact")
    if not dataset_path.is_file():
        raise ValueError(f"evaluation dataset does not exist: {dataset_path}")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    if samples_per_question <= 0:
        raise ValueError("samples_per_question must be positive")
    if tool_timeout_seconds <= 0:
        raise ValueError("tool_timeout_seconds must be positive")
    if max_tokens_per_turn <= 0:
        raise ValueError("max_tokens_per_turn must be positive")
    if not gpu_hours_source.strip() or not api_cost_source.strip():
        raise ValueError("usage provenance sources must be non-empty")

    records = load_predictions(predictions_path)
    if not allow_unscored and any(record.score is None for record in records):
        raise ValueError("formal evidence requires Judge score coverage for every prediction")
    dataset_rows = _row_count(dataset_path)
    primary_ids = {record.sample_id for record in records if record.sample_index == 0}
    if len(primary_ids) != dataset_rows:
        raise ValueError(
            f"predictions contain {len(primary_ids)} primary samples, dataset has {dataset_rows}"
        )
    expected_indices = set(range(samples_per_question))
    generations: dict[tuple[str, int], set[int]] = {}
    for record in records:
        if record.model != model or record.seed not in seeds:
            raise ValueError("prediction model/seed does not match the manifest request")
        if record.max_turns != max_turns or record.context_length != context_length:
            raise ValueError("prediction limits do not match the manifest request")
        generations.setdefault((record.sample_id, record.seed), set()).add(record.sample_index)
    expected_generation_groups = {
        (sample_id, seed) for sample_id in primary_ids for seed in seeds
    }
    if set(generations) != expected_generation_groups:
        raise ValueError("predictions do not cover every sample/seed combination")
    if any(indices != expected_indices for indices in generations.values()):
        raise ValueError("prediction generations do not match samples_per_question")
    prediction_cost_cny = round(sum(record.cost_cny for record in records), 6)
    if api_cost_cny + 1e-6 < prediction_cost_cny:
        raise ValueError(
            "api_cost_cny cannot be lower than the sum recorded in predictions "
            f"({prediction_cost_cny:.6f})"
        )

    identity_config: dict[str, Any] = {}
    judge_config: dict[str, Any] = {}
    if not allow_legacy_evidence:
        for field, expected in (
            ("temperature", temperature),
            ("top_p", top_p),
            ("tool_timeout_seconds", tool_timeout_seconds),
            ("max_tokens_per_turn", max_tokens_per_turn),
        ):
            actual = _uniform_metadata(records, field)
            if actual != expected:
                raise ValueError(
                    f"prediction metadata.{field}={actual!r} does not match requested {expected!r}"
                )
        actual_cache_mode = _uniform_metadata(records, "tool_cache_mode")
        if actual_cache_mode != tool_cache_mode:
            raise ValueError("prediction tool_cache_mode does not match the manifest request")
        actual_revision = _uniform_metadata(records, "checkpoint_revision")
        if actual_revision != checkpoint_revision:
            raise ValueError(
                "prediction checkpoint_revision does not match the manifest request"
            )
        checkpoint_sha = _uniform_metadata(
            records, "checkpoint_sha256", required=False
        )
        sampling_strategy = _uniform_metadata(records, "sampling_seed_strategy")
        fingerprints: dict[str, str] = {}
        for seed in seeds:
            seed_records = [record for record in records if record.seed == seed]
            fingerprints[str(seed)] = _uniform_metadata(
                seed_records, "run_fingerprint"
            )
        identity_config = {
            "checkpoint_sha256": checkpoint_sha,
            "run_fingerprints": fingerprints,
            "sampling_seed_strategy": sampling_strategy,
        }

        scored = [record for record in records if record.score is not None]
        if scored:
            judges = [record.metadata.get("judge") for record in scored]
            if any(not isinstance(judge, dict) for judge in judges):
                raise ValueError("scored formal evidence requires metadata.judge provenance")
            models = {str(judge.get("model")) for judge in judges}
            prompts = {str(judge.get("prompt_version")) for judge in judges}
            providers = {str(judge.get("provider")) for judge in judges}
            if len(models) != 1 or len(prompts) != 1 or len(providers) != 1:
                raise ValueError("predictions mix Judge model/prompt/provider provenance")
            actual_judge_model = next(iter(models))
            if not judge_model or actual_judge_model != judge_model:
                raise ValueError("prediction Judge model does not match --judge-model")
            judge_config = {
                "model": actual_judge_model,
                "prompt_version": next(iter(prompts)),
                "provider": next(iter(providers)),
            }

    commit = upstream_commit or _default_upstream_commit()
    personal_commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    if dirty and not allow_dirty:
        raise ValueError(
            "working tree is dirty; commit the exact implementation before creating formal evidence"
        )
    artifacts: dict[str, Any] = {
        "predictions": _artifact(predictions_path, output_path),
    }
    if raw_trajectories is not None:
        artifacts["raw_trajectories"] = _artifact(Path(raw_trajectories), output_path)
    if tool_cache is not None:
        artifacts["tool_cache"] = _artifact(Path(tool_cache), output_path)
    if judge_cache is not None:
        artifacts["judge_cache"] = _artifact(Path(judge_cache), output_path)

    requirements_path = ROOT / "requirements-repro.txt"
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "requirements_repro_sha256": sha256_file(requirements_path),
        "packages": _package_versions(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not-set"),
    }
    manifest = ExperimentManifest.from_mapping(
        {
            "schema_version": "1.0",
            "experiment_id": experiment_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "upstream_commit": commit,
            "checkpoint": checkpoint,
            "model": model,
            "evaluation_dataset": {
                "name": dataset_path.stem,
                "path": _relative_artifact(dataset_path, output_path),
                "sha256": sha256_file(dataset_path),
                "num_samples": dataset_rows,
            },
            "seeds": seeds,
            "hardware": {"gpu_type": gpu_type, "gpu_count": gpu_count},
            "environment": environment,
            "limits": {"max_turns": max_turns, "context_length": context_length},
            "usage": {
                "gpu_hours": gpu_hours,
                "api_cost_cny": api_cost_cny,
                "prediction_cost_cny": prediction_cost_cny,
            },
            "protocol": {
                "official_comparable": False,
                "label": (
                    "dirty-development-only"
                    if dirty
                    else protocol_label
                ),
                "tool_cache_mode": tool_cache_mode,
                "evidence_mode": (
                    "legacy-development-only" if allow_legacy_evidence else "formal-v1"
                ),
            },
            "artifacts": artifacts,
            "config": {
                "repository": {
                    "upstream_base_commit": commit,
                    "personal_commit": personal_commit,
                    "working_tree_dirty": dirty,
                    "origin_url": _try_git("config", "--get", "remote.origin.url")
                    or "not-configured",
                },
                "checkpoint_revision": checkpoint_revision,
                "checkpoint_sha256": identity_config.get("checkpoint_sha256"),
                "run_fingerprints": identity_config.get("run_fingerprints", {}),
                "judge_model": judge_model or "not-used",
                "judge": judge_config,
                "usage_provenance": {
                    "gpu_hours_source": gpu_hours_source,
                    "api_cost_source": api_cost_source,
                },
                "generation": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "samples_per_question": samples_per_question,
                    "tool_timeout_seconds": tool_timeout_seconds,
                    "max_tokens_per_turn": max_tokens_per_turn,
                    "sampling_seed_strategy": identity_config.get(
                        "sampling_seed_strategy", "not-recorded"
                    ),
                },
            },
        }
    )
    write_experiment_manifest(output_path, manifest)
    if not allow_legacy_evidence:
        try:
            validate_formal_evaluation_evidence(
                records, manifest, output_path, predictions_path, strict=True
            )
        except Exception:
            output_path.unlink(missing_ok=True)
            raise
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--checkpoint-revision",
        required=True,
        help="Immutable Hugging Face revision, checkpoint hash, or local artifact digest",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--evaluation-dataset", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--gpu-type", required=True)
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument("--gpu-hours", type=float, required=True)
    parser.add_argument("--api-cost-cny", type=float, required=True)
    parser.add_argument("--gpu-hours-source", default="declared-cli")
    parser.add_argument("--api-cost-source", default="declared-cli")
    parser.add_argument("--max-turns", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--tool-cache-mode", choices=["online", "record", "replay"], required=True)
    parser.add_argument("--raw-trajectories")
    parser.add_argument("--tool-cache")
    parser.add_argument("--judge-cache")
    parser.add_argument("--upstream-commit")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--samples-per-question", type=int, default=1)
    parser.add_argument("--tool-timeout-seconds", type=float, default=150.0)
    parser.add_argument("--max-tokens-per-turn", type=int, default=1024)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Development-only escape hatch; output is labelled non-reproducible.",
    )
    parser.add_argument("--judge-model")
    parser.add_argument("--protocol-label", default="budget-matched-non-official")
    parser.add_argument("--allow-unscored", action="store_true")
    parser.add_argument(
        "--allow-legacy-evidence",
        action="store_true",
        help="Development-only: omit canonical execution-identity authentication.",
    )
    args = parser.parse_args(argv)
    manifest = build_manifest(
        output_path=args.output,
        experiment_id=args.experiment_id,
        checkpoint=args.checkpoint,
        model=args.model,
        evaluation_dataset=args.evaluation_dataset,
        predictions=args.predictions,
        seeds=args.seed,
        gpu_type=args.gpu_type,
        gpu_count=args.gpu_count,
        gpu_hours=args.gpu_hours,
        api_cost_cny=args.api_cost_cny,
        gpu_hours_source=args.gpu_hours_source,
        api_cost_source=args.api_cost_source,
        max_turns=args.max_turns,
        context_length=args.context_length,
        tool_cache_mode=args.tool_cache_mode,
        raw_trajectories=args.raw_trajectories,
        tool_cache=args.tool_cache,
        judge_cache=args.judge_cache,
        upstream_commit=args.upstream_commit,
        checkpoint_revision=args.checkpoint_revision,
        temperature=args.temperature,
        top_p=args.top_p,
        samples_per_question=args.samples_per_question,
        tool_timeout_seconds=args.tool_timeout_seconds,
        max_tokens_per_turn=args.max_tokens_per_turn,
        allow_dirty=args.allow_dirty,
        judge_model=args.judge_model,
        protocol_label=args.protocol_label,
        allow_unscored=args.allow_unscored,
        allow_legacy_evidence=args.allow_legacy_evidence,
    )
    print(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
