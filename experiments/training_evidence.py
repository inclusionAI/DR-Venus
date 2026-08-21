#!/usr/bin/env python3
"""Create and finalize hash-bearing SFT/RL training evidence.

Snapshots are written before a paid run.  Finalization preserves the exit code
and measured usage for both successful and failed experiments.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Iterable

try:
    from drvenus_repro.manifests import sha256_file
    from Inference.evidence_identity import sha256_artifact
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from drvenus_repro.manifests import sha256_file
    from Inference.evidence_identity import sha256_artifact


ROOT = Path(__file__).resolve().parents[1]
_SENSITIVE_KEY = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|access[_-]?token|auth[_-]?token|bearer[_-]?token|secret|password|credential)(?:$|[_-])",
    re.I,
)


def _git(*args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=ROOT, check=check, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _try_git(*args: str) -> str | None:
    try:
        return _git(*args)
    except (OSError, subprocess.CalledProcessError):
        return None


def _upstream_base() -> str:
    for reference in ("upstream/master", "upstream/main", "origin/master", "origin/main"):
        if _try_git("rev-parse", "--verify", reference):
            merge_base = _try_git("merge-base", "HEAD", reference)
            if merge_base:
                return merge_base
    raise RuntimeError("cannot resolve upstream base; pass --upstream-commit")


def _relative(path: Path, anchor: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(), anchor.parent.resolve())).as_posix()
    except ValueError:  # Different Windows drives.
        return path.resolve().as_posix()


def _artifact(path: Path, anchor: Path, *, require_file: bool = False) -> dict[str, Any]:
    if not path.exists() or (require_file and not path.is_file()):
        raise ValueError(f"artifact does not exist: {path}")
    result: dict[str, Any] = {
        "path": _relative(path, anchor),
        "sha256": sha256_artifact(path),
        "kind": "directory" if path.is_dir() else "file",
    }
    if path.is_file():
        result["bytes"] = path.stat().st_size
    return result


def _parse_pairs(values: Iterable[str], *, label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} must use NAME=VALUE: {value!r}")
        name, item = value.split("=", 1)
        name = name.strip()
        if not name or name in result:
            raise ValueError(f"duplicate or empty {label} name: {name!r}")
        if label == "config" and _SENSITIVE_KEY.search(name):
            raise ValueError(f"refusing to persist sensitive config key: {name}")
        result[name] = item
    return result


def _typed(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _dirty_state() -> tuple[bool, str | None, list[str]]:
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if not status:
        return False, None, []
    # Hash the complete binary diff without persisting or printing its content;
    # this avoids leaking a deleted .env while still binding a pilot snapshot.
    tracked = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    untracked: list[tuple[str, str]] = []
    for line in status.splitlines():
        if line.startswith("?? "):
            path = ROOT / line[3:]
            if path.is_file():
                untracked.append((line[3:], sha256_file(path)))
    digest = hashlib.sha256(tracked)
    digest.update(json.dumps(sorted(untracked), separators=(",", ":")).encode())
    return True, digest.hexdigest(), [line[:2] for line in status.splitlines()]


def _environment(stage: str, output: Path, environment_lock: Path | None) -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in (
        "torch",
        "torchdata",
        "transformers",
        "peft",
        "accelerate",
        "flash-attn",
        "vllm",
        "ray",
        "hydra-core",
        "numpy",
        "pandas",
        "pyarrow",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    cuda_runtime = "not-available"
    try:
        import torch

        cuda_runtime = str(torch.version.cuda or "not-available")
    except (ImportError, OSError):
        pass
    driver = "not-available"
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        versions = sorted({line.strip() for line in completed.stdout.splitlines() if line.strip()})
        if versions:
            driver = ",".join(versions)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    requirements = ROOT / stage.upper() / "requirements.txt"
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda_runtime": cuda_runtime,
        "nvidia_driver": driver,
        "container_image_digest": os.environ.get(
            "CONTAINER_IMAGE_DIGEST", "not-recorded"
        ),
        "stage_requirements": _artifact(requirements, output, require_file=True),
    }
    if environment_lock is not None:
        result["environment_lock"] = _artifact(
            environment_lock, output, require_file=True
        )
    else:
        result["environment_lock"] = None
    return result


def create_snapshot(
    *,
    output: Path,
    stage: str,
    run_id: str,
    protocol_label: str,
    checkpoint_identity: Path,
    command_file: Path,
    data: Iterable[str],
    config: Iterable[str],
    gpu_type: str,
    gpu_count: int,
    upstream_commit: str | None = None,
    allow_dirty: bool = False,
    environment_lock: Path | None = None,
) -> dict[str, Any]:
    if stage not in {"sft", "rl"}:
        raise ValueError("stage must be sft or rl")
    if gpu_count <= 0:
        raise ValueError("gpu_count must be positive")
    data_pairs = _parse_pairs(data, label="data")
    config_pairs = _parse_pairs(config, label="config")
    dirty, dirty_hash, dirty_status_codes = _dirty_state()
    if dirty and not allow_dirty:
        raise ValueError(
            "working tree is dirty; formal training evidence requires committed code"
        )
    command = command_file.read_text(encoding="utf-8").strip()
    if not command:
        raise ValueError("resolved command file is empty")
    snapshot = {
        "schema_version": "1.0",
        "kind": "training_run_snapshot",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "run_id": run_id,
        "protocol": {
            "official_comparable": False,
            "label": "dirty-development-only" if dirty else protocol_label,
        },
        "repository": {
            "upstream_base_commit": upstream_commit or _upstream_base(),
            "personal_commit": _git("rev-parse", "HEAD"),
            "working_tree_dirty": dirty,
            "dirty_diff_sha256": dirty_hash,
            "dirty_status_codes": dirty_status_codes,
        },
        "hardware": {"gpu_type": gpu_type, "gpu_count": gpu_count},
        "environment": _environment(stage, output, environment_lock),
        "checkpoint_identity": _artifact(checkpoint_identity, output, require_file=True),
        "resolved_command": {
            **_artifact(command_file, output, require_file=True),
            "text": command,
        },
        "data": {
            name: _artifact(Path(path), output) for name, path in sorted(data_pairs.items())
        },
        "config": {name: _typed(value) for name, value in sorted(config_pairs.items())},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return snapshot


def finalize_snapshot(
    *,
    snapshot_path: Path,
    output: Path,
    exit_code: int,
    elapsed_seconds: float,
    api_cost_cny: float,
    log: Path | None = None,
    checkpoint: Path | None = None,
    extra_artifacts: Iterable[str] = (),
) -> dict[str, Any]:
    if elapsed_seconds < 0 or api_cost_cny < 0:
        raise ValueError("elapsed_seconds and api_cost_cny must be non-negative")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if snapshot.get("kind") != "training_run_snapshot":
        raise ValueError("not a training_run_snapshot")
    gpu_count = int(snapshot["hardware"]["gpu_count"])
    artifacts: dict[str, Any] = {
        "run_snapshot": _artifact(snapshot_path, output, require_file=True)
    }
    if log is not None and log.exists():
        artifacts["training_log"] = _artifact(log, output, require_file=True)
    if checkpoint is not None and checkpoint.exists():
        artifacts["checkpoint"] = _artifact(checkpoint, output)
    for name, path in sorted(_parse_pairs(extra_artifacts, label="artifact").items()):
        if name in artifacts:
            raise ValueError(f"duplicate final artifact name: {name}")
        artifacts[name] = _artifact(Path(path), output)
    manifest = {
        **snapshot,
        "kind": "training_experiment_manifest",
        "finalized_at": datetime.now(timezone.utc).isoformat(),
        "status": "succeeded" if exit_code == 0 else "failed",
        "exit_code": int(exit_code),
        "usage": {
            "wall_seconds": round(float(elapsed_seconds), 6),
            "gpu_hours": round(float(elapsed_seconds) * gpu_count / 3600.0, 6),
            "api_cost_cny": round(float(api_cost_cny), 6),
        },
        "artifacts": artifacts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--output", type=Path, required=True)
    snapshot.add_argument("--stage", choices=("sft", "rl"), required=True)
    snapshot.add_argument("--run-id", required=True)
    snapshot.add_argument("--protocol-label", default="budget-matched-non-official")
    snapshot.add_argument("--checkpoint-identity", type=Path, required=True)
    snapshot.add_argument("--command-file", type=Path, required=True)
    snapshot.add_argument("--data", action="append", default=[])
    snapshot.add_argument("--config", action="append", default=[])
    snapshot.add_argument("--gpu-type", required=True)
    snapshot.add_argument("--gpu-count", type=int, required=True)
    snapshot.add_argument("--upstream-commit")
    snapshot.add_argument("--allow-dirty", action="store_true")
    snapshot.add_argument(
        "--environment-lock",
        type=Path,
        help="pip-freeze/conda lock captured in the actual GPU runtime",
    )
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--snapshot", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--exit-code", type=int, required=True)
    finalize.add_argument("--elapsed-seconds", type=float, required=True)
    finalize.add_argument("--api-cost-cny", type=float, required=True)
    finalize.add_argument("--log", type=Path)
    finalize.add_argument("--checkpoint", type=Path)
    finalize.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="Additional immutable evidence as NAME=PATH (repeatable)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "snapshot":
        create_snapshot(
            output=args.output,
            stage=args.stage,
            run_id=args.run_id,
            protocol_label=args.protocol_label,
            checkpoint_identity=args.checkpoint_identity,
            command_file=args.command_file,
            data=args.data,
            config=args.config,
            gpu_type=args.gpu_type,
            gpu_count=args.gpu_count,
            upstream_commit=args.upstream_commit,
            allow_dirty=args.allow_dirty,
            environment_lock=args.environment_lock,
        )
    else:
        finalize_snapshot(
            snapshot_path=args.snapshot,
            output=args.output,
            exit_code=args.exit_code,
            elapsed_seconds=args.elapsed_seconds,
            api_cost_cny=args.api_cost_cny,
            log=args.log,
            checkpoint=args.checkpoint,
            extra_artifacts=args.artifact,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
