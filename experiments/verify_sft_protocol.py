#!/usr/bin/env python3
"""Fail closed unless S0/S1 inputs satisfy the formal equal-token protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import unicodedata

import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _question_key(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("every row must contain a non-empty question")
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def verify_sft_protocol(
    *,
    s0_path: str | Path,
    s1_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path,
    manifest_path: str | Path,
    data_manifest_path: str | Path,
    expected_token_budget: int = 20_000_000,
    expected_seed: int = 42,
    expected_max_length: int = 32_768,
    train_batch_size: int = 8,
    total_epochs: int = 1,
) -> dict[str, object]:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = {
        "s0_uniform": Path(s0_path).resolve(),
        "s1_turn_weighted": Path(s1_path).resolve(),
    }
    validation_path = Path(validation_path).resolve()
    test_path = Path(test_path).resolve()
    data_manifest_path = Path(data_manifest_path).resolve()
    for path in (*paths.values(), validation_path, test_path, data_manifest_path):
        if not path.is_file():
            raise ValueError(f"dataset does not exist: {path}")

    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    bound_manifest = manifest.get("data_manifest") or {}
    if bound_manifest.get("sha256") != _sha256(data_manifest_path):
        raise ValueError("SFT arms are not bound to the supplied data_manifest")
    correctness_mode = data_manifest.get("correctness", {}).get("mode")
    if correctness_mode not in {"field", "cache", "judge"}:
        raise ValueError("formal SFT requires explicit correctness filtering")
    split = data_manifest.get("split", {})
    if (
        split.get("unit") != "normalized_question"
        or split.get("ratios") != {"train": 0.9, "validation": 0.05, "test": 0.05}
        or split.get("resampling_applied_to") != "train_only"
    ):
        raise ValueError("formal SFT requires the deterministic 90/5/5 question split")
    data_artifacts = data_manifest.get("artifacts", {})
    if data_artifacts.get("train_clean", {}).get("sha256") != manifest.get("input", {}).get(
        "sha256"
    ):
        raise ValueError("SFT arm input is not data_manifest train_clean")
    if data_artifacts.get("validation", {}).get("sha256") != _sha256(validation_path):
        raise ValueError("validation artifact does not match data_manifest")
    if data_artifacts.get("test", {}).get("sha256") != _sha256(test_path):
        raise ValueError("test artifact does not match data_manifest")

    if manifest.get("target_assistant_tokens") != expected_token_budget:
        raise ValueError("SFT arms do not use the approved assistant-token budget")
    if manifest.get("seed") != expected_seed:
        raise ValueError("SFT arms do not use the approved seed")
    counter = manifest.get("token_counter", {})
    if counter.get("type") != "huggingface_multiturn_loss_mask_equivalent":
        raise ValueError("formal SFT requires the exact tokenizer/loss-mask counter")
    if not counter.get("tokenizer"):
        raise ValueError("formal SFT manifest must identify the tokenizer")
    if counter.get("max_length") != expected_max_length or counter.get("truncation") != "right":
        raise ValueError("SFT token counting must match 32K right-truncated training")
    if manifest.get("sampling_batch_multiple") != train_batch_size:
        raise ValueError("SFT arms were not sampled in complete formal train batches")

    artifact_metadata = manifest.get("artifacts", {})
    arm_metadata = manifest.get("arms", {})
    for arm, path in paths.items():
        artifact = artifact_metadata.get(arm, {})
        recorded_path = (manifest_path.parent / str(artifact.get("file", ""))).resolve()
        if recorded_path != path:
            raise ValueError(f"{arm} path does not match its manifest")
        if artifact.get("sha256") != _sha256(path):
            raise ValueError(f"{arm} hash does not match its manifest")
        realized = arm_metadata.get(arm, {}).get("realized_assistant_tokens")
        if not isinstance(realized, int) or realized < expected_token_budget:
            raise ValueError(f"{arm} does not reach the assistant-token target")
        if arm_metadata.get(arm, {}).get("effective_rows_after_drop_last") != arm_metadata.get(
            arm, {}
        ).get("rows") or arm_metadata.get(arm, {}).get("dropped_rows") != 0:
            raise ValueError(f"{arm} would lose rows under drop-last training")
        if arm_metadata.get(arm, {}).get("effective_assistant_tokens") != realized:
            raise ValueError(f"{arm} effective token evidence is inconsistent")

    if train_batch_size <= 0 or total_epochs <= 0:
        raise ValueError("train_batch_size and total_epochs must be positive")
    validation_questions = {
        _question_key(question) for question in pd.read_parquet(validation_path)["question"]
    }
    test_questions = {
        _question_key(question) for question in pd.read_parquet(test_path)["question"]
    }
    if not validation_questions or not test_questions:
        raise ValueError("validation/test set is empty")
    if validation_questions & test_questions:
        raise ValueError("validation and test questions overlap")
    row_counts: dict[str, int] = {}
    for arm, path in paths.items():
        arm_frame = pd.read_parquet(path)
        row_counts[arm] = len(arm_frame)
        train_questions = {_question_key(question) for question in arm_frame["question"]}
        overlap = train_questions & validation_questions
        if overlap:
            raise ValueError(f"{arm} leaks {len(overlap)} normalized validation questions")
        test_overlap = train_questions & test_questions
        if test_overlap:
            raise ValueError(f"{arm} leaks {len(test_overlap)} normalized test questions")

    optimizer_steps = {
        arm: (rows // train_batch_size) * total_epochs for arm, rows in row_counts.items()
    }
    if any(steps <= 0 for steps in optimizer_steps.values()):
        raise ValueError("an SFT arm is too small for the drop-last train batch")
    effective_tokens = {
        arm: int(arm_metadata[arm]["effective_assistant_tokens"]) for arm in paths
    }
    token_gap = abs(
        effective_tokens["s0_uniform"] - effective_tokens["s1_turn_weighted"]
    )
    maximum_token_gap = math.ceil(expected_token_budget * 0.001)
    if manifest.get("between_arm_token_gap") != token_gap:
        raise ValueError("SFT manifest between-arm token gap is inconsistent")
    if manifest.get("between_arm_token_gap_tolerance") != maximum_token_gap:
        raise ValueError("formal SFT arms must declare the 0.1% token-gap tolerance")
    if token_gap > maximum_token_gap:
        raise ValueError("S0/S1 effective token budgets differ by more than 0.1%")
    return {
        "s0_sha256": _sha256(paths["s0_uniform"]),
        "s1_sha256": _sha256(paths["s1_turn_weighted"]),
        "validation_sha256": _sha256(validation_path),
        "target_assistant_tokens": expected_token_budget,
        "seed": expected_seed,
        "tokenizer": counter["tokenizer"],
        "rows": row_counts,
        "optimizer_steps": optimizer_steps,
        "total_optimizer_steps": sum(optimizer_steps.values()),
        "effective_assistant_tokens": effective_tokens,
        "effective_token_gap": token_gap,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s0", required=True)
    parser.add_argument("--s1", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--token-budget", type=int, default=20_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=32_768)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--total-epochs", type=int, default=1)
    parser.add_argument("--print-total-steps", action="store_true")
    args = parser.parse_args(argv)
    result = verify_sft_protocol(
        s0_path=args.s0,
        s1_path=args.s1,
        validation_path=args.validation,
        test_path=args.test,
        manifest_path=args.manifest,
        data_manifest_path=args.data_manifest,
        expected_token_budget=args.token_budget,
        expected_seed=args.seed,
        expected_max_length=args.max_length,
        train_batch_size=args.train_batch_size,
        total_epochs=args.total_epochs,
    )
    if args.print_total_steps:
        print(result["total_optimizer_steps"])
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
