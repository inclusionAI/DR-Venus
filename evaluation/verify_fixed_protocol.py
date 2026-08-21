#!/usr/bin/env python3
"""Verify that a final evaluation invocation matches a frozen suite contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drvenus_repro.manifests import sha256_file
from evaluation.cohort_io import load_rows


DATA_DIR = Path(__file__).resolve().parent / "data"
SUITES = {
    "eval-core-100": {
        "rows": 100,
        "samples": 1,
        "max_turns": 12,
        "file": "eval-core-100.parquet",
        "sha256": "b7d6f405d104110a3c56dcaab6acf68b95631a53a0e0f98666dee11fef70b155",
        "manifest_sha256": "c75d1fa2f0b52582c21e8ae9027275bb599bdf97816819201b80f3983bc7d5df",
    },
    "pass4-20": {
        "rows": 20,
        "samples": 4,
        "max_turns": 12,
        "file": "pass4-20.parquet",
        "sha256": "c4fceeae246427cb16b2cfecd40086d1e5e0214be10ee2aff987773c800a2de5",
        "manifest_sha256": "d43cb5b45e47fe5011a096f5f7f8cef75af44a22f9dc495dc6ed47168f020f04",
        "parent_sha256": "b7d6f405d104110a3c56dcaab6acf68b95631a53a0e0f98666dee11fef70b155",
    },
    # eval-web-60 is deliberately absent until real BrowseComp and
    # BrowseComp-ZH inputs are provided and the generated cohort is committed
    # with its immutable hashes. A caller cannot self-assert a formal sidecar.
}


def verify_fixed_protocol(
    cohort_path: str | Path,
    *,
    max_turns: int,
    context_length: int,
    samples_per_question: int,
    seed: int,
    max_tokens_per_turn: int,
) -> dict:
    cohort = Path(cohort_path).resolve()
    manifest_path = cohort.with_name(cohort.name + ".manifest.json")
    if not manifest_path.is_file():
        raise ValueError(f"frozen cohort manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    suite = manifest.get("suite")
    if suite not in SUITES:
        raise ValueError(f"unsupported formal evaluation suite: {suite!r}")
    contract = SUITES[suite]
    expected_cohort = (DATA_DIR / contract["file"]).resolve()
    if cohort != expected_cohort:
        raise ValueError(
            f"formal suite {suite} must use the committed cohort: {expected_cohort}"
        )
    actual_sha = sha256_file(cohort)
    actual_manifest_sha = sha256_file(manifest_path)
    if actual_sha != contract["sha256"] or actual_manifest_sha != contract["manifest_sha256"]:
        raise ValueError("committed cohort or sidecar hash differs from the frozen contract")
    rows = load_rows(cohort)
    if len(rows) != contract["rows"] or manifest.get("num_samples") != contract["rows"]:
        raise ValueError("cohort row count does not match the frozen suite contract")
    if manifest.get("output", {}).get("sha256") != actual_sha:
        raise ValueError("cohort SHA-256 does not match its sidecar")
    if "parent_sha256" in contract:
        if manifest.get("parent", {}).get("cohort_sha256") != contract["parent_sha256"]:
            raise ValueError("Pass@4 cohort is not bound to the frozen eval-core parent")
    if samples_per_question != contract["samples"]:
        raise ValueError("samples_per_question does not match the frozen suite contract")
    if max_turns != contract["max_turns"]:
        raise ValueError("max_turns does not match the frozen suite contract")
    if context_length != 32_768 or max_tokens_per_turn != 1_024:
        raise ValueError("formal evaluation requires 32K context and 1024 tokens per turn")
    if seed != 42:
        raise ValueError("formal evaluation requires seed 42")
    return {
        "suite": suite,
        "rows": len(rows),
        "samples_per_question": samples_per_question,
        "max_turns": max_turns,
        "context_length": context_length,
        "max_tokens_per_turn": max_tokens_per_turn,
        "cohort_sha256": actual_sha,
        "manifest_sha256": actual_manifest_sha,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--max-turns", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--samples-per-question", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-tokens-per-turn", type=int, required=True)
    args = parser.parse_args(argv)
    result = verify_fixed_protocol(
        args.cohort,
        max_turns=args.max_turns,
        context_length=args.context_length,
        samples_per_question=args.samples_per_question,
        seed=args.seed,
        max_tokens_per_turn=args.max_tokens_per_turn,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
