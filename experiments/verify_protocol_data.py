#!/usr/bin/env python3
"""Verify the fixed 96-train/100-eval cohorts and question isolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import unicodedata

try:
    from drvenus_repro.manifests import sha256_file
except ModuleNotFoundError as exc:
    if exc.name != "drvenus_repro":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from drvenus_repro.manifests import sha256_file


def _question(prompt) -> str:
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if not isinstance(prompt, list):
        raise ValueError("prompt must be a message list")
    parts = [
        message.get("content", "")
        for message in prompt
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    text = "\n".join(part for part in parts if isinstance(part, str) and part.strip())
    if not text:
        raise ValueError("prompt has no user question")
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def _load_and_verify(path: Path, *, suite: str, count: int):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("protocol verification requires pandas and pyarrow") from exc
    if not path.is_file():
        raise ValueError(f"cohort does not exist: {path}")
    sidecar = path.with_name(path.name + ".manifest.json")
    if not sidecar.is_file():
        raise ValueError(f"cohort manifest does not exist: {sidecar}")
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    if manifest.get("suite") != suite:
        raise ValueError(f"expected suite {suite}, found {manifest.get('suite')}")
    if manifest.get("num_samples") != count:
        raise ValueError(f"expected {count} manifest samples")
    if manifest.get("output", {}).get("sha256") != sha256_file(path):
        raise ValueError(f"cohort SHA-256 mismatch: {path}")
    frame = pd.read_parquet(path)
    if len(frame) != count:
        raise ValueError(f"expected {count} parquet rows, found {len(frame)}")
    questions = [_question(prompt) for prompt in frame["prompt"]]
    if len(set(questions)) != len(questions):
        raise ValueError(f"cohort contains duplicate normalized questions: {path}")
    return manifest, set(questions)


def verify_protocol(
    train_path: str | Path,
    eval_path: str | Path,
    validation_path: str | Path | None = None,
) -> dict:
    train_path = Path(train_path).resolve()
    eval_path = Path(eval_path).resolve()
    train_manifest, train_questions = _load_and_verify(
        train_path, suite="rl-train-96", count=96
    )
    eval_manifest, eval_questions = _load_and_verify(
        eval_path, suite="eval-core-100", count=100
    )
    result = {
        "train_sha256": train_manifest["output"]["sha256"],
        "eval_sha256": eval_manifest["output"]["sha256"],
        "train_questions": len(train_questions),
        "eval_questions": len(eval_questions),
        "overlap": 0,
    }
    overlap = train_questions & eval_questions
    if overlap:
        raise ValueError(f"train/eval question leakage detected ({len(overlap)} questions)")
    if validation_path is not None:
        validation_path = Path(validation_path).resolve()
        validation_manifest, validation_questions = _load_and_verify(
            validation_path, suite="rl-val-32", count=32
        )
        train_val_overlap = train_questions & validation_questions
        val_eval_overlap = validation_questions & eval_questions
        if train_val_overlap or val_eval_overlap:
            raise ValueError(
                "RL validation leakage detected: "
                f"train={len(train_val_overlap)}, final_eval={len(val_eval_overlap)}"
            )
        result.update(
            validation_sha256=validation_manifest["output"]["sha256"],
            validation_questions=len(validation_questions),
        )
    return result


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=str(root / "experiments/data/rl-train-96.parquet"))
    parser.add_argument("--eval", default=str(root / "evaluation/data/eval-core-100.parquet"))
    parser.add_argument("--validation")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            verify_protocol(args.train, args.eval, args.validation),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
