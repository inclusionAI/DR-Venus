#!/usr/bin/env python3
"""Verify a formal experiment's immutable starting-checkpoint identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

try:
    from Inference.evidence_identity import sha256_artifact
except ModuleNotFoundError as exc:
    if exc.name != "Inference":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Inference.evidence_identity import sha256_artifact


def verify_checkpoint(
    checkpoint: str,
    *,
    expected_model_id: str,
    revision: str,
    expected_sha256: str | None = None,
) -> dict[str, str]:
    if not expected_model_id.strip():
        raise ValueError("expected_model_id is required")
    path = Path(checkpoint).expanduser()
    if path.exists():
        if not expected_sha256 or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
            raise ValueError("local checkpoints require a 64-hex expected SHA-256")
        actual = sha256_artifact(path)
        if actual.casefold() != expected_sha256.casefold():
            raise ValueError("local checkpoint SHA-256 mismatch")
        if revision.casefold() != actual.casefold():
            raise ValueError("local MODEL_REVISION must equal the checkpoint tree SHA-256")
        return {
            "kind": "local",
            "path": str(path.resolve()),
            "declared_model_id": expected_model_id,
            "revision": actual,
            "sha256": actual,
        }

    if checkpoint != expected_model_id:
        raise ValueError(
            f"formal remote checkpoint must be {expected_model_id!r}, got {checkpoint!r}"
        )
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
        raise ValueError("remote MODEL_REVISION must be an immutable 40-64 hex commit")
    return {
        "kind": "remote",
        "path": checkpoint,
        "declared_model_id": expected_model_id,
        "revision": revision.casefold(),
        "sha256": "remote-revision",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = verify_checkpoint(
        args.checkpoint,
        expected_model_id=args.expected_model_id,
        revision=args.revision,
        expected_sha256=args.expected_sha256,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8", newline="\n")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
