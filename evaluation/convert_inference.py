"""Convert raw Inference/run_demo.py trajectories to canonical predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Mapping

from drvenus_repro.manifests import write_predictions
from drvenus_repro.schemas import PredictionRecord


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def _format_valid(messages: Any) -> bool:
    if not isinstance(messages, list):
        return False
    for message in reversed(messages):
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        return isinstance(content, str) and bool(
            (match := ANSWER_RE.search(content)) and match.group(1).strip()
        )
    return False


def convert_raw_record(
    raw: Mapping[str, Any], *, trajectory_path: str, line_number: int
) -> PredictionRecord:
    termination = str(raw.get("termination", "exception"))
    prediction = raw.get("prediction")
    answer = prediction if isinstance(prediction, str) else ""
    completed = termination.startswith("answered") and bool(answer.strip())
    strict_format_valid = raw.get("final_answer_strict_format_valid")
    if not isinstance(strict_format_valid, bool):
        raise ValueError(
            "raw record is missing boolean final_answer_strict_format_valid evidence"
        )
    return PredictionRecord.from_mapping(
        {
            "schema_version": "1.0",
            "sample_id": raw["sample_id"],
            "data_source": raw["data_source"],
            "model": raw["model"],
            "seed": raw["seed"],
            "sample_index": raw.get("sample_index", 0),
            "answer": answer,
            "references": raw["references"],
            "score": None,
            "completed": completed,
            "format_valid": strict_format_valid,
            "turns": raw.get("turns", 0),
            "tool_calls": raw.get("tool_call_count", 0),
            "browse_calls": raw.get("browse_call_count", 0),
            "timeout_count": raw.get("timeout_count", 0),
            "input_tokens": raw.get("input_tokens", 0),
            "output_tokens": raw.get("output_tokens", 0),
            "latency_seconds": raw.get("elapsed_seconds", 0.0),
            "cost_cny": raw.get("cost_cny", 0.0),
            "max_turns": raw["max_turns"],
            "context_length": raw["context_length"],
            "trajectory_path": f"{trajectory_path}#L{line_number}",
            "metadata": {
                "termination": termination,
                "parse_error_count": raw.get("parse_error_count", 0),
                "dedup_count": raw.get("dedup_count", 0),
                "tool_error_count": raw.get("tool_error_count", 0),
                "final_context_tokens": raw.get("final_token_count", 0),
                "question": raw.get("question", ""),
                "strict_format_violation_count": raw.get(
                    "strict_format_violation_count", 0
                ),
                "legacy_sanitized_format_valid": _format_valid(raw.get("messages")),
                "tool_cache_mode": raw.get("tool_cache_mode"),
                "cache_hit_count": raw.get("cache_hit_count", 0),
                "cache_miss_count": raw.get("cache_miss_count", 0),
                "cached_tool_error_count": raw.get("cached_tool_error_count", 0),
                "run_fingerprint": raw.get("run_fingerprint"),
                "model_revision": raw.get("model_revision"),
                "checkpoint_revision": raw.get("checkpoint_revision"),
                "checkpoint_sha256": raw.get("checkpoint_sha256"),
                "tool_cache_sha256": raw.get("tool_cache_sha256"),
                "tool_timeout_seconds": raw.get("tool_timeout_seconds"),
                "temperature": raw.get("temperature"),
                "top_p": raw.get("top_p"),
                "max_tokens_per_turn": raw.get("max_tokens_per_turn"),
                "sampling_seed": raw.get("sampling_seed"),
                "sampling_seed_strategy": raw.get("sampling_seed_strategy"),
                "llm_sampling_seeds": raw.get("llm_sampling_seeds", []),
                "llm_call_count": raw.get("llm_call_count", 0),
                "llm_failed_call_count": raw.get("llm_failed_call_count", 0),
                "llm_usage_missing_count": raw.get("llm_usage_missing_count", 0),
                "transcript_input_tokens": raw.get("transcript_input_tokens", 0),
                "transcript_output_tokens": raw.get("transcript_output_tokens", 0),
                "tool_response_truncation_count": raw.get(
                    "tool_response_truncation_count", 0
                ),
                "tool_response_tokens_removed": raw.get(
                    "tool_response_tokens_removed", 0
                ),
                "server_reused": raw.get("server_reused"),
                "server_start_timeout": raw.get("server_start_timeout"),
            },
        }
    )


def convert_file(input_path: str | Path, output_path: str | Path) -> list[PredictionRecord]:
    input_path = Path(input_path)
    # A resumed run may append a successful retry after an exception/timeout.
    # Keep the latest attempt per evidence key while preserving fail-closed
    # parsing for every historical line.
    latest: dict[tuple[str, int, int], tuple[int, Mapping[str, Any]]] = {}
    fingerprints: set[str] = set()
    for line_number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            fingerprint = raw.get("run_fingerprint")
            if isinstance(fingerprint, str) and fingerprint:
                fingerprints.add(fingerprint)
            key = (str(raw["sample_id"]), int(raw.get("seed", 0)), int(raw.get("sample_index", 0)))
            latest[key] = (line_number, raw)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{input_path}:{line_number}: cannot convert record: {exc}") from exc
    if not latest:
        raise ValueError("raw inference file contains no records")
    if len(fingerprints) > 1:
        raise ValueError(
            "raw inference file mixes multiple run_fingerprint values; use a separate output file per configuration"
        )
    records = [
        convert_raw_record(raw, trajectory_path=input_path.as_posix(), line_number=line_number)
        for line_number, raw in sorted(latest.values(), key=lambda item: item[0])
    ]
    write_predictions(output_path, records)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args(argv)
    records = convert_file(args.input, args.output)
    print(f"Wrote {len(records)} canonical predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
