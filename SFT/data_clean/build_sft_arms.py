#!/usr/bin/env python3
"""Build token-budget-matched uniform and turn-weighted SFT arms.

The input should be ``train_clean.parquet`` from ``prepare_trajectories.py``.
Both arms draw complete trajectories with replacement until they reach the same
target assistant-token budget.  Because conversations are never truncated, the
realized budget may overshoot by at most one selected train batch; the manifest
records target, realized, and drop-last-effective counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from .data_governance import logical_records_sha256, sha256_file
except ImportError:  # Support direct execution.
    from data_governance import logical_records_sha256, sha256_file  # type: ignore[no-redef]


TextTokenCounter = Callable[[str], int]
AssistantTokenCounter = Callable[[Any], int]
_APPROX_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\s]", re.UNICODE)


def _messages_to_list(messages: Any) -> list[dict[str, Any]]:
    if isinstance(messages, np.ndarray):
        messages = messages.tolist()
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    result = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("every message must be an object")
        result.append(dict(message))
    return result


def approximate_text_tokens(text: str) -> int:
    """Dependency-free deterministic estimate for smoke tests only."""

    return len(_APPROX_TOKEN_RE.findall(text))


def count_assistant_tokens(messages: Any, text_counter: TextTokenCounter = approximate_text_tokens) -> int:
    """Count assistant content tokens; ``text_counter`` is injectable in tests."""

    total = 0
    for message in _messages_to_list(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("assistant content must be a string")
        count = text_counter(content)
        if not isinstance(count, int) or count < 0:
            raise ValueError("token counter must return a non-negative integer")
        total += count
    return total


def count_assistant_loss_tokens_with_tokenizer(
    messages: Any,
    tokenizer: Any,
    *,
    max_length: int | None = None,
    truncation: str = "right",
) -> int:
    """Mirror ``MultiTurnSFTDataset`` assistant loss-mask construction.

    Generation-prompt tokens are excluded; assistant message suffix tokens,
    including template terminators, are counted exactly as loss-mask ones.
    """

    if max_length is not None and max_length <= 0:
        raise ValueError("max_length must be positive")
    if truncation not in {"left", "right", "error"}:
        raise ValueError("truncation must be left, right, or error")

    message_list = _messages_to_list(messages)
    loss_mask: list[int] = []
    index = 0
    while index < len(message_list):
        message = message_list[index]
        end = index + 1
        if message.get("role") == "tool":
            while end < len(message_list) and message_list[end].get("role") == "tool":
                end += 1
        previous = tokenizer.apply_chat_template(
            message_list[:index], tokenize=False, add_generation_prompt=False, enable_thinking=None
        ) if index else ""
        current = tokenizer.apply_chat_template(
            message_list[:end], tokenize=False, add_generation_prompt=False, enable_thinking=None
        )
        if not all(isinstance(text, str) for text in (previous, current)):
            raise ValueError("tokenizer.apply_chat_template(..., tokenize=False) must return strings")
        if message.get("role") == "assistant":
            previous_with_prompt = tokenizer.apply_chat_template(
                message_list[:index], tokenize=False, add_generation_prompt=True, enable_thinking=None
            )
            if not isinstance(previous_with_prompt, str):
                raise ValueError("tokenizer.apply_chat_template(..., tokenize=False) must return strings")
            generation_prompt = previous_with_prompt[len(previous) :]
            assistant_suffix = current[len(previous_with_prompt) :]
            loss_mask.extend(
                [0] * len(tokenizer.encode(generation_prompt, add_special_tokens=False))
            )
            loss_mask.extend(
                [1] * len(tokenizer.encode(assistant_suffix, add_special_tokens=False))
            )
        else:
            message_text = current[len(previous) :]
            loss_mask.extend([0] * len(tokenizer.encode(message_text, add_special_tokens=False)))
        index = end

    if max_length is not None and len(loss_mask) > max_length:
        if truncation == "error":
            raise ValueError(f"trajectory has {len(loss_mask)} tokens, exceeding max_length={max_length}")
        loss_mask = loss_mask[-max_length:] if truncation == "left" else loss_mask[:max_length]
    return sum(loss_mask)


def compute_turn_weight(
    messages: Any,
    *,
    short_max: int = 50,
    medium_max: int = 100,
    short_weight: int = 1,
    medium_weight: int = 2,
    long_weight: int = 5,
) -> int:
    message_list = _messages_to_list(messages)
    turns = max(len(message_list) - 1, 0) // 2
    if turns <= short_max:
        return short_weight
    if turns <= medium_max:
        return medium_weight
    return long_weight


def sample_indices_to_budget(
    token_counts: Sequence[int],
    weights: Sequence[float],
    *,
    token_budget: int,
    seed: int,
    batch_multiple: int = 1,
) -> tuple[list[int], int]:
    """Draw full train batches until the common target is reached."""

    if token_budget <= 0:
        raise ValueError("assistant token budget must be positive")
    if len(token_counts) != len(weights) or not token_counts:
        raise ValueError("token counts and weights must be non-empty and aligned")
    if any(not isinstance(count, int) or count <= 0 for count in token_counts):
        raise ValueError("every source trajectory must contain at least one assistant token")
    if any(not math.isfinite(weight) or weight <= 0 for weight in weights):
        raise ValueError("sampling weights must be finite and positive")
    if not isinstance(batch_multiple, int) or batch_multiple <= 0:
        raise ValueError("batch_multiple must be a positive integer")

    rng = random.Random(seed)
    population = list(range(len(token_counts)))
    selected = []
    realized = 0
    while realized < token_budget:
        batch = rng.choices(population, weights=weights, k=batch_multiple)
        selected.extend(batch)
        realized += sum(token_counts[index] for index in batch)
    return selected, realized


def _ending_candidates(
    token_counts: Sequence[int],
    weights: Sequence[float],
    *,
    token_budget: int,
    seed: int,
    batch_multiple: int,
    search_trials: int,
) -> tuple[list[int], int, list[tuple[int, list[int], int]]]:
    """Draw a weighted prefix and alternative complete final batches."""

    if search_trials <= 0:
        raise ValueError("search_trials must be positive")
    # Reuse public sampler validation without consuming this RNG stream.
    sample_indices_to_budget(
        token_counts,
        weights,
        token_budget=1,
        seed=seed,
        batch_multiple=batch_multiple,
    )
    rng = random.Random(seed)
    population = list(range(len(token_counts)))
    prefix: list[int] = []
    realized = 0
    first_crossing: list[int] | None = None
    while first_crossing is None:
        batch = rng.choices(population, weights=weights, k=batch_multiple)
        batch_tokens = sum(token_counts[index] for index in batch)
        if realized + batch_tokens >= token_budget:
            first_crossing = batch
        else:
            prefix.extend(batch)
            realized += batch_tokens
    candidates = [(realized + sum(token_counts[index] for index in first_crossing), first_crossing, 0)]
    remaining = token_budget - realized
    for trial in range(1, search_trials):
        batch = rng.choices(population, weights=weights, k=batch_multiple)
        batch_tokens = sum(token_counts[index] for index in batch)
        if batch_tokens >= remaining:
            candidates.append((realized + batch_tokens, batch, trial))
    return prefix, realized, candidates


def sample_paired_arms_to_budget(
    token_counts: Sequence[int],
    first_weights: Sequence[float],
    second_weights: Sequence[float],
    *,
    token_budget: int,
    seed: int,
    batch_multiple: int,
    search_trials: int = 4096,
) -> tuple[list[int], int, list[int], int]:
    """Match two arms by jointly choosing their last complete train batch.

    Only the final batch is selected by token matching; all preceding batches
    follow the requested sampling distribution. This keeps the scientific
    comparison within a tight token tolerance without relying on drop-last.
    """

    first_prefix, _, first_candidates = _ending_candidates(
        token_counts,
        first_weights,
        token_budget=token_budget,
        seed=seed,
        batch_multiple=batch_multiple,
        search_trials=search_trials,
    )
    second_prefix, _, second_candidates = _ending_candidates(
        token_counts,
        second_weights,
        token_budget=token_budget,
        seed=seed,
        batch_multiple=batch_multiple,
        search_trials=search_trials,
    )
    left = sorted(first_candidates, key=lambda item: (item[0], item[2]))
    right = sorted(second_candidates, key=lambda item: (item[0], item[2]))
    i = j = 0
    best: tuple[tuple[int, int, int, int, int], tuple[int, list[int], int], tuple[int, list[int], int]] | None = None
    while i < len(left) and j < len(right):
        first = left[i]
        second = right[j]
        score = (
            abs(first[0] - second[0]),
            max(first[0], second[0]),
            first[0] + second[0],
            first[2],
            second[2],
        )
        if best is None or score < best[0]:
            best = (score, first, second)
        if first[0] < second[0]:
            i += 1
        elif second[0] < first[0]:
            j += 1
        else:
            break
    assert best is not None
    _, first, second = best
    return (
        first_prefix + first[1],
        first[0],
        second_prefix + second[1],
        second[0],
    )


def _selection_sha256(indices: Sequence[int]) -> str:
    encoded = json.dumps(list(indices), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_sft_arms(
    records: Sequence[dict[str, Any]],
    *,
    token_budget: int,
    seed: int,
    text_counter: TextTokenCounter = approximate_text_tokens,
    assistant_counter: AssistantTokenCounter | None = None,
    short_max: int = 50,
    medium_max: int = 100,
    short_weight: int = 1,
    medium_weight: int = 2,
    long_weight: int = 5,
    train_batch_size: int = 1,
    final_batch_search_trials: int = 4096,
    max_between_arm_token_gap_ratio: float = 0.001,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Return S0/S1 records and reproducibility metadata."""

    source = []
    for record in records:
        normalized_record = dict(record)
        normalized_record["messages"] = _messages_to_list(normalized_record.get("messages"))
        source.append(normalized_record)
    if not source:
        raise ValueError("input training data is empty")
    if short_max < 0 or medium_max < short_max:
        raise ValueError("turn thresholds must satisfy 0 <= short_max <= medium_max")
    token_counts = [
        (
            assistant_counter(record.get("messages"))
            if assistant_counter is not None
            else count_assistant_tokens(record.get("messages"), text_counter)
        )
        for record in source
    ]
    turn_weights = [
        compute_turn_weight(
            record.get("messages"),
            short_max=short_max,
            medium_max=medium_max,
            short_weight=short_weight,
            medium_weight=medium_weight,
            long_weight=long_weight,
        )
        for record in source
    ]
    if not (0 <= max_between_arm_token_gap_ratio <= 0.001):
        raise ValueError("max_between_arm_token_gap_ratio must be between 0 and 0.001")
    s0_indices, s0_tokens, s1_indices, s1_tokens = sample_paired_arms_to_budget(
        token_counts,
        [1.0] * len(source),
        turn_weights,
        token_budget=token_budget,
        seed=seed,
        batch_multiple=train_batch_size,
        search_trials=final_batch_search_trials,
    )
    token_gap = abs(s0_tokens - s1_tokens)
    token_gap_tolerance = math.ceil(token_budget * max_between_arm_token_gap_ratio)
    if token_gap > token_gap_tolerance:
        raise ValueError(
            "unable to match S0/S1 assistant-token totals within "
            f"{token_gap_tolerance} tokens after {final_batch_search_trials} final-batch trials; "
            "increase --final-batch-search-trials or inspect the trajectory-length distribution"
        )
    arms = {
        "s0_uniform": [dict(source[index]) for index in s0_indices],
        "s1_turn_weighted": [dict(source[index]) for index in s1_indices],
    }
    metadata = {
        "target_assistant_tokens": token_budget,
        "seed": seed,
        "source_rows": len(source),
        "source_assistant_tokens": int(sum(token_counts)),
        "max_source_assistant_tokens": int(max(token_counts)),
        "sampling_batch_multiple": train_batch_size,
        "final_batch_search_trials": final_batch_search_trials,
        "between_arm_token_gap": token_gap,
        "between_arm_token_gap_tolerance": token_gap_tolerance,
        "max_between_arm_token_gap_ratio": max_between_arm_token_gap_ratio,
        "weight_policy": {
            "short_max_turns": short_max,
            "medium_max_turns": medium_max,
            "short_weight": short_weight,
            "medium_weight": medium_weight,
            "long_weight": long_weight,
        },
        "arms": {
            "s0_uniform": {
                "rows": len(s0_indices),
                "realized_assistant_tokens": s0_tokens,
                "overshoot_tokens": s0_tokens - token_budget,
                "effective_rows_after_drop_last": len(s0_indices),
                "effective_assistant_tokens": s0_tokens,
                "dropped_rows": 0,
                "selection_sha256": _selection_sha256(s0_indices),
                "unique_source_rows": len(set(s0_indices)),
                "source_selection_counts": dict(sorted(Counter(s0_indices).items())),
            },
            "s1_turn_weighted": {
                "rows": len(s1_indices),
                "realized_assistant_tokens": s1_tokens,
                "overshoot_tokens": s1_tokens - token_budget,
                "effective_rows_after_drop_last": len(s1_indices),
                "effective_assistant_tokens": s1_tokens,
                "dropped_rows": 0,
                "selection_sha256": _selection_sha256(s1_indices),
                "unique_source_rows": len(set(s1_indices)),
                "source_selection_counts": dict(sorted(Counter(s1_indices).items())),
            },
        },
    }
    return arms, metadata


def _hf_assistant_counter(
    model_name_or_path: str,
    *,
    trust_remote_code: bool = False,
    max_length: int = 32768,
    truncation: str = "right",
) -> AssistantTokenCounter:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required when --tokenizer is specified") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

    def count(messages: Any) -> int:
        return count_assistant_loss_tokens_with_tokenizer(
            messages, tokenizer, max_length=max_length, truncation=truncation
        )

    return count


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build S0/S1 SFT arms with a shared assistant-token target.")
    parser.add_argument("--input", "-i", required=True, help="Unresampled train_clean.parquet.")
    parser.add_argument(
        "--data-manifest",
        help="data_manifest.json produced by prepare_trajectories.py; required by the formal verifier.",
    )
    parser.add_argument(
        "--final-batch-search-trials",
        type=int,
        default=4096,
        help="Candidate complete final batches searched jointly to match S0/S1 tokens.",
    )
    parser.add_argument(
        "--max-between-arm-token-gap-ratio",
        type=float,
        default=0.001,
        help="Fail if realized S0/S1 assistant tokens differ by more than this ratio (max 0.001).",
    )
    parser.add_argument("--output-dir", "-o", required=True)
    parser.add_argument("--assistant-token-budget", type=int, default=20_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional Hugging Face tokenizer. Without it, a deterministic estimate is used for smoke tests only.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-length", type=int, default=32768,
                        help="Training sequence length used when counting the truncated loss mask.")
    parser.add_argument("--truncation", choices=["left", "right", "error"], default="right")
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=8,
        help="Sample complete global batches so drop_last consumes every row.",
    )
    parser.add_argument("--short-max-turns", type=int, default=50)
    parser.add_argument("--medium-max-turns", type=int, default=100)
    parser.add_argument("--short-weight", type=int, default=1)
    parser.add_argument("--medium-weight", type=int, default=2)
    parser.add_argument("--long-weight", type=int, default=5)
    parser.add_argument("--write-jsonl", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataframe = pd.read_parquet(input_path)
    required_columns = {"question", "messages", "gt"}
    missing = required_columns - set(dataframe.columns)
    if missing:
        raise ValueError(f"input is missing required columns: {sorted(missing)}")
    source_records = dataframe.to_dict("records")
    for record in source_records:
        record["messages"] = _messages_to_list(record.get("messages"))
    assistant_counter = (
        _hf_assistant_counter(
            args.tokenizer,
            trust_remote_code=args.trust_remote_code,
            max_length=args.max_length,
            truncation=args.truncation,
        )
        if args.tokenizer
        else None
    )
    arms, manifest = build_sft_arms(
        source_records,
        token_budget=args.assistant_token_budget,
        seed=args.seed,
        text_counter=approximate_text_tokens,
        assistant_counter=assistant_counter,
        short_max=args.short_max_turns,
        medium_max=args.medium_max_turns,
        short_weight=args.short_weight,
        medium_weight=args.medium_weight,
        long_weight=args.long_weight,
        train_batch_size=args.train_batch_size,
        final_batch_search_trials=args.final_batch_search_trials,
        max_between_arm_token_gap_ratio=args.max_between_arm_token_gap_ratio,
    )
    manifest["schema_version"] = 1
    manifest["input"] = {
        "file": input_path.name,
        "sha256": sha256_file(input_path),
        "logical_sha256": logical_records_sha256(source_records),
    }
    if args.data_manifest:
        data_manifest_path = Path(args.data_manifest).expanduser().resolve()
        data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
        declared_train = data_manifest.get("artifacts", {}).get("train_clean", {})
        if declared_train.get("sha256") != manifest["input"]["sha256"]:
            raise ValueError("--input does not match data_manifest train_clean artifact")
        manifest["data_manifest"] = {
            "file": str(data_manifest_path),
            "sha256": sha256_file(data_manifest_path),
            "train_clean_sha256": declared_train.get("sha256"),
        }
    else:
        manifest["data_manifest"] = None
    manifest["token_counter"] = {
        "type": (
            "huggingface_multiturn_loss_mask_equivalent"
            if args.tokenizer
            else "approximate_assistant_content"
        ),
        "tokenizer": args.tokenizer,
        "max_length": args.max_length if args.tokenizer else None,
        "truncation": args.truncation if args.tokenizer else None,
    }
    manifest["artifacts"] = {}

    for arm_name, arm_records in arms.items():
        output_path = output_dir / f"{arm_name}.parquet"
        arm_df = pd.DataFrame(arm_records)[["question", "messages", "gt"]]
        arm_df.to_parquet(output_path, index=False)
        artifact = {
            "file": output_path.name,
            "sha256": sha256_file(output_path),
            "logical_sha256": logical_records_sha256(arm_records),
        }
        if args.write_jsonl:
            jsonl_path = output_dir / f"{arm_name}.jsonl"
            with jsonl_path.open("w", encoding="utf-8") as handle:
                for record in arm_records:
                    public_record = {key: record.get(key) for key in ("question", "messages", "gt")}
                    handle.write(json.dumps(public_record, ensure_ascii=False) + "\n")
            artifact["jsonl_file"] = jsonl_path.name
            artifact["jsonl_sha256"] = sha256_file(jsonl_path)
        manifest["artifacts"][arm_name] = artifact

    manifest_path = output_dir / "sft_arms_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved S0/S1 arms and manifest to: {output_dir}")


if __name__ == "__main__":
    main()
