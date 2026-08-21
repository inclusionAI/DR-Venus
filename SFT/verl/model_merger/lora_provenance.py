"""Persist and restore the exact PEFT configuration used by SFT LoRA runs.

The sharded FSDP state dict contains LoRA tensors, but it cannot recover
configuration values such as ``lora_alpha`` or ``lora_dropout``.  Guessing
those values changes the adapter semantics, so checkpoint conversion must be
backed by explicit provenance.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


LORA_PROVENANCE_FILENAME = "lora_adapter_config.json"
LORA_PROVENANCE_SCHEMA_VERSION = 1


class LoraProvenanceError(ValueError):
    """Raised when a LoRA checkpoint has missing or invalid provenance."""


def _json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((_json_ready(item) for item in value), key=str)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _require_positive_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LoraProvenanceError(f"LoRA provenance field {key!r} must be a positive integer; got {value!r}.")
    return value


def _require_finite_number(config: Mapping[str, Any], key: str, *, positive: bool) -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LoraProvenanceError(f"LoRA provenance field {key!r} must be numeric; got {value!r}.")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0) or (not positive and number < 0):
        qualifier = "positive" if positive else "non-negative"
        raise LoraProvenanceError(
            f"LoRA provenance field {key!r} must be a finite {qualifier} number; got {value!r}."
        )
    return number


def normalize_lora_config(raw_config: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Return a JSON-ready, validated PEFT LoRA configuration.

    A complete configuration is required deliberately.  In particular,
    ``lora_alpha`` and ``lora_dropout`` are not inferable from adapter weights.
    """

    if hasattr(raw_config, "to_dict"):
        raw_config = raw_config.to_dict()
    if not isinstance(raw_config, Mapping):
        raise LoraProvenanceError("LoRA provenance must be a PEFT config mapping or expose to_dict().")

    config = _json_ready(dict(raw_config))
    _require_positive_int(config, "r")
    _require_finite_number(config, "lora_alpha", positive=True)
    _require_finite_number(config, "lora_dropout", positive=False)

    target_modules = config.get("target_modules")
    if isinstance(target_modules, str):
        if not target_modules.strip():
            raise LoraProvenanceError("LoRA provenance field 'target_modules' must not be empty.")
    elif isinstance(target_modules, Sequence) and not isinstance(target_modules, (str, bytes)):
        if not target_modules or any(not isinstance(item, str) or not item.strip() for item in target_modules):
            raise LoraProvenanceError(
                "LoRA provenance field 'target_modules' must contain at least one non-empty module name."
            )
        config["target_modules"] = sorted(set(target_modules))
    else:
        raise LoraProvenanceError(
            "LoRA provenance field 'target_modules' must be 'all-linear' or a non-empty list of module names."
        )

    for key in ("bias", "task_type", "peft_type"):
        value = config.get(key)
        if not isinstance(value, str) or not value:
            raise LoraProvenanceError(f"LoRA provenance field {key!r} must be a non-empty string.")

    if config["peft_type"].upper() != "LORA":
        raise LoraProvenanceError(f"Expected PEFT type 'LORA'; got {config['peft_type']!r}.")
    try:
        json.dumps(config, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise LoraProvenanceError(f"LoRA provenance is not JSON-serializable: {exc}") from exc
    return config


def _extract_config(document: Any, source: Path) -> Mapping[str, Any]:
    if not isinstance(document, Mapping):
        raise LoraProvenanceError(f"LoRA provenance at {source} must contain a JSON object.")
    if "peft_config" not in document:
        # A standard PEFT adapter_config.json is also accepted as an explicit
        # compatibility override for historical checkpoints.
        return document
    if document.get("schema_version") != LORA_PROVENANCE_SCHEMA_VERSION:
        raise LoraProvenanceError(
            f"Unsupported LoRA provenance schema at {source}: {document.get('schema_version')!r}."
        )
    config = document.get("peft_config")
    if not isinstance(config, Mapping):
        raise LoraProvenanceError(f"LoRA provenance at {source} has no valid 'peft_config' object.")
    return config


def write_lora_provenance(checkpoint_dir: str | os.PathLike[str], peft_config: Mapping[str, Any] | Any) -> Path:
    """Atomically write complete LoRA provenance into a training checkpoint."""

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_path / LORA_PROVENANCE_FILENAME
    document = {
        "schema_version": LORA_PROVENANCE_SCHEMA_VERSION,
        "adapter_name": "default",
        "peft_config": normalize_lora_config(peft_config),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, destination)
    return destination


def write_adapter_config(adapter_dir: str | os.PathLike[str], peft_config: Mapping[str, Any] | Any) -> Path:
    """Write the validated standard PEFT ``adapter_config.json`` atomically."""

    adapter_path = Path(adapter_dir)
    adapter_path.mkdir(parents=True, exist_ok=True)
    destination = adapter_path / "adapter_config.json"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    config = normalize_lora_config(peft_config)
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, destination)
    return destination


def load_lora_provenance(
    checkpoint_dir: str | os.PathLike[str],
    *,
    explicit_config_path: str | os.PathLike[str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve the exact LoRA config for merging, failing closed if absent.

    ``explicit_config_path`` is the compatibility escape hatch for historical
    checkpoints.  It must point to a complete standard PEFT adapter config (or
    this module's provenance envelope); individual overrides are intentionally
    insufficient when the original provenance is unavailable.
    """

    source = (
        Path(explicit_config_path)
        if explicit_config_path is not None
        else Path(checkpoint_dir) / LORA_PROVENANCE_FILENAME
    )
    if not source.is_file():
        raise LoraProvenanceError(
            "LoRA tensors were found, but their PEFT configuration cannot be recovered from weights. "
            f"Expected {source}. For a historical checkpoint, pass --lora-config with a complete, "
            "verified adapter_config.json; conversion will not guess rank, alpha, targets, or dropout."
        )

    try:
        with source.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise LoraProvenanceError(f"Cannot read LoRA provenance at {source}: {exc}") from exc

    config = dict(_extract_config(document, source))
    for key, value in (overrides or {}).items():
        if value is not None:
            config[key] = value
    return normalize_lora_config(config)


def observed_lora_ranks(state_dict: Mapping[str, Any]) -> set[int]:
    """Extract adapter ranks from standard LoRA A/B tensor shapes."""

    ranks: set[int] = set()
    for name, tensor in state_dict.items():
        shape = getattr(tensor, "shape", None)
        if shape is None or len(shape) != 2:
            continue
        if ".lora_A." in name:
            ranks.add(int(shape[0]))
        elif ".lora_B." in name:
            ranks.add(int(shape[1]))
    return ranks


def validate_rank_matches_weights(config: Mapping[str, Any], state_dict: Mapping[str, Any]) -> None:
    ranks = observed_lora_ranks(state_dict)
    declared_rank = int(config["r"])
    if ranks and ranks != {declared_rank}:
        raise LoraProvenanceError(
            f"LoRA provenance declares rank {declared_rank}, but checkpoint tensors encode ranks {sorted(ranks)}."
        )
