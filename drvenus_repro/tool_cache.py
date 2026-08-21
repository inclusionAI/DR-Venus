"""Deterministic record/replay cache for external research tools.

The cache is intentionally independent from the Inference and RL tool stacks,
so both entrypoints can share the same JSONL artifact.  Runtime behaviour is
controlled by ``TOOL_CACHE_MODE`` and ``TOOL_CACHE_PATH``:

``online``
    Call the provider directly and do not read or write a cache (default).
``record``
    Reuse an existing successful entry, otherwise call the provider and append
    the result to the cache.
``replay``
    Return a recorded entry without contacting the provider.  A miss is an
    explicit error rather than an implicit online fallback.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, TypeVar
import unicodedata


LOGGER = logging.getLogger(__name__)

TOOL_CACHE_MODE_ENV = "TOOL_CACHE_MODE"
TOOL_CACHE_PATH_ENV = "TOOL_CACHE_PATH"
SUPPORTED_MODES = frozenset({"online", "record", "replay"})
SCHEMA_VERSION = 1

_REDACTED = "[REDACTED]"
_SENSITIVE_FIELD_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "api_token",
        "authorization",
        "auth_token",
        "access_token",
        "refresh_token",
        "bearer_token",
        "client_secret",
        "credential",
        "credentials",
        "password",
        "proxy_authorization",
        "secret",
        "serper_key_id",
        "jina_api_keys",
        "token",
        "x_api_key",
    }
)
_SENSITIVE_ENV_NAMES = (
    "API_KEY",
    "JUDGE_API_KEY",
    "SERPER_KEY_ID",
    "JINA_API_KEYS",
    "PROXY",
)

T = TypeVar("T")


class ToolCacheError(RuntimeError):
    """Base class for tool-cache failures."""


class ToolCacheConfigurationError(ToolCacheError):
    """Raised when cache-related environment configuration is invalid."""


class ToolCacheMissError(ToolCacheError):
    """Raised when replay mode cannot find a matching invocation."""


class CachedToolExecutionError(ToolCacheError):
    """Raised when replay reaches an invocation recorded as an error."""


def _normalized_field_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")


def _normalize_json_value(value: Any, *, active_ids: set[int]) -> Any:
    """Convert JSON-compatible input into a stable, secret-safe value."""

    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("tool arguments cannot contain NaN or infinity")
        return 0.0 if value == 0.0 else value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)

    value_id = id(value)
    if value_id in active_ids:
        raise ValueError("tool arguments cannot contain recursive values")

    if isinstance(value, Mapping):
        active_ids.add(value_id)
        try:
            normalized: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError("tool argument object keys must be strings")
                normalized_key = unicodedata.normalize("NFC", key)
                # The upstream dispatcher adds ``params`` as a self-reference
                # immediately before invoking qwen-agent.  It is transport
                # metadata, not a model-supplied tool argument.
                if normalized_key == "params" and item is value:
                    continue
                if normalized_key in normalized:
                    raise ValueError(f"duplicate normalized argument key: {normalized_key!r}")
                if _normalized_field_name(normalized_key) in _SENSITIVE_FIELD_NAMES:
                    normalized[normalized_key] = _REDACTED
                else:
                    normalized[normalized_key] = _normalize_json_value(item, active_ids=active_ids)
            return {key: normalized[key] for key in sorted(normalized)}
        finally:
            active_ids.remove(value_id)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        active_ids.add(value_id)
        try:
            return [_normalize_json_value(item, active_ids=active_ids) for item in value]
        finally:
            active_ids.remove(value_id)

    raise TypeError(f"tool arguments must be JSON-compatible, got {type(value).__name__}")


def normalize_tool_arguments(arguments: Any) -> Any:
    """Return canonical JSON-compatible arguments without credential values.

    Mapping order and Unicode composition are normalized.  Array order remains
    significant because batched ``search`` and ``visit`` responses preserve it.
    """

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass
    return _normalize_json_value(arguments, active_ids=set())


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def build_cache_key(tool: str, arguments: Any) -> str:
    """Build a SHA-256 key from the tool name and normalized arguments."""

    normalized_tool = unicodedata.normalize("NFC", str(tool)).strip().casefold()
    if not normalized_tool:
        raise ValueError("tool name must not be empty")
    key_material = {"arguments": normalize_tool_arguments(arguments), "tool": normalized_tool}
    return hashlib.sha256(_canonical_json(key_material).encode("utf-8")).hexdigest()


def _response_hash(response: Any) -> str:
    return hashlib.sha256(_canonical_json(response).encode("utf-8")).hexdigest()


def _environment_secret_values() -> tuple[str, ...]:
    values: set[str] = set()
    for name in _SENSITIVE_ENV_NAMES:
        raw = os.environ.get(name, "")
        if len(raw) >= 4:
            values.add(raw)
        if name == "JINA_API_KEYS":
            values.update(part.strip() for part in raw.split(",") if len(part.strip()) >= 4)
    return tuple(sorted(values, key=len, reverse=True))


def _redact_environment_secrets(value: Any, secret_values: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        for secret in secret_values:
            value = value.replace(secret, _REDACTED)
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_redact_environment_secrets(item, secret_values) for item in value]
    if isinstance(value, Mapping):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            stored_key = _redact_environment_secrets(key, secret_values) if isinstance(key, str) else key
            if isinstance(key, str) and _normalized_field_name(key) in _SENSITIVE_FIELD_NAMES:
                redacted[stored_key] = _REDACTED
            else:
                redacted[stored_key] = _redact_environment_secrets(item, secret_values)
        return redacted
    return value


@dataclass
class _CacheIndex:
    lock: threading.RLock = field(default_factory=threading.RLock)
    identity: tuple[int, int] | None = None
    offset: int = 0
    line_number: int = 0
    mtime_ns: int | None = None
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)


_INDEXES: dict[Path, _CacheIndex] = {}
_INDEXES_LOCK = threading.Lock()


def _index_for(path: Path) -> _CacheIndex:
    with _INDEXES_LOCK:
        return _INDEXES.setdefault(path, _CacheIndex())


def _warn_bad_line(path: Path, line_number: int, reason: str) -> None:
    LOGGER.warning("Ignoring corrupt tool-cache line %s:%d (%s)", path, line_number, reason)


def _validated_record(raw: bytes, path: Path, line_number: int) -> dict[str, Any] | None:
    try:
        record = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _warn_bad_line(path, line_number, type(exc).__name__)
        return None

    if not isinstance(record, dict):
        _warn_bad_line(path, line_number, "record is not an object")
        return None
    required = {
        "schema_version",
        "cache_key",
        "tool",
        "arguments",
        "provider",
        "timestamp",
        "status",
        "response",
        "response_hash",
    }
    if not required.issubset(record):
        _warn_bad_line(path, line_number, "missing required fields")
        return None
    if record["schema_version"] != SCHEMA_VERSION or record["status"] not in {"ok", "error"}:
        _warn_bad_line(path, line_number, "unsupported schema or status")
        return None
    try:
        expected_key = build_cache_key(record["tool"], record["arguments"])
        expected_response_hash = _response_hash(record["response"])
    except (TypeError, ValueError) as exc:
        _warn_bad_line(path, line_number, str(exc))
        return None
    if record["cache_key"] != expected_key:
        _warn_bad_line(path, line_number, "cache-key mismatch")
        return None
    if record["response_hash"] != expected_response_hash:
        _warn_bad_line(path, line_number, "response-hash mismatch")
        return None
    return record


@contextmanager
def _interprocess_lock(cache_path: Path):
    """Serialize cache reads/appends across Python processes.

    A one-byte sidecar is used instead of locking the JSONL itself so a writer
    can safely replace or create the data file while all participants use the
    same lock target.
    """

    lock_path = cache_path.with_name(f"{cache_path.name}.lock")
    with lock_path.open("a+b") as lock_file:
        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b"\0")
            lock_file.flush()
        lock_file.seek(0)

        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _refresh_index(path: Path, index: _CacheIndex) -> None:
    if not path.exists():
        index.identity = None
        index.offset = 0
        index.line_number = 0
        index.mtime_ns = None
        index.entries.clear()
        return

    stat = path.stat()
    identity = (stat.st_dev, stat.st_ino)
    replaced = index.identity != identity or stat.st_size < index.offset
    rewritten_in_place = stat.st_size == index.offset and index.mtime_ns not in (None, stat.st_mtime_ns)
    if replaced or rewritten_in_place:
        index.identity = identity
        index.offset = 0
        index.line_number = 0
        index.entries.clear()

    with path.open("rb") as cache_file:
        cache_file.seek(index.offset)
        while raw_line := cache_file.readline():
            index.line_number += 1
            record = _validated_record(raw_line, path, index.line_number)
            if record is not None:
                index.entries[record["cache_key"]] = record
        index.offset = cache_file.tell()

    index.identity = identity
    index.mtime_ns = stat.st_mtime_ns


def _lookup(path: Path, cache_key: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    index = _index_for(path)
    with index.lock:
        with _interprocess_lock(path):
            _refresh_index(path, index)
            return index.entries.get(cache_key)


def _append(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Append a record unless another process already stored a success.

    Provider calls intentionally happen outside the file lock so unrelated
    searches can run concurrently.  This final check makes racing calls for
    the same key converge on the first successful recorded response.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (_canonical_json(record) + "\n").encode("utf-8")
    index = _index_for(path)
    with index.lock:
        with _interprocess_lock(path):
            _refresh_index(path, index)
            existing = index.entries.get(record["cache_key"])
            if existing is not None and existing["status"] == "ok":
                return existing
            with path.open("a+b") as cache_file:
                cache_file.seek(0, os.SEEK_END)
                if cache_file.tell() > 0:
                    cache_file.seek(-1, os.SEEK_END)
                    if cache_file.read(1) != b"\n":
                        # Recover from a process that died mid-append.  The
                        # partial line remains auditable and will be ignored.
                        cache_file.write(b"\n")
                cache_file.write(serialized)
                cache_file.flush()
                os.fsync(cache_file.fileno())
            _refresh_index(path, index)
            return record


def _settings_from_environment(
    *, mode_env: str = TOOL_CACHE_MODE_ENV, path_env: str = TOOL_CACHE_PATH_ENV
) -> tuple[str, Path | None]:
    mode = os.environ.get(mode_env, "online").strip().casefold() or "online"
    if mode not in SUPPORTED_MODES:
        supported = ", ".join(sorted(SUPPORTED_MODES))
        raise ToolCacheConfigurationError(
            f"{mode_env} must be one of {supported}; got {mode!r}"
        )
    if mode == "online":
        return mode, None

    raw_path = os.environ.get(path_env, "").strip()
    if not raw_path:
        raise ToolCacheConfigurationError(
            f"{path_env} is required when {mode_env}={mode}"
        )
    return mode, Path(raw_path).expanduser().resolve(strict=False)


def _new_record(
    *,
    cache_key: str,
    tool: str,
    arguments: Any,
    provider: str,
    status: str,
    response: Any,
    status_code: int | None,
    latency_ms: float,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "cache_key": cache_key,
        "tool": unicodedata.normalize("NFC", tool).strip().casefold(),
        "arguments": arguments,
        "provider": provider,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": status,
        "status_code": status_code,
        "latency_ms": round(latency_ms, 3),
        "response": response,
        "response_hash": _response_hash(response),
    }


def _raise_cached_error(tool: str, record: Mapping[str, Any]) -> None:
    response = record.get("response")
    if isinstance(response, Mapping):
        error_type = response.get("type", "Error")
        message = response.get("message", "recorded tool invocation failed")
    else:
        error_type = "Error"
        message = str(response)
    raise CachedToolExecutionError(f"Cached {tool} invocation failed ({error_type}): {message}")


def execute_with_tool_cache(
    tool: str,
    arguments: Any,
    online_call: Callable[[], T],
    *,
    provider: str,
    mode_env: str = TOOL_CACHE_MODE_ENV,
    path_env: str = TOOL_CACHE_PATH_ENV,
) -> T:
    """Execute an external tool according to the configured cache mode."""

    mode, path = _settings_from_environment(mode_env=mode_env, path_env=path_env)
    if mode == "online":
        return online_call()

    assert path is not None
    normalized_arguments = normalize_tool_arguments(arguments)
    cache_key = build_cache_key(tool, normalized_arguments)
    cached = _lookup(path, cache_key)

    if cached is not None and cached["status"] == "ok":
        return cached["response"]  # type: ignore[return-value]
    if mode == "replay":
        if cached is not None:
            _raise_cached_error(tool, cached)
        raise ToolCacheMissError(
            f"No cached response for tool {tool!r} (cache_key={cache_key}); replay never falls back online"
        )

    secret_values = _environment_secret_values()
    started = time.perf_counter()
    try:
        response = online_call()
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        raw_status_code = getattr(getattr(exc, "response", None), "status_code", None)
        status_code = raw_status_code if isinstance(raw_status_code, int) else None
        cached_error = {
            "type": type(exc).__name__,
            "message": _redact_environment_secrets(str(exc), secret_values),
        }
        persisted = _append(
            path,
            _new_record(
                cache_key=cache_key,
                tool=tool,
                arguments=normalized_arguments,
                provider=provider,
                status="error",
                response=cached_error,
                status_code=status_code,
                latency_ms=latency_ms,
            ),
        )
        if persisted["status"] == "ok":
            return persisted["response"]  # type: ignore[return-value]
        try:
            sanitized_exception = type(exc)(cached_error["message"])
        except Exception:
            sanitized_exception = ToolCacheError(
                f"{cached_error['type']}: {cached_error['message']}"
            )
        raise sanitized_exception from None

    latency_ms = (time.perf_counter() - started) * 1000.0
    stored_response = _redact_environment_secrets(response, secret_values)
    persisted = _append(
        path,
        _new_record(
            cache_key=cache_key,
            tool=tool,
            arguments=normalized_arguments,
            provider=provider,
            status="ok",
            response=stored_response,
            # All supported HTTP adapters raise on non-success.  Their
            # successful Serper/Jina/OpenAI-compatible responses are HTTP 200.
            status_code=200,
            latency_ms=latency_ms,
        ),
    )
    # Record mode must behave exactly like later replay, including secret
    # redaction and winner selection under a concurrent same-key race.
    return persisted["response"]  # type: ignore[return-value]
