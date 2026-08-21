"""Deterministic split, correctness filtering, and manifest helpers.

The module deliberately uses only the Python standard library.  Network access
is possible only through :class:`OpenAICompatibleJudge`, which the CLI creates
exclusively when ``--correctness-mode judge`` is selected.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import re
import time
import unicodedata
import urllib.request
from urllib.parse import urlparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


DEFAULT_JUDGE_PROMPT_VERSION = "dr-venus-correctness-v1"
DEFAULT_SPLIT_RATIOS = {"train": 0.90, "validation": 0.05, "test": 0.05}
SPLIT_ALGORITHM = "normalized-question-sha256-sort-hamilton-v1"
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class CorrectnessDecision:
    """A traceable decision made by one correctness source."""

    keep: bool
    source: str
    reason: str
    score: float | None = None
    cache_key: str | None = None


def normalize_question_key(question: str) -> str:
    """Normalize a question only for grouping; the stored text is untouched."""

    normalized = unicodedata.normalize("NFKC", question)
    return " ".join(normalized.split()).casefold()


def _hamilton_sizes(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    names = list(ratios)
    if not names:
        raise ValueError("at least one split ratio is required")
    if any(not math.isfinite(value) or value < 0 for value in ratios.values()):
        raise ValueError("split ratios must be finite and non-negative")
    ratio_sum = sum(ratios.values())
    if not math.isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"split ratios must sum to 1.0, got {ratio_sum}")

    quotas = {name: total * ratios[name] for name in names}
    sizes = {name: int(math.floor(quotas[name])) for name in names}
    remaining = total - sum(sizes.values())
    order = sorted(names, key=lambda name: (-(quotas[name] - sizes[name]), names.index(name)))
    for name in order[:remaining]:
        sizes[name] += 1

    # Tiny datasets would otherwise silently have no validation/test data.  If
    # there are enough groups, give every requested split one group and take it
    # from the largest allocation.  This leaves normal 90/5/5 datasets intact.
    positive_names = [name for name in names if ratios[name] > 0]
    if total >= len(positive_names):
        for name in positive_names:
            if sizes[name] != 0:
                continue
            donors = [candidate for candidate in positive_names if sizes[candidate] > 1]
            if not donors:
                break
            donor = max(donors, key=lambda candidate: (sizes[candidate], ratios[candidate], -names.index(candidate)))
            sizes[donor] -= 1
            sizes[name] += 1
    return sizes


def split_records_by_question(
    records: Sequence[dict[str, Any]],
    *,
    seed: int,
    ratios: Mapping[str, float] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Split whole question groups deterministically.

    Returns ``(split_records, question_assignment)``.  Duplicate trajectories
    for the same normalized question are always assigned to the same split.
    """

    ratios = dict(ratios or DEFAULT_SPLIT_RATIOS)
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("every record must contain a non-empty string question")
        groups.setdefault(normalize_question_key(question), []).append(record)

    sizes = _hamilton_sizes(len(groups), ratios)
    ranked_questions = sorted(
        groups,
        key=lambda key: (hashlib.sha256(f"{seed}\0{key}".encode("utf-8")).hexdigest(), key),
    )
    assignments: dict[str, str] = {}
    cursor = 0
    for split_name in ratios:
        next_cursor = cursor + sizes[split_name]
        for key in ranked_questions[cursor:next_cursor]:
            assignments[key] = split_name
        cursor = next_cursor

    splits = {name: [] for name in ratios}
    for record in records:
        key = normalize_question_key(record["question"])
        splits[assignments[key]].append(record)
    return splits, assignments


def get_dotted_value(row: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = row
    for part in dotted_path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


_TRUE_STRINGS = {"1", "true", "yes", "correct", "pass", "passed", "ok"}
_FALSE_STRINGS = {"0", "false", "no", "incorrect", "wrong", "fail", "failed"}
_VALUE_KEYS = ("correct", "is_correct", "passed", "score", "reward", "judgement", "judgment", "label")


def parse_correctness_value(value: Any) -> float | None:
    """Convert common offline correctness encodings to a numeric score."""

    if isinstance(value, Mapping):
        for key in _VALUE_KEYS:
            if key in value:
                return parse_correctness_value(value[key])
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, numbers.Real):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in _TRUE_STRINGS:
            return 1.0
        if text in _FALSE_STRINGS:
            return 0.0
        try:
            numeric = float(text)
        except ValueError:
            return None
        return numeric if math.isfinite(numeric) else None
    return None


def extract_candidate_answer(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        matches = list(ANSWER_RE.finditer(content))
        if matches:
            return matches[-1].group(1).strip()
    return ""


def make_correctness_cache_key(
    question: str,
    reference: str,
    candidate: str,
    *,
    prompt_version: str = DEFAULT_JUDGE_PROMPT_VERSION,
    judge_model: str | None = None,
) -> str:
    payload = {
        "candidate": candidate.strip(),
        "prompt_version": prompt_version,
        "judge_model": (judge_model or "").strip(),
        "question": question.strip(),
        "reference": reference.strip(),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_record_key(record: Mapping[str, Any], prompt_version: str) -> str | None:
    question = record.get("question")
    reference = record.get("reference", record.get("gt"))
    candidate = record.get("candidate", record.get("prediction", record.get("answer")))
    if all(isinstance(value, str) for value in (question, reference, candidate)):
        version = record.get("prompt_version", prompt_version)
        computed = make_correctness_cache_key(
            question,
            reference,
            candidate,
            prompt_version=str(version),
            judge_model=str(record.get("judge_model") or ""),
        )
        explicit = record.get("cache_key")
        if explicit is not None and explicit != computed:
            return None
        return computed
    return None


def load_correctness_cache(
    path: Path,
    *,
    prompt_version: str = DEFAULT_JUDGE_PROMPT_VERSION,
) -> dict[str, dict[str, Any]]:
    """Load JSONL or JSON correctness records keyed by ``cache_key``."""

    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}

    records: list[Any]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        if isinstance(parsed, list):
            records = parsed
        elif isinstance(parsed, Mapping) and isinstance(parsed.get("entries"), list):
            records = parsed["entries"]
        elif isinstance(parsed, Mapping) and (
            "cache_key" in parsed or ("question" in parsed and any(key in parsed for key in _VALUE_KEYS))
        ):
            records = [parsed]
        elif isinstance(parsed, Mapping):
            records = []
            for key, value in parsed.items():
                record = dict(value) if isinstance(value, Mapping) else {"correct": value}
                record.setdefault("cache_key", key)
                records.append(record)
        else:
            raise ValueError(f"Unsupported correctness cache format: {path}")

    cache: dict[str, dict[str, Any]] = {}
    for item in records:
        if not isinstance(item, Mapping):
            continue
        record = dict(item)
        key = _cache_record_key(record, prompt_version)
        if key:
            cache[key] = record
    return cache


def _extract_first_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Judge response did not contain a JSON object")


def parse_judge_response(response: Mapping[str, Any]) -> tuple[float, str]:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("OpenAI-compatible response is missing choices[0].message.content") from exc
    if isinstance(content, Mapping):
        judgement = dict(content)
    elif isinstance(content, str):
        judgement = _extract_first_json_object(content)
    else:
        raise ValueError("Judge message content must be a string or object")
    score = parse_correctness_value(judgement)
    if score is None:
        raise ValueError("Judge JSON is missing a recognized correctness value")
    reason = judgement.get("reason", judgement.get("reasoning", ""))
    return score, str(reason) if reason is not None else ""


Transport = Callable[[str, dict[str, str], bytes, float], Mapping[str, Any] | bytes | str]


def _urllib_transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> Mapping[str, Any]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - explicit judge mode only
        return json.loads(response.read().decode("utf-8"))


class OpenAICompatibleJudge:
    """Small dependency-free client with bounded retries."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        retries: int = 2,
        transport: Transport | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not base_url:
            raise ValueError("judge base URL is required")
        if not model:
            raise ValueError("judge model is required")
        if retries < 0:
            raise ValueError("judge retries must be non-negative")
        if timeout <= 0 or not math.isfinite(timeout):
            raise ValueError("judge timeout must be finite and positive")
        parsed_url = urlparse(base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("judge base URL must be an absolute HTTP(S) URL")
        self.endpoint = (
            base_url.rstrip("/")
            if base_url.rstrip("/").endswith("/chat/completions")
            else f"{base_url.rstrip('/')}/chat/completions"
        )
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.transport = transport or _urllib_transport
        self.sleeper = sleeper

    def evaluate(self, question: str, reference: str, candidate: str) -> tuple[float, str]:
        user_payload = {"question": question, "reference_answer": reference, "candidate_answer": candidate}
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 256,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Decide whether the candidate answer is semantically equivalent to the reference answer "
                        "for the question. Return JSON only: {\"correct\": true|false, \"reason\": \"brief\"}."
                    ),
                },
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, sort_keys=True)},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self.transport(self.endpoint, headers, body, self.timeout)
                if isinstance(response, bytes):
                    response = json.loads(response.decode("utf-8"))
                elif isinstance(response, str):
                    response = json.loads(response)
                if not isinstance(response, Mapping):
                    raise ValueError("Judge transport returned a non-object response")
                return parse_judge_response(response)
            except Exception as exc:  # network/parser failures share the same bounded retry policy
                last_error = exc
                if attempt < self.retries:
                    self.sleeper(min(2**attempt, 8))
        raise RuntimeError(f"Judge failed after {self.retries + 1} attempts") from last_error


class CorrectnessEvaluator:
    """Apply disabled, field, offline-cache, or online-judge filtering."""

    MODES = {"none", "field", "cache", "judge"}

    def __init__(
        self,
        *,
        mode: str = "none",
        field: str = "correctness",
        threshold: float = 0.5,
        cache_path: Path | None = None,
        prompt_version: str = DEFAULT_JUDGE_PROMPT_VERSION,
        judge: OpenAICompatibleJudge | None = None,
        judge_model: str | None = None,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Unsupported correctness mode: {mode}")
        if not math.isfinite(threshold):
            raise ValueError("correctness threshold must be finite")
        if mode == "cache" and cache_path is None:
            raise ValueError("--judge-cache is required in cache mode")
        if mode == "cache" and cache_path is not None and not cache_path.exists():
            raise FileNotFoundError(f"Correctness cache does not exist: {cache_path}")
        if mode == "judge" and judge is None:
            raise ValueError("judge mode requires an OpenAI-compatible judge")
        self.mode = mode
        self.field = field
        self.threshold = threshold
        self.cache_path = cache_path
        self.prompt_version = prompt_version
        self.judge = judge
        self.judge_model = judge_model or (judge.model if judge is not None else None)
        self.cache = load_correctness_cache(cache_path, prompt_version=prompt_version) if cache_path else {}
        self.used_cache_judge_models: set[str] = set()
        self.used_cache_prompt_versions: set[str] = set()

    def evaluate(self, raw_row: Mapping[str, Any], clean_row: Mapping[str, Any]) -> CorrectnessDecision:
        if self.mode == "none":
            return CorrectnessDecision(True, "disabled", "correctness_not_requested")

        if self.mode == "field":
            raw_value = get_dotted_value(raw_row, self.field)
            if raw_value is None:
                return CorrectnessDecision(False, "field", "correctness_field_missing")
            score = parse_correctness_value(raw_value)
            if score is None:
                return CorrectnessDecision(False, "field", "correctness_field_invalid")
            keep = score >= self.threshold
            return CorrectnessDecision(keep, "field", "correctness_pass" if keep else "correctness_incorrect", score)

        question = str(clean_row.get("question", ""))
        reference = str(clean_row.get("gt", ""))
        candidate = extract_candidate_answer(clean_row.get("messages"))
        key = make_correctness_cache_key(
            question,
            reference,
            candidate,
            prompt_version=self.prompt_version,
            judge_model=self.judge_model,
        )
        cached = self.cache.get(key)
        if cached is not None:
            cached_version = str(cached.get("prompt_version") or "")
            cached_model = str(cached.get("judge_model") or "")
            if cached_version != self.prompt_version or cached_model != str(
                self.judge_model or ""
            ):
                if self.mode == "cache":
                    return CorrectnessDecision(
                        False,
                        "cache",
                        "correctness_cache_provenance_mismatch",
                        cache_key=key,
                    )
                cached = None
        if cached is not None:
            score = (
                parse_correctness_value(cached.get("score"))
                if "score" in cached
                else parse_correctness_value(cached)
            )
            if score is not None:
                self.used_cache_judge_models.add(str(cached.get("judge_model") or "unspecified"))
                self.used_cache_prompt_versions.add(str(cached.get("prompt_version") or "unspecified"))
                keep = score >= self.threshold
                return CorrectnessDecision(
                    keep,
                    "cache",
                    "correctness_pass" if keep else "correctness_incorrect",
                    score,
                    key,
                )
            if self.mode == "cache":
                return CorrectnessDecision(False, "cache", "correctness_cache_invalid", cache_key=key)
        elif self.mode == "cache":
            return CorrectnessDecision(False, "cache", "correctness_cache_miss", cache_key=key)

        assert self.judge is not None
        score, reason = self.judge.evaluate(question, reference, candidate)
        record = {
            "cache_key": key,
            "question": question,
            "reference": reference,
            "candidate": candidate,
            "correct": score >= self.threshold,
            "score": score,
            "reason": reason,
            "prompt_version": self.prompt_version,
            "judge_model": self.judge_model,
            "cache_key_version": 2,
        }
        self.cache[key] = record
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            if self.cache_path.suffix.casefold() == ".json":
                ordered_records = [self.cache[cache_key] for cache_key in sorted(self.cache)]
                self.cache_path.write_text(
                    json.dumps(ordered_records, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            else:
                with self.cache_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        keep = score >= self.threshold
        return CorrectnessDecision(
            keep,
            "judge",
            "correctness_pass" if keep else "correctness_incorrect",
            score,
            key,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_records_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    """Hash logical SFT fields independently from parquet metadata."""

    digest = hashlib.sha256()
    for record in records:
        payload = {key: record.get(key) for key in ("question", "messages", "gt")}
        digest.update(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def api_key_from_env(env_name: str) -> str | None:
    """Read a configured secret without ever returning it to manifests/logs."""

    value = os.environ.get(env_name)
    return value if value else None
