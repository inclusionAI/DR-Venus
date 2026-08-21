from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import ModuleType

import pytest

from drvenus_repro.tool_cache import (
    CachedToolExecutionError,
    ToolCacheConfigurationError,
    ToolCacheMissError,
    build_cache_key,
    execute_with_tool_cache,
    normalize_tool_arguments,
)


def _configure(monkeypatch: pytest.MonkeyPatch, mode: str, path: Path | None = None) -> None:
    monkeypatch.setenv("TOOL_CACHE_MODE", mode)
    if path is None:
        monkeypatch.delenv("TOOL_CACHE_PATH", raising=False)
    else:
        monkeypatch.setenv("TOOL_CACHE_PATH", str(path))


def test_online_is_the_default_and_does_not_require_a_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOOL_CACHE_MODE", raising=False)
    monkeypatch.delenv("TOOL_CACHE_PATH", raising=False)
    calls = 0

    def online() -> str:
        nonlocal calls
        calls += 1
        return f"response-{calls}"

    assert execute_with_tool_cache("search", {"query": ["x"]}, online, provider="serper") == "response-1"
    assert execute_with_tool_cache("search", {"query": ["x"]}, online, provider="serper") == "response-2"


def test_normalization_is_order_independent_but_preserves_array_order() -> None:
    composed = {"goal": "caf\u00e9", "url": ["b", "a"]}
    decomposed = {"url": ["b", "a"], "goal": "cafe\u0301"}

    assert normalize_tool_arguments(composed) == normalize_tool_arguments(decomposed)
    assert build_cache_key("VISIT", composed) == build_cache_key("visit", decomposed)
    assert build_cache_key("visit", composed) != build_cache_key(
        "visit", {"goal": "caf\u00e9", "url": ["a", "b"]}
    )


def test_record_writes_required_metadata_and_reuses_a_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_path = tmp_path / "tools.jsonl"
    _configure(monkeypatch, "record", cache_path)
    calls = 0

    def online() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"items": [1, 2], "source": "live"}

    arguments = {"query": ["deep research"], "options": {"lang": "en", "limit": 10}}
    expected = {"items": [1, 2], "source": "live"}
    assert execute_with_tool_cache("search", arguments, online, provider="serper") == expected
    reordered = {"options": {"limit": 10, "lang": "en"}, "query": ["deep research"]}
    assert execute_with_tool_cache("search", reordered, online, provider="serper") == expected
    assert calls == 1

    lines = cache_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["provider"] == "serper"
    assert record["status"] == "ok"
    assert record["timestamp"].endswith("Z")
    assert record["cache_key"] == build_cache_key("search", arguments)
    canonical_response = json.dumps(expected, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert record["response_hash"] == hashlib.sha256(canonical_response.encode()).hexdigest()


def test_replay_hit_never_invokes_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_path = tmp_path / "tools.jsonl"
    arguments = {"url": "https://example.test", "goal": "find the date"}
    _configure(monkeypatch, "record", cache_path)
    assert execute_with_tool_cache("visit", arguments, lambda: "recorded", provider="jina") == "recorded"

    _configure(monkeypatch, "replay", cache_path)

    def must_not_run() -> str:
        raise AssertionError("provider was contacted during replay")

    assert execute_with_tool_cache("visit", arguments, must_not_run, provider="jina") == "recorded"


def test_replay_miss_is_explicit_and_never_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure(monkeypatch, "replay", tmp_path / "missing.jsonl")
    called = False

    def online() -> str:
        nonlocal called
        called = True
        return "should not happen"

    with pytest.raises(ToolCacheMissError, match="never falls back online"):
        execute_with_tool_cache("search", {"query": ["miss"]}, online, provider="serper")
    assert called is False


def test_corrupt_lines_and_hash_mismatches_are_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cache_path = tmp_path / "tools.jsonl"
    arguments = {"query": ["valid"]}
    _configure(monkeypatch, "record", cache_path)
    execute_with_tool_cache("search", arguments, lambda: "valid response", provider="serper")
    valid_line = cache_path.read_text(encoding="utf-8")
    invalid_hash = json.loads(valid_line)
    invalid_hash["response_hash"] = "0" * 64
    cache_path.write_text(
        "not-json\n" + json.dumps(invalid_hash) + "\n" + valid_line,
        encoding="utf-8",
    )

    _configure(monkeypatch, "replay", cache_path)
    with caplog.at_level("WARNING", logger="drvenus_repro.tool_cache"):
        result = execute_with_tool_cache(
            "search", arguments, lambda: pytest.fail("unexpected provider call"), provider="serper"
        )
    assert result == "valid response"
    assert "Ignoring corrupt tool-cache line" in caplog.text


def test_record_recovers_after_a_truncated_jsonl_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_path = tmp_path / "tools.jsonl"
    cache_path.write_bytes(b'{"interrupted":')
    _configure(monkeypatch, "record", cache_path)
    arguments = {"query": ["recover"]}

    assert execute_with_tool_cache(
        "search", arguments, lambda: "recovered", provider="serper"
    ) == "recovered"
    _configure(monkeypatch, "replay", cache_path)
    assert execute_with_tool_cache(
        "search", arguments, lambda: pytest.fail("unexpected provider call"), provider="serper"
    ) == "recovered"


def test_api_keys_are_never_serialized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_path = tmp_path / "tools.jsonl"
    secret = "serper-super-secret-key"
    monkeypatch.setenv("SERPER_KEY_ID", secret)
    _configure(monkeypatch, "record", cache_path)

    response = f"provider accidentally echoed {secret}"
    arguments = {"query": ["x"], "api_key": secret}
    assert execute_with_tool_cache(
        "search", arguments, lambda: response, provider="serper"
    ) == "provider accidentally echoed [REDACTED]"
    cache_text = cache_path.read_text(encoding="utf-8")
    assert secret not in cache_text
    assert "[REDACTED]" in cache_text
    record = json.loads(cache_text)
    assert record["status_code"] == 200
    assert record["latency_ms"] >= 0


def test_recorded_errors_are_retried_in_record_and_raised_in_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_path = tmp_path / "tools.jsonl"
    arguments = {"query": ["transient"]}
    _configure(monkeypatch, "record", cache_path)

    with pytest.raises(ValueError, match="temporary outage"):
        execute_with_tool_cache(
            "search",
            arguments,
            lambda: (_ for _ in ()).throw(ValueError("temporary outage")),
            provider="serper",
        )

    _configure(monkeypatch, "replay", cache_path)
    with pytest.raises(CachedToolExecutionError, match="temporary outage"):
        execute_with_tool_cache("search", arguments, lambda: "online", provider="serper")


def test_threaded_appends_produce_complete_json_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_path = tmp_path / "tools.jsonl"
    _configure(monkeypatch, "record", cache_path)

    def invoke(index: int) -> str:
        return execute_with_tool_cache(
            "search", {"query": [f"query-{index}"]}, lambda: f"response-{index}", provider="serper"
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        assert list(executor.map(invoke, range(24))) == [f"response-{index}" for index in range(24)]

    records = [json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 24
    assert {record["cache_key"] for record in records} == {
        build_cache_key("search", {"query": [f"query-{index}"]}) for index in range(24)
    }


def test_racing_calls_for_one_key_converge_on_first_recorded_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_path = tmp_path / "tools.jsonl"
    _configure(monkeypatch, "record", cache_path)
    barrier = threading.Barrier(8)

    def invoke(index: int) -> str:
        def online() -> str:
            barrier.wait(timeout=5)
            return f"response-{index}"

        return execute_with_tool_cache(
            "search", {"query": ["same query"]}, online, provider="serper"
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        responses = list(executor.map(invoke, range(8)))

    assert len(set(responses)) == 1
    assert len(cache_path.read_text(encoding="utf-8").splitlines()) == 1


def test_non_online_modes_require_an_explicit_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure(monkeypatch, "record")
    with pytest.raises(ToolCacheConfigurationError, match="TOOL_CACHE_PATH"):
        execute_with_tool_cache("search", {"query": ["x"]}, lambda: "x", provider="serper")


@pytest.mark.parametrize("subproject", ["Inference", "RL"])
def test_both_tool_server_dispatchers_use_the_shared_cache(
    subproject: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"search": 0, "visit": 0}

    class StubSearch:
        def call(self, arguments: dict[str, object], **kwargs: object) -> str:
            calls["search"] += 1
            query = arguments["query"][0]  # type: ignore[index]
            return (
                f"A Google search for '{query}' found 1 results:\n\n## Web Results\n"
                "1. [Result](https://example.test)\nA deterministic snippet"
            )

    class StubVisit:
        def call(self, arguments: dict[str, object], **kwargs: object) -> str:
            calls["visit"] += 1
            return "deterministic page summary"

    package = ModuleType("tool_server")
    package.__path__ = []  # type: ignore[attr-defined]
    search_module = ModuleType("tool_server.tool_search")
    search_module.Search = StubSearch  # type: ignore[attr-defined]
    visit_module = ModuleType("tool_server.tool_visit")
    visit_module.Visit = StubVisit  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tool_server", package)
    monkeypatch.setitem(sys.modules, "tool_server.tool_search", search_module)
    monkeypatch.setitem(sys.modules, "tool_server.tool_visit", visit_module)

    module_path = Path(__file__).parents[1] / subproject / "tool_server" / "execute_tools.py"
    module_name = f"_tool_cache_integration_{subproject.casefold()}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    dispatcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dispatcher)

    cache_path = tmp_path / f"{subproject}.jsonl"
    _configure(monkeypatch, "record", cache_path)
    search_call = {"name": "search", "arguments": {"query": ["cache me"]}}
    visit_call = {
        "name": "visit",
        "arguments": {"url": "https://example.test", "goal": "summarize"},
    }
    recorded_search = dispatcher.custom_call_tool(search_call)
    recorded_visit = dispatcher.custom_call_tool(visit_call)
    assert calls == {"search": 1, "visit": 1}

    _configure(monkeypatch, "replay", cache_path)
    assert dispatcher.custom_call_tool(search_call) == recorded_search
    assert dispatcher.custom_call_tool(visit_call) == recorded_visit
    assert calls == {"search": 1, "visit": 1}
