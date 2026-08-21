import importlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest

from RL.scrl.llm_agent.tool_runtime import execute_tool_call


ROOT = Path(__file__).parents[1]


def _install_qwen_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    qwen_agent = ModuleType("qwen_agent")
    qwen_agent.__path__ = []  # type: ignore[attr-defined]
    tools = ModuleType("qwen_agent.tools")
    tools.__path__ = []  # type: ignore[attr-defined]
    base = ModuleType("qwen_agent.tools.base")

    class BaseTool:
        def __init__(self, cfg=None):
            self.cfg = cfg

    def register_tool(name, allow_overwrite=False):
        del name, allow_overwrite
        return lambda cls: cls

    base.BaseTool = BaseTool  # type: ignore[attr-defined]
    base.register_tool = register_tool  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qwen_agent", qwen_agent)
    monkeypatch.setitem(sys.modules, "qwen_agent.tools", tools)
    monkeypatch.setitem(sys.modules, "qwen_agent.tools.base", base)


def _import_rl_tool(monkeypatch: pytest.MonkeyPatch, name: str):
    monkeypatch.syspath_prepend(str(ROOT / "RL"))
    for module_name in list(sys.modules):
        if module_name == "tool_server" or module_name.startswith("tool_server."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
    return importlib.import_module(f"tool_server.{name}")


def test_agent_runtime_passes_provider_timeout_and_classifies_timeout():
    seen = {}

    def dispatch(call, *, timeout):
        seen.update(call=call, timeout=timeout)
        raise TimeoutError("socket deadline expired")

    result = execute_tool_call(
        {"name": "search", "arguments": {"query": ["x"]}},
        dispatch,
        timeout_seconds=7.5,
    )

    assert seen["timeout"] == 7.5
    assert result.failed is True
    assert result.timed_out is True
    assert result.content.startswith("Tool execution timed out after 8 seconds")


def test_search_retries_share_one_decreasing_deadline(monkeypatch: pytest.MonkeyPatch):
    _install_qwen_stubs(monkeypatch)
    module = _import_rl_tool(monkeypatch, "tool_search")
    deadline_module = importlib.import_module("tool_server.deadline")
    clock = {"now": 100.0}
    monkeypatch.setattr(deadline_module.time, "monotonic", lambda: clock["now"])
    observed = []

    class Response:
        status = 200

        @staticmethod
        def read():
            return b'{"organic":[{"title":"T","link":"https://e.test","snippet":"S"}]}'

    class Connection:
        attempts = 0

        def __init__(self, host, *, timeout):
            assert host == "google.serper.dev"
            observed.append(timeout)

        def request(self, *args):
            type(self).attempts += 1
            clock["now"] += 2.0
            if type(self).attempts == 1:
                raise ConnectionError("retryable")

        @staticmethod
        def getresponse():
            return Response()

        @staticmethod
        def close():
            return None

    monkeypatch.setattr(module.http.client, "HTTPSConnection", Connection)
    result = module.Search().call({"query": ["x"]}, timeout=5)

    assert "A Google search for 'x'" in result
    assert observed == pytest.approx([5.0, 3.0])


def test_visit_jina_retries_share_deadline(monkeypatch: pytest.MonkeyPatch):
    _install_qwen_stubs(monkeypatch)
    openai_module = ModuleType("openai")
    openai_module.OpenAI = object  # type: ignore[attr-defined]
    tiktoken_module = ModuleType("tiktoken")
    tiktoken_module.get_encoding = lambda name: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", openai_module)
    monkeypatch.setitem(sys.modules, "tiktoken", tiktoken_module)
    module = _import_rl_tool(monkeypatch, "tool_visit")
    deadline_module = importlib.import_module("tool_server.deadline")
    clock = {"now": 10.0}
    monkeypatch.setattr(deadline_module.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        deadline_module.time,
        "sleep",
        lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
    )
    observed = []

    class Response:
        status_code = 200
        text = "page"

    def fake_get(url, *, proxies, headers, timeout):
        del url, proxies, headers
        observed.append(timeout)
        if len(observed) == 1:
            clock["now"] += 1.0
            raise ConnectionError("retryable")
        return Response()

    monkeypatch.setattr(module.requests, "get", fake_get)
    deadline = deadline_module.ToolDeadline.start(5)
    assert module.Visit().jina_readpage("https://e.test", deadline=deadline) == "page"
    assert observed == pytest.approx([5.0, 3.5])


def test_visit_summary_client_uses_remaining_deadline_for_each_retry(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_qwen_stubs(monkeypatch)
    tiktoken_module = ModuleType("tiktoken")
    tiktoken_module.get_encoding = lambda name: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tiktoken", tiktoken_module)

    clients = []
    clock = {"now": 20.0}

    class Completion:
        def create(self, **kwargs):
            del kwargs
            clock["now"] += 1.0
            if len(clients) == 1:
                raise RuntimeError("retryable")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"summary":"s"}'))]
            )

    class Client:
        def __init__(self, **kwargs):
            clients.append(kwargs["timeout"])
            assert kwargs["max_retries"] == 0
            self.chat = SimpleNamespace(completions=Completion())

        def close(self):
            return None

    openai_module = ModuleType("openai")
    openai_module.OpenAI = Client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", openai_module)
    module = _import_rl_tool(monkeypatch, "tool_visit")
    deadline_module = importlib.import_module("tool_server.deadline")
    monkeypatch.setattr(deadline_module.time, "monotonic", lambda: clock["now"])
    deadline = deadline_module.ToolDeadline.start(5)

    result = module.Visit().call_server(
        [{"role": "user", "content": "x"}], max_retries=2, deadline=deadline
    )

    assert result == '{"summary":"s"}'
    assert clients == pytest.approx([5.0, 4.0])


def test_record_mode_timeout_never_writes_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    seen = []

    class StubSearch:
        def call(self, arguments, **kwargs):
            del arguments
            seen.append(kwargs["timeout"])
            raise TimeoutError("provider expired")

    class StubVisit:
        pass

    package = ModuleType("tool_server")
    package.__path__ = []  # type: ignore[attr-defined]
    search_module = ModuleType("tool_server.tool_search")
    search_module.Search = StubSearch  # type: ignore[attr-defined]
    visit_module = ModuleType("tool_server.tool_visit")
    visit_module.Visit = StubVisit  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tool_server", package)
    monkeypatch.setitem(sys.modules, "tool_server.tool_search", search_module)
    monkeypatch.setitem(sys.modules, "tool_server.tool_visit", visit_module)

    path = ROOT / "RL" / "tool_server" / "execute_tools.py"
    spec = importlib.util.spec_from_file_location("_rl_timeout_dispatcher", path)
    assert spec is not None and spec.loader is not None
    dispatcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dispatcher)

    cache_path = tmp_path / "tools.jsonl"
    monkeypatch.setenv("TOOL_CACHE_MODE", "record")
    monkeypatch.setenv("TOOL_CACHE_PATH", str(cache_path))
    with pytest.raises(TimeoutError):
        dispatcher.custom_call_tool(
            {"name": "search", "arguments": {"query": ["x"]}}, timeout=4.5
        )

    records = [json.loads(line) for line in cache_path.read_text().splitlines()]
    assert seen == [4.5]
    assert [record["status"] for record in records] == ["error"]
    assert all(record["status"] != "ok" for record in records)
