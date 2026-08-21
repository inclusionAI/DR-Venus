import importlib
from types import ModuleType
import sys

import pytest

from Inference.tool_runtime import execute_tool_call
from Inference.tool_server.deadline import ToolDeadline, ToolTimeoutError, positive_timeout
from drvenus_repro.tool_cache import ToolCacheMissError


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


def test_deadline_rejects_invalid_timeout_and_expires(monkeypatch: pytest.MonkeyPatch):
    assert positive_timeout("2.5") == 2.5
    with pytest.raises(ValueError, match="positive"):
        positive_timeout(0)

    clock = {"now": 10.0}
    deadline_module = importlib.import_module("Inference.tool_server.deadline")
    monkeypatch.setattr(deadline_module.time, "monotonic", lambda: clock["now"])
    deadline = ToolDeadline.start(3)
    assert deadline.remaining() == pytest.approx(3)
    clock["now"] = 13.01
    with pytest.raises(ToolTimeoutError, match="3 seconds"):
        deadline.remaining()


def test_runtime_passes_timeout_and_recovers_with_typed_counter_signal():
    seen = {}

    def timed_out(call, *, timeout):
        seen.update(call=call, timeout=timeout)
        raise ToolTimeoutError("provider timed out")

    result = execute_tool_call(
        {"name": "search", "arguments": {"query": ["x"]}},
        timed_out,
        timeout_seconds=7.5,
    )

    assert seen["timeout"] == 7.5
    assert result.failed is True
    assert result.timed_out is True
    assert result.content.startswith("ERROR: tool timeout:")


def test_replay_cache_miss_is_classified_as_fatal():
    def miss(call, *, timeout):
        del call, timeout
        raise ToolCacheMissError("missing frozen response")

    result = execute_tool_call(
        {"name": "visit", "arguments": {"url": ["x"], "goal": "g"}},
        miss,
        timeout_seconds=3,
    )

    assert result.failed is True
    assert result.fatal_replay_error is True
    assert result.cache_miss is True


def test_search_sets_socket_timeout_from_shared_deadline(monkeypatch: pytest.MonkeyPatch):
    _install_qwen_stubs(monkeypatch)
    sys.modules.pop("Inference.tool_server.tool_search", None)
    module = importlib.import_module("Inference.tool_server.tool_search")
    observed = []

    class Response:
        status = 200

        @staticmethod
        def read():
            return b'{"organic":[{"title":"T","link":"https://e.test","snippet":"S"}]}'

    class Connection:
        def __init__(self, host, *, timeout):
            assert host == "google.serper.dev"
            observed.append(timeout)

        def request(self, *args):
            return None

        @staticmethod
        def getresponse():
            return Response()

        @staticmethod
        def close():
            return None

    monkeypatch.setattr(module.http.client, "HTTPSConnection", Connection)
    response = module.Search().call({"query": ["x"]}, timeout=12.5)

    assert observed and 0 < observed[0] <= 12.5
    assert "A Google search for 'x'" in response


def test_visit_sets_requests_timeout_from_shared_deadline(monkeypatch: pytest.MonkeyPatch):
    _install_qwen_stubs(monkeypatch)

    openai_module = ModuleType("openai")
    openai_module.OpenAI = object  # type: ignore[attr-defined]
    tiktoken_module = ModuleType("tiktoken")
    tiktoken_module.get_encoding = lambda name: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", openai_module)
    monkeypatch.setitem(sys.modules, "tiktoken", tiktoken_module)
    sys.modules.pop("Inference.tool_server.tool_visit", None)
    module = importlib.import_module("Inference.tool_server.tool_visit")
    observed = []

    class Response:
        status_code = 200
        text = "page"

    def fake_get(url, *, proxies, headers, timeout):
        del proxies, headers
        assert url == "https://r.jina.ai/https://example.test"
        observed.append(timeout)
        return Response()

    monkeypatch.setattr(module.requests, "get", fake_get)
    deadline = ToolDeadline.start(9)
    assert module.Visit().jina_readpage(
        "https://example.test", deadline=deadline
    ) == "page"
    assert observed and 0 < observed[0] <= 9
