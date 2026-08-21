import importlib.util
import json
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / "RL/verl/utils/reward_score/llm_judge.py"
SPEC = importlib.util.spec_from_file_location("rl_llm_judge_under_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": "correct: yes"}}]}


def _configure(monkeypatch, cache, mode):
    monkeypatch.setenv("JUDGE_API_BASE", "https://judge.invalid/v1")
    monkeypatch.setenv("JUDGE_API_KEY", "test-only-key")
    monkeypatch.setenv("JUDGE_MODEL_NAME", "judge-model")
    monkeypatch.setenv("JUDGE_CACHE_MODE", mode)
    monkeypatch.setenv("JUDGE_CACHE_PATH", str(cache))
    monkeypatch.setenv("JUDGE_SEED", "42")


def test_judge_record_then_replay_is_stable(monkeypatch, tmp_path):
    cache = tmp_path / "judge.jsonl"
    _configure(monkeypatch, cache, "record")
    calls = []

    def post(*args, **kwargs):
        calls.append(kwargs["json"])
        return _Response()

    monkeypatch.setattr(MODULE.requests, "post", post)
    assert MODULE.get_judge_response("prompt") == "correct: yes"
    assert calls[0]["temperature"] == 0.0
    assert calls[0]["seed"] == 42

    _configure(monkeypatch, cache, "replay")
    monkeypatch.setattr(
        MODULE.requests,
        "post",
        lambda *args, **kwargs: pytest.fail("replay contacted the provider"),
    )
    assert MODULE.get_judge_response("prompt") == "correct: yes"
    record = json.loads(cache.read_text(encoding="utf-8").splitlines()[0])
    assert record["arguments"]["model"] == "judge-model"
    assert record["arguments"]["prompt_version"] == MODULE.JUDGE_PROMPT_VERSION
    assert record["arguments"]["endpoint_identity"] == "https://judge.invalid/v1/chat/completions"
    assert record["arguments"]["enable_thinking"] is True


def test_judge_parse_failure_is_not_reward_zero(monkeypatch):
    monkeypatch.setattr(MODULE, "get_judge_response", lambda prompt: "unparseable")
    with pytest.raises(RuntimeError, match="refusing"):
        MODULE.compute_score(
            "user\nquestion\nassistant\n",
            "assistant\n<answer>candidate</answer>",
            "reference",
            "source",
            True,
            None,
        )


def test_judge_cache_isolated_by_thinking_and_provider(monkeypatch, tmp_path):
    cache = tmp_path / "judge.jsonl"
    _configure(monkeypatch, cache, "record")
    monkeypatch.setattr(MODULE.requests, "post", lambda *args, **kwargs: _Response())
    assert MODULE.get_judge_response("prompt") == "correct: yes"

    monkeypatch.setenv("JUDGE_CACHE_MODE", "replay")
    monkeypatch.setenv("ENABLE_JUDGE_THINKING", "false")
    with pytest.raises(Exception, match="No cached response"):
        MODULE.get_judge_response("prompt")

    monkeypatch.setenv("ENABLE_JUDGE_THINKING", "true")
    monkeypatch.setenv("JUDGE_PROVIDER_ID", "different-deployment")
    with pytest.raises(Exception, match="No cached response"):
        MODULE.get_judge_response("prompt")
