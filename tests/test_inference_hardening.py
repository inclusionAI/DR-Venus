import importlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest

from Inference.evidence_identity import sha256_artifact, sha256_file
from Inference.format_validation import strict_raw_format_valid


def test_strict_format_rejects_repaired_mixed_and_repeated_outputs():
    assert strict_raw_format_valid(
        "<think>x</think><answer>answer</answer>"
    )
    assert strict_raw_format_valid(
        '<think>x</think><tool_call>{"name":"search"}</tool_call>'
    )
    assert not strict_raw_format_valid("<think>x<answer>answer</answer>")
    assert not strict_raw_format_valid(
        "<think>x</think><answer>a</answer><tool_call>{}</tool_call>"
    )
    assert not strict_raw_format_valid(
        "<think>x</think><answer>a</answer><answer>b</answer>"
    )
    assert not strict_raw_format_valid(
        "prefix <think>x</think><answer>a</answer>"
    )


def test_artifact_hash_binds_relative_names_and_contents(tmp_path: Path):
    artifact = tmp_path / "model"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}", encoding="utf-8")
    first = sha256_artifact(artifact)
    assert sha256_file(artifact / "config.json")

    (artifact / "config.json").write_text('{"changed":true}', encoding="utf-8")
    assert sha256_artifact(artifact) != first


@pytest.fixture
def run_demo_module(monkeypatch: pytest.MonkeyPatch):
    openai_module = ModuleType("openai")
    openai_module.Client = object  # type: ignore[attr-defined]
    transformers_module = ModuleType("transformers")
    transformers_module.AutoTokenizer = object  # type: ignore[attr-defined]
    transformers_module.PreTrainedTokenizer = object  # type: ignore[attr-defined]
    dispatcher = ModuleType("Inference.tool_server.execute_tools")
    dispatcher.custom_call_tool = lambda call, **kwargs: "evidence"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", openai_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "Inference.tool_server.execute_tools", dispatcher)
    sys.modules.pop("Inference.run_demo", None)
    return importlib.import_module("Inference.run_demo")


class FakeTokenizer:
    @staticmethod
    def apply_chat_template(messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        rendered = "".join(
            f"<{message['role']}>{message['content']}" for message in messages
        )
        return rendered + ("<assistant>" if add_generation_prompt else "")

    @staticmethod
    def encode(text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text)

    def __call__(self, text, *, add_special_tokens):
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}

    @staticmethod
    def decode(tokens, *, skip_special_tokens):
        assert skip_special_tokens is False
        return "".join(tokens)


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def _client(*responses):
    completions = FakeCompletions(responses)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def test_token_limit_preserves_tool_response_and_turn_bound(run_demo_module, monkeypatch):
    module = run_demo_module
    client, _ = _client(
        '<think>look</think><tool_call>{"name":"search","arguments":{"query":["x"]}}</tool_call>',
        "<think>done</think><answer>answer</answer>",
    )
    monkeypatch.setattr(module, "custom_call_tool", lambda call, **kwargs: "EVIDENCE")
    solver = module.ResearchProblemSolver(
        question="q",
        tokenizer=FakeTokenizer(),
        max_len=99_000,
        model_context_len=100_000,
        max_tokens_per_turn=1024,
        openai_client=client,
        model_name="model",
        max_steps=2,
        max_retries=1,
    )
    first_output = (
        '<think>look</think><tool_call>{"name":"search",'
        '"arguments":{"query":["x"]}}</tool_call>'
    )
    solver.max_len = solver._count_messages(
        [
            *solver.messages,
            {"role": "assistant", "content": first_output},
            {"role": "user", "content": "<tool_response>\nEVIDENCE\n</tool_response>"},
        ]
    )

    result = solver.solve()

    user_messages = [
        message["content"] for message in solver.messages if message["role"] == "user"
    ]
    assert any("EVIDENCE" in content for content in user_messages)
    assert any("maximum context length" in content for content in user_messages)
    assert result["turns"] == 2
    assert result["turns"] <= solver.max_steps
    assert result["final_answer_strict_format_valid"] is True


def test_generation_is_dynamically_capped_by_remaining_context(run_demo_module):
    module = run_demo_module
    client, completions = _client("<think>x</think><answer>a</answer>")
    solver = module.ResearchProblemSolver(
        question="q",
        tokenizer=FakeTokenizer(),
        max_len=100,
        model_context_len=200,
        max_tokens_per_turn=50,
        openai_client=client,
        model_name="model",
        max_steps=1,
        max_retries=1,
    )
    solver.messages = [{"role": "user", "content": "x"}]
    solver.model_context_len = solver.count_tokens() + 7

    assert solver._call_llm_once(use_forced_answer_reserve=True)
    assert completions.calls[0]["max_tokens"] == 7


def test_large_tool_response_is_head_tail_truncated_within_solver_limit(
    run_demo_module,
):
    module = run_demo_module
    client, _ = _client("<think>x</think><answer>a</answer>")
    solver = module.ResearchProblemSolver(
        question="q",
        tokenizer=FakeTokenizer(),
        max_len=10_000,
        model_context_len=20_000,
        max_tokens_per_turn=1024,
        openai_client=client,
        model_name="model",
        max_steps=1,
        max_retries=1,
    )
    solver.messages = [{"role": "user", "content": "short"}]
    solver.max_len = solver._count_messages(solver.messages) + 120
    result = "HEAD" + ("x" * 1000) + "TAIL"

    fitted = solver.fit_tool_response(result)

    assert fitted.startswith("HEAD")
    assert fitted.endswith("TAIL")
    assert "tool response truncated" in fitted
    candidate = {
        "role": "user",
        "content": f"<tool_response>\n{fitted}\n</tool_response>",
    }
    assert solver._count_messages([*solver.messages, candidate]) <= solver.max_len
    assert solver.tool_response_truncation_count == 1


def test_retry_seeds_change_but_are_reproducible(run_demo_module, monkeypatch):
    module = run_demo_module
    monkeypatch.setattr(module.time, "sleep", lambda _: None)

    def run_once():
        client, completions = _client(
            "<think>x</think>bad",
            "<think>x</think><answer>a</answer>",
        )
        solver = module.ResearchProblemSolver(
            question="q",
            tokenizer=FakeTokenizer(),
            max_len=9_000,
            model_context_len=10_000,
            max_tokens_per_turn=100,
            openai_client=client,
            model_name="model",
            max_steps=1,
            max_retries=2,
            seed=42,
        )
        content, status, *_ = solver._generate(step=0)
        assert content and status == 0
        return [call["seed"] for call in completions.calls]

    assert run_once() == [42, 43]
    assert run_once() == [42, 43]


def test_raw_repair_is_counted_as_strict_format_violation(run_demo_module):
    module = run_demo_module
    client, _ = _client("<think>x<answer>a</answer>")
    solver = module.ResearchProblemSolver(
        question="q",
        tokenizer=FakeTokenizer(),
        max_len=100,
        model_context_len=10_000,
        max_tokens_per_turn=50,
        openai_client=client,
        model_name="model",
        max_steps=1,
        max_retries=1,
    )

    result = solver.solve()
    assert result["prediction"] == "a"
    assert result["strict_format_violation_count"] == 1
    assert result["final_answer_strict_format_valid"] is False


def test_existing_server_requires_opt_in_and_verified_metadata(
    run_demo_module, monkeypatch
):
    module = run_demo_module

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "data": [
                    {
                        "id": "model",
                        "root": "org/checkpoint",
                        "max_model_len": 32768,
                    }
                ]
            }

    monkeypatch.setattr(module.requests, "get", lambda *args, **kwargs: Response())
    monkeypatch.setattr(module, "_port_is_open", lambda port: True)
    args = SimpleNamespace(
        deploy_mode="tp",
        base_port=6000,
        num_gpus=2,
        reuse_server=False,
        model_path="org/checkpoint",
        model_revision=None,
        max_model_len=32768,
    )
    with pytest.raises(RuntimeError, match="refusing to reuse"):
        module.start_vllm(args)

    args.reuse_server = True
    assert module.start_vllm(args) == ([], [6000], True)


def test_vllm_start_failure_cleans_up_spawned_process(run_demo_module, monkeypatch):
    module = run_demo_module

    class Process:
        pid = 123

        def __init__(self):
            self.terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout):
            return 0

    process = Process()
    monkeypatch.setattr(module, "_port_is_open", lambda port: False)
    monkeypatch.setattr(
        module, "start_vllm_tp", lambda args: ([process], [args.base_port])
    )
    monkeypatch.setattr(
        module,
        "wait_for_servers",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("OOM")),
    )
    args = SimpleNamespace(
        deploy_mode="tp",
        base_port=6000,
        num_gpus=2,
        reuse_server=False,
        server_start_timeout=10,
    )

    with pytest.raises(RuntimeError, match="OOM"):
        module.start_vllm(args)
    assert process.terminated is True
