#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import logging
import math
import os
import random
import re
import socket
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import filelock
import json5
import openai
import requests
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

#from tools_server.http_client import MessageClient
try:
    from .tool_server.execute_tools import custom_call_tool
    from .eval_inputs import ad_hoc_item, build_run_fingerprint, completed_keys, expand_samples, load_evaluation_items
    from .token_accounting import count_input_output_tokens
    from .tool_runtime import ReplayToolEvidenceError, execute_tool_call
    from .format_validation import strict_raw_format_valid
    from .evidence_identity import sha256_artifact, sha256_file
except ImportError:  # Support ``python Inference/run_demo.py``.
    from tool_server.execute_tools import custom_call_tool
    from eval_inputs import ad_hoc_item, build_run_fingerprint, completed_keys, expand_samples, load_evaluation_items
    from token_accounting import count_input_output_tokens
    from tool_runtime import ReplayToolEvidenceError, execute_tool_call
    from format_validation import strict_raw_format_valid
    from evidence_identity import sha256_artifact, sha256_file

# Suppress verbose library logging (only keep WARNING+)
for _lib in ("openai", "httpx", "urllib3", "requests", "filelock", "httpcore"):
    logging.getLogger(_lib).setLevel(logging.WARNING)


SYSTEM_PROMPT = """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For each user request, you must actively seek out and **cross-check information** from credible and diverse sources, then integrate the findings into a response that is comprehensive, accurate, well-structured, and objective. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer in `<answer></answer>` tags.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "search", "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "visit", "description": "Visit webpage(s) and return the summary of the content.", "parameters": {"type": "object", "properties": {"url": {"type": "array", "items": {"type": "string"}, "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."}, "goal": {"type": "string", "description": "The specific information goal for visiting webpage(s)."}}, "required": ["url", "goal"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Current date: 2026-03-01
"""


ALLOWED_ARGS = {
    "search": {"query"},
    "visit": {"url", "goal"},}

ALLOWED_TOOLS = set(ALLOWED_ARGS.keys())
SAMPLE_SEED_STRIDE = 100_000


def _filter_args(tool_name: str, args: dict) -> dict:
    if not isinstance(args, dict):
        return {}
    allowed = ALLOWED_ARGS.get(tool_name, set())
    return {k: v for k, v in args.items() if k in allowed}


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class ResearchProblemSolver:
    def __init__(
        self,
        question: str,
        tokenizer: PreTrainedTokenizer,
        max_len: int,
        openai_client: openai.Client,
        # tool_client: MessageClient,
        model_name: str,
        max_steps: int,
        time_limit: int = 9000,
        verbose: bool = False,
        max_retries: int = 5,
        seed: int = 42,
        tool_timeout: float = 150.0,
        temperature: float = 1.0,
        top_p: float = 0.95,
        model_context_len: int = 261000,
        max_tokens_per_turn: int = 1024,
        tool_cache_mode: str = "online",
    ):
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        self.question = question
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.openai_client = openai_client
        self.model_name = model_name
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.max_steps = max_steps
        # self.tool_client = tool_client
        self.time_limit = time_limit
        self.verbose = verbose
        self.max_retries = max(max_retries, 1)  # at least 1 attempt
        self.seed = seed
        if not math.isfinite(tool_timeout) or tool_timeout <= 0:
            raise ValueError("tool_timeout must be positive")
        self.tool_timeout = float(tool_timeout)
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        if model_context_len < 1 or max_tokens_per_turn < 1:
            raise ValueError("context and per-turn generation limits must be positive")
        if max_len >= model_context_len:
            raise ValueError("max_len must leave context space for a forced answer")
        self.model_context_len = int(model_context_len)
        self.max_tokens_per_turn = int(max_tokens_per_turn)
        if tool_cache_mode not in {"online", "record", "replay"}:
            raise ValueError("invalid tool_cache_mode")
        self.tool_cache_mode = tool_cache_mode

        self.start_time = time.time()
        self.turns = 0
        self.parse_error_count = 0
        self.tool_call_count = 0
        self.browse_call_count = 0
        self.tool_error_count = 0
        self.timeout_count = 0
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.cached_tool_error_count = 0
        self.tool_response_truncation_count = 0
        self.tool_response_tokens_removed = 0
        self.dedup_count = 0
        self.assistant_format_validity: list[bool] = []
        self.final_answer_strict_format_valid = False
        self.llm_call_count = 0
        self.llm_failed_call_count = 0
        self.llm_usage_missing_count = 0
        self.inference_prompt_tokens = 0
        self.inference_completion_tokens = 0
        self.llm_sampling_seeds: list[int] = []

        # Tool call dedup history: key -> True
        self._tool_call_history: dict[str, bool] = {}

    def log(self, msg: str):
        if self.verbose:
            print(f"[Solver] {msg}")

    # ---- tool call dedup ----

    @staticmethod
    def _tool_call_key(tool_call: dict) -> str:
        """Generate a hashable key from tool name + arguments for dedup."""
        return json.dumps(
            {"name": tool_call["name"], "arguments": tool_call["arguments"]},
            sort_keys=True, ensure_ascii=False,
        )

    # ---- repetition detection ----

    @staticmethod
    def _is_repetitive(content: str, tail_len: int = 50, threshold: int = 5) -> bool:
        """Check if the last `tail_len` chars appear more than `threshold` times."""
        if not content or len(content) < tail_len:
            return False
        tail = content[-tail_len:]
        return content.count(tail) > threshold

    # ---- token counting ----

    def _count_messages(self, messages: list[dict]) -> int:
        rendered = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        return len(
            self.tokenizer(rendered, add_special_tokens=False)["input_ids"]
        )

    def count_tokens(self) -> int:
        return self._count_messages(self.messages)

    # ---- single LLM call (no retry) ----

    def _call_llm_once(self, *, use_forced_answer_reserve: bool = False) -> str | None:
        """Single API call. Returns stripped content or None on failure."""
        try:
            prompt_tokens = self.count_tokens()
            context_ceiling = (
                self.model_context_len
                if use_forced_answer_reserve
                else self.max_len
            )
            available_tokens = context_ceiling - prompt_tokens
            if available_tokens < 1:
                self.log(
                    f"No generation space remains: prompt={prompt_tokens}, "
                    f"context_ceiling={context_ceiling}"
                )
                return None
            max_tokens = min(self.max_tokens_per_turn, available_tokens)
            remaining_problem_time = self.time_limit - (
                time.time() - self.start_time
            )
            if remaining_problem_time <= 0:
                self.log("Per-problem time limit exhausted before LLM call")
                return None
            call_seed = self.seed + self.llm_call_count
            self.llm_call_count += 1
            self.llm_sampling_seeds.append(call_seed)
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                stop=["<tool_response>", "\n<tool_response>"],
                temperature=self.temperature,
                max_tokens=max_tokens,
                top_p=self.top_p,
                presence_penalty=1.1,
                seed=call_seed,
                extra_body={
                    "top_k": 20,
                },
                timeout=max(0.001, remaining_problem_time),
            )
            content = response.choices[0].message.content
            usage = getattr(response, "usage", None)
            usage_prompt = getattr(usage, "prompt_tokens", None)
            usage_completion = getattr(usage, "completion_tokens", None)
            if isinstance(usage_prompt, int) and isinstance(usage_completion, int):
                self.inference_prompt_tokens += usage_prompt
                self.inference_completion_tokens += usage_completion
            else:
                self.llm_usage_missing_count += 1
                self.inference_prompt_tokens += prompt_tokens
                if isinstance(content, str):
                    self.inference_completion_tokens += len(
                        self.tokenizer.encode(content, add_special_tokens=False)
                    )
            return content.strip() if content and content.strip() else None
        except Exception as e:
            self.llm_failed_call_count += 1
            self.log(f"API error: {e}")
            return None

    # ---- response parsing (regex + status code) ----

    def parse_llm_response(self, content: str, step: int) -> tuple[int, str, object]:
        """
        Returns (status, thinking_or_error, payload):
          0 = final answer   -> payload is answer text
          1 = tool call      -> payload is tool_call dict
          2 = parse error    -> payload is ""
        """
        THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

        self.log(f"step={step}, content_len={len(content)}")

        think_match = THINK_RE.search(content)
        if not think_match:
            return 2, "Missing <think></think>", ""

        thinking = think_match.group(1).strip()

        answer_match = ANSWER_RE.search(content)
        tool_match = TOOL_RE.search(content)

        # Final answer
        if answer_match and not tool_match:
            return 0, thinking, answer_match.group(1).strip()

        # Tool call
        if tool_match and not answer_match:
            raw_tool_json = tool_match.group(1).strip()

            # Other tools: JSON parse
            try:
                tool_call = json5.loads(raw_tool_json)

                if not isinstance(tool_call, dict):
                    return 2, f"Tool call should be dict, got {type(tool_call)}", ""
                if "name" not in tool_call or "arguments" not in tool_call:
                    return 2, f"Tool call missing fields, raw={raw_tool_json}", ""

                if tool_call["name"] not in ALLOWED_TOOLS:
                    return 2, f"Unknown tool '{tool_call['name']}'. Available tools: {', '.join(sorted(ALLOWED_TOOLS))}", ""

                tool_call["arguments"] = _filter_args(
                    tool_call["name"], tool_call["arguments"]
                )
                return 1, thinking, tool_call

            except Exception as e:
                return 2, f"Tool call parse error: {repr(e)}, raw={raw_tool_json}", ""

        return 2, "Ambiguous or incomplete output", ""

    # ---- tool execution via tongyi tool ----

    def execute_tool(self, tool_call: dict) -> str:
        if tool_call.get("name") in {"search", "visit"}:
            self.browse_call_count += 1
        remaining_problem_time = self.time_limit - (time.time() - self.start_time)
        if remaining_problem_time <= 0:
            self.tool_error_count += 1
            self.timeout_count += 1
            return "ERROR: tool timeout: per-problem time limit exhausted"
        outcome = execute_tool_call(
            tool_call,
            custom_call_tool,
            timeout_seconds=min(self.tool_timeout, remaining_problem_time),
        )
        if outcome.failed:
            self.tool_error_count += 1
        if outcome.timed_out:
            self.timeout_count += 1
        if outcome.cache_miss:
            self.cache_miss_count += 1
        if outcome.cached_error:
            self.cached_tool_error_count += 1
        if outcome.fatal_replay_error:
            raise ReplayToolEvidenceError(outcome)
        if self.tool_cache_mode == "replay":
            self.cache_hit_count += 1
        return outcome.content

    def fit_tool_response(self, result: str) -> str:
        """Truncate a large result head+tail so the forced answer still fits."""

        def wrapped(content: str) -> dict:
            return {
                "role": "user",
                "content": f"<tool_response>\n{content}\n</tool_response>",
            }

        if self._count_messages([*self.messages, wrapped(result)]) <= self.max_len:
            return result

        result_tokens = self.tokenizer.encode(result, add_special_tokens=False)
        marker_template = "\n...[tool response truncated: {removed} tokens removed]...\n"
        keep = len(result_tokens)
        candidate = result
        while keep > 0:
            removed = len(result_tokens) - keep
            marker = marker_template.format(removed=removed)
            head_count = (keep + 1) // 2
            tail_count = keep // 2
            head = self.tokenizer.decode(
                result_tokens[:head_count], skip_special_tokens=False
            )
            tail = self.tokenizer.decode(
                result_tokens[-tail_count:] if tail_count else [],
                skip_special_tokens=False,
            )
            candidate = f"{head}{marker}{tail}"
            overflow = (
                self._count_messages([*self.messages, wrapped(candidate)])
                - self.max_len
            )
            if overflow <= 0:
                break
            keep = max(0, keep - overflow - 8)

        if keep == 0:
            candidate = marker_template.format(removed=len(result_tokens)).strip()
        removed = len(result_tokens) - keep
        self.tool_response_truncation_count += 1
        self.tool_response_tokens_removed += removed
        return candidate

    # ---- extract answer from message history ----

    def extract_final_answer(self) -> str | None:
        ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
        for msg in reversed(self.messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if not isinstance(content, str) or not content.strip():
                    return None
                stripped = THINK_RE.sub("", content).strip()
                m = ANSWER_RE.search(stripped)
                if m and m.group(1).strip():
                    return m.group(1).strip()
                return content.strip()
        return None

    # ---- structured result builder ----

    def _make_result(self, termination: str, prediction: str | None = None) -> dict:
        if prediction is None:
            prediction = self.extract_final_answer() or "No answer found"
        print(prediction)
        transcript_input_tokens, transcript_output_tokens = count_input_output_tokens(
            self.messages, self.tokenizer
        )
        if termination == "timeout":
            self.timeout_count += 1
        return {
            "prediction": prediction,
            "termination": termination,
            "turns": self.turns,
            "parse_error_count": self.parse_error_count,
            "tool_call_count": self.tool_call_count,
            "browse_call_count": self.browse_call_count,
            "tool_error_count": self.tool_error_count,
            "timeout_count": self.timeout_count,
            "cache_hit_count": self.cache_hit_count,
            "cache_miss_count": self.cache_miss_count,
            "cached_tool_error_count": self.cached_tool_error_count,
            "tool_response_truncation_count": self.tool_response_truncation_count,
            "tool_response_tokens_removed": self.tool_response_tokens_removed,
            "dedup_count": self.dedup_count,
            "assistant_output_count": len(self.assistant_format_validity),
            "strict_format_violation_count": sum(
                not valid for valid in self.assistant_format_validity
            ),
            "all_assistant_outputs_strict_format_valid": bool(
                self.assistant_format_validity
            ) and all(self.assistant_format_validity),
            "final_answer_strict_format_valid": self.final_answer_strict_format_valid,
            "final_token_count": self.count_tokens(),
            "input_tokens": self.inference_prompt_tokens,
            "output_tokens": self.inference_completion_tokens,
            "transcript_input_tokens": transcript_input_tokens,
            "transcript_output_tokens": transcript_output_tokens,
            "llm_call_count": self.llm_call_count,
            "llm_failed_call_count": self.llm_failed_call_count,
            "llm_usage_missing_count": self.llm_usage_missing_count,
            "llm_sampling_seeds": list(self.llm_sampling_seeds),
            "elapsed_seconds": round(time.time() - self.start_time, 2),
        }

    # ---- sanitize LLM output ----

    @staticmethod
    def _sanitize_content(content: str) -> str:
        if "<tool_response>" in content:
            content = content[: content.find("<tool_response>")]
        if "<think>" in content and "</think>" not in content:
            for tag in ("<answer>", "<tool_call>"):
                pos = content.find(tag)
                if pos != -1:
                    content = content[:pos] + "\n</think>\n" + content[pos:]
                    break
            else:
                content = content + "\n</think>"
        if "<think>" not in content and "</think>" in content:
            content = "<think>\n" + content
        return content

    # ---- unified generate with retry ----

    def _generate(self, step: int) -> tuple[str | None, int, str, object, bool]:
        """
        Call LLM and validate output quality with unified retry for:
          1. API failure / empty response
          2. Repetitive output
          3. Format parse error (status=2)
          4. Duplicate tool call (dedup)

        Returns (content, status, thinking, payload).
        content=None means LLM is completely unavailable.
        """
        # Track last results for fallback when retries are exhausted
        last_content = None
        last_status = -1
        last_thinking = ""
        last_payload = None
        last_strict_valid = False

        for attempt in range(self.max_retries):
            is_last = (attempt == self.max_retries - 1)
            backoff = min(2 ** attempt + random.uniform(0, 1), 30)

            # 1. API call
            content = self._call_llm_once(
                use_forced_answer_reserve=(step == -1)
            )
            if not content:
                self.log(f"[_generate] API failure (attempt {attempt + 1}/{self.max_retries})")
                if not is_last:
                    time.sleep(backoff)
                    continue
                # All retries exhausted with API failure
                return None, -1, "", None, False

            # 2. Repetition check
            if self._is_repetitive(content):
                self.log(f"[_generate] Repeat detected (attempt {attempt + 1}/{self.max_retries})")
                last_content = content
                if not is_last:
                    time.sleep(backoff)
                    continue
                # Last attempt: use repetitive content anyway
                self.log("[_generate] Repeat detected after all retries, using anyway")

            # 3. Sanitize + parse
            raw_strict_valid = strict_raw_format_valid(content)
            content = self._sanitize_content(content)
            status, thinking, payload = self.parse_llm_response(content, step)
            accepted_strict_valid = raw_strict_valid and status in {0, 1}

            # 4. Parse error → retry
            if status == 2:
                self.parse_error_count += 1
                self.log(f"[_generate] Parse error (attempt {attempt + 1}/{self.max_retries}): {thinking}")
                last_content, last_status, last_thinking, last_payload = content, status, thinking, payload
                last_strict_valid = False
                if not is_last:
                    time.sleep(backoff)
                    continue
                # Last attempt: return parse error for solve() to handle
                return content, status, thinking, payload, False

            # 5. Duplicate tool call → retry
            if status == 1:
                key = self._tool_call_key(payload)
                if key in self._tool_call_history:
                    self.dedup_count += 1
                    self.log(f"[_generate] Duplicate tool call (attempt {attempt + 1}/{self.max_retries}): {payload['name']}")
                    last_content, last_status, last_thinking, last_payload = content, status, thinking, payload
                    last_strict_valid = accepted_strict_valid
                    if not is_last:
                        time.sleep(backoff)
                        continue
                    # Last attempt: execute the duplicate call anyway
                    return content, status, thinking, payload, accepted_strict_valid

            # All checks passed
            return content, status, thinking, payload, accepted_strict_valid

        # Should not reach here, but return last result as fallback
        if last_content is not None:
            return last_content, last_status, last_thinking, last_payload, last_strict_valid
        return None, -1, "", None, False

    # ---- force answer (token limit / max steps) ----

    def _force_answer(self, termination_ok: str, termination_fail: str) -> dict:
        """Force LLM to produce a final answer (used at token/step limits)."""
        content, status, thinking, payload, strict_valid = self._generate(step=-1)
        if content is None:
            return self._make_result(termination_fail)

        self.messages.append({"role": "assistant", "content": content})
        self.turns += 1
        self.assistant_format_validity.append(strict_valid)

        if status == 0:
            self.final_answer_strict_format_valid = strict_valid
            return self._make_result(termination_ok, payload)

        # status != 0: LLM didn't produce <answer>, fall back to raw content.
        return self._make_result(termination_fail)

    # ---- main loop ----

    def solve(self) -> dict:
        # Reserve the final turn for a forced answer so the configured limit is
        # a true upper bound rather than ``max_steps + 1``.
        for step in range(max(0, self.max_steps - 1)):
            # Timeout check
            if time.time() - self.start_time > self.time_limit:
                return self._make_result("timeout")

            # Unified generate with retry
            content, status, thinking, payload, strict_valid = self._generate(step)
            if content is None:
                return self._make_result("llm_error")

            self.messages.append({"role": "assistant", "content": content})
            self.assistant_format_validity.append(strict_valid)

            if status == 0:  # final answer
                self.turns += 1
                self.final_answer_strict_format_valid = strict_valid
                return self._make_result("answered", payload)

            if status == 1:  # tool call
                self.turns += 1
                self.tool_call_count += 1
                self._tool_call_history[self._tool_call_key(payload)] = True
                result = self.fit_tool_response(self.execute_tool(payload))
                self.messages.append({
                    "role": "user",
                    "content": f"<tool_response>\n{result}\n</tool_response>",
                })

            if status == 2:  # parse error (retries exhausted)
                self.turns += 1
                self.log(f"Parse error at step {step}: {thinking}")
                self.messages.append({
                    "role": "user",
                    "content": (
                        f"<tool_response>\nERROR!\n"
                        "Please call a tool with the correct format, or if you are ready to answer, provide the entire final answer in `<answer></answer>` tags.\n</tool_response>"
                    ),
                })

            # Token limit check
            token_count = self.count_tokens()
            if token_count >= self.max_len:
                # Preserve the tool response / parse-recovery message in the
                # auditable trajectory and append a separate limit notice.
                self.messages.append({
                    "role": "user",
                    "content": (
                        "<tool_response>\nYou have reached the maximum context length. "
                        "Based on the information and knowledge you have gathered, please provide the entire final answer in `<answer></answer>` tags now.\n</tool_response>"
                    ),
                })
                return self._force_answer("answered_token_limit", "token_limit_no_answer")

        # max_steps exhausted, force a final answer
        self.messages.append({
            "role": "user",
            "content": (
                "<tool_response>\nYou've reached the maximum number of tool calls. "
                "Based on the information and knowledge you have gathered, please provide the entire final answer in `<answer></answer>` tags now.\n</tool_response>"
            ),
        })
        return self._force_answer("answered_max_steps", "max_steps")


# ---------------------------------------------------------------------------
# vLLM server management
# ---------------------------------------------------------------------------

def start_vllm_tp(args):
    cmd = [
        "vllm", "serve", args.model_path,
        "--served-model-name", "model",
        "--gpu-memory-utilization", str(args.gpu_util),
        "-tp", str(args.tp_size),
        "--max-num-seqs", str(args.batch_size),
        "--max-model-len", str(args.max_model_len),
        "--seed", str(args.seed),
        "--enforce-eager",
        #"--disable-cuda-graph",
        # "--disable-log-stats",
        #"--disable-log-requests",0
        "--port", str(args.base_port),
    ]
    if args.model_revision:
        cmd.extend(["--revision", args.model_revision])
    proc = subprocess.Popen(cmd)
    return [proc], [args.base_port]


def start_vllm_multi(args):
    procs = []
    ports = []
    try:
        for i in range(args.num_gpus):
            port = args.base_port + i
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(i)
            cmd = [
                "vllm", "serve", args.model_path,
                "--served-model-name", "model",
                "--gpu-memory-utilization", str(args.gpu_util),
                "--max-num-seqs", str(args.batch_size),
                "--max-model-len", str(args.max_model_len),
                "--seed", str(args.seed),
                # "--disable-log-stats",
                # "--disable-log-requests",
                "--port", str(port),
            ]
            if args.model_revision:
                cmd.extend(["--revision", args.model_revision])
            proc = subprocess.Popen(cmd, env=env)
            procs.append(proc)
            ports.append(port)
    except Exception:
        stop_vllm(procs)
        raise
    return procs, ports


def _normalized_model_reference(value: str) -> str:
    candidate = os.path.expanduser(value)
    if os.path.exists(candidate):
        return os.path.normcase(os.path.realpath(candidate))
    return value.rstrip("/\\")


def _validate_reused_server(response: requests.Response, args, port: int) -> None:
    if response.status_code != 200:
        raise RuntimeError(
            f"port {port} is occupied but /v1/models returned HTTP {response.status_code}"
        )
    try:
        payload = response.json()
        models = payload["data"]
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"port {port} did not return vLLM model metadata") from exc
    if not isinstance(models, list):
        raise RuntimeError(f"port {port} returned invalid vLLM model metadata")
    model = next(
        (item for item in models if isinstance(item, dict) and item.get("id") == "model"),
        None,
    )
    if model is None:
        raise RuntimeError(f"port {port} does not serve the required model alias 'model'")

    root = model.get("root")
    if not isinstance(root, str) or not root.strip():
        raise RuntimeError(
            f"port {port} does not expose model root; reuse cannot be attributed safely"
        )
    if _normalized_model_reference(root) != _normalized_model_reference(args.model_path):
        raise RuntimeError(
            f"port {port} serves root {root!r}, expected {args.model_path!r}"
        )

    advertised_context = model.get("max_model_len")
    if not isinstance(advertised_context, int) or advertised_context < args.max_model_len:
        raise RuntimeError(
            f"port {port} cannot verify requested context length {args.max_model_len}"
        )
    if args.model_revision:
        advertised_revision = model.get("revision", model.get("model_revision"))
        if advertised_revision != args.model_revision:
            raise RuntimeError(
                f"port {port} cannot verify model revision {args.model_revision!r}"
            )


def _port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _probe_server(port: int) -> requests.Response | None:
    if not _port_is_open(port):
        return None
    try:
        return requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=3)
    except requests.RequestException as exc:
        # The TCP port is open: resets/timeouts are occupied, not "free".
        raise RuntimeError(f"port {port} is occupied or unhealthy: {exc}") from exc


def start_vllm(args) -> tuple[list, list[int], bool]:
    """Start vLLM or explicitly reuse a metadata-verified deployment."""
    if args.deploy_mode == "tp":
        expected_ports = [args.base_port]
    else:
        expected_ports = [args.base_port + i for i in range(args.num_gpus)]

    responses = {port: _probe_server(port) for port in expected_ports}
    occupied = [port for port, response in responses.items() if response is not None]
    if occupied:
        if not args.reuse_server:
            raise RuntimeError(
                f"refusing to reuse occupied vLLM port(s) {occupied}; "
                "stop them or pass --reuse_server for strict metadata validation"
            )
        if len(occupied) != len(expected_ports):
            raise RuntimeError(
                f"cannot reuse a partial deployment: occupied={occupied}, expected={expected_ports}"
            )
        for port, response in responses.items():
            assert response is not None
            _validate_reused_server(response, args, port)
        print(f"Reusing verified vLLM servers on ports {expected_ports}")
        return [], expected_ports, True

    print(f"Starting vLLM servers (mode={args.deploy_mode})...")
    procs = []
    try:
        if args.deploy_mode == "tp":
            procs, ports = start_vllm_tp(args)
        else:
            procs, ports = start_vllm_multi(args)
        wait_for_servers(
            ports, procs=procs, timeout=args.server_start_timeout
        )
        return procs, ports, False
    except Exception:
        stop_vllm(procs)
        raise


def wait_for_servers(ports: list[int], *, procs: list, timeout: int = 900):
    start = time.time()
    ready = set()
    while time.time() - start < timeout:
        for proc in procs:
            return_code = proc.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"vLLM process {proc.pid} exited before readiness "
                    f"with code {return_code}"
                )
        for port in ports:
            if port in ready:
                continue
            try:
                response = requests.get(
                    f"http://127.0.0.1:{port}/v1/models", timeout=3
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload.get("data"), list):
                    raise ValueError("missing model data")
                print(f"vLLM server on port {port} is ready.")
                ready.add(port)
            except Exception:
                pass
        if len(ready) == len(ports):
            for proc in procs:
                return_code = proc.poll()
                if return_code is not None:
                    raise RuntimeError(
                        f"vLLM process {proc.pid} exited at readiness "
                        f"with code {return_code}"
                    )
            print("All vLLM servers are ready.")
            return
        time.sleep(5)
    raise RuntimeError(
        f"vLLM servers failed to start within {timeout}s. "
        f"Ready: {ready}, Expected: {set(ports)}"
    )


def stop_vllm(procs: list):
    if not procs:
        return
    for proc in procs:
        try:
            proc.terminate()
        except Exception:
            pass
    for proc in procs:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
    print("All vLLM processes stopped.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_openai_client(port: int) -> openai.Client:
    return openai.Client(
        api_key="123",
        base_url=f"http://127.0.0.1:{port}/v1",
        max_retries=0,
    )


# def build_tool_client() -> MessageClient:
#     return MessageClient(isconsumer=True)


def append_jsonl_safely(output_path: str, lock_path: str, data: dict):
    with filelock.FileLock(lock_path):
        with open(output_path, "a+", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


def prepare_evaluation_run(args):
    """Validate/load the cohort and resolve resume state before renting GPUs."""

    output_path = args.output_file
    lock_path = output_path + ".lock"
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    base_items = (
        load_evaluation_items(args.input_file, limit=args.limit)
        if args.input_file
        else [ad_hoc_item(args.question)]
    )
    model_evidence_name = args.model_name or os.path.basename(
        args.model_path.rstrip("/\\")
    )
    run_fingerprint = build_run_fingerprint(
        {
            "model_path": args.model_path,
            "model": model_evidence_name,
            "seed": args.seed,
            "sampling_seed_strategy": f"base+sample_index*{SAMPLE_SEED_STRIDE}+llm_call_index",
            "samples_per_question": args.samples_per_question,
            "max_steps": args.max_steps,
            "max_model_len": args.max_model_len,
            "solver_max_len": args.solver_max_len,
            "max_tokens_per_turn": args.max_tokens_per_turn,
            "time_limit": args.time_limit,
            "max_retries": args.max_retries,
            "server_start_timeout": args.server_start_timeout,
            "model_revision": args.model_revision,
            "checkpoint_revision": args.checkpoint_revision,
            "checkpoint_identity": args.checkpoint_identity,
            "tool_cache_mode": args.tool_cache_mode,
            "tool_cache_path": os.environ.get("TOOL_CACHE_PATH", ""),
            "tool_cache_sha256": args.tool_cache_sha256,
            "tool_timeout": args.tool_timeout,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "system_prompt": SYSTEM_PROMPT,
        },
        base_items,
    )
    requested_items = expand_samples(base_items, args.samples_per_question)
    already_done = completed_keys(
        output_path, seed=args.seed, run_fingerprint=run_fingerprint
    )
    items_to_process = [
        item for item in requested_items if item.key not in already_done
    ]
    return (
        output_path,
        lock_path,
        model_evidence_name,
        run_fingerprint,
        requested_items,
        items_to_process,
    )


def verify_immutable_artifacts(args) -> None:
    if args.tool_cache_sha256:
        current_cache_sha = sha256_file(os.environ["TOOL_CACHE_PATH"])
        if current_cache_sha != args.tool_cache_sha256:
            raise RuntimeError("replay tool cache changed during evaluation")
    if args.checkpoint_sha256:
        current_checkpoint_sha = sha256_artifact(args.model_path)
        if current_checkpoint_sha != args.checkpoint_sha256:
            raise RuntimeError("checkpoint artifact changed during evaluation")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="vLLM Batch Inference Pipeline (v2)")

    # Deployment
    parser.add_argument("--deploy_mode", type=str, default="tp",
                        choices=["tp", "multi"],
                        help="tp: single server with tensor parallelism; "
                             "multi: one server per GPU")
    parser.add_argument("--tp_size", type=int, default=2,
                        help="Tensor parallel size (tp mode only)")
    parser.add_argument("--num_gpus", type=int, default=8,
                        help="Number of GPUs (multi mode only)")
    parser.add_argument("--base_port", type=int, default=6000,
                        help="Base port for vLLM servers")
    parser.add_argument(
        "--reuse_server",
        action="store_true",
        help="reuse occupied ports only after strict /v1/models validation",
    )
    parser.add_argument(
        "--server_start_timeout",
        type=int,
        default=900,
        help="maximum seconds to wait for vLLM readiness",
    )

    # Model
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_revision", type=str, default=None)
    parser.add_argument(
        "--checkpoint_revision",
        type=str,
        default=os.environ.get("CHECKPOINT_REVISION") or None,
        help="immutable checkpoint commit/version; required for remote cohort evaluation",
    )
    parser.add_argument("--gpu_util", type=float, default=0.95)
    parser.add_argument("--max_model_len", type=int, default=261000)

    # Data
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--question", type=str)
    input_group.add_argument("--input_file", type=str,
                             help="eval cohort parquet/jsonl; all rows share one vLLM deployment")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples_per_question", type=int, default=1,
                        help="use 4 only for the fixed Pass@4 subset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default=None,
                        help="name stored in evidence; defaults to model_path basename")
    parser.add_argument("--cost_cny_per_sample", type=float, default=0.0,
                        help="measured amortized API/tool cost written to raw evidence")
    parser.add_argument(
        "--tool_cache_mode",
        choices=["online", "record", "replay"],
        default=os.environ.get("TOOL_CACHE_MODE", "online").strip().casefold() or "online",
    )
    parser.add_argument(
        "--tool_cache_path",
        default=os.environ.get("TOOL_CACHE_PATH", ""),
        help="required for record/replay; replay never falls back online",
    )

    # Inference
    parser.add_argument("--batch_size", type=int, default=32,
                        help="vLLM max-num-seqs")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Thread pool concurrency")
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Max steps per problem")
    parser.add_argument("--solver_max_len", type=int, default=240000,
                        help="Solver token limit")
    parser.add_argument("--time_limit", type=int, default=9000,
                        help="Per-problem time limit in seconds (default 150min)")
    parser.add_argument(
        "--max_tokens_per_turn",
        type=int,
        default=1024,
        help="generation cap per assistant turn; dynamically reduced near context limit",
    )
    parser.add_argument(
        "--tool_timeout",
        type=float,
        default=float(os.environ.get("TOOL_TIMEOUT_SECONDS", "150")),
        help="end-to-end timeout for one search/visit invocation in seconds",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--verbose", action="store_true")

    # Robustness (unified)
    parser.add_argument("--max_retries", type=int, default=5,
                        help="Max retries for API errors, repetition, parse errors, "
                             "and duplicate tool calls (0 = no retry)")

    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    positive_integer_fields = (
        "tp_size",
        "num_gpus",
        "batch_size",
        "num_workers",
        "max_steps",
        "max_model_len",
        "solver_max_len",
        "time_limit",
        "samples_per_question",
        "max_tokens_per_turn",
        "server_start_timeout",
    )
    for name in positive_integer_fields:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name} must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if not math.isfinite(args.cost_cny_per_sample) or args.cost_cny_per_sample < 0:
        raise ValueError("--cost_cny_per_sample must be non-negative")
    if not math.isfinite(args.tool_timeout) or args.tool_timeout <= 0:
        raise ValueError("--tool_timeout must be positive")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if not math.isfinite(args.top_p) or not 0 < args.top_p <= 1:
        raise ValueError("--top_p must be in (0, 1]")
    if not math.isfinite(args.gpu_util) or not 0 < args.gpu_util <= 1:
        raise ValueError("--gpu_util must be in (0, 1]")
    if args.max_retries < 0:
        raise ValueError("--max_retries must be non-negative")
    if args.deploy_mode == "tp" and args.tp_size > args.num_gpus:
        raise ValueError("--tp_size cannot exceed --num_gpus in tp mode")
    if not 1 <= args.base_port <= 65535:
        raise ValueError("--base_port must be between 1 and 65535")
    if args.solver_max_len + args.max_tokens_per_turn > args.max_model_len:
        raise ValueError(
            "--solver_max_len plus --max_tokens_per_turn cannot exceed --max_model_len"
        )
    if args.input_file and not os.path.isfile(args.input_file):
        raise FileNotFoundError(f"evaluation input does not exist: {args.input_file}")

    os.environ["TOOL_CACHE_MODE"] = args.tool_cache_mode
    os.environ["TOOL_TIMEOUT_SECONDS"] = str(args.tool_timeout)
    if args.tool_cache_mode in {"record", "replay"}:
        if not args.tool_cache_path.strip():
            raise ValueError(
                "--tool_cache_path is required in record/replay mode"
            )
        cache_path = os.path.abspath(os.path.expanduser(args.tool_cache_path))
        if args.tool_cache_mode == "replay" and not os.path.isfile(cache_path):
            raise FileNotFoundError(f"replay tool cache does not exist: {cache_path}")
        os.environ["TOOL_CACHE_PATH"] = cache_path
    else:
        os.environ.pop("TOOL_CACHE_PATH", None)
    args.tool_cache_sha256 = (
        sha256_file(os.environ["TOOL_CACHE_PATH"])
        if args.tool_cache_mode == "replay"
        else None
    )

    checkpoint_path = os.path.abspath(os.path.expanduser(args.model_path))
    args.checkpoint_sha256 = None
    if os.path.exists(checkpoint_path):
        if args.input_file:
            print("Hashing local checkpoint for immutable evaluation evidence...")
            args.checkpoint_sha256 = sha256_artifact(checkpoint_path)
        args.checkpoint_identity = (
            f"sha256:{args.checkpoint_sha256}"
            if args.checkpoint_sha256
            else f"local-path:{os.path.realpath(checkpoint_path)}"
        )
    else:
        if (
            args.checkpoint_revision
            and args.model_revision
            and args.checkpoint_revision != args.model_revision
        ):
            raise ValueError(
                "remote --checkpoint_revision and --model_revision must match"
            )
        revision = args.checkpoint_revision or args.model_revision
        if args.input_file and not revision:
            raise ValueError(
                "remote cohort evaluation requires --checkpoint_revision or --model_revision"
            )
        if revision:
            # The evidence revision must be the revision vLLM/tokenizer load.
            args.model_revision = revision
            args.checkpoint_revision = revision
        args.checkpoint_identity = f"revision:{revision or 'unversioned'}"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    validate_args(args)

    (
        output_path,
        lock_path,
        model_evidence_name,
        run_fingerprint,
        requested_items,
        items_to_process,
    ) = prepare_evaluation_run(args)
    resumed_count = len(requested_items) - len(items_to_process)
    print(f"Pending samples: {len(items_to_process)} (resumed: {resumed_count})")
    if not items_to_process:
        verify_immutable_artifacts(args)
        print(f"Nothing to run. Existing evidence: {output_path}")
        raise SystemExit(0)

    vllm_procs, ports, server_reused = start_vllm(args)

    try:
        print(f"Loading tokenizer from: {args.model_path}")
        tokenizer_kwargs = (
            {"revision": args.model_revision} if args.model_revision else {}
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, **tokenizer_kwargs)

        print(f"Workers: {args.num_workers}, Ports: {ports}")

        stats_lock = Lock()
        stats = {"success": 0, "failed": 0, "fatal_replay_errors": 0}
        progress = tqdm(total=len(items_to_process), desc="Processing")

        def solve_problem(item, worker_index: int):
            port = ports[worker_index % len(ports)]
            solver = None
            try:
                local_openai_client = build_openai_client(port)
                # local_tool_client = build_tool_client()

                solver = ResearchProblemSolver(
                    question=item.question,
                    tokenizer=tokenizer,
                    max_len=args.solver_max_len,
                    max_steps=args.max_steps,
                    openai_client=local_openai_client,
                    # tool_client=local_tool_client,
                    model_name="model",
                    time_limit=args.time_limit,
                    verbose=args.verbose,
                    max_retries=args.max_retries,
                    seed=args.seed + item.sample_index * SAMPLE_SEED_STRIDE,
                    tool_timeout=args.tool_timeout,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    model_context_len=args.max_model_len,
                    max_tokens_per_turn=args.max_tokens_per_turn,
                    tool_cache_mode=args.tool_cache_mode,
                )

                result = solver.solve()

                record = {
                    "sample_id": item.sample_id,
                    "sample_index": item.sample_index,
                    "data_source": item.data_source,
                    "question": item.question,
                    "references": list(item.references),
                    "model": model_evidence_name,
                    "seed": args.seed,
                    "sampling_seed": args.seed + item.sample_index * SAMPLE_SEED_STRIDE,
                    "sampling_seed_strategy": f"base+sample_index*{SAMPLE_SEED_STRIDE}+llm_call_index",
                    "run_fingerprint": run_fingerprint,
                    "messages": solver.messages,
                    "prediction": result["prediction"],
                    "termination": result["termination"],
                    "turns": result["turns"],
                    "parse_error_count": result["parse_error_count"],
                    "tool_call_count": result["tool_call_count"],
                    "browse_call_count": result["browse_call_count"],
                    "tool_error_count": result["tool_error_count"],
                    "timeout_count": result["timeout_count"],
                    "cache_hit_count": result["cache_hit_count"],
                    "cache_miss_count": result["cache_miss_count"],
                    "cached_tool_error_count": result["cached_tool_error_count"],
                    "tool_response_truncation_count": result[
                        "tool_response_truncation_count"
                    ],
                    "tool_response_tokens_removed": result[
                        "tool_response_tokens_removed"
                    ],
                    "dedup_count": result["dedup_count"],
                    "assistant_output_count": result["assistant_output_count"],
                    "strict_format_violation_count": result["strict_format_violation_count"],
                    "all_assistant_outputs_strict_format_valid": result[
                        "all_assistant_outputs_strict_format_valid"
                    ],
                    "final_answer_strict_format_valid": result[
                        "final_answer_strict_format_valid"
                    ],
                    "final_token_count": result["final_token_count"],
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "transcript_input_tokens": result["transcript_input_tokens"],
                    "transcript_output_tokens": result["transcript_output_tokens"],
                    "llm_call_count": result["llm_call_count"],
                    "llm_failed_call_count": result["llm_failed_call_count"],
                    "llm_usage_missing_count": result["llm_usage_missing_count"],
                    "llm_sampling_seeds": result["llm_sampling_seeds"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "max_turns": args.max_steps,
                    "context_length": args.max_model_len,
                    "cost_cny": args.cost_cny_per_sample,
                    "tool_timeout_seconds": args.tool_timeout,
                    "tool_cache_mode": args.tool_cache_mode,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens_per_turn": args.max_tokens_per_turn,
                    "model_revision": args.model_revision,
                    "checkpoint_revision": args.checkpoint_revision,
                    "checkpoint_sha256": args.checkpoint_sha256,
                    "tool_cache_sha256": args.tool_cache_sha256,
                    "server_reused": server_reused,
                    "server_start_timeout": args.server_start_timeout,
                }
                append_jsonl_safely(output_path, lock_path, record)

                with stats_lock:
                    if result["termination"].startswith("answered"):
                        stats["success"] += 1
                    else:
                        stats["failed"] += 1

            except Exception as e:
                tb = traceback.format_exc()
                fatal_replay_error = isinstance(e, ReplayToolEvidenceError)
                record = {
                    "sample_id": item.sample_id,
                    "sample_index": item.sample_index,
                    "data_source": item.data_source,
                    "question": item.question,
                    "references": list(item.references),
                    "model": model_evidence_name,
                    "seed": args.seed,
                    "sampling_seed": args.seed + item.sample_index * SAMPLE_SEED_STRIDE,
                    "sampling_seed_strategy": f"base+sample_index*{SAMPLE_SEED_STRIDE}+llm_call_index",
                    "run_fingerprint": run_fingerprint,
                    "messages": solver.messages if solver else None,
                    "prediction": None,
                    "termination": (
                        "replay_cache_miss"
                        if fatal_replay_error and e.cache_miss
                        else "replay_cached_error"
                        if fatal_replay_error
                        else "exception"
                    ),
                    "turns": solver.turns if solver else 0,
                    "parse_error_count": solver.parse_error_count if solver else 0,
                    "tool_call_count": solver.tool_call_count if solver else 0,
                    "browse_call_count": solver.browse_call_count if solver else 0,
                    "tool_error_count": solver.tool_error_count if solver else 0,
                    "timeout_count": solver.timeout_count if solver else 0,
                    "cache_hit_count": solver.cache_hit_count if solver else 0,
                    "cache_miss_count": solver.cache_miss_count if solver else 0,
                    "cached_tool_error_count": solver.cached_tool_error_count if solver else 0,
                    "tool_response_truncation_count": (
                        solver.tool_response_truncation_count if solver else 0
                    ),
                    "tool_response_tokens_removed": (
                        solver.tool_response_tokens_removed if solver else 0
                    ),
                    "dedup_count": solver.dedup_count if solver else 0,
                    "assistant_output_count": len(solver.assistant_format_validity) if solver else 0,
                    "strict_format_violation_count": (
                        sum(not valid for valid in solver.assistant_format_validity)
                        if solver else 0
                    ),
                    "all_assistant_outputs_strict_format_valid": (
                        bool(solver.assistant_format_validity)
                        and all(solver.assistant_format_validity)
                        if solver else False
                    ),
                    "final_answer_strict_format_valid": (
                        solver.final_answer_strict_format_valid if solver else False
                    ),
                    "final_token_count": 0,
                    "input_tokens": solver.inference_prompt_tokens if solver else 0,
                    "output_tokens": solver.inference_completion_tokens if solver else 0,
                    "transcript_input_tokens": 0,
                    "transcript_output_tokens": 0,
                    "llm_call_count": solver.llm_call_count if solver else 0,
                    "llm_failed_call_count": solver.llm_failed_call_count if solver else 0,
                    "llm_usage_missing_count": solver.llm_usage_missing_count if solver else 0,
                    "llm_sampling_seeds": list(solver.llm_sampling_seeds) if solver else [],
                    "elapsed_seconds": round(time.time() - solver.start_time, 2) if solver else 0,
                    "max_turns": args.max_steps,
                    "context_length": args.max_model_len,
                    "cost_cny": args.cost_cny_per_sample,
                    "tool_timeout_seconds": args.tool_timeout,
                    "tool_cache_mode": args.tool_cache_mode,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens_per_turn": args.max_tokens_per_turn,
                    "model_revision": args.model_revision,
                    "checkpoint_revision": args.checkpoint_revision,
                    "checkpoint_sha256": args.checkpoint_sha256,
                    "tool_cache_sha256": args.tool_cache_sha256,
                    "server_reused": server_reused,
                    "server_start_timeout": args.server_start_timeout,
                    "error": repr(e),
                    "traceback": tb,
                }
                append_jsonl_safely(output_path, lock_path, record)
                print(f"\n[ERROR] {item.question[:80]}...: {repr(e)}")
                if args.verbose:
                    print(tb)
                with stats_lock:
                    stats["failed"] += 1
                    if fatal_replay_error:
                        stats["fatal_replay_errors"] += 1

            finally:
                progress.update(1)

        if items_to_process:
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = [
                    executor.submit(solve_problem, item, i)
                    for i, item in enumerate(items_to_process)
                ]
                for future in as_completed(futures):
                    future.result()

        progress.close()

        print(
            f"\nDone. Success: {stats['success']}, Failed: {stats['failed']}, "
            f"Fatal replay errors: {stats['fatal_replay_errors']}"
        )
        print(f"Output: {output_path}")
        verify_immutable_artifacts(args)
        if stats["fatal_replay_errors"]:
            raise RuntimeError(
                "replay cache miss/recorded error detected; refusing successful evaluation exit"
            )

    finally:
        stop_vllm(vllm_procs)
