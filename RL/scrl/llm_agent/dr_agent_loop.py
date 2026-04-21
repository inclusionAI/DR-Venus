"""
DeepResearcher async agent loop for verl's AgentLoopBase framework.

Adapted from scrl/llm_agent/generation.py (sync version). Core logic
(system prompt, response parsing, tool calling, context management) is
kept identical; only the execution model is changed from synchronous
batch processing to per-sample async processing.
"""

import asyncio
import copy
import itertools
import json
import json5
import os
import random
import re
import threading
import time
from typing import Any, List, Tuple
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)
from verl.utils.profiler import simple_timer

# ---------- Constants (mirrored from generation.py) ----------

ALLOWED_ARGS = {
    "search": {"query"},
    "visit": {"url", "goal"},
    "PythonInterpreter": {"code"},
}

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


class _Flag:
    END = 0
    CALL = 1
    ERROR = 2


def _filter_args(tool_name: str, args: dict) -> dict:
    if not isinstance(args, dict):
        return {}
    if tool_name not in ALLOWED_ARGS:
        return args
    allowed = ALLOWED_ARGS[tool_name]
    return {k: v for k, v in args.items() if k in allowed}


def _is_repetitive(content: str, tail_len: int = 50, threshold: int = 5) -> bool:
    """Check if content ends with a repeating pattern (degenerate generation)."""
    if not content or len(content) < tail_len:
        return False
    tail = content[-tail_len:]
    return content.count(tail) > threshold


def _tool_call_key(tool_call: dict) -> str:
    """Hashable key for tool call deduplication."""
    return json.dumps(
        {"name": tool_call["name"], "arguments": tool_call["arguments"]},
        sort_keys=True, ensure_ascii=False,
    )


def _parse_single_response(content: str, think: bool = False) -> Tuple[int, str, Any]:
    """Parse a single LLM response string. Returns (flag, thinking, payload).
    Mirrors the per-sample logic in LLMGenerationManager.parse_response.
    """
    if think:
        content = "<think>" + content

    if "<tool_response>" in content:
        content = content[: content.find("<tool_response>")]

    if "<think>" in content and "</think>" not in content:
        content = content + "\n</think>"
    if "<think>" not in content and "</think>" in content:
        content = "<think>\n" + content

    think_match = THINK_RE.search(content)
    if not think_match:
        return (_Flag.ERROR, "Missing <think></think>", "")

    thinking = think_match.group(1)

    answer_match = ANSWER_RE.search(content)
    tool_match = TOOL_RE.search(content)

    if answer_match and not tool_match:
        return (_Flag.END, thinking, answer_match.group(1))

    if tool_match and not answer_match:
        raw_tool_str = tool_match.group(1).strip()
        try:
            if (
                "pythoninterpreter" in raw_tool_str.lower()
                and "<code>" in raw_tool_str
                and "</code>" in raw_tool_str
            ):
                code_match = re.search(r"<code>(.*?)</code>", raw_tool_str, re.DOTALL)
                if not code_match:
                    return (_Flag.ERROR, f"PythonInterpreter no <code> block, raw={raw_tool_str}", "")
                tool_call = {
                    "name": "PythonInterpreter",
                    "arguments": {"code": code_match.group(1).strip()},
                }
            else:
                try:
                    tool_call = json.loads(raw_tool_str)
                except (json.JSONDecodeError, ValueError):
                    tool_call = json5.loads(raw_tool_str)
                if not isinstance(tool_call, dict):
                    return (_Flag.ERROR, f"Tool call not dict: {type(tool_call)}", "")
                if "name" not in tool_call or "arguments" not in tool_call:
                    return (_Flag.ERROR, f"Tool call missing fields, raw={raw_tool_str}", "")
                tool_call["arguments"] = _filter_args(tool_call["name"], tool_call.get("arguments", {}))
            return (_Flag.CALL, thinking, tool_call)
        except Exception as e:
            return (_Flag.ERROR, f"Tool call parse error: {repr(e)}, raw={raw_tool_str}", "")

    return (_Flag.ERROR, "Ambiguous or incomplete output", "")


class _LockedProc:
    """Thread-safe wrapper for tokenizer/processor. Rust-based fast tokenizers
    are not thread-safe when called from multiple threads concurrently."""

    __slots__ = ("_proc", "_lock")

    def __init__(self, proc, lock):
        self._proc = proc
        self._lock = lock

    def __call__(self, *args, **kwargs):
        with self._lock:
            return self._proc(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        with self._lock:
            return self._proc.apply_chat_template(*args, **kwargs)

    def decode(self, *args, **kwargs):
        with self._lock:
            return self._proc.decode(*args, **kwargs)

    def encode(self, *args, **kwargs):
        with self._lock:
            return self._proc.encode(*args, **kwargs)


# ---------- Agent Loop ----------


@register("dr_agent")
class DRAgentLoop(AgentLoopBase):
    """DeepResearcher multi-turn async agent loop."""

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return

        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_turns_base = config.max_turns
        cls.max_len = config.data.max_model_len
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.codeact_env_disabled = config.codeact_env_disabled

        from tool_server.tool_prompt import SYSTEM_PROMPT
        cls.system_prompt = SYSTEM_PROMPT

        _base_proc = processor if processor is not None else tokenizer
        _pool_size = max(1, min(16, os.cpu_count() or 8))
        try:
            cls._tok_pool = [
                (copy.deepcopy(_base_proc), threading.Lock())
                for _ in range(_pool_size)
            ]
        except Exception:
            cls._tok_pool = [(_base_proc, threading.Lock())]
        cls._tok_pool_idx = itertools.count()

        cls._tool_timeout = float(config.get("tool_timeout", 120))

        cls._trace_save_interval = config.get("trace_save_interval", 0)
        cls._trace_max_samples = config.get("trace_max_samples", 0)
        cls._trace_save_dir = os.path.join(
            config.trainer.get("default_local_dir", "/tmp"),
            "rollout_traces",
        )
        import shutil
        shutil.rmtree(cls._trace_save_dir, ignore_errors=True)

        # ── Rollout-time IG scoring config (read from env vars) ──
        cls._rollout_ig = os.environ.get("IGPO_ROLLOUT_IG", "0") == "1"
        if cls._rollout_ig:
            cls._ig_compute_freq = max(1, int(os.environ.get("IGPO_IG_COMPUTE_FREQ", "1")))
            _tf = os.environ.get("IGPO_IG_TOOL_FILTER", "")
            cls._ig_tool_filter = set(
                t.strip().lower() for t in _tf.split(",") if t.strip()
            ) if _tf.strip() else set()
            cls._ig_info_gain_type = os.environ.get("IGPO_INFO_GAIN_TYPE", "log_prob_diff")
            cls._ig_gt_prefix = "Now there's enough information to answer\n</think>\n<answer>\n"
            cls._ig_gt_suffix = "\n</answer><|im_end|>"
            import sys
            print(f"[DRAgentLoop] Rollout-IG ENABLED: freq={cls._ig_compute_freq}, "
                  f"filter={cls._ig_tool_filter or 'none'}, type={cls._ig_info_gain_type}",
                  file=sys.stderr)

        # ── Rollout robustness config (retry / dedup / repetition detection) ──
        cls._robust_enabled = os.environ.get("ROLLOUT_ROBUST", "0") == "1"
        if cls._robust_enabled:
            cls._robust_max_retries = max(0, int(os.environ.get("ROLLOUT_ROBUST_MAX_RETRIES", "3")))
            cls._robust_dedup = os.environ.get("ROLLOUT_ROBUST_DEDUP", "1") == "1"
            cls._robust_repetition = os.environ.get("ROLLOUT_ROBUST_REPETITION", "1") == "1"
        else:
            cls._robust_max_retries = 0
            cls._robust_dedup = False
            cls._robust_repetition = False

        cls._class_initialized = True

        import sys
        print(f"[DRAgentLoop] init_class done: max_turns={cls.max_turns_base}, "
              f"max_len={cls.max_len}, response_length={cls.response_length}, "
              f"codeact_disabled={cls.codeact_env_disabled}, "
              f"tok_pool_size={len(cls._tok_pool)}",
              file=sys.stderr)
        if cls._robust_enabled:
            print(f"[DRAgentLoop] Robustness ENABLED: max_retries={cls._robust_max_retries}, "
                  f"dedup={cls._robust_dedup}, repetition={cls._robust_repetition}",
                  file=sys.stderr)

    def _save_trace_snapshot(self, request_id, turn, num_turns, messages,
                                global_step="unknown", finished=False,
                                sample_idx=0):
        """Write a single-sample trace snapshot to disk."""
        save_dir = os.path.join(self._trace_save_dir, f"step_{global_step}")
        os.makedirs(save_dir, exist_ok=True)
        trace_path = os.path.join(save_dir, f"{sample_idx:04d}.json")
        record = {
            "request_id": request_id,
            "sample_idx": sample_idx,
            "turn": turn,
            "num_turns": num_turns,
            "messages": messages,
        }
        if finished:
            record["finished"] = True
        with open(trace_path, "w") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    # ── Rollout-time IG helpers ──

    def _prepare_gt_tokens(self, gt_text: str, _proc):
        """Tokenize ground-truth into pseudo-response tokens and locate the GT span.

        Returns (gt_full_token_ids, gs, ge) where gs/ge are the start/end
        indices of the actual GT tokens within gt_full_token_ids, or
        ``([], 0, 0)`` if GT is empty.
        """
        import math
        if not gt_text:
            return [], 0, 0
        full_text = f"{self._ig_gt_prefix}{gt_text}{self._ig_gt_suffix}"
        encoding = _proc(full_text, return_tensors="pt", padding=False,
                         return_offsets_mapping=True)
        token_ids = encoding["input_ids"].squeeze(0).tolist()
        offset_mapping = encoding["offset_mapping"].squeeze(0).tolist()
        if len(token_ids) == 0:
            return [], 0, 0
        gt_char_start = len(self._ig_gt_prefix)
        gt_char_end = gt_char_start + len(gt_text)
        gs = ge = None
        for ti, (cs, ce) in enumerate(offset_mapping):
            if gs is None and ce > gt_char_start:
                gs = ti
            if cs < gt_char_end and ce > 0:
                ge = ti + 1
        if gs is None:
            gs = len(token_ids)
        if ge is None:
            ge = len(token_ids)
        return token_ids, gs, ge

    async def _score_ig_at_turn(self, request_id, current_ids, gt_full_tokens,
                                gs, ge):
        """Send a prompt-logprobs scoring request and return mean log-prob
        of the GT tokens, or ``None`` on failure.

        *current_ids* is the conversation context at the current turn boundary.
        The scoring prompt is ``current_ids + gt_full_tokens``.  With vLLM
        prefix-caching, *current_ids* should be a cache hit (it was just used
        for generation), so only the ~15 GT tokens require new computation.
        """
        import math
        if gs >= ge or not gt_full_tokens:
            return None
        scoring_ids = current_ids + gt_full_tokens
        try:
            prompt_logprobs = await self.server_manager.compute_prompt_logprobs(
                request_id=request_id,
                prompt_ids=scoring_ids,
            )
        except Exception:
            return None
        if prompt_logprobs is None:
            return None
        offset = len(current_ids)
        lps = []
        for k in range(gs, ge):
            pos = offset + k
            if pos >= len(prompt_logprobs) or prompt_logprobs[pos] is None:
                continue
            tok_id = gt_full_tokens[k]
            entry = prompt_logprobs[pos]
            if tok_id in entry:
                lps.append(entry[tok_id].logprob)
        if not lps:
            return None
        mean_lp = sum(lps) / len(lps)
        if math.isnan(mean_lp) or math.isinf(mean_lp):
            return None
        return mean_lp

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        request_id = uuid4().hex
        metrics = {}
        _pool_entry = self._tok_pool[next(self._tok_pool_idx) % len(self._tok_pool)]
        _proc = _LockedProc(_pool_entry[0], _pool_entry[1])

        raw_prompt = kwargs.get("raw_prompt")
        if raw_prompt is None:
            raise ValueError("DRAgentLoop requires 'raw_prompt' in dataset. "
                             "Set data.return_raw_chat=true in config.")

        messages = list(raw_prompt)
        if not any(m.get("role") == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        prompt_ids: list[int] = []
        response_ids: list[int] = []
        response_mask: list[int] = []
        turn_boundaries: list[dict] = []

        max_turns = max(1, int(random.random() * 10 - 5) + self.max_turns_base)
        t_gen_total = 0.0
        t_tool_total = 0.0
        t_prefill_total = 0.0
        t_decode_total = 0.0
        _tool_timeout_count = 0
        _tool_call_count = 0
        num_turns = 0
        t_sample_start = time.time()
        _sid = request_id[:8]
        self.__class__._live_progress[_sid] = 0
        _prev_ctx_len = 0
        _max_prefill = (0.0, 0, 0)   # (time, ctx_len, delta_ctx)
        _max_decode = (0.0, 0)        # (time, gen_tokens)
        _max_tool = 0.0
        _was_cutoff = False

        # ── Robustness state ──
        _robust_repetition_count = 0
        _robust_parse_retry_count = 0
        _robust_dedup_count = 0
        _tool_call_history: dict[str, bool] = {}

        _sample_idx = kwargs.get("_sample_idx", 0)
        _save_trace = (
            self._trace_save_interval > 0
            and (self._trace_max_samples <= 0
                 or _sample_idx < self._trace_max_samples)
        )

        # ── Rollout-time IG state ──
        _ig_enabled = False
        _ig_step_rewards: dict[int, float] = {}  # step → ig_value
        _ig_prev_value = None
        _gt_full_tokens: list[int] = []
        _gt_gs = _gt_ge = 0
        if self._rollout_ig:
            import math as _ig_math
            import json as _ig_json
            _rm = kwargs.get("reward_model")
            if isinstance(_rm, dict):
                _gt_text = _rm.get("ground_truth", "")
            else:
                _gt_text = ""
            if "<|answer_split|>" in _gt_text:
                _gt_text = _gt_text.split("<|answer_split|>")[0]
            _gt_text = _gt_text.strip()
            if _gt_text.startswith("["):
                try:
                    _parsed = _ig_json.loads(_gt_text)
                    if isinstance(_parsed, list):
                        label = 'true'
                        for entry in _parsed:
                            if entry.get('label', '').lower() == 'false':
                                label = 'false'
                                break
                        _gt_text = label
                except Exception:
                    pass
            if _gt_text:
                _gt_full_tokens, _gt_gs, _gt_ge = await self.loop.run_in_executor(
                    None,
                    lambda: self._prepare_gt_tokens(_gt_text, _proc),
                )
                _ig_enabled = _gt_gs < _gt_ge and len(_gt_full_tokens) > 0

        for step in range(max_turns):
            # ── Tokenize current messages ──
            chat_text = await self.loop.run_in_executor(
                None,
                lambda: _proc.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                ),
            )
            think = chat_text.rstrip().endswith("<think>")

            model_inputs = await self.loop.run_in_executor(
                None,
                lambda: _proc(
                    chat_text, return_tensors="pt", padding=False,
                    truncation=True, max_length=self.max_len,
                ),
            )
            current_ids = model_inputs["input_ids"].squeeze(0).tolist()

            if step == 0:
                prompt_ids = list(current_ids)

            # ── Context length check ──
            is_last = False
            if step > 0 and len(current_ids) > self.max_len - self.response_length:
                messages[-1] = {
                    "role": "user",
                    "content": (
                        "You have reached the maximum context length. "
                        "Provide your final answer in <answer></answer> tags now."
                    ),
                }
                is_last = True
                chat_text = await self.loop.run_in_executor(
                    None,
                    lambda: _proc.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    ),
                )
                think = chat_text.rstrip().endswith("<think>")
                model_inputs = await self.loop.run_in_executor(
                    None,
                    lambda: _proc(
                        chat_text, return_tensors="pt", padding=False,
                        truncation=True, max_length=self.max_len,
                    ),
                )
                current_ids = model_inputs["input_ids"].squeeze(0).tolist()

            if not is_last and self.__class__._cutoff_triggered:
                is_last = True
                _was_cutoff = True
                _cutoff_hint = (
                    "The rollout budget has been reached. "
                    "Provide your final answer in <answer></answer> tags now."
                )
                if step == 0:
                    messages[-1] = {
                        "role": messages[-1].get("role", "user"),
                        "content": messages[-1].get("content", "") + "\n\n" + _cutoff_hint,
                    }
                else:
                    messages[-1] = {"role": "user", "content": _cutoff_hint}
                chat_text = await self.loop.run_in_executor(
                    None,
                    lambda: _proc.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    ),
                )
                think = chat_text.rstrip().endswith("<think>")
                model_inputs = await self.loop.run_in_executor(
                    None,
                    lambda: _proc(
                        chat_text, return_tensors="pt", padding=False,
                        truncation=True, max_length=self.max_len,
                    ),
                )
                current_ids = model_inputs["input_ids"].squeeze(0).tolist()

            if step == max_turns - 1 and not is_last:
                is_last = True
                if step > 0:
                    messages[-1] = {
                        "role": "user",
                        "content": (
                            "You've reached the maximum number of tool calls. "
                            "Provide your final answer in <answer></answer> tags now."
                        ),
                    }
                    chat_text = await self.loop.run_in_executor(
                        None,
                        lambda: _proc.apply_chat_template(
                            messages, add_generation_prompt=True, tokenize=False
                        ),
                    )
                    think = chat_text.rstrip().endswith("<think>")
                    model_inputs = await self.loop.run_in_executor(
                        None,
                        lambda: _proc(
                            chat_text, return_tensors="pt", padding=False,
                            truncation=True, max_length=self.max_len,
                        ),
                    )
                    current_ids = model_inputs["input_ids"].squeeze(0).tolist()

            # Record turn boundary AFTER is_last modifications so token_len
            # reflects the actual (possibly shortened) conversation used for
            # generation. Matches sync mode (generation.py L686-697).
            # tool_name is filled after parsing; None = no tool call this turn.
            turn_boundaries.append({
                "step": step,
                "token_len": len(current_ids),
                "msg_count": len(messages),
                "tool_name": None,
            })

            # ── Generate (with optional robustness retry) ──
            gen_sampling_params = dict(sampling_params)
            gen_sampling_params["max_tokens"] = self.response_length

            _ctx_len = len(current_ids)
            _delta_ctx = _ctx_len - _prev_ctx_len
            _lt = self.__class__._live_timing
            _max_gen_attempts = (self._robust_max_retries + 1) if self._robust_enabled else 1

            for _gen_attempt in range(_max_gen_attempts):
                _is_last_attempt = (_gen_attempt == _max_gen_attempts - 1)

                t0 = time.time()
                output = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=current_ids,
                    sampling_params=gen_sampling_params,
                )
                t_gen = time.time() - t0
                t_gen_total += t_gen
                t_prefill_total += output.prefill_time
                t_decode_total += output.decode_time

                gen_token_ids = output.token_ids
                _n_gen_tok = len(gen_token_ids)

                if output.prefill_time > _max_prefill[0]:
                    _max_prefill = (output.prefill_time, _ctx_len, _delta_ctx)
                if output.decode_time > _max_decode[0]:
                    _max_decode = (output.decode_time, _n_gen_tok)

                _lt["prefill"] += output.prefill_time
                _lt["decode"] += output.decode_time
                _lt["wall_gen"] += t_gen
                if output.prefill_time > _lt["max_prefill"][0]:
                    _lt["max_prefill"] = (output.prefill_time, _ctx_len, _delta_ctx)
                if output.decode_time > _lt["max_decode"][0]:
                    _lt["max_decode"] = (output.decode_time, _n_gen_tok)

                # ── Decode and strip terminal tokens ──
                _raw_decoded = await self.loop.run_in_executor(
                    None,
                    lambda: _proc.decode(gen_token_ids, skip_special_tokens=False),
                )
                _raw_decoded = _raw_decoded.replace("<|endoftext|>", "")
                if _raw_decoded.rstrip().endswith("<|im_end|>"):
                    _raw_decoded = _raw_decoded.rstrip()[: -len("<|im_end|>")]

                # ── Repetition check (before tag fixing, matching inference) ──
                if (self._robust_enabled and self._robust_repetition
                        and not _is_last_attempt and _is_repetitive(_raw_decoded)):
                    _robust_repetition_count += 1
                    continue

                # ── Tag fixing ──
                raw_content = ("<think>" if think else "") + _raw_decoded
                if "<tool_response>" in raw_content:
                    raw_content = raw_content[: raw_content.find("<tool_response>")]
                if "<think>" in raw_content and "</think>" not in raw_content:
                    raw_content = raw_content + "\n</think>"
                if "<think>" not in raw_content and "</think>" in raw_content:
                    raw_content = "<think>\n" + raw_content

                # ── Parse response ──
                flag, thinking, payload = _parse_single_response(raw_content, think=False)

                # ── Other robustness checks (retry if not last attempt) ──
                if self._robust_enabled and not _is_last_attempt:
                    if flag == _Flag.ERROR:
                        _robust_parse_retry_count += 1
                        continue
                    if self._robust_dedup and flag == _Flag.CALL:
                        if _tool_call_key(payload) in _tool_call_history:
                            _robust_dedup_count += 1
                            continue

                break  # all checks passed or last attempt

            _prev_ctx_len = _ctx_len
            _lt["turns"] += 1
            num_turns += 1
            self.__class__._live_progress[_sid] = num_turns

            # ── Rollout IG: decide whether to score this turn ──
            _ig_score_this_turn = False
            if _ig_enabled:
                if step == 0:
                    _ig_score_this_turn = True
                elif step % self._ig_compute_freq == 0:
                    if not self._ig_tool_filter:
                        _ig_score_this_turn = True
                    elif len(turn_boundaries) >= 2:
                        _prev_tool = turn_boundaries[-2].get("tool_name")
                        if _prev_tool and _prev_tool.lower() in self._ig_tool_filter:
                            _ig_score_this_turn = True

            _ig_coro = None
            if _ig_score_this_turn:
                _ig_coro = self._score_ig_at_turn(
                    request_id, current_ids, _gt_full_tokens, _gt_gs, _gt_ge,
                )

            def _process_ig_lp(lp):
                nonlocal _ig_prev_value
                if lp is None:
                    return
                if _ig_prev_value is None:
                    _ig_prev_value = lp if self._ig_info_gain_type == "log_prob_diff" else _ig_math.exp(lp)
                else:
                    _cur = lp if self._ig_info_gain_type == "log_prob_diff" else _ig_math.exp(lp)
                    _ig = _cur - _ig_prev_value
                    if not (_ig_math.isnan(_ig) or _ig_math.isinf(_ig)):
                        _ig_step_rewards[step] = _ig
                    _ig_prev_value = _cur

            if flag == _Flag.END or is_last:
                if _ig_coro:
                    _process_ig_lp(await _ig_coro)
                messages.append({"role": "assistant", "content": raw_content})
                break

            elif flag == _Flag.ERROR:
                if _ig_coro:
                    _process_ig_lp(await _ig_coro)
                messages.append({"role": "assistant", "content": raw_content})
                error_msg = (
                    "<tool_response>\nYour response was malformed. "
                    "Please call a tool or provide a final answer.\n</tool_response>"
                )
                messages.append({"role": "user", "content": error_msg})

            elif flag == _Flag.CALL:
                turn_boundaries[-1]["tool_name"] = payload.get("name") if isinstance(payload, dict) else None
                # ── Execute tool call (concurrent with IG scoring) ──
                if _ig_coro:
                    _ig_task = asyncio.create_task(_ig_coro)
                t0 = time.time()
                tool_result = await self._call_tool(
                    question=messages[1]["content"] if len(messages) > 1 else "",
                    thinking=thinking,
                    tool_call=payload,
                    total_number=1,
                )
                t_tool = time.time() - t0
                t_tool_total += t_tool
                _tool_call_count += 1
                if self._robust_dedup:
                    _tool_call_history[_tool_call_key(payload)] = True
                if tool_result.startswith("Tool execution timed out"):
                    _tool_timeout_count += 1
                if _ig_coro:
                    _process_ig_lp(await _ig_task)
                if t_tool > _max_tool:
                    _max_tool = t_tool
                _lt = self.__class__._live_timing
                _lt["tool"] += t_tool
                if t_tool > _lt["max_tool"]:
                    _lt["max_tool"] = t_tool

                if self.codeact_env_disabled:
                    messages.append({"role": "assistant", "content": raw_content})
                    tool_response_str = f"<tool_response>\n{tool_result}\n</tool_response>"
                    messages.append({"role": "user", "content": tool_response_str})
                else:
                    code_str = str(payload.get("arguments", {}).get("code", ""))
                    messages.append({
                        "role": "assistant",
                        "content": f"<think>{thinking}</think>\n<code>{code_str}</code>",
                    })
                    messages.append({
                        "role": "tool",
                        "content": f"<code_response>{tool_result}</code_response>",
                    })

            if _save_trace and (step + 1) % self._trace_save_interval == 0:
                self._save_trace_snapshot(
                    request_id, step, num_turns, messages,
                    global_step=kwargs.get("global_step", "unknown"),
                    sample_idx=_sample_idx)

        if _save_trace:
            self._save_trace_snapshot(
                request_id, step, num_turns, messages,
                global_step=kwargs.get("global_step", "unknown"),
                finished=True, sample_idx=_sample_idx)

        # ── Finalize: re-tokenize from messages to include role markers ──
        # Mirrors sync mode (generation.py L983-1016): reconstruct prompt
        # and response via apply_chat_template so the response tensor
        # contains inter-turn role markers identical to sync mode.
        j = 2
        while j < len(messages):
            if messages[j]["role"] == "assistant":
                break
            j += 1
        else:
            j = len(messages)

        initial_prompt_str = await self.loop.run_in_executor(
            None,
            lambda: _proc.apply_chat_template(
                messages[:j], add_generation_prompt=True, tokenize=False
            ),
        )
        full_str = await self.loop.run_in_executor(
            None,
            lambda: _proc.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False
            ),
        )
        response_str = full_str[len(initial_prompt_str):]

        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: _proc(
                initial_prompt_str, return_tensors="pt", padding=False,
                truncation=True, max_length=self.max_len,
            ),
        )
        prompt_ids = prompt_ids["input_ids"].squeeze(0).tolist()

        response_enc = await self.loop.run_in_executor(
            None,
            lambda: _proc(response_str, return_tensors="pt", padding=False),
        )
        response_ids = response_enc["input_ids"].squeeze(0).tolist()

        # All-1s mask: ray_trainer.py L2003 unconditionally overwrites
        # response_mask with compute_response_mask (all 1s for valid tokens)
        # before any consumer reads it.  Fine-grained tool-response masking
        # for the policy loss is handled by process_response_mask in
        # update_actor, which independently reconstructs the mask from text.
        response_mask = [1] * len(response_ids)

        self.__class__._live_progress.pop(_sid, None)

        metrics = AgentLoopMetrics(
            generate_sequences=t_gen_total,
            tool_calls=t_tool_total,
            prefill_total=t_prefill_total,
            decode_total=t_decode_total,
            num_turns=num_turns,
            max_prefill_time=_max_prefill[0],
            max_prefill_ctx=_max_prefill[1],
            max_prefill_delta=_max_prefill[2],
            max_decode_time=_max_decode[0],
            max_decode_tokens=_max_decode[1],
            max_tool_time=_max_tool,
            tool_timeout_count=_tool_timeout_count,
            tool_call_count=_tool_call_count,
        )

        _extra = {
            "turn_boundaries": turn_boundaries,
            "messages": messages,
        }
        _extra["cutoff"] = _was_cutoff
        if self._robust_enabled:
            _extra["robust_repetition_retries"] = _robust_repetition_count
            _extra["robust_parse_retries"] = _robust_parse_retry_count
            _extra["robust_dedup_retries"] = _robust_dedup_count
            _extra["robust_total_retries"] = (
                _robust_repetition_count + _robust_parse_retry_count + _robust_dedup_count
            )
        if _ig_enabled:
            _total_non_baseline = max(0, num_turns - 1)
            _ig_full = [None] * _total_non_baseline
            for _s, _v in _ig_step_rewards.items():
                _idx = _s - 1
                if 0 <= _idx < _total_non_baseline:
                    _ig_full[_idx] = _v
            _extra["rollout_ig_rewards"] = _ig_full

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            num_turns=num_turns,
            metrics=metrics,
            extra_fields=_extra,
        )

    async def _call_tool(
        self, question: str, thinking: str, tool_call: dict, total_number: int
    ) -> str:
        """Execute a single tool call via the open-source tool_server
        (synchronous ``custom_call_tool``, wrapped in ``run_in_executor``).

        The ``question`` / ``thinking`` / ``total_number`` args are kept for
        signature-compat with older callers but are unused by the new
        function-based tool interface (only ``tool_call`` is consumed).
        """

        def _sync_call():
            try:
                from tool_server.execute_tools import custom_call_tool
                content = custom_call_tool(tool_call)
                if content is None:
                    return "Tool returned empty response"
                if isinstance(content, list):
                    content = "\n".join(str(c) for c in content)
                return content if isinstance(content, str) else str(content)
            except Exception as e:
                return f"Tool execution error: {repr(e)}"

        _timeout = self.__class__._tool_timeout
        try:
            result = await asyncio.wait_for(
                self.loop.run_in_executor(None, _sync_call),
                timeout=_timeout,
            )
        except asyncio.TimeoutError:
            result = f"Tool execution timed out after {_timeout:.0f} seconds."
        return result