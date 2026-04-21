# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import heapq
import logging
import os
import queue
import random
import threading
from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import Any, Optional

import hydra
import numpy as np
import ray
import torch
from cachetools import LRUCache
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_attr, rollout_trace_op
from verl.workers.rollout.async_server import TokenOutput, async_server_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AsyncLLMServerManager:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least requests load balancing
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    """

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], max_cache_size: int = 10000):
        """Initialize the AsyncLLMServerManager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            max_cache_size (int, optional): max cache size for request_id to server mapping. Defaults to 10000.
        """
        self.config = config
        self.server_handles = server_handles
        random.shuffle(self.server_handles)

        # Least requests load balancing
        self.weighted_serveres = [[0, (hash(server), server)] for server in server_handles]
        heapq.heapify(self.weighted_serveres)

        # LRU cache to map request_id to server
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

    def _choose_server(self, request_id: str) -> ray.actor.ActorHandle:
        # TODO: implement server pressure awareness load balancing
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]

        server = self.weighted_serveres[0][1][1]
        self.weighted_serveres[0][0] += 1
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])
        self.request_id_to_server[request_id] = server
        return server

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

        Returns:
            TokenOutput: token output
        """
        server = self._choose_server(request_id)
        output = await server.generate.remote(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
        )
        return output

    async def compute_prompt_logprobs(
        self,
        request_id: str,
        prompt_ids: list[int],
    ) -> list:
        """Forward to the sticky server's ``compute_prompt_logprobs``."""
        server = self._choose_server(request_id)
        return await server.compute_prompt_logprobs.remote(
            prompt_ids=prompt_ids,
            request_id=request_id,
        )


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""

    generate_sequences: float = 0.0
    tool_calls: float = 0.0
    prefill_total: float = 0.0
    decode_total: float = 0.0
    num_turns: int = 0
    max_prefill_time: float = 0.0
    max_prefill_ctx: int = 0
    max_prefill_delta: int = 0
    max_decode_time: float = 0.0
    max_decode_tokens: int = 0
    max_tool_time: float = 0.0
    tool_timeout_count: int = 0
    tool_call_count: int = 0


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    prompt_ids: list[int]
    """Prompt token ids."""
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    response_logprobs: Optional[list[float]] = None
    """Log probabilities for the response tokens."""
    multi_modal_data: Optional[dict[str, Any]] = None
    """Multi-modal data for multi-modal tools."""
    reward_score: Optional[float] = None
    """Reward score for the trajectory."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class _InternalAgentLoopOutput(AgentLoopOutput):
    """Internal agent loop output with padded sequences."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    """Padded prompt token ids."""
    response_ids: torch.Tensor
    """Padded response token ids."""
    input_ids: torch.Tensor
    """Padded input ids(prompt_ids + response_ids)."""
    position_ids: torch.Tensor
    """Padded position ids."""
    response_mask: torch.Tensor
    """Padded response mask."""
    attention_mask: torch.Tensor
    """Padded attention mask."""
    response_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities for the response tokens."""
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    """Multi-modal inputs for processors (e.g., pixel_values, image_grid_thw)."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


# make hydra.utils.instantiate happy
class _DummyConfig:
    def __init__(self, config: DictConfig) -> None:
        self.config = config


class AgentLoopBase(ABC):
    """An agent loop takes a input message, chat with OpenAI compatible LLM server and interact with various
    environments."""

    _class_initialized = False
    _live_progress: dict[str, int] = {}
    """Shared progress tracker: {sample_id: current_turn}. Updated by run(), read by worker progress reporter."""
    _live_timing: dict = {"turns": 0, "prefill": 0.0, "decode": 0.0, "tool": 0.0,
                          "max_prefill": (0.0, 0, 0), "max_decode": (0.0, 0), "max_tool": 0.0}
    """Real-time timing accumulator: updated after every turn by all running samples."""
    _cutoff_triggered: bool = False
    """Set to True when completion ratio reaches the cutoff threshold. Agent loops check this at turn boundaries."""

    def __init__(
        self,
        trainer_config: _DummyConfig,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        """Initialize agent loop, each sample will have its own loop instance.

        Args:
            trainer_config (_DummyConfig): trainer config.
            server_manager (AsyncLLMServerManager): OpenAI compatible LLM server manager.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
            processor (AutoProcessor): Processor for process messages.
        """
        self.init_class(config=trainer_config.config, tokenizer=tokenizer, processor=processor, **kwargs)
        self.config = trainer_config.config
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.loop = asyncio.get_running_loop()

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer, processor: AutoProcessor, **kwargs):
        """This is used to do heavy initialization work that should shared across all instances. It's only called once.

        Args:
            config (DictConfig): trainer config.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
            processor (AutoProcessor): Processor for process multi_modal data.
            **kwargs: extra kwargs from config file passed in by `hydra.utils.instantiate`.
        """
        if cls._class_initialized:
            return
        cls._class_initialized = True

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run agent loop to interact with LLM server and environment.

        Args:
            sampling_params (Dict[str, Any]): LLM sampling params.
            **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

        Returns:
            AgentLoopOutput: Agent loop output.
        """
        raise NotImplementedError


"""Agent loop registry: key is agent_name, value is a dict of agent loop config
used by hydra.utils.instantiate to initialize agent loop instance.

https://hydra.cc/docs/advanced/instantiate_objects/overview/
"""
_agent_loop_registry: dict[str, dict] = {}


def register(agent_name: str):
    """Register agent loop class."""

    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_loop_registry[agent_name] = {"_target_": fqdn}
        return subclass

    return decorator


@ray.remote(num_cpus=1)
class BatchExecutor:
    """Batch executor is used to collect requests into a batch execution"""

    def __init__(self, batch_func, micro_batch_size=1, max_batch_size=None):
        """

        Args:
            batch_func: batch processing function.
            micro_batch_size (int, optional): micro batch size. Defaults to 1.
            max_batch_size: batch size for batching.
        """
        self._q = queue.Queue()
        self._batch_func = batch_func
        self._max_batch = max_batch_size
        self._micro_batch_size = micro_batch_size

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    async def submit_task(self, item):
        """
        Blocking submission, returning Future
        Args:
            item: function input

        Returns:
            fut: function output
        """
        fut = Future()
        self._q.put((item, fut))
        async_fut = asyncio.wrap_future(fut)
        res = await async_fut
        return res

    def _worker_loop(self):
        while True:
            # 1. Fetch a full batch (block until at least one)
            first, first_fut = self._q.get()
            items = [first]
            futs = [first_fut]

            # Take the remaining tasks at once
            while True:
                try:
                    next_item, next_fut = self._q.get_nowait()
                    items.append(next_item)
                    futs.append(next_fut)
                    if self._max_batch and len(items) >= self._max_batch:
                        break
                except queue.Empty:
                    while len(items) % self._micro_batch_size != 0:
                        try:
                            next_item, next_fut = self._q.get(timeout=300)
                        except queue.Empty:
                            break
                        items.append(next_item)
                        futs.append(next_fut)
                        if self._max_batch and len(items) >= self._max_batch:
                            break
                    break

            try:
                results = self._batch_func(items)
            except Exception as e:
                for f in futs:
                    f.set_exception(e)
            else:
                if len(results) != len(futs):
                    err = RuntimeError(
                        f"BatchExecutor: batch_func returned {len(results)} results "
                        f"for {len(futs)} items")
                    for f in futs:
                        if not f.done():
                            f.set_exception(err)
                else:
                    for f, r in zip(futs, results, strict=True):
                        f.set_result(r)


@ray.remote(num_cpus=1)
class RewardManagerWorker:
    """Reward manager worker to compute reward score asynchronously to overlap with agent loop."""

    def __init__(self, config: DictConfig, local_path: str, rm_executor: BatchExecutor = None) -> None:
        tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.reward_manager = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        self.rm_executor = rm_executor

    def compute_score(
        self,
        data: DataProto,
    ) -> dict:
        """Compute reward score for agent loop output.

        Args:
            data: reward function input

        Returns:
            dict: Reward score and reward extra info.
        """
        if self.rm_executor is not None:
            res = ray.get(self.rm_executor.submit_task.remote(data))
            data = data.union(res)

        result = self.reward_manager(data, return_dict=True)
        reward_score = result["reward_tensor"].sum(dim=-1).item()
        reward_extra_info = {k: v[0] for k, v in result.get("reward_extra_info", {}).items()}
        return {"reward_score": reward_score, "reward_extra_info": reward_extra_info}


@ray.remote
class AgentLoopWorker:
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(
        self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], rm_executor: BatchExecutor = None
    ):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
        """
        self.config = config
        self.server_manager = AsyncLLMServerManager(config, server_handles)
        self.rm_executor = rm_executor

        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.processor = hf_processor(local_path, trust_remote_code=True)

        agent_loop_config_path = config.actor_rollout_ref.rollout.agent.agent_loop_config_path
        if agent_loop_config_path:
            agent_loop_configs = OmegaConf.load(agent_loop_config_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config
        if self.config.actor_rollout_ref.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.actor_rollout_ref.model.custom_chat_template
            self.tokenizer.chat_template = self.config.actor_rollout_ref.model.custom_chat_template

        self.reward_manager_worker = RewardManagerWorker.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False,
            ),
        ).remote(self.config, local_path, self.rm_executor)

        trace_config = self.config.actor_rollout_ref.rollout.get("trace", {})
        RolloutTraceConfig.init(
            self.config.trainer.project_name,
            self.config.trainer.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
        )

        # Redirect stdout to a per-worker log file instead of /dev/null,
        # so that C-level crash messages and library output are preserved.
        import sys
        _log_dir = os.path.join(
            self.config.trainer.get("default_local_dir", "/tmp"),
            "worker_stdout_logs",
        )
        os.makedirs(_log_dir, exist_ok=True)
        _log_path = os.path.join(_log_dir, f"worker_{os.getpid()}.log")
        _log_fd = os.open(_log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        os.dup2(_log_fd, 1)
        os.close(_log_fd)
        sys.stdout = open(_log_path, 'a')
        print(f"[AgentLoopWorker pid={os.getpid()}] stdout redirected to {_log_path}", file=sys.stderr)

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            batch.non_tensor_batch["agent_name"] = np.array(["single_turn_agent"] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        import sys
        import time as _time

        _log = sys.stderr

        n_samples = len(batch)
        _t0 = _time.time()
        _done_count = 0
        _done_turns = []
        _pid = os.getpid()
        AgentLoopBase._live_timing = {"turns": 0, "prefill": 0.0, "decode": 0.0, "tool": 0.0,
                                      "wall_gen": 0.0,
                                      "max_prefill": (0.0, 0, 0), "max_decode": (0.0, 0), "max_tool": 0.0}
        AgentLoopBase._live_progress.clear()
        AgentLoopBase._cutoff_triggered = False

        _cutoff_ratio = config.agent.get("completion_cutoff", None)
        _cutoff_ratio = float(_cutoff_ratio) if _cutoff_ratio else None
        _cutoff_logged = False

        print(
            f"[Worker pid={_pid}] Starting: {n_samples} samples"
            + (f" (cutoff={_cutoff_ratio:.0%})" if _cutoff_ratio else ""),
            file=_log, flush=True,
        )

        async def _tracked_run(idx, sp, tj, **kw):
            nonlocal _done_count, _cutoff_logged
            result = await self._run_agent_loop(sp, tj, **kw)
            _done_count += 1
            _done_turns.append(result.num_turns)
            if (_cutoff_ratio
                    and not AgentLoopBase._cutoff_triggered
                    and _done_count / n_samples >= _cutoff_ratio):
                AgentLoopBase._cutoff_triggered = True
                _n_remaining = n_samples - _done_count
                if not _cutoff_logged:
                    _cutoff_logged = True
                    print(
                        f"[Worker pid={_pid}] Cutoff triggered: "
                        f"{_done_count}/{n_samples} done ({_cutoff_ratio:.0%}), "
                        f"signalling {_n_remaining} remaining samples to wrap up",
                        file=_log, flush=True,
                    )
            return result

        def _fmt_ctx(n):
            return f"{n/1000:.0f}K" if n >= 1000 else str(n)

        async def _progress_reporter():
            while _done_count < n_samples:
                await asyncio.sleep(30)
                if _done_count < n_samples:
                    _elapsed = _time.time() - _t0
                    _n_active = n_samples - _done_count
                    _turns = list(AgentLoopBase._live_progress.values())
                    if _turns:
                        _info = (f"min_turns={min(_turns)}, max_turns={max(_turns)}, "
                                 f"avg_turns={sum(_turns)/len(_turns):.1f}")
                    else:
                        _info = "no active samples"
                    _line1 = (
                        f"[Worker pid={_pid}] {_done_count}/{n_samples} done, "
                        f"{_n_active} active ({_elapsed:.0f}s)"
                    )
                    _lt = AgentLoopBase._live_timing
                    _lt_turns = _lt["turns"]
                    if _lt_turns > 0:
                        _avg_pf = _lt["prefill"] / _lt_turns
                        _avg_dc = _lt["decode"] / _lt_turns
                        _avg_gen = _avg_pf + _avg_dc
                        _avg_wall = _lt["wall_gen"] / _lt_turns
                        _avg_queue = max(0.0, _avg_wall - _avg_gen)
                        _avg_tl = _lt["tool"] / _lt_turns
                        _mp = _lt["max_prefill"]
                        _md = _lt["max_decode"]
                        _line2 = (
                            f"  turns: [{_info}] | "
                            f"gen: {_avg_wall:.1f}s/t (prefill={_avg_pf:.1f}s decode={_avg_dc:.1f}s "
                            f"queue={_avg_queue:.1f}s) "
                            f"max_prefill={_mp[0]:.1f}s"
                            f"(ctx={_fmt_ctx(_mp[1])},delta={_fmt_ctx(_mp[2])}) "
                            f"max_decode={_md[0]:.1f}s({_fmt_ctx(_md[1])}tok) | "
                            f"tool: {_avg_tl:.1f}s/t max={_lt['max_tool']:.1f}s"
                        )
                    else:
                        _line2 = f"  turns: [{_info}] | timing: awaiting first turn"
                    print(f"{_line1}\n{_line2}", file=_log, flush=True)

        _max_conc = config.agent.get("max_concurrent_samples", None)
        _sem = asyncio.Semaphore(_max_conc) if _max_conc else None
        if _sem:
            print(
                f"[Worker pid={_pid}] Concurrency limited to {_max_conc} samples",
                file=_log, flush=True,
            )

        def _make_dummy_output():
            """Construct a minimal no-op output for a failed sample.

            The dummy has: response_mask=0 everywhere (zero loss contribution),
            reward_score=0.0, empty turn_boundaries (skipped by IG/penalty),
            and a minimal messages list (safe for apply_chat_template).
            """
            _prompt_len = self.config.actor_rollout_ref.rollout.prompt_length
            _pad = self.tokenizer.pad_token_id or 0
            _p = torch.full((1, _prompt_len), _pad, dtype=torch.long)
            _r = torch.full((1, 1), _pad, dtype=torch.long)
            _z = torch.zeros(1, 1, dtype=torch.long)
            _a_prompt = torch.zeros(1, _prompt_len, dtype=torch.long)
            _a_resp = torch.ones(1, 1, dtype=torch.long)
            return _InternalAgentLoopOutput(
                prompt_ids=_p,
                response_ids=_r,
                input_ids=torch.cat([_p, _r], dim=1),
                position_ids=torch.zeros(1, _prompt_len + 1, dtype=torch.long),
                response_mask=_z,
                attention_mask=torch.cat([_a_prompt, _a_resp], dim=1),
                response_logprobs=None,
                multi_modal_inputs=None,
                multi_modal_data=None,
                reward_score=0.0,
                num_turns=0,
                metrics=AgentLoopMetrics(),
                extra_fields={
                    "reward_extra_info": {},
                    "turn_boundaries": [],
                    "messages": [{"role": "system", "content": ""}],
                },
            )

        async def _guarded_run(idx, sp, tj, **kw):
            nonlocal _done_count
            try:
                if _sem:
                    async with _sem:
                        return await _tracked_run(idx, sp, tj, **kw)
                return await _tracked_run(idx, sp, tj, **kw)
            except Exception as exc:
                import traceback
                print(
                    f"[Worker pid={_pid}] Sample {idx} FAILED: {exc!r}\n"
                    f"{traceback.format_exc()}",
                    file=_log, flush=True,
                )
                _done_count += 1
                _done_turns.append(0)
                _rid = kw.get("request_id", "")
                if _rid:
                    AgentLoopBase._live_progress.pop(_rid[:8], None)
                return _make_dummy_output()

        _global_steps = batch.meta_info.get("global_steps", -1)

        tasks = []
        for i in range(n_samples):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            kwargs["global_step"] = _global_steps
            kwargs["_sample_idx"] = int(kwargs.pop("_global_sample_idx", i))
            tasks.append(asyncio.create_task(
                _guarded_run(i, sampling_params, trajectory_info[i], **kwargs)
            ))
        reporter = asyncio.create_task(_progress_reporter())

        outputs = await asyncio.gather(*tasks)
        reporter.cancel()

        _elapsed = _time.time() - _t0
        _avg_turns = sum(_done_turns) / len(_done_turns) if _done_turns else 0
        _max_turns = max(_done_turns) if _done_turns else 0
        _cutoff_msg = ""
        if _cutoff_ratio and _cutoff_logged:
            _n_cutoff = sum(1 for o in outputs
                           if getattr(o, 'extra_fields', {}).get('cutoff', False))
            _cutoff_msg = f", cutoff={_n_cutoff} samples"
        print(
            f"[Worker pid={os.getpid()}] Done: {n_samples}/{n_samples} in {_elapsed:.1f}s, "
            f"avg_turns={_avg_turns:.1f}, max_turns={_max_turns}, "
            f"avg_per_sample={_elapsed / n_samples:.1f}s{_cutoff_msg}",
            file=_log, flush=True,
        )

        output = self._postprocess(outputs)
        return output

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        **kwargs,
    ) -> _InternalAgentLoopOutput:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=_DummyConfig(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
            )
            output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)

            # Some AgentLoop may have already computed the reward score, e.g SWE-agent.

            # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
            # prompt_ids: left padded with zeros (e.g., [0,0,0,0,1,2,3,4])
            # response_ids: right padded with zeros (e.g., [5,6,7,8,0,0,0,0])
            # input_ids: concatenation of prompt + response
            # Mask:
            # For example, if the prompt is [1,2,3,4] and the response is [5,6,7,(tool start)8,9(tool end),10,11,12]
            # - prompt_attention_mask: 0s for padding, 1s for tokens
            #   e.g., [0,0,0,0,1,1,1,1]
            # - response_attention_mask: 0s for padding, 1s for tokens
            #   e.g., [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
            # attention_mask: concatenation of prompt_attention_mask and response_attention_mask
            #   e.g., [0,0,0,0,1,1,1,1(prompt),1,1,1,1,1,1,1,1,1,1,1,0,0,0,0(response)]
            # - response_mask: 1s for LLM generated tokens, 0 for tool response/padding tokens
            #   e.g., [1,1,1,1,1,1,1,(tool start),0,0(tool end),1,1,0,0,0,0]
            # - position_ids: sequential positions for tokens, starting at 0
            #   e.g., [0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,0,0,0]

            self.tokenizer.padding_side = "left"
            prompt_output = self.tokenizer.pad(
                {"input_ids": output.prompt_ids},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.prompt_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            if prompt_output["input_ids"].dim() == 1:
                prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
                prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

            response_ids_t = torch.tensor(output.response_ids, dtype=torch.long).unsqueeze(0)
            response_attn_t = torch.ones_like(response_ids_t)
            response_output = {"input_ids": response_ids_t, "attention_mask": response_attn_t}

            response_mask_t = torch.tensor(output.response_mask, dtype=torch.long).unsqueeze(0)

            response_logprobs = None
            if output.response_logprobs is not None:
                response_logprobs = torch.tensor(output.response_logprobs, dtype=torch.float32).unsqueeze(0)

            response_mask = response_mask_t * response_output["attention_mask"]
            attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
            input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

            # Handle multi-modal inputs and position_ids calculation
            # Only support Qwen2VLImageProcessor for multi-modal processing currently
            # TODO: support other multi-modal inputs
            multi_modal_inputs = None
            if (
                self.processor is not None
                and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
            ):
                from verl.models.transformers.qwen2_vl import get_rope_index

                images = output.multi_modal_data.get("image", None)
                current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
                multi_modal_inputs = self.processor(text=[current_text], images=images, return_tensors="pt")
                multi_modal_inputs.pop("input_ids", None)
                multi_modal_inputs.pop("attention_mask", None)

                # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
                # because np.array() only keeps the keys for BatchFeature.
                multi_modal_inputs = dict(multi_modal_inputs)

                image_grid_thw = multi_modal_inputs.get("image_grid_thw")
                video_grid_thw = multi_modal_inputs.get("video_grid_thw")
                second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts")

                position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids.squeeze(0),
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask.squeeze(0),
                ).unsqueeze(0)  # (1, 3, seq_len)
            else:
                position_ids = compute_position_id_with_mask(attention_mask)  # (1, seq_len)
            enable_async_reward = (
                self.rm_executor is not None and self.config.reward_model.enable_resource_pool
            ) or not self.config.reward_model.enable
            if output.reward_score is None and enable_async_reward:
                batch = TensorDict(
                    {
                        "prompts": prompt_output["input_ids"],  # [1, prompt_length]
                        "responses": response_output["input_ids"],  # [1, response_length]
                        "attention_mask": attention_mask,  # [1, prompt_length + response_length]
                        "input_ids": input_ids,  # [1, prompt_length + response_length]
                        "position_ids": position_ids,
                    },
                    batch_size=1,
                )
                non_tensor_batch = {
                    **{k: np.array([v]) for k, v in kwargs.items()},
                    "__num_turns__": np.array([output.num_turns]),
                }
                data = DataProto(
                    batch=batch,
                    non_tensor_batch=non_tensor_batch,
                )
                result = await self.reward_manager_worker.compute_score.remote(data)
                output.reward_score = result["reward_score"]
                output.extra_fields["reward_extra_info"] = result["reward_extra_info"]

            return _InternalAgentLoopOutput(
                prompt_ids=prompt_output["input_ids"],
                response_ids=response_output["input_ids"],
                input_ids=input_ids,
                position_ids=position_ids,
                response_mask=response_mask,
                attention_mask=attention_mask,
                response_logprobs=response_logprobs,
                multi_modal_inputs=multi_modal_inputs,
                multi_modal_data=output.multi_modal_data,
                reward_score=output.reward_score,
                num_turns=output.num_turns,
                metrics=output.metrics,
                extra_fields=output.extra_fields,
            )

    def _postprocess(self, inputs: list[_InternalAgentLoopOutput]) -> DataProto:
        """Process the padded outputs from _run_agent_loop and combine them into a batch."""
        import torch.nn.functional as F

        max_response_len = max(inp.response_ids.size(-1) for inp in inputs)
        pad_token_id = self.tokenizer.pad_token_id or 0
        for inp in inputs:
            r_pad = max_response_len - inp.response_ids.size(-1)
            if r_pad > 0:
                inp.response_ids = F.pad(inp.response_ids, (0, r_pad), value=pad_token_id)
                inp.response_mask = F.pad(inp.response_mask, (0, r_pad), value=0)
                inp.input_ids = F.pad(inp.input_ids, (0, r_pad), value=pad_token_id)
                inp.attention_mask = F.pad(inp.attention_mask, (0, r_pad), value=0)
                if inp.position_ids.dim() == 3:
                    inp.position_ids = F.pad(inp.position_ids, (0, r_pad), value=0)
                else:
                    inp.position_ids = F.pad(inp.position_ids, (0, r_pad), value=0)
                if inp.response_logprobs is not None:
                    inp.response_logprobs = F.pad(inp.response_logprobs, (0, r_pad), value=0.0)

        # Convert lists back to tensors and stack them to create a batch.
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                # position_ids: [bsz, 3, prompt_length + response_length] or [bsz, prompt_length + response_length]
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            prompt_length = prompt_ids.size(1)
            response_length = (attention_mask[:, prompt_length:].sum(dim=1) - 1).clamp(min=0)
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }

        # add reward_extra_info to non_tensor_batch
        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = sorted(set(k for info in reward_extra_infos for k in info))
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info.get(key) for info in reward_extra_infos])

        # Add multi_modal_inputs to non_tensor_batch if any samples have them
        multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        metrics = [input.metrics.model_dump() for input in inputs]
        # Collect extra fields from all inputs and convert them to np.ndarray
        extra_fields = {}
        all_keys = set(key for input_item in inputs for key in input_item.extra_fields)
        for key in all_keys:
            extra_fields[key] = np.array([input.extra_fields.get(key) for input in inputs], dtype=object)

        non_tensor_batch.update(extra_fields)
        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={"metrics": metrics, "reward_extra_keys": reward_extra_keys},
        )


async def get_trajectory_info(step, index, validate):
    """Get trajectory info.

    Args:
        step (int): global steps in the trainer.
        index (list): form datastore extra_info.index column.
        validate (bool): whether is a validate step.

    Returns:
        list: trajectory.
    """
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info


class AgentLoopManager:
    """Agent loop manager that manages a group of agent loop workers."""

    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup, rm_wg: RayWorkerGroup = None):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): trainer config.
            worker_group (RayWorkerGroup): ActorRolloutRef worker group.
        """
        self.config = config
        self.worker_group = worker_group
        self.rm_executor = None
        self.rm_micro_batch_size = None
        if rm_wg:

            def batch_fn(data_list: list[DataProto]) -> list[torch.Tensor]:
                new_data_list = []
                for data in data_list:
                    temp_non_tensor_batch = {"__num_turns__": data.non_tensor_batch["__num_turns__"]}
                    temp_data = DataProto(batch=data.batch, non_tensor_batch=temp_non_tensor_batch)
                    new_data_list.append(temp_data)

                new_batch = DataProto.concat(new_data_list)
                out_data = rm_wg.compute_rm_score(new_batch)
                return out_data.split(1)

            self.rm_executor = BatchExecutor.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
            ).remote(batch_fn, rm_wg.world_size)

            self.rm_micro_batch_size = rm_wg.world_size

        _local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self._pad_token_id = hf_tokenizer(_local_path, trust_remote_code=True).pad_token_id or 0

        self._initialize_llm_servers()
        self._init_agent_loop_workers()

        # Initially we're in sleep mode.
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

    def _initialize_llm_servers(self):
        self.rollout_tp_size = self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        self.rollout_dp_size = self.worker_group.world_size // self.rollout_tp_size

        workers_info = ray.get(
            [
                worker.__ray_call__.remote(lambda self: ray.get_runtime_context().get_node_id())
                for worker in self.worker_group.workers
            ]
        )
        assert len(workers_info) == self.worker_group.world_size

        self.async_llm_servers = [None] * self.rollout_dp_size
        self.server_addresses = [None] * self.rollout_dp_size

        if self.config.actor_rollout_ref.rollout.agent.custom_async_server:
            server_class = async_server_class(
                rollout_backend=self.config.actor_rollout_ref.rollout.name,
                rollout_backend_module=self.config.actor_rollout_ref.rollout.agent.custom_async_server.path,
                rollout_backend_class=self.config.actor_rollout_ref.rollout.agent.custom_async_server.name,
            )
        else:
            server_class = async_server_class(rollout_backend=self.config.actor_rollout_ref.rollout.name)

        # Start all server instances, restart if address already in use.
        unready_dp_ranks = set(range(self.rollout_dp_size))
        while len(unready_dp_ranks) > 0:
            servers = {
                rollout_dp_rank: server_class.options(
                    # make sure AsyncvLLMServer colocates with its corresponding workers
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=workers_info[rollout_dp_rank * self.rollout_tp_size],
                        soft=False,
                    ),
                    name=f"async_llm_server_{rollout_dp_rank}",
                ).remote(self.config, self.rollout_dp_size, rollout_dp_rank, self.worker_group.name_prefix)
                for rollout_dp_rank in unready_dp_ranks
            }

            for rollout_dp_rank, server in servers.items():
                try:
                    address = ray.get(server.get_server_address.remote())
                    self.server_addresses[rollout_dp_rank] = address
                    self.async_llm_servers[rollout_dp_rank] = server
                    unready_dp_ranks.remove(rollout_dp_rank)
                except Exception:
                    ray.kill(server)
                    print(f"rollout server {rollout_dp_rank} failed, maybe address already in use, restarting...")

        # All server instances are ready, init AsyncLLM engine.
        ray.get([server.init_engine.remote() for server in self.async_llm_servers])

    def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                AgentLoopWorker.options(
                    name=f"agent_loop_worker_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(self.config, self.async_llm_servers, self.rm_executor)
            )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """

        import datetime
        import time as _time

        if self.rm_micro_batch_size and len(prompts) % self.rm_micro_batch_size != 0:
            raise ValueError(
                f"The length of prompts {len(prompts)} cannot divide the world size of rm_wg {self.rm_micro_batch_size}"
            )

        n_total = len(prompts)
        n_workers = len(self.agent_loop_workers)
        print(
            f"\n{'=' * 72}\n"
            f"  [AsyncRollout] {datetime.datetime.now().strftime('%H:%M:%S')} "
            f"Starting: {n_total} samples → {n_workers} workers "
            f"({n_total // n_workers} samples/worker)\n"
            f"{'=' * 72}",
            flush=True,
        )
        _t_rollout_start = _time.time()

        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()
        prompts.non_tensor_batch["_global_sample_idx"] = np.arange(len(prompts), dtype=np.int32)
        chunkes = prompts.chunk(n_workers)
        outputs = ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        _t_rollout_gen = _time.time() - _t_rollout_start

        import torch.nn.functional as F

        max_response_len = max(out.batch["responses"].size(1) for out in outputs)
        for out in outputs:
            r_pad = max_response_len - out.batch["responses"].size(1)
            if r_pad > 0:
                for key in list(out.batch.keys()):
                    if key == "prompts":
                        continue
                    t = out.batch[key]
                    if key in ("rollout_log_probs", "rm_scores"):
                        out.batch[key] = F.pad(t, (0, r_pad), value=0.0)
                    elif key in ("responses", "input_ids"):
                        out.batch[key] = F.pad(t, (0, r_pad), value=self._pad_token_id)
                    else:
                        out.batch[key] = F.pad(t, (0, r_pad), value=0)

        output = DataProto.concat(outputs)
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

        # calculate performance metrics
        metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)

        _t_rollout_total = _time.time() - _t_rollout_start
        _pf_avg = timing.get('agent_loop/prefill_total/mean', 0)
        _dc_avg = timing.get('agent_loop/decode_total/mean', 0)
        print(
            f"\n{'=' * 72}\n"
            f"  [AsyncRollout] {datetime.datetime.now().strftime('%H:%M:%S')} "
            f"Complete: {n_total} samples in {_t_rollout_total:.1f}s\n"
            f"  gen={timing.get('agent_loop/generate_sequences/mean', 0):.1f}s(avg) "
            f"{timing.get('agent_loop/generate_sequences/max', 0):.1f}s(max) | "
            f"prefill={_pf_avg:.1f}s(avg) decode={_dc_avg:.1f}s(avg)\n"
            f"  max_prefill={timing.get('agent_loop/max_prefill_time', 0):.1f}s"
            f"(ctx={timing.get('agent_loop/max_prefill_ctx', 0) / 1000:.0f}K,"
            f"delta={timing.get('agent_loop/max_prefill_delta', 0) / 1000:.0f}K,"
            f"{timing.get('agent_loop/max_prefill_throughput', 0):.0f}tok/s) "
            f"max_decode={timing.get('agent_loop/max_decode_time', 0):.1f}s"
            f"({timing.get('agent_loop/max_decode_tokens', 0):.0f}tok,"
            f"{timing.get('agent_loop/max_decode_per_tok', 0):.2f}s/tok)\n"
            f"  tool={timing.get('agent_loop/tool_calls/mean', 0):.1f}s(avg) "
            f"{timing.get('agent_loop/tool_calls/max', 0):.1f}s(max) | "
            f"timeout={timing.get('agent_loop/tool_timeout/total', 0):.0f}/"
            f"{timing.get('agent_loop/tool_calls/total', 0):.0f}"
            f"({timing.get('agent_loop/tool_timeout/rate', 0):.1%})\n"
            f"  slowest: gen={timing.get('agent_loop/slowest/generate_sequences', 0):.1f}s, "
            f"tool={timing.get('agent_loop/slowest/tool_calls', 0):.1f}s, "
            f"prompt={timing.get('agent_loop/slowest/prompt_length', 0):.0f}tok, "
            f"response={timing.get('agent_loop/slowest/response_length', 0):.0f}tok\n"
            f"{'=' * 72}",
            flush=True,
        )

        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()

        t_prefill = np.array([metric.get("prefill_total", 0) for chunk in metrics for metric in chunk])
        t_decode = np.array([metric.get("decode_total", 0) for chunk in metrics for metric in chunk])
        timing["agent_loop/prefill_total/mean"] = t_prefill.mean()
        timing["agent_loop/decode_total/mean"] = t_decode.mean()

        _max_prefill_time = np.array([metric.get("max_prefill_time", 0) for chunk in metrics for metric in chunk])
        _max_prefill_ctx = np.array([metric.get("max_prefill_ctx", 0) for chunk in metrics for metric in chunk])
        _max_prefill_delta = np.array([metric.get("max_prefill_delta", 0) for chunk in metrics for metric in chunk])
        _max_decode_time = np.array([metric.get("max_decode_time", 0) for chunk in metrics for metric in chunk])
        _max_decode_tokens = np.array([metric.get("max_decode_tokens", 0) for chunk in metrics for metric in chunk])
        _gmax_pf_idx = int(_max_prefill_time.argmax())
        _gmax_dc_idx = int(_max_decode_time.argmax())
        timing["agent_loop/max_prefill_time"] = float(_max_prefill_time[_gmax_pf_idx])
        timing["agent_loop/max_prefill_ctx"] = int(_max_prefill_ctx[_gmax_pf_idx])
        timing["agent_loop/max_prefill_delta"] = int(_max_prefill_delta[_gmax_pf_idx])
        timing["agent_loop/max_decode_time"] = float(_max_decode_time[_gmax_dc_idx])
        timing["agent_loop/max_decode_tokens"] = int(_max_decode_tokens[_gmax_dc_idx])
        _pf_time = float(_max_prefill_time[_gmax_pf_idx])
        _pf_ctx = int(_max_prefill_ctx[_gmax_pf_idx])
        _dc_time = float(_max_decode_time[_gmax_dc_idx])
        _dc_tok = int(_max_decode_tokens[_gmax_dc_idx])
        timing["agent_loop/max_prefill_throughput"] = _pf_ctx / _pf_time if _pf_time > 0 else 0.0
        timing["agent_loop/max_decode_per_tok"] = _dc_time / _dc_tok if _dc_tok > 0 else 0.0

        _timeout_counts = np.array([metric.get("tool_timeout_count", 0) for chunk in metrics for metric in chunk])
        _tool_counts = np.array([metric.get("tool_call_count", 0) for chunk in metrics for metric in chunk])
        timing["agent_loop/tool_timeout/total"] = int(_timeout_counts.sum())
        timing["agent_loop/tool_calls/total"] = int(_tool_counts.sum())
        _total_calls = int(_tool_counts.sum())
        timing["agent_loop/tool_timeout/rate"] = (
            float(_timeout_counts.sum()) / _total_calls if _total_calls > 0 else 0.0
        )

        # batch sequence generation is bounded by the slowest sample
        slowest = np.argmax(t_generate_sequences + t_tool_calls)
        attention_mask = output.batch["attention_mask"][slowest]
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
        timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing

    def wake_up(self):
        """Wake up all rollout server instances."""
        ray.get([server.wake_up.remote() for server in self.async_llm_servers])

    def sleep(self):
        """Sleep all rollout server instances."""
        ray.get([server.sleep.remote() for server in self.async_llm_servers])
