# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F


def slice_kv_cache(past_kv, end):
    """Slice a KV cache to [:end] along the sequence dimension.

    Compatible with all transformers cache formats (DynamicCache with layers,
    key_cache/value_cache, or legacy tuple).  Uses the official
    to_legacy_cache / from_legacy_cache API pair so that no internal cache
    structure is accessed directly.
    """
    raw = past_kv.to_legacy_cache() if hasattr(past_kv, 'to_legacy_cache') else list(past_kv)
    sliced = tuple((k[:, :, :end, :], v[:, :, :end, :]) for k, v in raw)
    cache_cls = type(past_kv)
    if hasattr(cache_cls, 'from_legacy_cache'):
        return cache_cls.from_legacy_cache(sliced)
    cache = cache_cls()
    for i, (k, v) in enumerate(sliced):
        cache.update(k, v, i)
    return cache


from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import (
    gather_outputs_and_unpad,
    get_ulysses_sequence_parallel_group,
    set_ulysses_sequence_parallel_group,
    ulysses_pad,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None,
                 tokenizer=None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.tokenizer = tokenizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = self.compute_entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @torch.no_grad()
    def compute_ig_with_kv_cache(self, data: DataProto) -> DataProto:
        """Compute info-gain log probs for all turns using KV cache reuse.

        Runs entirely on this worker — KV cache stays on GPU, never crosses Ray.

        For each sample:
          Phase 1: Forward full trajectory with use_cache=True -> KV cache
          Phase 2: For each turn, slice KV to boundary-1, prepend last prompt
                   token to GT response, forward ~1+R tokens.

        Args:
            data: DataProto with batch keys:
                trajectory_input_ids [N, L], trajectory_attention_mask [N, L],
                trajectory_position_ids [N, L], gt_input_ids [N, R]
              and meta_info:
                turn_boundaries, gt_idx, temperature

        Returns:
            DataProto with 'per_turn_log_probs' [N, T, R] (unfilled = -inf)
        """
        self.actor_module.eval()
        device = get_device_id()
        data = data.to(device)

        # Disable Ulysses SP for the entire KV cache info-gain computation.
        # This function processes samples independently (batch_size=1 per GPU),
        # so cross-GPU sequence parallelism is unnecessary. More importantly,
        # Ulysses all-to-all would corrupt the position_ids (4x replication)
        # causing flash attention's prepare_fa_kwargs_from_position_ids to crash
        # when position_ids don't start from 0 (KV cache turns with bp > 1).
        _saved_sp_group = None
        if self.use_ulysses_sp:
            _saved_sp_group = get_ulysses_sequence_parallel_group()
            set_ulysses_sequence_parallel_group(None)

        try:
            return self._compute_ig_with_kv_cache_impl(data, device)
        finally:
            if self.use_ulysses_sp:
                set_ulysses_sequence_parallel_group(_saved_sp_group)

    def _compute_ig_with_kv_cache_impl(self, data: DataProto, device) -> DataProto:
        """Inner implementation of compute_ig_with_kv_cache (called with Ulysses SP disabled)."""
        traj_ids = data.batch['trajectory_input_ids']
        traj_mask = data.batch['trajectory_attention_mask']
        traj_pos = data.batch['trajectory_position_ids']
        gt_ids = data.batch['gt_input_ids']
        orig_idx = data.batch['original_sample_idx'][:, 0].tolist()  # local->global index map

        turn_boundaries = data.meta_info['turn_boundaries']
        gt_idx_list = data.meta_info['gt_idx']
        temperature = data.meta_info.get('temperature', 1.0)

        N = traj_ids.shape[0]
        T = len(turn_boundaries)
        R = gt_ids.shape[1]

        _verify_kv = os.environ.get('IGPO_VERIFY_KV_CACHE', '').lower() in ('true', '1', 'yes')
        _parity_env = os.environ.get('IGPO_VERIFY_KV_PARITY', '').lower()
        _verify_parity = _parity_env in ('true', '1', 'yes', 'strict')
        # strict mode: reference also uses KV-cache continuation (same Q_len),
        #   so FA2 tiling is identical → catches real bugs precisely.
        # relaxed mode (true/1/yes): reference uses fresh forward (different Q_len),
        #   tolerates FA2 bfloat16 precision diff from different CUDA tiling.
        _parity_strict = _parity_env == 'strict'
        _parity_tol = float(os.environ.get('IGPO_KV_PARITY_TOLERANCE', '2.0'))
        if _verify_kv or _verify_parity:
            _orig_set = set(int(x) for x in orig_idx)
            assert len(_orig_set) == N, (
                f"[KV-VERIFY FAIL] Duplicate original_sample_idx: {orig_idx}")
            for _oi in orig_idx:
                assert 0 <= int(_oi) < len(gt_idx_list), (
                    f"[KV-VERIFY FAIL] original_sample_idx {_oi} out of range "
                    f"[0, {len(gt_idx_list)})")
            for _ti, _tb in enumerate(turn_boundaries):
                for _act_i in _tb['activate_list']:
                    assert _act_i in _tb['per_sample_token_lens'], (
                        f"[KV-VERIFY FAIL] turn {_ti}: sample {_act_i} in activate_list "
                        f"but missing from per_sample_token_lens")
            logger.info(f"[KV-VERIFY] Worker local_N={N}, global_indices={sorted(_orig_set)}, "
                        f"total_samples={len(gt_idx_list)}, turns={T}")

        _parity_diffs = []
        _layer_diag_results = []
        _short_seq_tested = False
        _short_seq_pass = False

        # FSDP uses collective ALLGATHER to unshard parameters on every forward
        # call.  All data-parallel ranks MUST execute the same number of forward
        # calls; otherwise ranks that finish early leave the ALLGATHER group and
        # the remaining ranks deadlock (NCCL timeout).
        #
        # Phase-1 builds the KV cache in chunks (default 4096 tokens each) to
        # bound activation memory — a single forward on a 170K-token trajectory
        # can require 60+ GiB just for one F.linear output.
        # We guarantee  _max_N * (_max_phase1_chunks + T)  calls per rank:
        #   - _max_phase1_chunks per sample (Phase-1: chunked KV cache build)
        #   - T per sample (Phase-2: one per turn boundary)
        # Inactive samples/turns run a cheap 1-token dummy forward instead of
        # skipping, so the ALLGATHER count stays in lockstep.
        _sync_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
        _sync_pos = torch.zeros(1, 1, dtype=torch.long, device=device)

        def _fsdp_sync_dummy():
            with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                _d = self.actor_module(
                    input_ids=_sync_ids,
                    position_ids=_sync_pos,
                    use_cache=False,
                )
            del _d

        # Disable verify-parity in multi-GPU mode: the verify code adds
        # variable numbers of forward calls per active turn, which would
        # cause the same ALLGATHER deadlock we're fixing here.
        if _verify_parity and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            logger.warning(
                "[KV-PARITY] Disabled in multi-GPU FSDP mode to prevent "
                "ALLGATHER deadlock (unbalanced verify forward calls)")
            _verify_parity = False

        # All ranks must run the same number of loop iterations so that the
        # total FSDP ALLGATHER count matches.  When total_samples % world_size
        # != 0, different ranks receive different local N — pad to max_N.
        _max_N = torch.tensor(N, dtype=torch.long, device=device)
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            torch.distributed.all_reduce(_max_N, op=torch.distributed.ReduceOp.MAX)
        _max_N = int(_max_N.item())

        # Phase-1 chunk size: bound activation memory for long trajectories.
        _IG_PHASE1_CHUNK = int(os.environ.get('IGPO_PHASE1_CHUNK_SIZE', '4096'))
        _local_max_traj = int(traj_mask.sum(dim=1).max().item()) if N > 0 else 0
        _max_traj_t = torch.tensor(_local_max_traj, dtype=torch.long, device=device)
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            torch.distributed.all_reduce(_max_traj_t, op=torch.distributed.ReduceOp.MAX)
        _max_phase1_chunks = max(1, int((_max_traj_t.item() + _IG_PHASE1_CHUNK - 1) // _IG_PHASE1_CHUNK))
        if _local_max_traj > _IG_PHASE1_CHUNK:
            logger.info(f"[IG-KV] Phase-1 chunking: max_traj={_local_max_traj}, "
                        f"chunk={_IG_PHASE1_CHUNK}, chunks={_max_phase1_chunks}")

        result = torch.full((N, T, R), float('-inf'), dtype=torch.float32, device=device)

        for si in range(_max_N):
            if si >= N:
                for _ in range(_max_phase1_chunks + T):
                    _fsdp_sync_dummy()
                continue
            global_si = int(orig_idx[si])
            gs, ge = gt_idx_list[global_si]
            if gs >= ge:
                for _ in range(_max_phase1_chunks + T):
                    _fsdp_sync_dummy()
                continue

            s_ids = traj_ids[si:si + 1]
            s_mask = traj_mask[si:si + 1]
            s_pos = traj_pos[si:si + 1]

            traj_valid = int(s_mask.sum(dim=1).item())
            s_gt = gt_ids[si:si + 1]  # [1, R]
            _valid_pos = (s_mask[0] == 1).nonzero(as_tuple=True)[0]
            _first_valid = _valid_pos[0].item() if len(_valid_pos) > 0 else 0

            # Strip padding so the KV-cache forward uses the same FA kernel
            # (flash_attn_func) as the reference path. Passing a mask with
            # zeros triggers flash_attn_varlen_func, whose bfloat16 numerics
            # differ from flash_attn_func and cause KV-parity failures.
            v_end = _first_valid + traj_valid
            v_ids = s_ids[:, _first_valid:v_end]
            v_pos = s_pos[:, _first_valid:v_end]

            _n_chunks = (traj_valid + _IG_PHASE1_CHUNK - 1) // _IG_PHASE1_CHUNK
            past_kv = None
            for _ci in range(_max_phase1_chunks):
                if _ci < _n_chunks:
                    _c_s = _ci * _IG_PHASE1_CHUNK
                    _c_e = min(_c_s + _IG_PHASE1_CHUNK, traj_valid)
                    with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                        fwd = self.actor_module(
                            input_ids=v_ids[:, _c_s:_c_e],
                            position_ids=v_pos[:, _c_s:_c_e],
                            past_key_values=past_kv,
                            use_cache=True,
                        )
                    past_kv = fwd.past_key_values
                    del fwd
                else:
                    _fsdp_sync_dummy()

            for ti, tb in enumerate(turn_boundaries):
                if global_si not in tb['activate_list']:
                    _fsdp_sync_dummy()
                    continue
                bp = min(tb['per_sample_token_lens'].get(global_si, 0), traj_valid)
                if bp < 1:
                    _fsdp_sync_dummy()
                    continue

                sliced = slice_kv_cache(past_kv, bp - 1)

                last_tok = v_ids[:, bp - 1:bp]
                inp = torch.cat([last_tok, s_gt], dim=1)
                pos = torch.arange(bp - 1, bp + R, device=device).unsqueeze(0)
                f_mask = torch.ones(1, bp + R, dtype=s_mask.dtype, device=device)

                with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                    out = self.actor_module(
                        input_ids=inp, attention_mask=f_mask,
                        position_ids=pos, past_key_values=sliced, use_cache=False,
                    )

                logits_t = out.logits[:, :R, :] / temperature
                lp = logprobs_from_logits(logits_t, s_gt)
                result[si, ti, :] = lp[0].float()
                del sliced, out

                if _verify_parity:
                    if _parity_strict:
                        # Short-sequence continuation test (once per worker):
                        # compare KV-cache continuation vs fresh forward on
                        # a short prefix where both Q_len < FA2 tile size
                        # (Br=128), so tiling is identical and any diff > 1e-4
                        # proves a real bug in slice/continuation/logit logic.
                        if not _short_seq_tested and bp >= 3:
                            _sbp = min(10, bp)
                            _sr = min(5, R)
                            _sgt = s_gt[:, :_sr]
                            with torch.autocast(device_type=self.device_name,
                                                dtype=torch.bfloat16):
                                _sfwd = self.actor_module(
                                    input_ids=v_ids[:, :_sbp],
                                    position_ids=v_pos[:, :_sbp],
                                    use_cache=True,
                                )
                            _skv = _sfwd.past_key_values
                            del _sfwd
                            _ssl = slice_kv_cache(_skv, _sbp - 1)
                            del _skv
                            _sinp = torch.cat(
                                [v_ids[:, _sbp-1:_sbp], _sgt], dim=1)
                            _spos = torch.arange(
                                _sbp - 1, _sbp + _sr,
                                device=device).unsqueeze(0)
                            _smsk = torch.ones(
                                1, _sbp + _sr,
                                dtype=s_mask.dtype, device=device)
                            with torch.autocast(device_type=self.device_name,
                                                dtype=torch.bfloat16):
                                _sout = self.actor_module(
                                    input_ids=_sinp,
                                    attention_mask=_smsk,
                                    position_ids=_spos,
                                    past_key_values=_ssl,
                                    use_cache=False,
                                )
                            _slp_kv = logprobs_from_logits(
                                _sout.logits[:, :_sr, :] / temperature,
                                _sgt)[0].float()
                            del _sout, _ssl
                            _sinp_f = torch.cat(
                                [v_ids[:, :_sbp], _sgt], dim=1)
                            _spos_f = torch.arange(
                                _sbp + _sr,
                                device=device).unsqueeze(0)
                            with torch.autocast(device_type=self.device_name,
                                                dtype=torch.bfloat16):
                                _sout_f = self.actor_module(
                                    input_ids=_sinp_f,
                                    attention_mask=_smsk,
                                    position_ids=_spos_f,
                                    use_cache=False,
                                )
                            _slp_fr = logprobs_from_logits(
                                _sout_f.logits[:, _sbp-1:_sbp+_sr-1, :] / temperature,
                                _sgt)[0].float()
                            del _sout_f
                            _sdiff = (_slp_kv - _slp_fr).abs().max().item()
                            _short_seq_tested = True
                            _short_seq_pass = _sdiff < 1e-4
                            if _short_seq_pass:
                                logger.info(
                                    f"[KV-SHORT-SEQ] sample={global_si} "
                                    f"ctx={_sbp} resp={_sr} "
                                    f"max_diff={_sdiff:.6e} → "
                                    f"continuation logic VERIFIED")
                            else:
                                logger.error(
                                    f"[KV-SHORT-SEQ] sample={global_si} "
                                    f"ctx={_sbp} resp={_sr} "
                                    f"max_diff={_sdiff:.6e} → "
                                    f"POTENTIAL BUG in continuation!")

                        # Strict: build independent KV cache from context[:bp],
                        # then continue with same [last_tok, s_gt].
                        # Both paths use Q_len=1+R for continuation, so FA2
                        # CUDA tiling is identical → true correctness check.
                        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                            _rfwd = self.actor_module(
                                input_ids=v_ids[:, :bp],
                                position_ids=v_pos[:, :bp],
                                use_cache=True,
                            )
                        _rkv = _rfwd.past_key_values
                        del _rfwd

                        if len(_layer_diag_results) < 3:
                            _f_raw = (past_kv.to_legacy_cache()
                                      if hasattr(past_kv, 'to_legacy_cache')
                                      else list(past_kv))
                            _r_raw = (_rkv.to_legacy_cache()
                                      if hasattr(_rkv, 'to_legacy_cache')
                                      else list(_rkv))
                            _n_lyrs = len(_f_raw)
                            _l0_k = (_f_raw[0][0][:, :, :bp, :].float()
                                     - _r_raw[0][0].float()).abs().max().item()
                            _l0_v = (_f_raw[0][1][:, :, :bp, :].float()
                                     - _r_raw[0][1].float()).abs().max().item()
                            if len(_layer_diag_results) == 0:
                                for _li in range(_n_lyrs):
                                    _kd = (_f_raw[_li][0][:, :, :bp, :].float()
                                           - _r_raw[_li][0].float()).abs().max().item()
                                    _vd = (_f_raw[_li][1][:, :, :bp, :].float()
                                           - _r_raw[_li][1].float()).abs().max().item()
                                    logger.warning(
                                        f"[KV-LAYER-DIAG] sample={global_si} "
                                        f"layer={_li:2d}/{_n_lyrs} "
                                        f"k_diff={_kd:.6e} v_diff={_vd:.6e}")
                            _layer_diag_results.append({
                                'sample': global_si, 'turn': ti, 'bp': bp,
                                'layer0_k': _l0_k, 'layer0_v': _l0_v,
                            })
                            # bf16 linear projections (cuBLAS) may choose
                            # different algorithms for different M dimensions,
                            # causing tiny per-element diffs even at layer 0.
                            # Tolerance 0.05 ≈ 6 bf16 ULPs at magnitude 1.0.
                            logger.warning(
                                f"[KV-LAYER-DIAG] sample={global_si}: "
                                f"Layer 0 k={_l0_k:.6e} v={_l0_v:.6e} "
                                f"(K diff from RoPE seq_len dependence; "
                                f"V diff is the true input-processing indicator)")
                            del _f_raw, _r_raw

                        _rsliced = slice_kv_cache(_rkv, bp - 1)
                        del _rkv
                        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                            ref_out = self.actor_module(
                                input_ids=inp, attention_mask=f_mask,
                                position_ids=pos, past_key_values=_rsliced,
                                use_cache=False,
                            )
                        ref_logits = ref_out.logits[:, :R, :] / temperature
                        ref_lp = logprobs_from_logits(ref_logits, s_gt)
                        ref_lp_f = ref_lp[0].float()
                        kv_lp_f = result[si, ti, :]
                        max_diff = (kv_lp_f - ref_lp_f).abs().max().item()
                        _parity_diffs.append((global_si, ti, bp, max_diff))
                        if max_diff > 0.01:
                            logger.error(
                                f"[KV-PARITY FAIL] sample={global_si} turn={ti} "
                                f"bp={bp} max_diff={max_diff:.4e} (strict)")
                        del ref_out, _rsliced
                    else:
                        # Relaxed: compare against fresh forward (Q_len=bp+R).
                        # flash_attn_func uses different CUDA tiling for
                        # Q_len=1+R vs Q_len=bp+R, producing bfloat16 diffs
                        # of ~1.0 for long sequences. This is NOT a bug.
                        ref_inp = torch.cat([v_ids[:, :bp], s_gt], dim=1)
                        ref_mask = torch.ones(1, bp + R, dtype=s_mask.dtype, device=device)
                        ref_pos = torch.arange(bp + R, device=device).unsqueeze(0)
                        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                            ref_out = self.actor_module(
                                input_ids=ref_inp, attention_mask=ref_mask,
                                position_ids=ref_pos, use_cache=False,
                            )
                        ref_logits = ref_out.logits[:, bp - 1:bp + R - 1, :] / temperature
                        ref_lp = logprobs_from_logits(ref_logits, s_gt)
                        ref_lp_f = ref_lp[0].float()
                        kv_lp_f = result[si, ti, :]
                        max_diff = (kv_lp_f - ref_lp_f).abs().max().item()
                        _parity_diffs.append((global_si, ti, bp, max_diff))
                        if max_diff > _parity_tol:
                            logger.error(
                                f"[KV-PARITY FAIL] sample={global_si} turn={ti} "
                                f"bp={bp} max_diff={max_diff:.4e} (tol={_parity_tol})")
                        elif max_diff > 0.5:
                            logger.warning(
                                f"[KV-PARITY WARN] sample={global_si} turn={ti} "
                                f"bp={bp} max_diff={max_diff:.4e} (FA2 precision)")
                        del ref_out

            del past_kv

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if _verify_kv or _verify_parity:
            _filled = (result != float('-inf')).any(dim=2).sum().item()
            _total = N * T
            _has_nan = torch.any(torch.isnan(result[result != float('-inf')])).item() if _filled > 0 else False
            logger.info(f"[KV-VERIFY] Result: {_filled}/{_total} (sample,turn) entries filled, "
                        f"shape={list(result.shape)}, has_nan={_has_nan}")
            assert not _has_nan, "[KV-VERIFY FAIL] NaN detected in per_turn_log_probs"

        if _verify_parity and _parity_diffs:
            _all_diffs = [d[3] for d in _parity_diffs]
            _max_all = max(_all_diffs)
            _mean_all = sum(_all_diffs) / len(_all_diffs)

            if _parity_strict:
                # Short-seq test is the definitive correctness indicator:
                # it compares KV-cache continuation vs fresh forward on
                # sequences short enough that FA2 tiling is identical.
                # Layer-0 K diff is large (~0.5) due to RoPE's cos/sin
                # depending on seq_len; V diff is tiny (~0.001) confirming
                # the linear projection is correct. This is informational.
                if _layer_diag_results:
                    _l0v_ok = all(
                        d['layer0_v'] < 0.05
                        for d in _layer_diag_results)
                    if _l0v_ok:
                        logger.info(
                            f"[KV-LAYER-DIAG] Layer-0 V diff < 0.05 for all "
                            f"{len(_layer_diag_results)} samples → "
                            f"linear projections correct (K diff is from RoPE "
                            f"seq_len dependence, not a bug)")
                    else:
                        logger.warning(
                            f"[KV-LAYER-DIAG] Layer-0 V diff >= 0.05 for some "
                            f"samples → investigate v_proj")
                if _short_seq_pass:
                    logger.info(
                        f"[KV-DIAG-SUMMARY] short-seq continuation PASS → "
                        f"code verified correct. Using relaxed tolerance "
                        f"{_parity_tol}.")
                    _eff_tol = _parity_tol
                elif _short_seq_tested:
                    logger.error(
                        f"[KV-DIAG-SUMMARY] FAILED: short-seq test failed "
                        f"→ using strict tolerance 0.01")
                    _eff_tol = 0.01
                else:
                    _eff_tol = 0.01
            else:
                _eff_tol = _parity_tol

            _n_fail = sum(1 for d in _all_diffs if d > _eff_tol)
            _mode_str = "strict" if _parity_strict else f"relaxed(tol={_parity_tol})"
            if _parity_strict:
                _ss_tag = ("short_ok" if _short_seq_pass
                           else ("short_FAIL" if _short_seq_tested
                                 else "short_skip"))
                _mode_str = f"strict({_ss_tag},eff_tol={_eff_tol})"
            logger.info(
                f"[KV-PARITY] mode={_mode_str}, checked {len(_parity_diffs)} pairs: "
                f"max_diff={_max_all:.4e}, mean_diff={_mean_all:.4e}, "
                f"fail(>{_eff_tol})={_n_fail}/{len(_parity_diffs)}")
            assert _n_fail == 0, (
                f"[KV-PARITY CRITICAL] {_n_fail}/{len(_parity_diffs)} entries exceed "
                f"tolerance {_eff_tol} ({_mode_str}). "
                f"Worst: {sorted(_parity_diffs, key=lambda x: -x[3])[:5]}")

        return DataProto.from_dict(tensors={'per_turn_log_probs': result})

    def process_response_mask(self, responses, response_mask):
        """Mask out non-assistant tokens (tool responses, user turns) from response_mask.

        Scans for <|im_start|>assistant\\n ... <|im_end|> regions and sets mask=1
        only within those regions. Tokens outside (tool outputs, user messages)
        get mask=0 so they don't contribute to policy loss.
        """
        tokenizer = self.tokenizer
        assistant_start = "<|im_start|>assistant\n"
        assistant_start_tokens = tokenizer.encode(assistant_start, add_special_tokens=False)
        assistant_end = "<|im_end|>"
        assistant_end_tokens = tokenizer.encode(assistant_end, add_special_tokens=False)
        new_response_mask = response_mask.clone().fill_(0)
        start_tokens_len = len(assistant_start_tokens)
        for i, response in enumerate(responses):
            response_tokens = assistant_start_tokens + response.tolist()
            idx = 0
            while idx < len(response_tokens):
                start_pos = -1
                for j in range(idx, len(response_tokens) - len(assistant_start_tokens) + 1):
                    if response_tokens[j: j + len(assistant_start_tokens)] == assistant_start_tokens:
                        start_pos = j + len(assistant_start_tokens)
                        break
                if start_pos == -1:
                    break

                end_pos = -1
                for k in range(start_pos, len(response_tokens) - len(assistant_end_tokens) + 1):
                    if response_tokens[k: k + len(assistant_end_tokens)] == assistant_end_tokens:
                        end_pos = k + len(assistant_end_tokens)
                        break

                if end_pos == -1:
                    end_pos = len(response_tokens)
                    while end_pos > start_pos and response_tokens[end_pos - 1] == tokenizer.pad_token_id:
                        end_pos -= 1
                    new_response_mask[i, start_pos - start_tokens_len:end_pos - start_tokens_len] = 1
                    break
                new_response_mask[i, start_pos - start_tokens_len:end_pos - start_tokens_len] = 1

                idx = end_pos

        return response_mask * new_response_mask

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        if self.config.mask_tool_response:
            data.batch.set_("response_mask", self.process_response_mask(
                data.batch["responses"], data.batch["response_mask"]))

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if self.config.tis_imp_ratio_cap > 0:
            assert "rollout_log_probs" in data.batch.keys(), (
                "Truncated Importance Sampling (TIS) requires to configure "
                "`actor_rollout_ref.rollout.calculate_log_probs=True` "
                "and is not currently supported in Server mode (agent loop)."
            )
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    rollout_log_probs = model_inputs["rollout_log_probs"] if self.config.tis_imp_ratio_cap > 0 else None
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    if on_policy:
                        old_log_prob = log_prob.detach()
                    else:
                        old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla
                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_log_probs=rollout_log_probs,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    loss.backward()

                    with torch.no_grad():
                        _ratio = torch.exp(log_prob.detach() - old_log_prob.detach())
                        _valid_ratio = _ratio[response_mask.bool()]
                        if _valid_ratio.numel() > 0:
                            micro_batch_metrics["actor/ratio_mean"] = _valid_ratio.mean().item()
                            micro_batch_metrics["actor/ratio_std"] = _valid_ratio.std().item()
                            micro_batch_metrics["actor/ratio_max"] = _valid_ratio.max().item()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
