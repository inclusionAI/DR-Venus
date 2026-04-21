"""
Prealigned Vectorized GT LogProb Computation

This module implements the prealigned prompt vectorization strategy for GT LogProb computation.
It is completely independent and does not affect the original computation mode.

Key Design Principles:
1. Complete decoupling: This module only processes collected data, never modifies original data flow
2. Mathematical rigor: Prealigned prompts ensure response position_ids are identical to original mode
3. Strict validation: Built-in checkpoints to verify results match original mode exactly
4. Minimal footprint: Only called when vectorized mode is enabled, zero impact otherwise

Usage:
    from scrl.llm_agent.prealigned_vectorized import compute_vectorized_gt_logprob

    results = compute_vectorized_gt_logprob(
        pseudo_outputs_per_turn=collected_outputs,
        activate_lists_per_turn=collected_activate_lists,
        gt_idx=gt_idx,
        actor_rollout_wg=self.actor_rollout_wg,
        tokenizer=self.tokenizer,
        info_gain_type=self.config.info_gain_type,
        enable_strict_validation=True,
    )
"""

import os
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import math
import copy

try:
    from verl.protocol import DataProto
except ImportError:
    from verl import DataProto


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PrealignedVectorizedConfig:
    """Configuration for prealigned vectorized computation."""
    enable_strict_validation: bool = False
    validation_tolerance: float = 1e-6
    debug_print: bool = False
    debug_samples: List[int] = None


def get_config_from_env() -> PrealignedVectorizedConfig:
    """Get configuration from environment variables."""
    return PrealignedVectorizedConfig(
        enable_strict_validation=os.environ.get('IGPO_VECTORIZED_STRICT_VALIDATION', '').lower() in ('true', '1'),
        debug_print=os.environ.get('IGPO_VECTORIZED_DEBUG', '').lower() in ('true', '1'),
    )


# ============================================================================
# Core Functions: Prealigned Prompt Processing
# ============================================================================

def prealign_single_turn(
    pseudo_output: Any,  # DataProto
    target_prompt_len: int,
    pad_token_id: int,
) -> Any:  # DataProto
    """
    Prealign a single turn's pseudo_output to target_prompt_len.

    Applies RIGHT padding to the prompt part, keeping the response at the end.
    This ensures response's position_ids are identical to the original mode.

    Original format:   [Prompt_actual][Response]
    Prealigned format: [Prompt_actual][PAD...][Response]
    """
    prompts = pseudo_output.batch['prompts']
    responses = pseudo_output.batch['responses']
    input_ids = pseudo_output.batch['input_ids']
    attention_mask = pseudo_output.batch['attention_mask']
    position_ids = pseudo_output.batch['position_ids']

    batch_size = prompts.shape[0]
    actual_prompt_len = prompts.shape[1]
    response_len = responses.shape[1]
    device = input_ids.device

    pad_len = target_prompt_len - actual_prompt_len

    if pad_len <= 0:
        return DataProto.from_dict({
            'prompts': prompts.clone(),
            'responses': responses.clone(),
            'input_ids': input_ids.clone(),
            'attention_mask': attention_mask.clone(),
            'position_ids': position_ids.clone(),
        })

    # Step 1: Prealign prompts (right padding)
    aligned_prompts = F.pad(prompts, (0, pad_len), value=pad_token_id)

    # Step 2: Rebuild input_ids = [aligned_prompts][responses]
    aligned_input_ids = torch.cat([aligned_prompts, responses], dim=1)

    # Step 3: Rebuild attention_mask
    prompt_mask = attention_mask[:, :actual_prompt_len]
    pad_mask = torch.zeros(batch_size, pad_len, dtype=attention_mask.dtype, device=device)
    response_mask = attention_mask[:, actual_prompt_len:]
    aligned_attention_mask = torch.cat([prompt_mask, pad_mask, response_mask], dim=1)

    # Step 4: Rebuild position_ids — CRITICAL: keep response position_ids unchanged
    #
    # Original:   [0, 1, ..., k, k+1, ..., k+m]
    #              ↑ prompt ↑   ↑ response ↑
    #
    # Prealigned: [0, 1, ..., k, 0, 0, ..., 0, k+1, ..., k+m]
    #              ↑ prompt ↑  ↑ PAD (ignored) ↑  ↑ response (unchanged!) ↑
    prompt_pos = position_ids[:, :actual_prompt_len]
    pad_pos = torch.zeros(batch_size, pad_len, dtype=position_ids.dtype, device=device)
    response_pos = position_ids[:, actual_prompt_len:]
    aligned_position_ids = torch.cat([prompt_pos, pad_pos, response_pos], dim=1)

    return DataProto.from_dict({
        'prompts': aligned_prompts,
        'responses': responses.clone(),
        'input_ids': aligned_input_ids,
        'attention_mask': aligned_attention_mask,
        'position_ids': aligned_position_ids,
    })


def merge_prealigned_turns(
    aligned_outputs: List[Any],  # List[DataProto]
) -> Any:  # DataProto
    """
    Merge all prealigned turns into a single batch.
    Since all turns are prealigned to the same seq_len, we can simply concatenate.
    """
    merged_input_ids = torch.cat([o.batch['input_ids'] for o in aligned_outputs], dim=0)
    merged_attention_mask = torch.cat([o.batch['attention_mask'] for o in aligned_outputs], dim=0)
    merged_position_ids = torch.cat([o.batch['position_ids'] for o in aligned_outputs], dim=0)
    merged_responses = torch.cat([o.batch['responses'] for o in aligned_outputs], dim=0)
    merged_prompts = torch.cat([o.batch['prompts'] for o in aligned_outputs], dim=0)

    return DataProto.from_dict({
        'prompts': merged_prompts,
        'responses': merged_responses,
        'input_ids': merged_input_ids,
        'attention_mask': merged_attention_mask,
        'position_ids': merged_position_ids,
    })


# ============================================================================
# Core Function: Main Entry Point
# ============================================================================

def compute_vectorized_gt_logprob(
    pseudo_outputs_per_turn: List[Any],      # List[DataProto]
    activate_lists_per_turn: List[List[int]],
    gt_idx: List[List[int]],
    actor_rollout_wg: Any,
    tokenizer: Any,
    info_gain_type: str = "prob_diff",
    enable_strict_validation: bool = False,
) -> Dict[str, Any]:
    """
    Main entry point for prealigned vectorized GT LogProb computation.

    1. Prealigns all turns' prompts to the same length
    2. Merges all turns into a single batch
    3. Calls compute_log_prob ONCE
    4. Extracts results and computes info_gain
    5. (Optional) Validates results against original mode
    """
    num_turns = len(pseudo_outputs_per_turn)
    if num_turns == 0:
        return {
            'gt_values': {},
            'info_gain_rewards': [],
            'gt_log_probs_per_turn': [],
            'gt_entropys_per_turn': [],
        }

    num_samples = pseudo_outputs_per_turn[0].batch['input_ids'].shape[0]
    pad_token_id = tokenizer.pad_token_id

    print(f"[PREALIGNED VECTORIZED] Starting: {num_turns} turns, {num_samples} samples/turn")

    # ========== Step 1: Determine max_prompt_len ==========
    max_prompt_len = 0
    response_len = pseudo_outputs_per_turn[0].batch['responses'].shape[1]

    for pseudo_output in pseudo_outputs_per_turn:
        prompt_len = pseudo_output.batch['prompts'].shape[1]
        max_prompt_len = max(max_prompt_len, prompt_len)

    max_seq_len = max_prompt_len + response_len
    print(f"[PREALIGNED VECTORIZED] max_prompt_len={max_prompt_len}, response_len={response_len}, max_seq_len={max_seq_len}")

    # ========== Step 2: Prealign all turns ==========
    aligned_outputs = []
    for turn_idx, pseudo_output in enumerate(pseudo_outputs_per_turn):
        aligned_output = prealign_single_turn(
            pseudo_output=pseudo_output,
            target_prompt_len=max_prompt_len,
            pad_token_id=pad_token_id,
        )
        aligned_outputs.append(aligned_output)

        if turn_idx < 3:
            orig_seq_len = pseudo_output.batch['input_ids'].shape[1]
            aligned_seq_len = aligned_output.batch['input_ids'].shape[1]
            print(f"[PREALIGNED] Turn {turn_idx}: original_seq_len={orig_seq_len}, aligned_seq_len={aligned_seq_len}")

    # ========== Step 3: Merge all turns ==========
    merged_batch = merge_prealigned_turns(aligned_outputs)
    total_batch_size = merged_batch.batch['input_ids'].shape[0]

    print(f"[PREALIGNED VECTORIZED] Merged: {total_batch_size} total samples (= {num_turns} turns x {num_samples})")

    # ========== Step 4: Call compute_log_prob ONCE ==========
    print(f"[PREALIGNED VECTORIZED] Calling compute_log_prob ONCE...")
    merged_log_probs_result = actor_rollout_wg.compute_log_prob(merged_batch)
    merged_old_log_probs = merged_log_probs_result.batch['old_log_probs']
    merged_entropys = merged_log_probs_result.batch['entropys']

    print(f"[PREALIGNED VECTORIZED] compute_log_prob completed, shape={merged_old_log_probs.shape}")

    # ========== Step 5: Extract results per turn ==========
    gt_values = {}
    info_gain_rewards = [[] for _ in range(num_samples)]
    gt_log_probs_per_turn = [[] for _ in range(num_samples)]
    gt_entropys_per_turn = [[] for _ in range(num_samples)]

    vectorized_mean_log_probs = [[] for _ in range(num_samples)]

    for turn_idx in range(num_turns):
        start_idx = turn_idx * num_samples
        end_idx = (turn_idx + 1) * num_samples

        turn_old_log_probs = merged_old_log_probs[start_idx:end_idx]
        turn_entropys = merged_entropys[start_idx:end_idx]
        activate_list = activate_lists_per_turn[turn_idx]

        if turn_idx == 0:
            for global_idx in activate_list:
                if gt_idx[global_idx][0] >= gt_idx[global_idx][1]:
                    continue

                log_probs = turn_old_log_probs[global_idx, gt_idx[global_idx][0]:gt_idx[global_idx][1]]
                mean_log_prob = log_probs.mean().item()

                if math.isnan(mean_log_prob) or math.isinf(mean_log_prob):
                    continue

                vectorized_mean_log_probs[global_idx].append(mean_log_prob)

                if info_gain_type == "log_prob_diff":
                    gt_values[global_idx] = mean_log_prob
                else:
                    gt_values[global_idx] = math.exp(mean_log_prob)

                gt_log_probs_per_turn[global_idx].append(log_probs.tolist())
                gt_entropys_per_turn[global_idx].append(
                    turn_entropys[global_idx, gt_idx[global_idx][0]:gt_idx[global_idx][1]].tolist()
                )
        else:
            for global_idx in activate_list:
                if gt_idx[global_idx][0] >= gt_idx[global_idx][1]:
                    continue
                if global_idx not in gt_values:
                    continue

                log_probs = turn_old_log_probs[global_idx, gt_idx[global_idx][0]:gt_idx[global_idx][1]]
                mean_log_prob = log_probs.mean().item()

                if math.isnan(mean_log_prob) or math.isinf(mean_log_prob):
                    info_gain_rewards[global_idx].append(0.0)
                    continue

                vectorized_mean_log_probs[global_idx].append(mean_log_prob)

                if info_gain_type == "log_prob_diff":
                    cur_value = mean_log_prob
                    info_gain = cur_value - gt_values[global_idx]
                else:
                    cur_value = math.exp(mean_log_prob)
                    info_gain = cur_value - gt_values[global_idx]

                if math.isnan(info_gain) or math.isinf(info_gain):
                    info_gain_rewards[global_idx].append(0.0)
                    continue

                info_gain_rewards[global_idx].append(info_gain)
                gt_values[global_idx] = cur_value

                gt_log_probs_per_turn[global_idx].append(log_probs.tolist())
                gt_entropys_per_turn[global_idx].append(
                    turn_entropys[global_idx, gt_idx[global_idx][0]:gt_idx[global_idx][1]].tolist()
                )

    total_info_gains = sum(len(r) for r in info_gain_rewards)
    print(f"[PREALIGNED VECTORIZED] COMPLETED: {num_turns} turns, {total_info_gains} info_gains, 1 compute_log_prob call")

    result = {
        'gt_values': gt_values,
        'info_gain_rewards': info_gain_rewards,
        'gt_log_probs_per_turn': gt_log_probs_per_turn,
        'gt_entropys_per_turn': gt_entropys_per_turn,
        'vectorized_mean_log_probs': vectorized_mean_log_probs,
    }

    # ========== Step 6: Strict Validation (if enabled) ==========
    if enable_strict_validation:
        print(f"[PREALIGNED VECTORIZED] Running strict validation...")
        print(f"[PREALIGNED VECTORIZED] This will make {num_turns} additional compute_log_prob calls for verification.")
        validation_result = _run_strict_validation(
            pseudo_outputs_per_turn=pseudo_outputs_per_turn,
            activate_lists_per_turn=activate_lists_per_turn,
            gt_idx=gt_idx,
            actor_rollout_wg=actor_rollout_wg,
            info_gain_type=info_gain_type,
            vectorized_mean_log_probs=vectorized_mean_log_probs,
        )
        result['validation_passed'] = validation_result['passed']
        result['validation_details'] = validation_result['details']
        result['validation_total_compared'] = validation_result['total_compared']
        result['validation_total_matched'] = validation_result['total_matched']
        result['validation_total_mismatched'] = validation_result['total_mismatched']
        result['validation_max_diff'] = validation_result['max_diff']

        if validation_result['passed']:
            print(f"[PREALIGNED VECTORIZED] Validation PASSED! Max diff: {validation_result['max_diff']:.2e}")
        else:
            print(f"[PREALIGNED VECTORIZED] Validation FAILED! Max diff: {validation_result['max_diff']:.2e}")
            print(f"[PREALIGNED VECTORIZED] Details: {validation_result['details']}")

    return result


# ============================================================================
# Validation Functions
# ============================================================================

def _run_strict_validation(
    pseudo_outputs_per_turn: List[Any],
    activate_lists_per_turn: List[List[int]],
    gt_idx: List[List[int]],
    actor_rollout_wg: Any,
    info_gain_type: str,
    vectorized_mean_log_probs: List[List[float]],
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """
    Run strict validation by computing original mode results and comparing.

    Computes GT LogProb using the original mode (one compute_log_prob call per turn)
    and compares with vectorized results.
    """
    num_turns = len(pseudo_outputs_per_turn)
    num_samples = len(vectorized_mean_log_probs)

    original_mean_log_probs = [[] for _ in range(num_samples)]
    orig_gt_values = {}

    print(f"[VALIDATION] Computing original mode results for {num_turns} turns...")

    for turn_idx, pseudo_output in enumerate(pseudo_outputs_per_turn):
        log_probs_result = actor_rollout_wg.compute_log_prob(pseudo_output)
        old_log_probs = log_probs_result.batch['old_log_probs']

        activate_list = activate_lists_per_turn[turn_idx]

        if turn_idx == 0:
            for global_idx in activate_list:
                if gt_idx[global_idx][0] >= gt_idx[global_idx][1]:
                    continue
                log_probs = old_log_probs[global_idx, gt_idx[global_idx][0]:gt_idx[global_idx][1]]
                mean_log_prob = log_probs.mean().item()
                if not math.isnan(mean_log_prob) and not math.isinf(mean_log_prob):
                    original_mean_log_probs[global_idx].append(mean_log_prob)
                    orig_gt_values[global_idx] = mean_log_prob
        else:
            for global_idx in activate_list:
                if gt_idx[global_idx][0] >= gt_idx[global_idx][1]:
                    continue
                if global_idx not in orig_gt_values:
                    continue
                log_probs = old_log_probs[global_idx, gt_idx[global_idx][0]:gt_idx[global_idx][1]]
                mean_log_prob = log_probs.mean().item()
                if not math.isnan(mean_log_prob) and not math.isinf(mean_log_prob):
                    original_mean_log_probs[global_idx].append(mean_log_prob)

    total_compared = 0
    total_matched = 0
    total_mismatched = 0
    max_diff = 0.0
    mismatch_details = []

    for sample_idx in range(num_samples):
        vec_probs = vectorized_mean_log_probs[sample_idx]
        orig_probs = original_mean_log_probs[sample_idx]

        if len(vec_probs) != len(orig_probs):
            total_mismatched += 1
            if len(mismatch_details) < 10:
                mismatch_details.append({
                    'type': 'length_mismatch',
                    'sample': sample_idx,
                    'vec_len': len(vec_probs),
                    'orig_len': len(orig_probs),
                })
            continue

        for turn_idx, (v, o) in enumerate(zip(vec_probs, orig_probs)):
            total_compared += 1
            diff = abs(v - o)
            max_diff = max(max_diff, diff)

            if diff <= tolerance:
                total_matched += 1
            else:
                total_mismatched += 1
                if len(mismatch_details) < 10:
                    mismatch_details.append({
                        'type': 'value_mismatch',
                        'sample': sample_idx,
                        'turn': turn_idx,
                        'vectorized': v,
                        'original': o,
                        'diff': diff,
                    })

    passed = (total_mismatched == 0) and (total_compared > 0)

    print(f"[VALIDATION] Compared {total_compared} values: {total_matched} matched, {total_mismatched} mismatched")
    print(f"[VALIDATION] Max absolute difference: {max_diff:.2e}")

    return {
        'passed': passed,
        'total_compared': total_compared,
        'total_matched': total_matched,
        'total_mismatched': total_mismatched,
        'max_diff': max_diff,
        'details': mismatch_details,
    }


def validate_prealignment_correctness(
    original_output: Any,  # DataProto
    aligned_output: Any,   # DataProto
    sample_idx: int = 0,
) -> Dict[str, Any]:
    """
    Validate that prealignment preserves response position_ids correctly.
    Debugging utility to verify the prealignment logic.
    """
    orig_pos = original_output.batch['position_ids'][sample_idx]
    aligned_pos = aligned_output.batch['position_ids'][sample_idx]

    orig_prompt_len = original_output.batch['prompts'].shape[1]
    aligned_prompt_len = aligned_output.batch['prompts'].shape[1]

    orig_response_pos = orig_pos[orig_prompt_len:].tolist()
    aligned_response_pos = aligned_pos[aligned_prompt_len:].tolist()

    response_pos_match = (orig_response_pos == aligned_response_pos)

    return {
        'response_position_ids_match': response_pos_match,
        'original_prompt_len': orig_prompt_len,
        'aligned_prompt_len': aligned_prompt_len,
        'original_response_pos': orig_response_pos[:10],
        'aligned_response_pos': aligned_response_pos[:10],
    }


# ============================================================================
# KV Cache Mode: Single trajectory forward + per-turn GT forward
# ============================================================================

def compute_ig_with_kv_cache(
    trajectory_input_ids: torch.Tensor,
    trajectory_attention_mask: torch.Tensor,
    trajectory_position_ids: torch.Tensor,
    gt_token_ids: List[List[int]],
    gt_idx: List[List[int]],
    turn_boundaries: List[Dict],
    actor_rollout_wg: Any,
    tokenizer: Any,
    info_gain_type: str = "prob_diff",
    num_samples: int = 1,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """
    Compute info-gain rewards using KV cache reuse via a single worker call.

    Packs trajectory + GT data into a DataProto, sends it to the worker where
    the KV cache is built and reused entirely on GPU (never crosses Ray).

    Returns info_gain_rewards (list of lists) and gt_values (dict).
    """
    num_turns = len(turn_boundaries)
    if num_turns == 0:
        return {'gt_values': {}, 'info_gain_rewards': [[] for _ in range(num_samples)]}

    pad_token_id = tokenizer.pad_token_id

    print(f"[KV-CACHE IG] Starting: {num_turns} turns, {num_samples} samples")

    gt_max_len = max(len(gt) for gt in gt_token_ids)
    gt_padded = [list(gt) + [pad_token_id] * (gt_max_len - len(gt)) for gt in gt_token_ids]
    gt_tensor = torch.tensor(gt_padded, dtype=torch.long)  # CPU — will be sent to worker

    # Include original sample indices so the worker can map local->global after DP split
    orig_indices = torch.arange(num_samples, dtype=torch.long)

    kv_data = DataProto.from_dict(
        tensors={
            'trajectory_input_ids': trajectory_input_ids,
            'trajectory_attention_mask': trajectory_attention_mask,
            'trajectory_position_ids': trajectory_position_ids,
            'gt_input_ids': gt_tensor,
            'original_sample_idx': orig_indices.unsqueeze(1),  # [N, 1] for consistent batch dim
        },
        meta_info={
            'turn_boundaries': turn_boundaries,
            'gt_idx': gt_idx,
            'temperature': temperature,
        },
    )

    result_proto = actor_rollout_wg.compute_ig_with_kv_cache(kv_data)

    # per_turn_log_probs: [N, T, R] — log P(gt[j] | context_at_turn_t)
    per_turn_lp = result_proto.batch['per_turn_log_probs']  # [N, T, R]

    info_gain_rewards = [[] for _ in range(num_samples)]
    gt_values = {}

    for si in range(num_samples):
        gs, ge = gt_idx[si]
        if gs >= ge:
            continue

        for ti in range(num_turns):
            if si not in turn_boundaries[ti]['activate_list']:
                continue

            if ti > 0 and si not in gt_values:
                continue

            lp_slice = per_turn_lp[si, ti, gs:ge]
            if torch.all(lp_slice == float('-inf')):
                if ti > 0:
                    info_gain_rewards[si].append(0.0)
                continue

            mean_lp = lp_slice.mean().item()
            if math.isnan(mean_lp) or math.isinf(mean_lp):
                if ti > 0:
                    info_gain_rewards[si].append(0.0)
                continue

            if ti == 0:
                if info_gain_type == "log_prob_diff":
                    gt_values[si] = mean_lp
                else:
                    gt_values[si] = math.exp(mean_lp)
            else:

                if info_gain_type == "log_prob_diff":
                    cur_val = mean_lp
                else:
                    cur_val = math.exp(mean_lp)

                ig = cur_val - gt_values[si]
                if math.isnan(ig) or math.isinf(ig):
                    info_gain_rewards[si].append(0.0)
                    continue

                info_gain_rewards[si].append(ig)
                gt_values[si] = cur_val

    total_ig = sum(len(r) for r in info_gain_rewards)
    print(f"[KV-CACHE IG] COMPLETED: {num_turns} turns, {total_ig} info_gains, "
          f"{num_samples} samples processed")

    return {
        'gt_values': gt_values,
        'info_gain_rewards': info_gain_rewards,
    }
