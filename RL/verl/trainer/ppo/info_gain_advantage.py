"""
Info-Gain GRPO advantage estimator.

Registers as "grpo_info_gain" via the DR advantage estimator registry.
"""

from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import register_adv_est


def _compute_turn_level_advantage(
    normalized_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
    bsz: int,
    seq_len: int,
    device: torch.device,
    turn_boundary_mask: torch.Tensor = None,
    adaptive_gamma_min_weight: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """
    Turn-level discounted accumulation + broadcast.

    When adaptive_gamma_min_weight > 0, gamma is dynamically adjusted per sample
    so that gamma^num_turns = adaptive_gamma_min_weight, ensuring early-turn
    signals retain a minimum proportion of their original magnitude regardless
    of trajectory length.

    Returns (discounted_returns, stats) where stats contains per-sample
    effective_gamma and num_turns for diagnostic logging.
    """
    discounted_returns = torch.zeros(bsz, seq_len, device=device, dtype=normalized_rewards.dtype)
    per_sample_gammas = []
    per_sample_num_turns = []

    for sample_idx in range(bsz):
        sample_rewards = normalized_rewards[sample_idx]
        sample_mask = response_mask[sample_idx]

        if turn_boundary_mask is not None:
            reward_positions = turn_boundary_mask[sample_idx].nonzero(as_tuple=True)[0].tolist()
        else:
            reward_positions = (sample_rewards != 0).nonzero(as_tuple=True)[0].tolist()

        if len(reward_positions) == 0:
            continue

        num_turns = len(reward_positions)
        effective_gamma = gamma
        if adaptive_gamma_min_weight > 0 and num_turns > 1:
            effective_gamma = adaptive_gamma_min_weight ** (1.0 / num_turns)

        per_sample_gammas.append(effective_gamma)
        per_sample_num_turns.append(num_turns)

        turn_data = []
        next_turn_adv = 0.0

        for pos in reversed(reward_positions):
            turn_reward = sample_rewards[pos].item()
            turn_adv = turn_reward + effective_gamma * next_turn_adv
            turn_data.append((pos, turn_adv))
            next_turn_adv = turn_adv

        turn_data.reverse()

        prev_end = 0
        for i, (reward_pos, adv) in enumerate(turn_data):
            discounted_returns[sample_idx, prev_end:reward_pos + 1] = adv * sample_mask[prev_end:reward_pos + 1]
            prev_end = reward_pos + 1

    stats = {"gammas": per_sample_gammas, "num_turns": per_sample_num_turns}
    return discounted_returns, stats


@register_adv_est("grpo_info_gain")
def compute_grpo_info_gain_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GRPO advantage with info-gain: F1/IG mask separation, curriculum weights, turn-level discount.
    """
    _VALID_NORM_MODES = {"joint", "separate", "scaled_separate", "raw_ig"}
    gamma = config.gamma if config else 1.0
    info_gain_norm_mode = config.info_gain_norm_mode if config else "joint"
    if info_gain_norm_mode not in _VALID_NORM_MODES:
        raise ValueError(
            f"info_gain_norm_mode={info_gain_norm_mode!r} is invalid. "
            f"Must be one of {_VALID_NORM_MODES}"
        )
    adaptive_gamma_min_weight = float(getattr(config, "adaptive_gamma_min_weight", 0.0)) if config else 0.0
    if config:
        norm_adv_by_std_in_grpo = getattr(config, "norm_adv_by_std_in_grpo", True)
    ig_weight = float(getattr(config, "ig_weight", 1.0)) if config else 1.0
    curriculum_f1_weight = 1.0
    curriculum_ig_weight = 1.0
    if config and getattr(config, "use_curriculum", False):
        curriculum_f1_weight = getattr(config, "_curriculum_f1_weight", 1.0)
        curriculum_ig_weight = getattr(config, "_curriculum_ig_weight", 1.0)

    bsz, seq_len = token_level_rewards.shape
    device = token_level_rewards.device

    with torch.no_grad():
        last_valid_pos = (seq_len - 1) - response_mask.flip(dims=[1]).to(torch.long).argmax(dim=1)

        position_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        f1_mask = (position_indices == last_valid_pos.unsqueeze(1)) & (response_mask == 1)
        ig_mask = (response_mask == 1) & (~f1_mask) & (token_level_rewards != 0)

    unique_indices, inverse_indices = np.unique(index, return_inverse=True)
    group_ids = torch.tensor(inverse_indices, device=device, dtype=torch.long)
    num_groups = len(unique_indices)

    group_ids_expanded = group_ids.unsqueeze(1).expand(-1, seq_len)

    def compute_group_stats(mask):
        flat_mask = mask.view(-1)
        flat_rewards = token_level_rewards.view(-1)
        flat_group_ids = group_ids_expanded.reshape(-1)

        valid_idx = flat_mask.nonzero(as_tuple=True)[0]
        if valid_idx.numel() == 0:
            return torch.zeros(num_groups, device=device), torch.ones(num_groups, device=device)

        valid_rewards = flat_rewards[valid_idx]
        valid_groups = flat_group_ids[valid_idx]

        group_sum = torch.zeros(num_groups, device=device).scatter_add_(0, valid_groups, valid_rewards)
        group_count = torch.zeros(num_groups, device=device).scatter_add_(0, valid_groups, torch.ones_like(valid_rewards))

        group_mean = group_sum / group_count.clamp(min=1.0)

        expanded_mean = group_mean[valid_groups]
        sq_diff = (valid_rewards - expanded_mean) ** 2
        group_sq_sum = torch.zeros(num_groups, device=device).scatter_add_(0, valid_groups, sq_diff)
        group_var = group_sq_sum / group_count.clamp(min=1.0)
        group_std = torch.sqrt(group_var + 1e-8)

        group_std = torch.where(group_count <= 1, torch.ones_like(group_std), group_std)

        return group_mean, group_std

    normalized_rewards = torch.zeros_like(token_level_rewards)

    # Track scaling diagnostics (populated by scaled_separate branch)
    _diag_ig_scale = None
    _diag_f1_abs_mean = None
    _diag_ig_abs_mean = None
    _diag_f1_std = None

    if info_gain_norm_mode == "separate":
        f1_mean, f1_std = compute_group_stats(f1_mask)
        _diag_f1_std = f1_std
        f1_mean_map = f1_mean[group_ids_expanded]
        f1_std_map = f1_std[group_ids_expanded]

        norm_f1 = (token_level_rewards - f1_mean_map)
        if norm_adv_by_std_in_grpo:
            norm_f1 = norm_f1 / (f1_std_map + epsilon)
        normalized_rewards = torch.where(f1_mask, norm_f1, normalized_rewards)

        ig_mean, ig_std = compute_group_stats(ig_mask)
        ig_mean_map = ig_mean[group_ids_expanded]
        ig_std_map = ig_std[group_ids_expanded]

        norm_ig = (token_level_rewards - ig_mean_map)
        if norm_adv_by_std_in_grpo:
            norm_ig = norm_ig / (ig_std_map + epsilon)
        normalized_rewards = torch.where(ig_mask, norm_ig, normalized_rewards)

    elif info_gain_norm_mode == "scaled_separate":
        f1_mean, f1_std = compute_group_stats(f1_mask)
        _diag_f1_std = f1_std
        f1_mean_map = f1_mean[group_ids_expanded]
        f1_std_map = f1_std[group_ids_expanded]

        norm_f1 = (token_level_rewards - f1_mean_map)
        if norm_adv_by_std_in_grpo:
            norm_f1 = norm_f1 / (f1_std_map + epsilon)
        normalized_rewards = torch.where(f1_mask, norm_f1, normalized_rewards)

        ig_mean, ig_std = compute_group_stats(ig_mask)
        ig_mean_map = ig_mean[group_ids_expanded]
        ig_std_map = ig_std[group_ids_expanded]

        norm_ig = (token_level_rewards - ig_mean_map)
        if norm_adv_by_std_in_grpo:
            norm_ig = norm_ig / (ig_std_map + epsilon)

        f1_abs_mean = normalized_rewards[f1_mask].abs().mean() if f1_mask.any() else torch.tensor(1.0, device=device)
        f1_abs_mean = torch.clamp(f1_abs_mean, min=0.3)
        ig_abs_mean = norm_ig[ig_mask].abs().mean() if ig_mask.any() else torch.tensor(1.0, device=device)
        ig_scale = f1_abs_mean / (ig_abs_mean + epsilon)
        ig_scale = ig_scale.clamp(max=10.0)
        _diag_ig_scale = ig_scale.item()
        _diag_f1_abs_mean = f1_abs_mean.item()
        _diag_ig_abs_mean = ig_abs_mean.item()
        norm_ig = norm_ig * ig_scale
        normalized_rewards = torch.where(ig_mask, norm_ig, normalized_rewards)

    elif info_gain_norm_mode == "raw_ig":
        # F1: standard GRPO (mean subtracted + std divided)
        f1_mean, f1_std = compute_group_stats(f1_mask)
        _diag_f1_std = f1_std
        f1_mean_map = f1_mean[group_ids_expanded]
        f1_std_map = f1_std[group_ids_expanded]

        norm_f1 = (token_level_rewards - f1_mean_map)
        if norm_adv_by_std_in_grpo:
            norm_f1 = norm_f1 / (f1_std_map + epsilon)
        normalized_rewards = torch.where(f1_mask, norm_f1, normalized_rewards)

        # IG: only divide by std (no mean subtraction)
        _ig_mean, ig_std = compute_group_stats(ig_mask)
        ig_std_map = ig_std[group_ids_expanded]

        norm_ig = token_level_rewards / (ig_std_map + epsilon)

        # Apply ig_weight to IG component before F1 broadcast
        if ig_weight != 1.0:
            norm_ig = norm_ig * ig_weight
        normalized_rewards = torch.where(ig_mask, norm_ig, normalized_rewards)

        # Broadcast F1_norm to all IG positions (uniform outcome credit)
        f1_per_sample = (normalized_rewards * f1_mask.float()).sum(dim=1, keepdim=True)
        normalized_rewards = normalized_rewards + f1_per_sample * ig_mask.float()

        # Force gamma=0: no inter-turn accumulation
        gamma = 0.0
        adaptive_gamma_min_weight = 0.0

    else:  # joint
        f1_mean, f1_std = compute_group_stats(f1_mask)
        _diag_f1_std = f1_std
        joint_mask = f1_mask | ig_mask
        g_mean, g_std = compute_group_stats(joint_mask)
        mean_map = g_mean[group_ids_expanded]
        std_map = g_std[group_ids_expanded]

        norm_val = (token_level_rewards - mean_map)
        if norm_adv_by_std_in_grpo:
            norm_val = norm_val / (std_map + epsilon)
        normalized_rewards = torch.where(joint_mask, norm_val, normalized_rewards)

    if info_gain_norm_mode != "raw_ig" and ig_weight != 1.0:
        normalized_rewards = torch.where(
            ig_mask, normalized_rewards * ig_weight, normalized_rewards)

    if curriculum_f1_weight != 1.0 or curriculum_ig_weight != 1.0:
        normalized_rewards = torch.where(
            f1_mask, normalized_rewards * curriculum_f1_weight, normalized_rewards)
        normalized_rewards = torch.where(
            ig_mask, normalized_rewards * curriculum_ig_weight, normalized_rewards)

    turn_boundary_mask = f1_mask | ig_mask

    ig_active_samples = (ig_mask.any(dim=1)).sum().item()
    ig_total_positions = ig_mask.sum().item()
    nonzero_reward_positions = ((token_level_rewards != 0) & (~f1_mask)).sum().item()
    _warmup_gate_off = bool(getattr(config, '_ig_warmup_gate_off', False)) if config else False
    if ig_active_samples == 0:
        if _warmup_gate_off:
            pass  # IG warmup gate is off — pure GRPO behavior is intentional
        else:
            msg = (f"[IGPO-CRITICAL] ig_mask is entirely empty — "
                   f"grpo_info_gain has DEGRADED to standard GRPO for this batch "
                   f"(bsz={bsz}, f1_positions={f1_mask.sum().item()}, "
                   f"nonzero_non_f1_rewards={nonzero_reward_positions})")
            if nonzero_reward_positions > 0:
                raise RuntimeError(
                    f"{msg}. IG rewards exist at {nonzero_reward_positions} positions "
                    f"but response_mask is 0 there — likely response_mask was not "
                    f"set to all-1s before compute_advantage (see ray_trainer.py "
                    f"compute_response_mask).")
            print(msg)

    discounted_returns, turn_stats = _compute_turn_level_advantage(
        normalized_rewards=normalized_rewards,
        response_mask=response_mask,
        gamma=gamma,
        bsz=bsz,
        seq_len=seq_len,
        device=device,
        turn_boundary_mask=turn_boundary_mask,
        adaptive_gamma_min_weight=adaptive_gamma_min_weight,
    )

    # ── Build diagnostic metrics ──
    adv_metrics = {
        "igpo/ig_active_samples": ig_active_samples,
        "igpo/ig_active_ratio": ig_active_samples / max(bsz, 1),
        "igpo/ig_total_positions": ig_total_positions,
        "igpo/f1_total_positions": f1_mask.sum().item(),
    }

    # Raw IG value distribution (pre-normalization)
    if ig_mask.any():
        raw_ig = token_level_rewards[ig_mask]
        adv_metrics["igpo/raw_ig_mean"] = raw_ig.mean().item()
        adv_metrics["igpo/raw_ig_std"] = raw_ig.std().item() if raw_ig.numel() > 1 else 0.0
        adv_metrics["igpo/raw_ig_positive_ratio"] = (raw_ig > 0).float().mean().item()
        adv_metrics["igpo/raw_ig_negative_count"] = int((raw_ig < 0).sum().item())
        adv_metrics["igpo/raw_ig_abs_mean"] = raw_ig.abs().mean().item()

        if info_gain_norm_mode == "raw_ig":
            adv_metrics["igpo/ig_mean_over_std"] = (
                adv_metrics["igpo/raw_ig_mean"] / max(adv_metrics["igpo/raw_ig_std"], 1e-8)
            )

    # Outcome value at last valid position (includes answer format penalty if active)
    if f1_mask.any():
        raw_f1 = token_level_rewards[f1_mask]
        adv_metrics["igpo/outcome_reward_mean"] = raw_f1.mean().item()

    # scaled_separate specific: ig_scale and its inputs
    if _diag_ig_scale is not None:
        adv_metrics["igpo/ig_scale"] = _diag_ig_scale
        adv_metrics["igpo/outcome_norm_abs_mean"] = _diag_f1_abs_mean
        adv_metrics["igpo/ig_norm_abs_mean"] = _diag_ig_abs_mean

    # Group outcome diversity: fraction of groups where all outcomes are identical
    # (f1_std ≈ 0 means no variance → all samples in the group have the same outcome)
    if _diag_f1_std is not None:
        f1_count = torch.zeros(num_groups, device=device)
        if f1_mask.any():
            flat_f1_mask = f1_mask.view(-1)
            flat_group_ids = group_ids_expanded.reshape(-1)
            valid_f1_idx = flat_f1_mask.nonzero(as_tuple=True)[0]
            if valid_f1_idx.numel() > 0:
                f1_count.scatter_add_(0, flat_group_ids[valid_f1_idx],
                                      torch.ones(valid_f1_idx.numel(), device=device))
        groups_with_f1 = (f1_count > 1).sum().item()
        all_same_groups = ((f1_count > 1) & (_diag_f1_std < 1e-6)).sum().item()
        adv_metrics["igpo/all_same_outcome_ratio"] = all_same_groups / max(groups_with_f1, 1)
        adv_metrics["igpo/num_groups"] = num_groups

    # Adaptive gamma & trajectory length statistics
    gammas = turn_stats["gammas"]
    num_turns_list = turn_stats["num_turns"]
    if gammas:
        adv_metrics["igpo/effective_gamma_mean"] = sum(gammas) / len(gammas)
        adv_metrics["igpo/effective_gamma_min"] = min(gammas)
        adv_metrics["igpo/effective_gamma_max"] = max(gammas)
    if num_turns_list:
        adv_metrics["igpo/num_turns_mean"] = sum(num_turns_list) / len(num_turns_list)
        adv_metrics["igpo/num_turns_min"] = min(num_turns_list)
        adv_metrics["igpo/num_turns_max"] = max(num_turns_list)

    # Final advantage magnitude (after discounting)
    active_adv = discounted_returns[response_mask == 1]
    if active_adv.numel() > 0:
        adv_metrics["igpo/adv_abs_mean"] = active_adv.abs().mean().item()
        adv_metrics["igpo/adv_std"] = active_adv.std().item() if active_adv.numel() > 1 else 0.0
        adv_metrics["igpo/adv_positive_ratio"] = (active_adv > 0).float().mean().item()

    # Advantage decomposition by position type
    if ig_mask.any():
        ig_adv = discounted_returns[ig_mask]
        adv_metrics["igpo/adv_ig_positions_mean"] = ig_adv.mean().item()
    if f1_mask.any():
        f1_adv = discounted_returns[f1_mask]
        adv_metrics["igpo/adv_f1_positions_mean"] = f1_adv.mean().item()

    return discounted_returns, discounted_returns, adv_metrics
