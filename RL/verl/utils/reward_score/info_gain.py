"""
Token-level reward scoring with info-gain support.
"""

import os
import re
import string
import json
from collections import Counter


def check_tags_balance(solution_str: str) -> bool:
    tags_to_check = ['code', 'tool_call', 'think', 'answer']
    for tag in tags_to_check:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start_count = solution_str.count(start_tag)
        end_count = solution_str.count(end_tag)
        if start_count != end_count:
            return False
        last_pos = -1
        while True:
            start_pos = solution_str.find(start_tag, last_pos + 1)
            if start_pos == -1:
                break
            end_pos = solution_str.find(end_tag, start_pos)
            if end_pos == -1:
                return False
            last_pos = end_pos
    return True


def preprocess_text(text: str) -> str:
    for punct in string.punctuation:
        text = text.replace(punct, ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def compute_f1(solution_str, ground_truth, data_source, val_type='f1') -> float:
    solution_str = solution_str.lower()
    ground_truth = ground_truth.lower()
    ground_truths = ground_truth.split("<|answer_split|>")
    if not check_tags_balance(solution_str):
        if val_type == 'noformatf1':
            return 0
        else:
            return -2.0

    try:
        answer_match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            answer_content = preprocess_text(answer_content)
        else:
            if val_type == 'noformatf1':
                return 0
            else:
                return -2.0
    except Exception as e:
        print(f"Error extracting answer content: {e}")
        if val_type == 'noformatf1':
            return 0
        else:
            return -2.0

    max_score = 0.0
    for gt in ground_truths:
        gt = preprocess_text(gt)
        if val_type == 'em':
            if gt == answer_content:
                return 1.0
        else:
            pred_tokens = Counter(answer_content.split())
            gt_tokens = Counter(gt.split())
            if not gt_tokens:
                continue
            if not pred_tokens:
                continue
            common_tokens = sum((pred_tokens & gt_tokens).values())
            num_pred = sum(pred_tokens.values())
            num_gt = sum(gt_tokens.values())
            precision = common_tokens / num_pred if num_pred else 0
            recall = common_tokens / num_gt if num_gt else 0
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                max_score = max(max_score, f1)
    return max_score


def _char_pos_to_token_idx(char_pos, offset_mapping):
    for i, (start, end) in enumerate(offset_mapping):
        if start <= char_pos < end:
            return i
        if char_pos < start:
            return max(0, i - 1)
    return len(offset_mapping) - 1


def _find_turn_boundaries_by_tokens(token_ids, tokenizer):
    """Find non-final turn boundary positions using special token anchors.

    Searches for ``<|im_end|>`` tokens that are followed (within a small
    window) by ``<|im_start|>assistant``, indicating a turn transition.
    Returns a list of token positions (indices into *token_ids*) where
    ``<|im_end|>`` marks the end of a non-final turn.

    This operates entirely in the **original token space** and is immune
    to BPE context-dependency issues because ``<|im_start|>`` / ``<|im_end|>``
    are never-split special tokens with fixed IDs.
    """
    try:
        IM_START = tokenizer.convert_tokens_to_ids("<|im_start|>")
        IM_END = tokenizer.convert_tokens_to_ids("<|im_end|>")
    except Exception:
        return []

    boundaries = []
    n = len(token_ids)
    for pos, tid in enumerate(token_ids):
        if tid != IM_END:
            continue
        for nxt in range(pos + 1, min(pos + 5, n)):
            if token_ids[nxt] == IM_START:
                role_toks = token_ids[nxt + 1:nxt + 4]
                if role_toks:
                    role_text = tokenizer.decode(
                        role_toks, skip_special_tokens=False)
                    if role_text.lstrip().startswith("assistant"):
                        boundaries.append(pos)
                break
    return boundaries


_crosscheck_cache = {}


def _get_crosscheck_state():
    """Lazy-init crosscheck state from env var IGPO_CROSSCHECK.

    IGPO_CROSSCHECK controls dual-path cross-validation:
      unset / "0"  → disabled
      "1" / "true" → check first 200 samples (default when IGPO active)
      N (int)      → check first N samples
      "all"        → check all samples (expensive)
    """
    if "state" not in _crosscheck_cache:
        import os
        env = os.environ.get("IGPO_CROSSCHECK", "").strip().lower()
        if env in ("0", "false", ""):
            _crosscheck_cache["state"] = {
                "enabled": False, "remaining": 0,
                "match": 0, "diverge": 0, "bpe_drift": 0,
            }
        elif env in ("all", "always"):
            _crosscheck_cache["state"] = {
                "enabled": True, "remaining": float("inf"),
                "match": 0, "diverge": 0, "bpe_drift": 0,
            }
        else:
            try:
                n = int(env)
            except ValueError:
                n = 200
            _crosscheck_cache["state"] = {
                "enabled": True, "remaining": n,
                "match": 0, "diverge": 0, "bpe_drift": 0,
            }
    return _crosscheck_cache["state"]


def _compute_legacy_path(solution_str, tokenizer, info_gain_reward,
                         f1_score, alpha):
    """Run the legacy decode→reencode path for cross-validation."""
    encoding = tokenizer(solution_str, return_offsets_mapping=True,
                         add_special_tokens=False)
    token_ids = encoding['input_ids']
    offset_mapping = encoding['offset_mapping']
    tokens_size = len(token_ids)
    if tokens_size == 0:
        return None

    scores = [0.0] * tokens_size
    separator = "\n<|im_start|>assistant\n"

    turn_start_positions = []
    turn_end_positions = []
    sep_positions = []
    search_pos = 0
    while True:
        sep_pos = solution_str.find(separator, search_pos)
        if sep_pos == -1:
            break
        sep_positions.append(sep_pos)
        search_pos = sep_pos + 1

    if len(sep_positions) == 0:
        turn_start_positions = [0]
        turn_end_positions = [len(solution_str)]
    else:
        if sep_positions[0] > 0:
            turn_start_positions.append(0)
            turn_end_positions.append(sep_positions[0])
        for i, sep_pos in enumerate(sep_positions):
            turn_start = sep_pos + len(separator)
            turn_start_positions.append(turn_start)
            if i + 1 < len(sep_positions):
                turn_end = sep_positions[i + 1]
            else:
                turn_end = len(solution_str)
            turn_end_positions.append(turn_end)

    chats_size = len(turn_start_positions)
    if len(info_gain_reward) == 0 or chats_size == 1:
        scores[-1] = alpha * f1_score
        return scores

    ig_work = list(info_gain_reward)
    if len(ig_work) > chats_size - 1:
        ig_work = ig_work[:chats_size - 1]

    ig_diminish_rate = float(os.environ.get("IGPO_IG_DIMINISH_RATE", "0"))
    for i in range(chats_size):
        turn_end_char = turn_end_positions[i]
        last_token_idx = _char_pos_to_token_idx(
            turn_end_char - 1, offset_mapping) if turn_end_char > 0 else 0
        last_token_idx = min(last_token_idx, tokens_size - 1)
        if i < chats_size - 1:
            if i < len(ig_work):
                ig_value = ig_work[i]
                if ig_value is None:
                    continue
                if ig_value == 0.0:
                    ig_value = 1e-10
                if ig_diminish_rate > 0 and ig_value > 0:
                    ig_value = ig_value / (1.0 + i * ig_diminish_rate)
                scores[last_token_idx] = ig_value
        else:
            scores[last_token_idx] = alpha * f1_score
    return scores


def compute_score(solution_str, ground_truth, data_source, val_type='f1',
                  info_gain_reward=None, tokenizer=None, is_validation=False,
                  outcome_score=None, original_token_ids=None):
    """
    Compute token-level reward scores with info-gain mapping.

    Args:
        outcome_score: If provided, use this as the outcome reward at the last
            turn instead of computing F1 internally.  This allows callers
            (e.g. NaiveBatchRewardManager) to supply an externally computed
            score such as an LLM-judge result.
        original_token_ids: If provided, use these token IDs directly to find
            turn boundaries via special-token anchors (<|im_end|> / <|im_start|>),
            completely bypassing the decode→reencode roundtrip and eliminating
            BPE context-dependency issues.
    """
    if tokenizer is None:
        raise ValueError("tokenizer cannot be None")
    if info_gain_reward is None:
        info_gain_reward = []

    alpha = 1.0

    if outcome_score is not None:
        f1_score = outcome_score
        if is_validation:
            em_score = compute_f1(solution_str, ground_truth, data_source, val_type='em')
            noformatf1_score = compute_f1(solution_str, ground_truth, data_source, val_type='noformatf1')
    elif is_validation:
        f1_score = compute_f1(solution_str, ground_truth, data_source, val_type='f1')
        em_score = compute_f1(solution_str, ground_truth, data_source, val_type='em')
        noformatf1_score = compute_f1(solution_str, ground_truth, data_source, val_type='noformatf1')
    else:
        f1_score = compute_f1(solution_str, ground_truth, data_source, val_type)

    # ── Primary path: use original token IDs with special-token anchors ──
    if original_token_ids is not None:
        tokens_size = len(original_token_ids)
        scores = [0.0] * tokens_size

        if tokens_size == 0:
            if is_validation:
                return {"f1": f1_score, "em": em_score, "noformatf1": noformatf1_score, "scores": scores}
            return scores

        turn_ends = _find_turn_boundaries_by_tokens(
            original_token_ids, tokenizer)
        chats_size = len(turn_ends) + 1

        if len(info_gain_reward) == 0 or chats_size == 1:
            scores[-1] = alpha * f1_score
            if is_validation:
                return {"f1": f1_score, "em": em_score, "noformatf1": noformatf1_score, "scores": scores}
            return scores

        ig_reward_work = list(info_gain_reward)
        if len(ig_reward_work) > chats_size - 1:
            print(f"info_gain.py: IG list too long (truncating) - chats_size={chats_size}, "
                  f"info_gain_len={len(ig_reward_work)}")
            ig_reward_work = ig_reward_work[:chats_size - 1]

        ig_diminish_rate = float(os.environ.get("IGPO_IG_DIMINISH_RATE", "0"))
        for i, end_pos in enumerate(turn_ends):
            if i < len(ig_reward_work):
                ig_value = ig_reward_work[i]
                if ig_value is None:
                    continue
                if ig_value == 0.0:
                    ig_value = 1e-10
                if ig_diminish_rate > 0 and ig_value > 0:
                    ig_value = ig_value / (1.0 + i * ig_diminish_rate)
                scores[end_pos] = ig_value

        scores[-1] = alpha * f1_score

        # ── Dual-path cross-validation ──
        # When IGPO_CROSSCHECK is enabled, also run the legacy path and compare.
        # If both agree → high confidence in primary path correctness.
        # If they disagree → BPE drift detected; primary path is correct by
        # construction (operates in original token space).
        _crosscheck_state = _get_crosscheck_state()
        if _crosscheck_state["enabled"] and _crosscheck_state["remaining"] > 0:
            _crosscheck_state["remaining"] -= 1
            try:
                legacy = _compute_legacy_path(
                    solution_str, tokenizer, info_gain_reward,
                    f1_score, alpha)
                if legacy is not None:
                    enc_len = len(legacy)
                    if enc_len == tokens_size:
                        primary_nz = {i: v for i, v in enumerate(scores) if v != 0.0}
                        legacy_nz = {i: v for i, v in enumerate(legacy) if v != 0.0}
                        if primary_nz == legacy_nz:
                            _crosscheck_state["match"] += 1
                        else:
                            _crosscheck_state["diverge"] += 1
                            if _crosscheck_state["diverge"] <= 3:
                                print(
                                    f"[IGPO-CROSSCHECK] Primary≠Legacy "
                                    f"(enc_len={enc_len}=orig_len OK). "
                                    f"primary_nz={primary_nz}, "
                                    f"legacy_nz={legacy_nz}")
                    else:
                        _crosscheck_state["bpe_drift"] += 1
                        if _crosscheck_state["bpe_drift"] <= 3:
                            print(
                                f"[IGPO-CROSSCHECK] BPE drift detected: "
                                f"orig_len={tokens_size} vs enc_len={enc_len}. "
                                f"Primary path used (correct by construction).")
                    total = (_crosscheck_state["match"]
                             + _crosscheck_state["diverge"]
                             + _crosscheck_state["bpe_drift"])
                    if _crosscheck_state["remaining"] == 0:
                        print(
                            f"[IGPO-CROSSCHECK] Summary: "
                            f"{_crosscheck_state['match']} match, "
                            f"{_crosscheck_state['diverge']} diverge, "
                            f"{_crosscheck_state['bpe_drift']} BPE-drift "
                            f"(total={total})")
            except Exception as e:
                if _crosscheck_state.get("_err_count", 0) < 2:
                    print(f"[IGPO-CROSSCHECK] Error in legacy path: {e}")
                    _crosscheck_state["_err_count"] = \
                        _crosscheck_state.get("_err_count", 0) + 1

        if is_validation:
            return {"f1": f1_score, "em": em_score, "noformatf1": noformatf1_score, "scores": scores}
        return scores

    # ── Legacy path: decode→reencode (for callers without token IDs) ──
    scores = _compute_legacy_path(
        solution_str, tokenizer, info_gain_reward, f1_score, alpha)
    if scores is None:
        scores = []
    if is_validation:
        return {"f1": f1_score, "em": em_score, "noformatf1": noformatf1_score, "scores": scores}
    return scores
