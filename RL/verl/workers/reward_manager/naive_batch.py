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

from collections import defaultdict

import torch
import numpy as np
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from verl.workers.reward_manager import register


@register("naive_batch")
class NaiveBatchRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **reward_kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.deepthink_disabled = reward_kwargs.get("deepthink_disabled", True)

    def __call__(self, data: DataProto, return_dict=False, default_batchsize=16,
                 val_type='llm', info_gain_rewards=None):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score AND no IG rewards needed, short-circuit.
        # When IG rewards are provided, we must proceed to Phase 2 even if
        # rm_scores exist (rollout-time RewardManagerWorker computes rm_scores
        # without IG data; the IG overlay happens here).
        _has_rm_scores = "rm_scores" in data.batch.keys()
        if _has_rm_scores and info_gain_rewards is None:
            rm_scores = data.batch["rm_scores"]
            if return_dict:
                _prompt_len = data.batch["prompts"].size(1)
                outcome_scores = []
                for i in range(len(data)):
                    resp_valid = int(data.batch["attention_mask"][i, _prompt_len:].sum().item())
                    idx = max(0, resp_valid - 1)
                    outcome_scores.append(rm_scores[i, idx].item())
                return {
                    "reward_tensor": rm_scores,
                    "reward_extra_info": {"outcome_score": outcome_scores},
                }
            else:
                return rm_scores

        if _has_rm_scores:
            reward_tensor = data.batch["rm_scores"].clone().to(torch.float32)
        else:
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        batch_size = min(default_batchsize, len(data))  # limit batch size in case of smaller data

        # Split data into chunks according to batch_size
        batched_inputs = []
        for start_idx in range(0, len(data), batch_size):
            if start_idx + batch_size < len(data):
                batch = data[start_idx:start_idx + batch_size]
            else:
                batch = data[start_idx:]

            prompts, responses, ground_truths, data_sources, extra_infos = [], [], [], [], []
            valid_response_lengths = []

            # Process each item in the batch
            for data_item in batch:
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                # Decode prompts and responses
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

                valid_response_lengths.append(valid_response_length)
                
                prompts.append(prompt_str)
                responses.append(response_str)
                ground_truths.append(data_item.non_tensor_batch["reward_model"]["ground_truth"])
                data_sources.append(data_item.non_tensor_batch.get(self.reward_fn_key, "unknown"))
                extra_infos.append(data_item.non_tensor_batch.get("extra_info", None))

            # Collect inputs for compute_score for the current batch
            inputs = {
                "valid_response_lengths": valid_response_lengths,
                "prompts": prompts,
                "responses": responses,
                "ground_truths": ground_truths,
                "data_sources": data_sources,
                "extra_infos": extra_infos,
                "start_idx": start_idx,
            }
            batched_inputs.append(inputs)

        response_idx_dict = {}

        if _has_rm_scores:
            # rm_scores already contain Phase-1 outcome scores from the
            # rollout-time RewardManagerWorker.  Derive response_idx_dict
            # from the attention mask instead of re-running Phase 1.
            _prompt_len = data.batch["prompts"].size(1)
            for i in range(len(data)):
                _resp_valid = int(
                    data.batch["attention_mask"][i, _prompt_len:].sum().item())
                response_idx_dict[i] = max(0, _resp_valid - 1)
            print(
                f"[IGPO-Phase1] Skipped Phase-1 scoring: rm_scores present "
                f"(bsz={len(data)}), proceeding to Phase-2 IG overlay",
                flush=True)
        else:
            # Phase 1: compute scores via self.compute_score
            for inputs in batched_inputs:
                scores = self.compute_score(
                    data_source=inputs["data_sources"],
                    prompt_str=inputs["prompts"],
                    solution_str=inputs["responses"],
                    ground_truth=inputs["ground_truths"],
                    extra_info=inputs["extra_infos"],
                    val_type=val_type,
                    batch_size=batch_size,
                    is_valid = data.meta_info.get("validate", False),
                    tokenizer = self.tokenizer

                )
                start_idx = inputs['start_idx']
                # Map back scores to the reward tensor
                for idx, score in enumerate(scores):
                    if type(score) != float and type(score) != int:
                        score_size = len(score)
                        response_idx = inputs["valid_response_lengths"][idx] - 1
                        input_ids = self.tokenizer(inputs["prompts"][idx], add_special_tokens=False)['input_ids']
                        assert len(score) == response_idx + 1
                        reward_tensor[start_idx + idx, :response_idx+1] = torch.tensor(
                            score, 
                            dtype=reward_tensor.dtype,     
                            device=reward_tensor.device    
                        )
                    else:
                        response_idx = inputs["valid_response_lengths"][idx] - 1  # last token
                        reward_tensor[start_idx + idx, response_idx] = score
                    response_idx_dict[start_idx + idx] = response_idx

                    data_source = inputs["data_sources"][idx]
                    if data_source not in already_print_data_sources:
                        already_print_data_sources[data_source] = 0

                    if already_print_data_sources[data_source] < self.num_examine:
                        already_print_data_sources[data_source] += 1
                        print("[prompt]", inputs["prompts"][idx])
                        print("[response]", inputs["responses"][idx])
                        print("[data_source]", data_source, "[ground_truth]", inputs["ground_truths"][idx])
                        if isinstance(score, dict):
                            for key, value in score.items():
                                print(f"[{key}]", value)
                        else:
                            print("[score]", score)

        # Save per-sample outcome scores before IG overwrites reward_tensor.
        outcome_scores = []
        for i in range(len(data)):
            if i in response_idx_dict:
                outcome_scores.append(reward_tensor[i, response_idx_dict[i]].item())
            else:
                outcome_scores.append(0.0)
        reward_extra_info["outcome_score"] = outcome_scores

        # ── Info-gain post-processing ──
        # For samples with info-gain rewards, recompute token-level rewards
        # via info_gain.compute_score.  By default info_gain computes its own
        # F1 as the outcome reward (for exact parity with NaiveRewardManager).
        # When use_llm_outcome is True, the Phase-1 score (e.g. LLM-judge)
        # is forwarded as outcome_score so the last turn uses it instead.
        if info_gain_rewards is not None:
            from verl.utils.reward_score import info_gain as ig_module
            use_llm_outcome = (val_type == 'llm')
            _answer_fmt_pen = data.non_tensor_batch.get('answer_format_penalty', None)
            _ig_processed = 0
            _ig_skipped = 0
            _ig_placed_total = 0
            _ig_first_detail = None
            for inputs in batched_inputs:
                start_idx = inputs["start_idx"]
                for idx in range(len(inputs["responses"])):
                    global_idx = start_idx + idx
                    ig_reward = info_gain_rewards[global_idx]
                    if ig_reward is None or len(ig_reward) == 0:
                        _ig_skipped += 1
                        # Even without IG data, apply answer format penalty
                        # directly to reward_tensor (e.g. single-turn trajectory).
                        if (_answer_fmt_pen is not None
                                and global_idx < len(_answer_fmt_pen)
                                and _answer_fmt_pen[global_idx] < 0):
                            _resp_idx = response_idx_dict.get(global_idx)
                            if _resp_idx is not None:
                                reward_tensor[global_idx, _resp_idx] = float(
                                    _answer_fmt_pen[global_idx])
                        continue
                    _ig_processed += 1
                    ig_kwargs = dict(
                        val_type=val_type,
                        info_gain_reward=ig_reward,
                        tokenizer=self.tokenizer,
                        is_validation=False,
                    )
                    if use_llm_outcome:
                        response_idx = response_idx_dict[global_idx]
                        ig_kwargs["outcome_score"] = reward_tensor[
                            global_idx, response_idx
                        ].item()
                    if (_answer_fmt_pen is not None
                            and global_idx < len(_answer_fmt_pen)
                            and _answer_fmt_pen[global_idx] < 0):
                        ig_kwargs["outcome_score"] = float(_answer_fmt_pen[global_idx])
                    orig_len = int(inputs["valid_response_lengths"][idx])
                    resp_tensor = data.batch["responses"][global_idx]
                    orig_ids = resp_tensor[:orig_len].tolist()
                    ig_kwargs["original_token_ids"] = orig_ids

                    token_scores = ig_module.compute_score(
                        inputs["responses"][idx],
                        inputs["ground_truths"][idx],
                        inputs["data_sources"][idx],
                        **ig_kwargs,
                    )

                    _nz_before = int((reward_tensor[global_idx] != 0).sum().item())
                    reward_tensor[global_idx, :] = 0.0
                    reward_tensor[global_idx, :orig_len] = torch.tensor(
                        token_scores[:orig_len],
                        dtype=reward_tensor.dtype,
                        device=reward_tensor.device,
                    )
                    _nz_after = int((reward_tensor[global_idx] != 0).sum().item())
                    _ig_placed_total += _nz_after

                    if _ig_first_detail is None and _ig_processed <= 3:
                        _nz_positions = [
                            i for i, v in enumerate(token_scores[:orig_len])
                            if v != 0.0
                        ]
                        _ig_first_detail = (
                            f"sample={global_idx} ig_len={len(ig_reward)} "
                            f"orig_len={orig_len} scores_len={len(token_scores)} "
                            f"nz_positions={_nz_positions[:10]} "
                            f"nz_before={_nz_before} nz_after={_nz_after} "
                            f"has_original_token_ids=True "
                            f"reward_tensor_shape={list(reward_tensor.shape)}"
                        )
            print(
                f"[IGPO-Phase2] IG post-processing: "
                f"processed={_ig_processed} skipped={_ig_skipped} "
                f"total_nz_placed={_ig_placed_total} "
                f"reward_tensor_id={id(reward_tensor)} "
                f"detail=[{_ig_first_detail}]",
                flush=True)
        else:
            if not getattr(self, '_phase2_skip_logged', False):
                print(
                    f"[IGPO-Phase2] info_gain_rewards is None — Phase 2 SKIPPED "
                    f"(this message will not repeat)",
                    flush=True)
                self._phase2_skip_logged = True

        if not self.deepthink_disabled:
            reward_dict = {}

            # First, collect rewards from final-answer turns.
            for i in range(len(data.non_tensor_batch['agent_grpo_idx_deepthink'])):
                ids = data.non_tensor_batch['agent_grpo_idx_deepthink'][i].split('||')
                if 'answer' in data.non_tensor_batch['agent_grpo_idx_deepthink'][i]:
                    for grpo_id in ids[:1]:
                        reward_dict[grpo_id] = reward_dict.get(grpo_id,[])
                        response_idx = response_idx_dict[i]
                        reward_dict[grpo_id].append(reward_tensor[i,response_idx])

            # Propagate rewards backward through turns until every sample has a reward.
            agent_grpo_idx_deepthink_full = []
            for i in range(len(data.non_tensor_batch['agent_grpo_idx_deepthink'])):
                ids = data.non_tensor_batch['agent_grpo_idx_deepthink'][i].split('||')
                next_node = ids[0]
                for pre_node in ids[1:]:
                    if int(next_node.split('_')[2]) <= int(pre_node.split('_')[2]) + 1:
                        agent_grpo_idx_deepthink_full.append((next_node,pre_node))
            agent_grpo_idx_deepthink_full = sorted(agent_grpo_idx_deepthink_full,key = lambda x:-int(x[0].split('_')[2]))

            for next_node, pre_node in agent_grpo_idx_deepthink_full:
                reward_dict[pre_node] = reward_dict.get(pre_node,[])
                reward_dict[pre_node].append(float(np.mean(reward_dict.get(next_node,[0.0]))))

            # Final-answer trajectories are grouped by sample-level uid; all other
            # turns are grouped by (sample_uid, turn_idx).
            data.non_tensor_batch['uid'] = np.array([x.split('||')[0].split('_')[0] if '_answer' in x else x.split('||')[0].split('_')[0] + '_' + x.split('||')[0].split('_')[2] for x in data.non_tensor_batch['agent_grpo_idx_deepthink']])
            for i in range(len(data.non_tensor_batch['agent_grpo_idx_deepthink'])):
                last_id = data.non_tensor_batch['agent_grpo_idx_deepthink'][i]
                id_parts = data.non_tensor_batch['agent_grpo_idx_deepthink'][i].split('||')
                last_id = id_parts[0]
                response_idx = response_idx_dict[i]
                reward_tensor[i,response_idx] = float(np.mean(reward_dict.get(last_id,[0.0])))


        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
