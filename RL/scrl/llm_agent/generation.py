# =============================================================================
# Based on the Search-R1 example from the Search-R1 project.
#
# Original Authors: Jin Bowen, Zeng Hansi, Yue Zhenrui, Wang Dong, Zamani Hamed, Han Jiawei
#
# License: Apache 2.0
# Project URL: https://github.com/PeterGriffinJin/Search-R1
# =============================================================================

import math
import torch
import re
import os
import json
import json5
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scrl.llm_agent.tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
import numpy as np

import copy
import datetime
import random
import time
try:
    from verl.utils.debug import marked_timer
except ImportError:
    from verl.utils.profiler import marked_timer
from enum import IntEnum
from tool_server.execute_tools import custom_call_tool
from tool_server.tool_prompt import SYSTEM_PROMPT, SUMMARY_PROMPT
from tensordict import TensorDict

_THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL)
_ANSWER_RE = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
_TOOL_RE = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)

ALLOWED_ARGS = {
    "search": {"query"},
    "visit": {"url", "goal"},
    "PythonInterpreter": {"code"},
}


def _filter_args(tool_name: str, args: dict) -> dict:
    if not isinstance(args, dict):
        return {}
    if tool_name not in ALLOWED_ARGS:
        return args
    allowed = ALLOWED_ARGS[tool_name]
    return {k: v for k, v in args.items() if k in allowed}


class GenerationFlag(IntEnum):
    ERROR = 2
    CALL = 1
    END = 0


@dataclass
class GenerationConfig:
    max_turns: int
    num_gpus: int
    data_writing_path: str = None
    model_name: str = None
    n: int = 1
    project_name: str = None
    experiment_name: str = None
    search_engine: str = "rag"
    nnodes: int = 1
    system_prompt: Optional[str] = SYSTEM_PROMPT
    codeact_env_disabled: bool = True
    summary_prompt: Optional[str] = SUMMARY_PROMPT
    deepthink_disabled: bool = False
    max_len: int = 32000
    info_gain_type: str = ""
    use_vectorized_gt_logprob: bool = False
    ig_compute_freq: int = 1
    use_kv_cache_ig: bool = False
    ig_tool_filter: str = ""


class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
        processor=None,
    ):
        self.tokenizer = tokenizer
        if processor is None:
            self.processor = tokenizer
        else:
            self.processor = processor
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        self.system_prompt = config.system_prompt
        self.summary_prompt = config.summary_prompt
        self.deepthink_disabled = config.deepthink_disabled
        self.codeact_env_disabled = config.codeact_env_disabled

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id
        ))

    def _update_right_side(self, original_right_side: Dict,
                           cur_responses: torch.Tensor,
                           next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side of rollings."""
        if next_obs_ids is not None:
            responses = self.tensor_fn.concatenate_with_padding(
                [original_right_side['responses'], cur_responses, next_obs_ids],
                pad_to_left=False
            )
        else:
            responses = self.tensor_fn.concatenate_with_padding(
                [original_right_side['responses'], cur_responses],
                pad_to_left=False
            )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()

        return {'responses': responses[:, :effective_len]}

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, next_obs_ids: torch.Tensor) -> DataProto:
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])

        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        return DataProto.from_dict({
            'input_ids': new_input_ids[:, -effective_len:],
            'position_ids': new_position_ids[:, -effective_len:],
            'attention_mask': new_attention_mask[:, -effective_len:]
        })

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        next_obs_ids = self.processor(
            next_obs,
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']
        return next_obs_ids


    def execute_predictions(
        self, tool_call_list, total_number
    ):
        """Serially dispatch each tool call to the open-source ``tool_server``.

        Input tuples use the legacy layout ``(idx, question, think, tool_call_dict)``
        produced by ``run_llm_loop``; the returned list preserves the original
        shape ``[{'idx','question','think','tool_call','total_number','content'}, ...]``
        expected by downstream code (see the ``tool_call_list[i]['content']`` /
        ``['idx']`` / ``['think']`` / ``['tool_call']`` accesses later in this
        file).  ``total_number`` is retained purely for signature/compat; the
        new function-based tool interface does not consume it.
        """
        query_contents = []
        for tc in tool_call_list:
            idx, question, think, tool_call = tc[0], tc[1], tc[2], tc[3]
            try:
                content = custom_call_tool(tool_call)
                if content is None:
                    content = "Tool returned empty response"
                elif isinstance(content, list):
                    content = "\n".join(str(c) for c in content)
                elif not isinstance(content, str):
                    content = str(content)
            except Exception as e:
                content = f"Tool execution error: {repr(e)}"
            query_contents.append({
                "idx": idx,
                "question": question,
                "think": think,
                "tool_call": tool_call,
                "total_number": total_number,
                "content": content,
            })
        return query_contents

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences_continue(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        if not hasattr(self.actor_rollout_wg, 'generate_sequences_continue'):
            # Fall back to the legacy single-shot generation API.
            gfunc = self.actor_rollout_wg.generate_sequences
        else:
            gfunc = self.actor_rollout_wg.generate_sequences_continue
        active_batch.meta_info = {'validate': self.is_validation}
        num_gpus = self.config.num_gpus * self.config.nnodes
        if num_gpus <= 1:
            return gfunc(active_batch)

        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        if remainder == 0:
            output = gfunc(active_batch)
            return output
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}

        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)
        padded_active_batch = DataProto.from_dict(padded_batch)
        padded_active_batch.meta_info = {'validate':self.is_validation}

        # Generate with padded batch
        padded_output = gfunc(padded_active_batch)
        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}

        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta

        padded_output.batch = trimmed_batch
        return padded_output

    def parse_question(self, input_ids: torch.Tensor, is_multimodal=True) -> str:
        """Parse question to get the query content."""
        query_contents = self.tokenizer.batch_decode(input_ids)
        query_contents = [re.sub(r'^(<\|endoftext\|>)+', '', content) for content in query_contents]
        query_contents = [content.split("<|im_start|>user\n")[1].split("<|im_end|>")[0] for content in query_contents]
        if is_multimodal:
            query_contents = [content.split("<|vision_end|>")[-1] for content in query_contents]
            return query_contents
        return query_contents

    def parse_response(self, input_ids: torch.Tensor, think: bool = False) -> List[Tuple[bool, str, str]]:
        response_contents = self.tokenizer.batch_decode(input_ids)
        results = []
        THINK_RE = _THINK_RE
        ANSWER_RE = _ANSWER_RE
        TOOL_RE = _TOOL_RE

        for i, content in enumerate(response_contents):
            if think:
                content = "<think>" + content

            if "<tool_response>" in content:
                content = content[:content.find("<tool_response>")]

            if "<think>" in content and "</think>" not in content:
                content = content + "\n</think>"
            if "<think>" not in content and "</think>" in content:
                content = "<think>\n" + content

            think_match = THINK_RE.search(content)
            if not think_match:
                results.append((GenerationFlag.ERROR, "Missing <think></think>", ""))
                continue

            thinking = think_match.group(1)

            answer_match = ANSWER_RE.search(content)
            tool_match = TOOL_RE.search(content)

            if answer_match and not tool_match:
                results.append((GenerationFlag.END, thinking, answer_match.group(1)))
                continue

            if tool_match and not answer_match:
                raw_tool_str = tool_match.group(1).strip()
                try:
                    if "pythoninterpreter" in raw_tool_str.lower() and "<code>" in raw_tool_str and "</code>" in raw_tool_str:
                        code_match = re.search(r'<code>(.*?)</code>', raw_tool_str, re.DOTALL)
                        if not code_match:
                            results.append((GenerationFlag.ERROR, f"PythonInterpreter detected but no <code> block found, raw={raw_tool_str}", ""))
                            continue
                        tool_call = {
                            "name": "PythonInterpreter",
                            "arguments": {"code": code_match.group(1).strip()}
                        }
                    else:
                        tool_call = json5.loads(raw_tool_str)

                        if not isinstance(tool_call, dict):
                            results.append((GenerationFlag.ERROR, f"Tool call should be a dict, got: {type(tool_call)}", ""))
                            continue

                        if "name" not in tool_call or "arguments" not in tool_call:
                            results.append((GenerationFlag.ERROR, f"Tool call missing required fields, raw={raw_tool_str}", ""))
                            continue

                        tool_call["arguments"] = _filter_args(tool_call["name"], tool_call.get("arguments", {}))

                    results.append((GenerationFlag.CALL, thinking, tool_call))
                except Exception as e:
                    results.append((GenerationFlag.ERROR, f"Tool call parse error: {repr(e)}, raw={raw_tool_str}", ""))
                continue

            results.append((GenerationFlag.ERROR, "Ambiguous or incomplete output", ""))
        return results

    def pseudo_generate_sequences(self, prompts, response):
        """Build a pseudo DataProto that looks like generation output with GT answer as response.
        Used for compute_log_prob to get P(GT|context) at each turn."""
        from verl.utils.torch_functional import pad_2d_list_to_length, get_response_mask
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        batch_size = idx.size(0)
        eos_token_id = self.tokenizer.eos_token_id
        non_tensor_batch = prompts.non_tensor_batch
        response = pad_2d_list_to_length(response, self.tokenizer.pad_token_id).to(idx.device)

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)

        last_valid_pos_ids = (attention_mask.sum(dim=1, keepdim=True).long() - 1).clamp(min=0)
        response_position_ids = last_valid_pos_ids + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    def run_llm_loop(self, gen_batch: DataProto, global_steps: int,
                     ground_truths: list = None) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        explit_control_flag = hasattr(self.actor_rollout_wg, 'convert_to_rollout') and hasattr(
            self.actor_rollout_wg, 'convert_to_trainer')
        if explit_control_flag:
            print('Enabling standalone controller to toggle rollout and reuse KV cache')
            self.actor_rollout_wg.convert_to_rollout()
        node_rank = int(os.environ["PET_NODE_RANK"])
        print(f"node {node_rank} gains {len(gen_batch.batch['input_ids'])} * {self.config.n} datas!", flush=True)
        query_contents = self.parse_question(gen_batch.batch['input_ids'])
        non_tensor_batch = gen_batch.non_tensor_batch

        images = []
        images_input = []
        if "multi_modal_data" in non_tensor_batch:
            for multi_modal_data in non_tensor_batch.pop("multi_modal_data", None):
                images.append(multi_modal_data)
            for multi_modal_input in non_tensor_batch.pop("multi_modal_inputs", None):
                images_input.append(multi_modal_input)
        messages_list = []
        agent_grpo_idx = []
        agent_roll_idx = []
        messages_list_deepthink = []
        messages_list_deepthink_string = []
        has_image_flag = True
        if len(images) == 0:
            has_image_flag = False
            # When no images are provided, fill with None placeholders.
            images = [None for _ in range(len(query_contents))]
            images_input = [None for _ in range(len(query_contents))]
        for idx, (query_content, image) in enumerate(zip(query_contents, images)):
            for rollid in range(self.config.n):
                if self.system_prompt:
                    messages = [{"role": "system", "content": self.system_prompt},]
                    mm_user_content = []
                    if image is not None:
                        mm_user_content += [{"type": "image", "image": image["image"][0]}]
                    mm_user_content += [{"type": "text",  "text": query_content}]
                    if has_image_flag:
                        messages += [{"role": "user", "content": mm_user_content}]
                    else:
                        messages += [{"role": "user", "content": query_content}]
                else:
                    messages = [
                        {"role": "user", "content": query_content}
                    ]
                messages_list.append(messages)
                agent_grpo_idx.append(idx)
                agent_roll_idx.append(rollid)
        activate_list = [i for i in range(len(messages_list))]
        # When no image is provided, this collapses to an array of Nones.
        images = [image for image in images for _ in range(self.config.n)]
        images_input = [image_input for image_input in images_input for _ in range(self.config.n)]

        message_string_list = []
        agent_grpo_idx_deepthink_activate_list = [
            f'{agent_grpo_idx[activate_list[i]]}_{agent_roll_idx[activate_list[i]]}_0' for i in range(len(messages_list))]  # ids for all currently active trajectories
        agent_grpo_idx_deepthink = []  # ids for all rollout trajectories (accumulated across turns)

        # Ensure the rollout output directory exists.
        output_dir = f"./outputs/{self.config.project_name}/{self.config.experiment_name}/rollout"
        if not os.path.exists(output_dir):
            print(f"Directory not exist, create at {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

        # --- Info-gain initialization (IGPO) ---
        _use_info_gain = bool(self.config.info_gain_type) and ground_truths is not None
        if _use_info_gain:
            import math, json as _json
            import torch.nn.functional as F

            ground_truths_rolling = []
            for gt in ground_truths:
                _gt_copy = dict(gt)
                if "<|answer_split|>" in _gt_copy['ground_truth']:
                    _gt_copy['ground_truth'] = _gt_copy['ground_truth'].split("<|answer_split|>")[0]
                _gt_text = _gt_copy['ground_truth'].strip()
                if _gt_text.startswith('['):
                    try:
                        parsed = _json.loads(_gt_text)
                        if isinstance(parsed, list):
                            label = 'true'
                            for item in parsed:
                                if item['label'].lower() == 'false':
                                    label = 'false'
                                    break
                            _gt_copy['ground_truth'] = label
                    except Exception:
                        pass
                for _ in range(self.config.n):
                    ground_truths_rolling.append(_gt_copy)

            PREFIX = "Now there's enough information to answer\n</think>\n<answer>\n"
            SUFFIX = "\n</answer><|im_end|>"
            pseudo_resps_with_gt = []
            gt_idx = []
            for _gt in ground_truths_rolling:
                gt_text = _gt['ground_truth']
                full_text = f"{PREFIX}{gt_text}{SUFFIX}"
                encoding = self.tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True)
                token_ids = encoding['input_ids'].tolist()[0]
                offset_mapping = encoding['offset_mapping'].tolist()[0]
                pseudo_resps_with_gt.append(token_ids)
                if len(token_ids) == 0:
                    gt_idx.append([0, 0])
                    continue
                gt_char_start = len(PREFIX)
                gt_char_end = len(PREFIX) + len(gt_text)
                gt_token_start = None
                gt_token_end = None
                for ti, (cs, ce) in enumerate(offset_mapping):
                    if gt_token_start is None and ce > gt_char_start:
                        gt_token_start = ti
                    if cs < gt_char_end and ce > 0:
                        gt_token_end = ti + 1
                if gt_token_start is None:
                    gt_token_start = len(token_ids)
                if gt_token_end is None:
                    gt_token_end = len(token_ids)
                gt_idx.append([gt_token_start, gt_token_end])

            ig_gt_values = {}
            ig_rewards = [[] for _ in range(len(messages_list))]
            ig_rollings_active = None
            _ig_turn_count = [0] * len(messages_list)  # per-sample non-baseline turn count

            _use_vectorized = bool(self.config.use_vectorized_gt_logprob)
            _use_kv_cache_ig = bool(self.config.use_kv_cache_ig)
            _ig_compute_freq = max(1, int(self.config.ig_compute_freq))
            _ig_tool_filter_raw = getattr(self.config, "ig_tool_filter", "") or ""
            _ig_tool_filter = set(
                t.strip().lower() for t in _ig_tool_filter_raw.split(",") if t.strip()
            ) if _ig_tool_filter_raw.strip() else set()
            _step_tool_names = {}  # {step: {sample_idx: tool_name}}
            _vec_data_collector = None
            _kv_turn_boundaries = [] if _use_kv_cache_ig else None
            if _use_kv_cache_ig:
                print(f"[IGPO] Info-gain enabled (KV-CACHE), type={self.config.info_gain_type}, "
                      f"freq={_ig_compute_freq}, samples={len(messages_list)}")
            elif _use_vectorized:
                _vec_data_collector = {
                    'pseudo_outputs_per_turn': [],
                    'activate_lists_per_turn': [],
                    'gt_idx': gt_idx,
                    'num_samples': len(messages_list),
                }
                print(f"[IGPO] Info-gain enabled (VECTORIZED), type={self.config.info_gain_type}, "
                      f"freq={_ig_compute_freq}, samples={len(messages_list)}")
            else:
                print(f"[IGPO] Info-gain enabled, type={self.config.info_gain_type}, "
                      f"freq={_ig_compute_freq}, samples={len(messages_list)}")
        # --- end info-gain init ---

        # Note: messages_list_for_kv is captured after the loop, before deepthink replaces messages_list

        max_turns = self.config.max_turns  # add random jitter around max_turns for generalization
        max_turns = max(1, int(random.random() * 10 - 5) + max_turns)
        _turn_timings = []  # [(step, llm_sec, tool_sec, total_sec), ...]
        _n_seqs_total = len(messages_list)
        _apc_dummy_msg = [{"role": "user", "content": "hi"}]
        _apc_dummy_text = self.processor.apply_chat_template(
            [_apc_dummy_msg], add_generation_prompt=True, tokenize=False)
        _apc_dummy_tokens = self.processor(_apc_dummy_text, return_tensors="pt")['input_ids'][0]
        _apc_dummy_tokens = _apc_dummy_tokens[_apc_dummy_tokens != self.tokenizer.pad_token_id]
        for step in range(max_turns):
            _turn_start = time.time()
            print(f"{datetime.datetime.now()} node {node_rank} step {step} start!")
            if activate_list == []:
                break

            last_list = []
            for index, i in enumerate(activate_list):
                messages_temp = self.processor.apply_chat_template(
                    messages_list[i],
                    add_generation_prompt=True,
                    tokenize=False
                )
                messages_temp = self.processor(messages_temp, return_tensors="pt", padding=True)['input_ids'][0]

                if step > 0 and len(messages_temp) > self.config.max_len - 10000:
                    messages_list[i][-1] = {
                        "role": "user",
                        "content": (
                            "You have reached the maximum context length. "
                            "Provide your final answer in <answer></answer> tags now."
                        ),
                    }
                    last_list.append(index)

            if step > 0 and step == max_turns - 1:
                for index, i in enumerate(activate_list):
                    if index not in last_list:
                        messages_list[i][-1] = {
                            "role": "user",
                            "content": (
                                "You've reached the maximum number of tool calls. "
                                "Provide your final answer in <answer></answer> tags now."
                            ),
                        }
                        last_list.append(index)

            activate_messages_list = [messages_list[i] for i in activate_list]
            # Initialize agent_grpo_idx_deepthink_activate_list for the current turn.
            for i in range(len(activate_list)):
                idlist = agent_grpo_idx_deepthink_activate_list[activate_list[i]].split('||')
                agent_grpo_idx_deepthink_activate_list[activate_list[i]] = '||'.join(idlist)

            # --DEEPTHINK SUMMARY PART--
            if not self.deepthink_disabled and step != 0 and step % 1 == 0 and step != max_turns - 1:
                messages_list_compress_input = [rolling for rolling in activate_messages_list]

                # Optionally crossover with the final turn of another rollout from the same prompt.
                merge_ids = []
                for idx, i in enumerate(activate_list):
                    merge_id = []
                    available_ids = [idx2 for idx2, j in enumerate(
                        activate_list) if i != j and agent_grpo_idx[i] == agent_grpo_idx[j]]
                    if len(available_ids) > 0:
                        j = random.choice(available_ids)
                        merge_id.append(agent_grpo_idx_deepthink_activate_list[activate_list[j]])
                        merge_ids.append(merge_id)
                        messages_list_compress_input[idx].extend(activate_messages_list[j][2:])
                    else:
                        merge_ids.append(merge_id)
                        pass

                messages_list_compress_input = [rolling + [{'role': 'tool',
                                                            'content': SUMMARY_PROMPT}] for rolling in messages_list_compress_input]

                messages_list_compress = self.processor.apply_chat_template(
                    messages_list_compress_input, add_generation_prompt=True, tokenize=False)
                messages_list_compress = [rolling for rolling in messages_list_compress]
                rollings_active = self.processor(messages_list_compress, return_tensors="pt", padding=True)
                pad_mask = rollings_active['input_ids'] != self.tokenizer.pad_token_id
                sorted_indices = pad_mask.to(torch.int64).argsort(dim=1, stable=True)
                rollings_active['input_ids'] = rollings_active['input_ids'].gather(1, sorted_indices)
                rollings_active['attention_mask'] = rollings_active['attention_mask'].gather(1, sorted_indices)

                attention_mask = rollings_active['attention_mask']
                rollings_active['position_ids'] = self.tensor_fn.create_position_ids(attention_mask)
                print(
                    f"{datetime.datetime.now()} node {node_rank}, turn {step} summary active is {len(rollings_active['input_ids'])} datas")
                rollings_active = DataProto.from_dict({
                    'input_ids': rollings_active['input_ids'],
                    'attention_mask': rollings_active['attention_mask'],
                    'position_ids': rollings_active['position_ids'],
                })
                gen_output = self._generate_with_gpu_padding(rollings_active)
                messages_list_compress_output = self.tokenizer.batch_decode(
                    gen_output.batch['responses'], skip_special_tokens=False)

                # Register intermediate nodes so gradients can flow through the summary turn.
                # The uid must encode not only the current turn/rollid but also the history uids.
                for i in range(len(activate_list)):
                    now_uid = f'{agent_grpo_idx[activate_list[i]]}_{agent_roll_idx[activate_list[i]]}_{step}'
                    idlist = [now_uid]
                    idlist.extend(agent_grpo_idx_deepthink_activate_list[activate_list[i]].split('||')[:1])
                    for e in merge_ids[i]:
                        idlist.extend(e.split('||')[:1])
                    agent_grpo_idx_deepthink_activate_list[activate_list[i]] = '||'.join(idlist)

                    agent_grpo_idx_deepthink.append(agent_grpo_idx_deepthink_activate_list[activate_list[i]])
                    messages_list_deepthink.append(messages_list_compress_input[i] + [{'role': 'assistant',
                                                                                       'content': "History summary:\n" + messages_list_compress_output[i].split('<|im_end|>')[0]}])
                    messages_list[activate_list[i]] = messages_list[activate_list[i]][:2] + [{'role': 'user',
                                                                                              'content': "History summary:\n" + messages_list_compress_output[i].split('<|im_end|>')[0]}]

                # Refresh the active message buffers after summarization.
                activate_messages_list = [messages_list[i] for i in activate_list]

                # ---------------
            try:
                rollings_active = self.processor.apply_chat_template(
                    activate_messages_list, add_generation_prompt=True, tokenize=False)
            except:
                json.dump(activate_messages_list, open('./debug.json', 'w'))
                rollings_active = []
                for e in activate_messages_list:
                    print(e)
                    result = self.tokenizer.apply_chat_template([e], add_generation_prompt=True, tokenize=False)
                    rollings_active.extend(result)
            _template_has_think = any(r.rstrip().endswith('<think>') for r in rollings_active)
            think = _template_has_think
            rollings_active = self.processor(rollings_active, return_tensors="pt", padding=True,
                                             truncation=True, max_length=self.config.max_len)

            pad_mask = rollings_active['input_ids'] != self.tokenizer.pad_token_id
            sorted_indices = pad_mask.to(torch.int64).argsort(dim=1, stable=True)
            rollings_active['input_ids'] = rollings_active['input_ids'].gather(1, sorted_indices)
            rollings_active['attention_mask'] = rollings_active['attention_mask'].gather(1, sorted_indices)

            attention_mask = rollings_active['attention_mask']
            rollings_active['position_ids'] = self.tensor_fn.create_position_ids(attention_mask)

            print(
                f"{datetime.datetime.now()} node {node_rank}, turn {step} rollings_active is {len(rollings_active['input_ids'])} datas")
            print(f"max  input_ids {max([len(x) for x in rollings_active['input_ids']])}")

            # [APC-DIAG] detect decode-reencode token mismatch between turns
            try:
                if not hasattr(self, '_apc_diag_prev'):
                    self._apc_diag_prev = {}
                _diag_sample = min(3, len(activate_list))
                if step > 0 and self._apc_diag_prev:
                    _n_checked = 0
                    _n_mismatch = 0
                    for _bi in range(_diag_sample):
                        _si = activate_list[_bi]
                        if _si not in self._apc_diag_prev:
                            continue
                        _prev_toks = self._apc_diag_prev[_si]
                        _cur_ids = rollings_active['input_ids'][_bi]
                        _cur_toks = _cur_ids[_cur_ids != self.tokenizer.pad_token_id].tolist()
                        _prev_len = len(_prev_toks)
                        _n_checked += 1
                        if _prev_len > len(_cur_toks):
                            print(f"[APC-DIAG] seq {_si}: cur({len(_cur_toks)}) "
                                  f"shorter than prev({_prev_len})", flush=True)
                            _n_mismatch += 1
                            continue
                        _prefix = _cur_toks[:_prev_len]
                        if _prefix == _prev_toks:
                            print(f"[APC-DIAG] seq {_si}: prefix MATCH "
                                  f"({_prev_len} tokens identical, "
                                  f"cur_total={len(_cur_toks)})", flush=True)
                        else:
                            _fd = next((_j for _j in range(_prev_len)
                                        if _prefix[_j] != _prev_toks[_j]), _prev_len)
                            _n_mismatch += 1
                            _ctx = 5
                            print(f"[APC-DIAG] seq {_si}: prefix MISMATCH "
                                  f"at pos {_fd}/{_prev_len} "
                                  f"prev={_prev_toks[max(0,_fd-_ctx):_fd+_ctx]} "
                                  f"curr={_prefix[max(0,_fd-_ctx):_fd+_ctx]}",
                                  flush=True)
                    if _n_checked > 0:
                        print(f"[APC-DIAG] turn {step}: {_n_checked} checked, "
                              f"{_n_mismatch} mismatch", flush=True)
                self._apc_diag_prev = {}
                for _bi in range(_diag_sample):
                    _si = activate_list[_bi]
                    _cur_ids = rollings_active['input_ids'][_bi]
                    _cur_toks = _cur_ids[_cur_ids != self.tokenizer.pad_token_id].tolist()
                    self._apc_diag_prev[_si] = _cur_toks
            except Exception as _e:
                print(f"[APC-DIAG] error: {_e}", flush=True)

            rollings_active = DataProto.from_dict({
                'input_ids': rollings_active['input_ids'],
                'attention_mask': rollings_active['attention_mask'],
                'position_ids': rollings_active['position_ids'],
            })
            if has_image_flag:
                activate_non_tensor_batch = {
                    "multi_modal_data": np.array([
                        {"image": [images[i]["image"][0]]} for i in activate_list
                    ], dtype=object)
                }
                rollings_active.non_tensor_batch = activate_non_tensor_batch

            # --- Info-gain: compute P(GT|context) at this turn ---
            if _use_info_gain and (step == 0 or step % _ig_compute_freq == 0):
                if _use_kv_cache_ig:
                    # KV-cache mode: record per-sample valid token lengths at this turn
                    _per_sample_lens = {}
                    for ai, gi in enumerate(activate_list):
                        _per_sample_lens[gi] = int(rollings_active.batch['attention_mask'][ai].sum().item())
                    _kv_turn_boundaries.append({
                        'step': step,
                        'per_sample_token_lens': _per_sample_lens,
                        'activate_list': list(activate_list),
                    })
                else:
                    if step == 0:
                        ig_rollings_active = copy.deepcopy(rollings_active)
                    else:
                        if ig_rollings_active.batch['input_ids'].shape[1] < rollings_active.batch['input_ids'].shape[1]:
                            diff = rollings_active.batch['input_ids'].shape[1] - ig_rollings_active.batch['input_ids'].shape[1]
                            ig_rollings_active.batch['input_ids'] = F.pad(ig_rollings_active.batch['input_ids'], (0, diff), value=self.tokenizer.pad_token_id)
                            ig_rollings_active.batch['attention_mask'] = F.pad(ig_rollings_active.batch['attention_mask'], (0, diff), value=0)
                            ig_rollings_active.batch['position_ids'] = F.pad(ig_rollings_active.batch['position_ids'], (0, diff), value=0)
                        src_len = rollings_active.batch['input_ids'].shape[1]
                        ig_len = ig_rollings_active.batch['input_ids'].shape[1]
                        for ai, gi in enumerate(activate_list):
                            ig_rollings_active.batch['input_ids'][gi, :src_len] = rollings_active.batch['input_ids'][ai]
                            ig_rollings_active.batch['attention_mask'][gi, :src_len] = rollings_active.batch['attention_mask'][ai]
                            ig_rollings_active.batch['position_ids'][gi, :src_len] = rollings_active.batch['position_ids'][ai]
                            if src_len < ig_len:
                                ig_rollings_active.batch['input_ids'][gi, src_len:] = self.tokenizer.pad_token_id
                                ig_rollings_active.batch['attention_mask'][gi, src_len:] = 0
                                ig_rollings_active.batch['position_ids'][gi, src_len:] = 0

                    pseudo_gen_output = self.pseudo_generate_sequences(ig_rollings_active, pseudo_resps_with_gt)

                    if _use_vectorized and _vec_data_collector is not None:
                        # Vectorized mode: collect data, defer computation to after the loop
                        pseudo_output_clone = DataProto.from_dict({
                            'prompts': pseudo_gen_output.batch['prompts'].clone(),
                            'responses': pseudo_gen_output.batch['responses'].clone(),
                            'input_ids': pseudo_gen_output.batch['input_ids'].clone(),
                            'attention_mask': pseudo_gen_output.batch['attention_mask'].clone(),
                            'position_ids': pseudo_gen_output.batch['position_ids'].clone(),
                        })
                        _vec_data_collector['pseudo_outputs_per_turn'].append(pseudo_output_clone)
                        _vec_data_collector['activate_lists_per_turn'].append(list(activate_list))
                    else:
                        # Original mode: immediate per-turn computation
                        pseudo_gen_output_log_probs = self.actor_rollout_wg.compute_log_prob(pseudo_gen_output)

                        info_gain_type = self.config.info_gain_type
                        if step == 0:
                            for i in activate_list:
                                if i >= len(gt_idx) or gt_idx[i][0] >= gt_idx[i][1]:
                                    continue
                                log_probs = pseudo_gen_output_log_probs.batch['old_log_probs'][i, gt_idx[i][0]:gt_idx[i][1]]
                                mean_lp = log_probs.mean().item()
                                if math.isnan(mean_lp) or math.isinf(mean_lp):
                                    continue
                                if info_gain_type == "log_prob_diff":
                                    ig_gt_values[i] = mean_lp
                                else:
                                    ig_gt_values[i] = math.exp(mean_lp)
                        else:
                            for i in activate_list:
                                if i >= len(gt_idx) or gt_idx[i][0] >= gt_idx[i][1]:
                                    continue
                                if i not in ig_gt_values:
                                    continue
                                if _ig_tool_filter:
                                    prev_tool = _step_tool_names.get(step - 1, {}).get(i)
                                    if not prev_tool or prev_tool.lower() not in _ig_tool_filter:
                                        ig_rewards[i].append(None)
                                        continue
                                log_probs = pseudo_gen_output_log_probs.batch['old_log_probs'][i, gt_idx[i][0]:gt_idx[i][1]]
                                mean_lp = log_probs.mean().item()
                                if math.isnan(mean_lp) or math.isinf(mean_lp):
                                    ig_rewards[i].append(0.0)
                                    continue
                                if info_gain_type == "log_prob_diff":
                                    cur_val = mean_lp
                                else:
                                    cur_val = math.exp(mean_lp)
                                ig = cur_val - ig_gt_values[i]
                                if math.isnan(ig) or math.isinf(ig):
                                    ig_rewards[i].append(0.0)
                                    continue
                                ig_rewards[i].append(ig)
                                ig_gt_values[i] = cur_val
            # Freq > 1: pad non-computed turns with None so ig_rewards stays aligned
            elif _use_info_gain and step > 0:
                for i in activate_list:
                    ig_rewards[i].append(None)
            # Per-step turn counting (for post-loop sparse expansion)
            if _use_info_gain and step > 0:
                for i in activate_list:
                    _ig_turn_count[i] += 1
            # --- end info-gain per-turn ---

            _llm_start = time.time()
            if len(activate_list) < _n_seqs_total and explit_control_flag:
                _active_set = set(activate_list)
                _seq_len = rollings_active.batch['input_ids'].shape[1]
                _full_ids = torch.full(
                    (_n_seqs_total, _seq_len), self.tokenizer.pad_token_id,
                    dtype=rollings_active.batch['input_ids'].dtype)
                _full_mask = torch.zeros(
                    (_n_seqs_total, _seq_len),
                    dtype=rollings_active.batch['attention_mask'].dtype)
                _full_pos = torch.zeros(
                    (_n_seqs_total, _seq_len),
                    dtype=rollings_active.batch['position_ids'].dtype)
                for _bi, _si in enumerate(activate_list):
                    _full_ids[_si] = rollings_active.batch['input_ids'][_bi]
                    _full_mask[_si] = rollings_active.batch['attention_mask'][_bi]
                    _full_pos[_si] = rollings_active.batch['position_ids'][_bi]
                _d_len = len(_apc_dummy_tokens)
                for _i in range(_n_seqs_total):
                    if _i not in _active_set:
                        _full_ids[_i, -_d_len:] = _apc_dummy_tokens
                        _full_mask[_i, -_d_len:] = 1
                        _full_pos[_i, -_d_len:] = torch.arange(
                            _d_len, dtype=_full_pos.dtype)
                _gen_batch = DataProto.from_dict({
                    'input_ids': _full_ids,
                    'attention_mask': _full_mask,
                    'position_ids': _full_pos,
                })
                if has_image_flag:
                    _gen_batch.non_tensor_batch = {
                        "multi_modal_data": np.array([
                            {"image": [images[_i]["image"][0]]}
                            if _i in _active_set
                            else {"image": [images[0]["image"][0]]}
                            for _i in range(_n_seqs_total)
                        ], dtype=object)
                    }
                print(
                    f"[APC] Padded batch: {len(activate_list)} active + "
                    f"{_n_seqs_total - len(activate_list)} dummy = "
                    f"{_n_seqs_total} total")
                _full_output = self._generate_with_gpu_padding(_gen_batch)
                gen_output = _full_output.select_idxs(activate_list)
            else:
                gen_output = self._generate_with_gpu_padding(rollings_active)
            _llm_elapsed = time.time() - _llm_start

            meta_info = gen_output.meta_info
            print(
                f"{datetime.datetime.now()} node {node_rank}, turn {step} gen_output {len(gen_output.batch['responses'])} datas (llm: {_llm_elapsed:.1f}s)")

            results = self.parse_response(gen_output.batch['responses'], think=think)
            assert len(results) == len(activate_list)

            raw_contents = {}
            for ri in range(len(results)):
                rc = ('<think>' if think else '') + self.tokenizer.decode(
                    gen_output.batch['responses'][ri], skip_special_tokens=False
                ).replace("<|endoftext|>", "")
                if rc.rstrip().endswith("<|im_end|>"):
                    rc = rc.rstrip()[:-len("<|im_end|>")]
                if "<tool_response>" in rc:
                    rc = rc[:rc.find("<tool_response>")]
                if "<think>" in rc and "</think>" not in rc:
                    rc = rc + "\n</think>"
                if "<think>" not in rc and "</think>" in rc:
                    rc = "<think>\n" + rc
                raw_contents[activate_list[ri]] = rc

            activate_list_copy = []
            tool_call_list = []
            for i in range(len(results)):
                if results[i][0] == GenerationFlag.END or i in last_list:
                    idx = agent_grpo_idx_deepthink_activate_list[activate_list[i]].split('||')
                    idx = [idx[0] + '_answer'] + idx[:1]
                    agent_grpo_idx_deepthink.append('||'.join(idx))
                    messages_list_deepthink.append(activate_messages_list[i] + [{'role': 'assistant',
                                                                                 'content': raw_contents[activate_list[i]]}])
                    messages_list_deepthink_string.append(self.tokenizer.decode(rollings_active.batch['input_ids'][i], skip_special_tokens=False).replace(
                        "<|endoftext|>", "") + self.tokenizer.decode(gen_output.batch['responses'][i], skip_special_tokens=False).replace("<|endoftext|>", ""))
                    messages_list[activate_list[i]].append(
                        {
                            "role": "assistant",
                            "content": raw_contents[activate_list[i]]
                        }
                    )
                elif (results[i][0] == GenerationFlag.ERROR):  # keep the sample active so it can retry
                    activate_list_copy.append(activate_list[i])

                    messages_list[activate_list[i]].append({
                        "role": "assistant",
                        "content": raw_contents[activate_list[i]]
                    })
                    messages_list[activate_list[i]].append({
                        "role": "user",
                        "content": (
                            "<tool_response>\nYour response was malformed. "
                            "Please call a tool or provide a final answer.\n</tool_response>"
                        )
                    })

                elif (results[i][0] == GenerationFlag.CALL):  # the model requested a tool call
                    activate_list_copy.append(activate_list[i])
                    tool_call_list.append((activate_list[i], messages_list[activate_list[i]]
                                          [1]["content"], results[i][1], results[i][2]))
                    if _use_info_gain and _ig_tool_filter:
                        tool_name = results[i][2].get("name") if isinstance(results[i][2], dict) else None
                        _step_tool_names.setdefault(step, {})[activate_list[i]] = tool_name
                else:
                    assert False, f"Unexcepted GenerationFlag {results[i][0]}"

            _tool_start = time.time()
            with marked_timer('toolcall', {}):
                tool_call_list = self.execute_predictions(tool_call_list, len(messages_list))
            _tool_elapsed = time.time() - _tool_start
            print(f"{datetime.datetime.now()} node {node_rank}, turn {step} tool_call_list {len(tool_call_list)} datas (tool: {_tool_elapsed:.1f}s)")
            for i in range(len(tool_call_list)):
                if not self.codeact_env_disabled:  # code-act environment is enabled
                    messages_list[tool_call_list[i]['idx']].append(
                        {
                            "role": "assistant",
                            "content": '<think>' + tool_call_list[i]['think'] + "</think>"+"\n<code>" + str(tool_call_list[i]['tool_call']['arguments']['code']) + "</code>",
                        }
                    )
                    try:
                        messages_list[tool_call_list[i]['idx']].append(
                            {
                                "role": "tool",
                                "content": "<code_response>" + tool_call_list[i]['content'] + "</code_response>",
                            }
                        )
                    except:
                        messages_list[tool_call_list[i]['idx']].append(
                            {
                                "role": "tool",
                                "content": "<code_response>" + 'Format error: code execution failed.' + "</code_response>",
                            }
                        )
                else:
                    messages_list[tool_call_list[i]['idx']].append(
                        {
                            "role": "assistant",
                            "content": raw_contents[tool_call_list[i]['idx']]
                        }
                    )
                    try:
                        messages_list[tool_call_list[i]['idx']].append(
                            {
                                "role": "user",
                                "content": f"<tool_response>\n{tool_call_list[i]['content']}\n</tool_response>"
                            }
                        )
                    except Exception:
                        messages_list[tool_call_list[i]['idx']].append(
                            {
                                "role": "user",
                                "content": "<tool_response>\nFormat error: tool call failed.\n</tool_response>"
                            }
                        )
            _turn_elapsed = time.time() - _turn_start
            _turn_timings.append((step, _llm_elapsed, _tool_elapsed, _turn_elapsed))
            print(f"{datetime.datetime.now()} turn {step} finished, node {node_rank} had {len(activate_list)} queries, now has {len(activate_list_copy)} queries "
                  f"[turn {step}: total={_turn_elapsed:.1f}s, llm={_llm_elapsed:.1f}s, tool={_tool_elapsed:.1f}s]")
            activate_list = activate_list_copy

        if _turn_timings:
            _total_llm = sum(t[1] for t in _turn_timings)
            _total_tool = sum(t[2] for t in _turn_timings)
            _total_all = sum(t[3] for t in _turn_timings)
            print(f"[TIMING] node {node_rank} rollout summary: {len(_turn_timings)} turns, "
                  f"total={_total_all:.1f}s, llm={_total_llm:.1f}s ({_total_llm/_total_all*100:.0f}%), "
                  f"tool={_total_tool:.1f}s ({_total_tool/_total_all*100:.0f}%), "
                  f"avg_per_turn={_total_all/len(_turn_timings):.1f}s")

        if activate_list != []:
            for i in activate_list:
                # message_string_list[i] = self.tokenizer.apply_chat_template(
                #     messages_list[i], add_generation_prompt=False, tokenize=False)
                idx = agent_grpo_idx_deepthink_activate_list[i].split('||')
                idx[0] = idx[0] + '_answer'
                agent_grpo_idx_deepthink.append('||'.join(idx))
                messages_list_deepthink.append(messages_list[i])
                messages_list_deepthink_string.append(self.processor.apply_chat_template(
                    messages_list[i], add_generation_prompt=False, tokenize=False))

        response_str_list = []
        initial_prompt_list = []
        order = sorted(range(len(agent_grpo_idx_deepthink)), key=lambda i: int(agent_grpo_idx_deepthink[i].split('_')[0]))
        agent_grpo_idx_deepthink = [agent_grpo_idx_deepthink[i] for i in order]
        messages_list_deepthink = [messages_list_deepthink[i] for i in order]
        if has_image_flag:
            images_deepthink = np.array([{"image": [images[i]["image"][0]]} for i in order], dtype=object)
            images_input_deepthink = np.array([{"pixel_values": images_input[i]["pixel_values"],
                                              "image_grid_thw": images_input[i]["image_grid_thw"]} for i in order], dtype=object)
        else:
            images_deepthink = np.array([None for i in order], dtype=object)
            images_input_deepthink = np.array([None for i in order], dtype=object)
            pass

        # Snapshot: before messages_list is replaced, capture the original per-sample
        # conversations for KV cache mode (indices align with activate_list/gt_idx).
        if _use_info_gain and _use_kv_cache_ig:
            messages_list_for_kv = [messages_list[i] for i in range(len(pseudo_resps_with_gt))]

        messages_list = messages_list_deepthink

        for i, messages in enumerate(messages_list):
            j = 2
            while j < len(messages):
                if messages[j]['role'] == 'assistant':
                    break
                j += 1
            else:
                print(f"No assistant message found in messages_list[{i}]:", messages)
                assert False, "No assistant message found"
            # if messages[-1]["content"].rstrip().endswith("<|im_end|>"):
            #     messages[-1]["content"] = messages[-1]["content"].rstrip()[:-len("<|im_end|>")]
            # prompt = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
            initial_prompt = self.processor.apply_chat_template(messages[:j], add_generation_prompt=True, tokenize=False)
            initial_prompt_list.append(initial_prompt)
            response_str_list.append(self.processor.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False)[len(initial_prompt):])
            message_string_list.append(initial_prompt_list[-1] + response_str_list[-1])

        if has_image_flag:
            images = [image["image"][0] for image in images_deepthink]
            prompts_tokenizered = self.processor(text=initial_prompt_list, images=images, return_tensors="pt", padding=True)
        else:
            prompts_tokenizered = self.processor(initial_prompt_list, return_tensors="pt", padding=True)
            images = [None for image in images_deepthink]

        prompts_repeated = prompts_tokenizered['input_ids']
        pad_mask = prompts_repeated != self.tokenizer.pad_token_id
        sorted_indices = pad_mask.to(torch.int64).argsort(dim=1, stable=True)

        prompts_repeated = prompts_repeated.gather(1, sorted_indices)
        prompts_attention_mask = prompts_tokenizered['attention_mask'].gather(1, sorted_indices)

        _resp_enc = self.processor(response_str_list, return_tensors="pt", padding=True)
        responses = _resp_enc['input_ids']
        responses_attention_mask = _resp_enc['attention_mask']
        attention_mask = torch.cat((prompts_attention_mask, responses_attention_mask), dim=-1)
        position_ids = self.tensor_fn.create_position_ids(attention_mask)

        message_tensor = DataProto.from_dict({
            'prompts': prompts_repeated,
            'responses': responses,
            'input_ids': torch.cat((prompts_repeated, responses), dim=-1),
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        })
        message_tensor.meta_info.update(meta_info)
        if has_image_flag:
            message_tensor.non_tensor_batch["multi_modal_data"] = images_deepthink
            message_tensor.non_tensor_batch["multi_modal_inputs"] = images_input_deepthink
        message_tensor.non_tensor_batch['agent_grpo_idx_deepthink'] = np.array(agent_grpo_idx_deepthink, dtype=object)

        # --- Vectorized GT LogProb batch computation (after loop) ---
        if _use_info_gain and _use_vectorized and _vec_data_collector is not None:
            num_turns_collected = len(_vec_data_collector['pseudo_outputs_per_turn'])
            if num_turns_collected > 0:
                from scrl.llm_agent.prealigned_vectorized import compute_vectorized_gt_logprob
                import os as _os
                _verify_vectorized = _os.environ.get('IGPO_VERIFY_VECTORIZED', '').lower() in ('true', '1', 'yes')

                vectorized_result = compute_vectorized_gt_logprob(
                    pseudo_outputs_per_turn=_vec_data_collector['pseudo_outputs_per_turn'],
                    activate_lists_per_turn=_vec_data_collector['activate_lists_per_turn'],
                    gt_idx=_vec_data_collector['gt_idx'],
                    actor_rollout_wg=self.actor_rollout_wg,
                    tokenizer=self.tokenizer,
                    info_gain_type=self.config.info_gain_type,
                    enable_strict_validation=_verify_vectorized,
                )

                ig_rewards = vectorized_result['info_gain_rewards']
                print(f"[IGPO] Vectorized GT LogProb COMPLETED: {num_turns_collected} turns, "
                      f"non-empty rewards: {sum(1 for r in ig_rewards if len(r) > 0)}")

                if _verify_vectorized:
                    _vlog_lines = [
                        f"",
                        f"[VERIFY] ========== Vectorized Validation (step {global_steps}) ==========",
                        f"[VERIFY] Turns: {num_turns_collected}, Samples: {len(ig_rewards)}",
                    ]
                    if 'validation_passed' in vectorized_result:
                        _passed = vectorized_result['validation_passed']
                        _total = vectorized_result.get('validation_total_compared', '?')
                        _matched = vectorized_result.get('validation_total_matched', '?')
                        _mismatched = vectorized_result.get('validation_total_mismatched', '?')
                        _max_diff = vectorized_result.get('validation_max_diff', 0.0)
                        _vlog_lines.append(f"[VERIFY] Result: {'PASSED' if _passed else 'FAILED'}")
                        _vlog_lines.append(f"[VERIFY] Compared: {_total}, Matched: {_matched}, Mismatched: {_mismatched}")
                        _vlog_lines.append(f"[VERIFY] Max absolute diff: {_max_diff:.2e}")
                        _details = vectorized_result.get('validation_details', [])
                        if _details:
                            _vlog_lines.append(f"[VERIFY] Mismatch details (up to 10):")
                            for _detail in _details:
                                _vlog_lines.append(f"[VERIFY]   {_detail}")
                    else:
                        _vlog_lines.append(f"[VERIFY] WARNING: no validation result returned")
                    _vlog_lines.append(f"[VERIFY] ====================================")
                    _verify_summary = "\n".join(_vlog_lines)
                    print(_verify_summary)
                    _verify_log_path = _os.path.join(output_dir, "verify_vectorized.log")
                    with open(_verify_log_path, 'a') as _vf:
                        _vf.write(_verify_summary + "\n")
            else:
                print(f"[IGPO] Vectorized GT LogProb: No turns collected (all samples may have finished early)")
        # --- end vectorized batch computation ---

        # --- KV Cache mode: post-loop computation ---
        # Requires deepthink_disabled=True so messages_list[i] is the unmodified
        # full conversation for original sample i. Indices in activate_list match
        # messages_list indices directly.
        if _use_info_gain and _use_kv_cache_ig and _kv_turn_boundaries is not None:
            num_turns_collected = len(_kv_turn_boundaries)
            if num_turns_collected > 0:
                from scrl.llm_agent.prealigned_vectorized import compute_ig_with_kv_cache
                from verl.utils.model import compute_position_id_with_mask

                num_orig_samples = len(pseudo_resps_with_gt)
                # Tokenize full conversations with add_generation_prompt=True to match
                # token positions from the generation loop (where boundaries were recorded).
                traj_texts = []
                for i in range(num_orig_samples):
                    traj_text = self.processor.apply_chat_template(
                        messages_list_for_kv[i], add_generation_prompt=True, tokenize=False)
                    traj_texts.append(traj_text)
                _saved_padding_side = getattr(self.processor, 'padding_side', 'right')
                self.processor.padding_side = 'right'
                traj_encoded = self.processor(
                    traj_texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=self.config.max_len)
                self.processor.padding_side = _saved_padding_side
                traj_input_ids = traj_encoded['input_ids']
                traj_attn_mask = traj_encoded['attention_mask']
                traj_pos_ids = compute_position_id_with_mask(traj_attn_mask)

                # Apply ig_tool_filter: remove samples from non-baseline turns
                # if previous turn didn't call a tool in the filter set.
                _kv_boundaries_for_ig = _kv_turn_boundaries
                if _ig_tool_filter and len(_kv_turn_boundaries) > 1:
                    _kv_boundaries_for_ig = [_kv_turn_boundaries[0]]  # baseline always kept
                    _tf_skipped = 0
                    for ti in range(1, len(_kv_turn_boundaries)):
                        prev_step = _kv_turn_boundaries[ti - 1]['step']
                        prev_tools = _step_tool_names.get(prev_step, {})
                        orig_activate = _kv_turn_boundaries[ti]['activate_list']
                        new_activate = [
                            si for si in orig_activate
                            if (prev_tools.get(si, "") or "").lower() in _ig_tool_filter
                        ]
                        _tf_skipped += len(orig_activate) - len(new_activate)
                        if new_activate:
                            new_lens = {
                                si: _kv_turn_boundaries[ti]['per_sample_token_lens'][si]
                                for si in new_activate
                            }
                            _kv_boundaries_for_ig.append({
                                'step': _kv_turn_boundaries[ti]['step'],
                                'per_sample_token_lens': new_lens,
                                'activate_list': new_activate,
                            })
                    print(f"[IGPO] ig_tool_filter={_ig_tool_filter}: "
                          f"kept {len(_kv_boundaries_for_ig)} turns, "
                          f"skipped {_tf_skipped} sample-turns")

                kv_result = compute_ig_with_kv_cache(
                    trajectory_input_ids=traj_input_ids,
                    trajectory_attention_mask=traj_attn_mask,
                    trajectory_position_ids=traj_pos_ids,
                    gt_token_ids=pseudo_resps_with_gt,
                    gt_idx=gt_idx,
                    turn_boundaries=_kv_boundaries_for_ig,
                    actor_rollout_wg=self.actor_rollout_wg,
                    tokenizer=self.tokenizer,
                    info_gain_type=self.config.info_gain_type,
                    num_samples=num_orig_samples,
                )
                ig_rewards = kv_result['info_gain_rewards']

                # Tool-filter expansion: sparse → full aligned with ALL turns.
                # _kv_boundaries_for_ig may have fewer turns than _kv_turn_boundaries;
                # compute_score expects ig_rewards aligned to ALL turn boundaries.
                if _ig_tool_filter and len(_kv_boundaries_for_ig) < len(_kv_turn_boundaries):
                    all_steps = [tb['step'] for tb in _kv_turn_boundaries]
                    kept_activate = {}
                    for tb in _kv_boundaries_for_ig:
                        if tb['step'] != 0:
                            for si in tb['activate_list']:
                                kept_activate.setdefault(si, set()).add(tb['step'])
                    total_non_baseline = len(all_steps) - 1
                    for si in range(len(ig_rewards)):
                        sparse_ig = ig_rewards[si]
                        if total_non_baseline <= 0 or len(sparse_ig) >= total_non_baseline:
                            continue
                        si_kept = kept_activate.get(si, set())
                        full_ig = []
                        sparse_idx = 0
                        for s in all_steps[1:]:
                            if s in si_kept and sparse_idx < len(sparse_ig):
                                full_ig.append(sparse_ig[sparse_idx])
                                sparse_idx += 1
                            else:
                                full_ig.append(None)
                        ig_rewards[si] = full_ig

                print(f"[IGPO] KV-Cache IG COMPLETED: {num_turns_collected} turns, "
                      f"non-empty rewards: {sum(1 for r in ig_rewards if len(r) > 0)}")
            else:
                print(f"[IGPO] KV-Cache IG: No turn boundaries collected")
        # --- end KV cache computation ---

        # --- Freq > 1: expand sparse ig_rewards for vectorized/KV-cache modes ---
        # For original mode, in-loop padding already produced full-length lists.
        # For vectorized/KV-cache, ig_rewards was replaced post-loop with sparse
        # lists (only computed turns). Expand them to full length with None at
        # non-computed positions so downstream info_gain.compute_score() alignment
        # is correct.
        if _use_info_gain and _ig_compute_freq > 1:
            for si in range(len(ig_rewards)):
                total = _ig_turn_count[si]
                sparse = ig_rewards[si]
                if total == 0 or len(sparse) >= total:
                    continue
                full = [None] * total
                for k, val in enumerate(sparse):
                    step_num = (k + 1) * _ig_compute_freq
                    idx = step_num - 1
                    if idx < total:
                        full[idx] = val
                ig_rewards[si] = full
            print(f"[IGPO] Freq>1 sparse->full expansion done: freq={_ig_compute_freq}")
        # --- end freq > 1 expansion ---

        # --- Store info-gain rewards in non_tensor_batch ---
        if _use_info_gain:
            n_samples = len(agent_grpo_idx_deepthink)
            ig_rewards_output = [[] for _ in range(n_samples)]
            answer_indices = [i for i, x in enumerate(agent_grpo_idx_deepthink) if 'answer' in x]
            for ai_idx, sample_global_idx in enumerate(answer_indices):
                uid_str = agent_grpo_idx_deepthink[sample_global_idx]
                parts = uid_str.split('_')
                try:
                    original_idx = int(parts[0]) * self.config.n + int(parts[1])
                    if original_idx < len(ig_rewards):
                        ig_rewards_output[sample_global_idx] = ig_rewards[original_idx]
                except (ValueError, IndexError):
                    pass
            ig_arr = np.empty(len(ig_rewards_output), dtype=object)
            for _i, _r in enumerate(ig_rewards_output):
                ig_arr[_i] = _r
            message_tensor.non_tensor_batch['info_gain_rewards'] = ig_arr
            print(f"[IGPO] Stored info_gain_rewards for {n_samples} samples, "
                  f"non-empty: {sum(1 for r in ig_rewards_output if len(r) > 0)}")
        # --- end info-gain store ---

        print("generation finished")
        if explit_control_flag:
            self.actor_rollout_wg.convert_to_trainer()

        print(f"node {node_rank} message_string_list {len(message_string_list)}")

        if self.is_validation or self.deepthink_disabled:  # during validation or when deepthink is disabled, keep only answer-bearing trajectories
            sample_idx = [i for i, x in enumerate(agent_grpo_idx_deepthink) if 'answer' in x]
            message_string_list = [message_string_list[i] for i, x in enumerate(agent_grpo_idx_deepthink) if 'answer' in x]
            message_tensor = message_tensor.select_idxs(sample_idx)
        return message_string_list, message_tensor
