# Copyright 2025 Individual Contributor: Mert Unsal
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

from verl import DataProto
from verl.workers.reward_manager import register
import asyncio

from openai import AsyncOpenAI
from typing import Dict, List, Any
import re
from functools import reduce
from operator import or_
import numpy as np
import json
from verl.utils.profiler import simple_timer
from verl.utils.import_utils import load_extern_type

def filter_think(text):
    import re
    think = ''
    if '<think>' in text and '</think>' in text:
        if re.match(r'<think>([\s\S]*)</think>', text, re.S) is not None:
            think = re.match(r'<think>([\s\S]*)</think>', text, re.S).group(0)
        ans = re.sub(r'<think>[\s\S]*</think>', '', text, re.S)
        return {"think": think, "ans": ans}
    else:
        return {'think': '', 'ans': text}

def extract_json_from_text(text):
    data = ''  # 使用None代替"none"表示空值更规范
    
    try:
        if not isinstance(text, str):
            raise TypeError("输入必须是字符串类型")

        pattern = re.compile(r'\{.*?\}', re.DOTALL)
        matches = pattern.finditer(text)
        
        last_match = None
        for match in matches:
            last_match = match

        if last_match:
            json_str = last_match.group()
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")
        else:
            print("未找到JSON数据")
            
    except TypeError as te:
        print(f"类型错误: {te}")
    except Exception as e:
        print(f"意外错误: {e}")
    
    return data

def format_reward(predict_str: str) -> float:
    pattern = re.compile(
        r'<think>.*?</think>\s*'
        r'(<answer>.*?</answer>|<tool_call>.*?</tool_call>)',
        re.DOTALL
    )
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else -1.0


@register("group_async_batch")
class GroupBatchRewardManager:
    """
    A batch reward manager that computes rewards for a batch of data.

    Args:
        tokenizer (Tokenizer): The tokenizer to use for decoding the responses.
        num_examine (int): The number of responses to examine.
        compute_score (callable): The function to compute the rewards.
        reward_fn_key (str): The key to use for the reward function.
        reward_kwargs (dict): The keyword arguments to pass to the reward function.
    """

    def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key="data_source", **reward_kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs
        self.reward_meta = ['data_source', 'solution_str', 'ground_truth', 'extra_info',]
        self.model = reward_kwargs.get("model", 'auto')
        self.use_single_judge = self.reward_kwargs.get('use_single_judge', False) and self.num_examine == 0
        self.use_group_rule = self.reward_kwargs.get('use_group_rule', False)  # and self.num_examine == 0
        self.build_message = load_extern_type(
            self.reward_kwargs.get('single_messge_fn_path', ''),
            self.reward_kwargs.get('single_messge_fn_name', '')
        ) or self._build_message

        self.build_grouped_messages = load_extern_type(
            self.reward_kwargs.get('group_messge_fn_path', ''),
            self.reward_kwargs.get('group_messge_fn_name', '')
        ) or self._build_grouped_messages

        self.use_group_judge = self.reward_kwargs.get('use_group_judge', False) and self.num_examine == 0
        print(f'init manager ok {self.reward_kwargs} {self.model }')
        self.llm_print = defaultdict(list)
        self.concurrency = self.reward_kwargs.get("concurrency", 10)

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    async def call_llm(self, messages, semaphore, use_llm=True, task=None):
        if not use_llm:
            return ''
        aclient = AsyncOpenAI(base_url=self.reward_kwargs.get(
            'url', 'http://localhost:8000/v1'), api_key='congyu', timeout=3600, max_retries=5
        )
        async with semaphore:
            try:
                response = await aclient.chat.completions.create(messages=messages, model=self.model, **self.reward_kwargs.get('chat_kwargs', {}))
                content = response.choices[0].message.content
                if content is None:
                    content = ''
                self.llm_print[task].append((messages, content))
                if len(self.llm_print[task]) < 3:
                    print(f"[llm judge {len(self.llm_print[task])}]")
                    print(f"[messages] {messages}")
                    print(f"[llm return] {content}")
                return content
            except Exception:
                import traceback
                print(traceback.format_exc())
                return ''

    def apply_group_length_penlty(self, grouped_data):
        raise NotImplementedError

    def reduce_grouped_data_pairwise(self, grouped_data: Dict[str, List], operator):
        # group wise reduce
        # 这里可以算很多东西，比如 lambda x,y: is_equal(x['solution_str'], x['ground_truth']) + is_equal(y['solution_str'], y['ground_truth'])
        #  lambda x,y: is_equal(x['solution_str'], x['ground_truth'])
        return reduce(operator, [param for uidr, param in grouped_data.items()])

    def reduce_grouped_data_listwise(self, grouped_data: Dict[str, List], operator):
        # group wise reduce
        # 这里可以算很多东西，比如 lambda x,y: is_equal(x['solution_str'], x['ground_truth']) + is_equal(y['solution_str'], y['ground_truth'])
        #  lambda x,y: is_equal(x['solution_str'], x['ground_truth'])
        return operator(filter(lambda x: x['result'], [param for uidr, param in grouped_data.items()]))

    def group_rule_score(self, uid, grouped_data: Dict[str, List]):
        # 存储中间变量
        label_dict = {}
        score_dict = {}
        pred_rate = {}

        # 1. 解析每个样本的 label, pred, score
        for uidr, param in grouped_data.items():
            # 解析真实标签
            label = self.default_parser(param['ground_truth'], 'result')
            label_dict[uidr] = label

            # 解析模型预测结果和score
            think, filtered_solution = filter_think(param['solution_str'])
            # calc to score_dict
            score_dict[uidr] = ...
        # pred_rate_avg = all_same(score_dict)
        all_same = 0 
        # 5. 结合交叉熵和群体奖励，使用factor加权
        total_rewards = {}
        for uidr in grouped_data.keys():
            if score_dict[uidr] is None:
                total_rewards[(uid, uidr)] = {"score": 0.1 * 0, 'gp_score_reward': 0}
            else:
                total_rewards[(uid, uidr)] = {"score": score_dict[uidr], 'gp_score_reward': 0}

        return total_rewards, all_same

    def _build_message(self, param):  # sample wise reward
        system, question, answer, ground_truth = param['system_str'], param['question_str'], param['answer_str'], param['ground_truth']
        think, answer = filter_think(answer)

        messages = [
            {"role": "system",
             "content": """你需要对输入的【用户画像】和【分析过程】进行分析，从以下维度进行分析，并给出该分析的质量判断
1. 判断该【分析过程】是否清晰合理，不存在前后矛盾，并且提及内容与输入一致
2. 判断【分析过程】的完整性，即分析过程中是否包含如下内容
    一、数据质量与完整性评估
    二、用户画像/付款方画像分析
    三、基础交易特征（金额/时间/闭环交易）分析
    四、异常行为标签化分析
    五、偏白/偏黑特征分析
3. 判断【分析过程】和【模型回复】是否一致

当各项分析一致时返回
<judge>yes</judge>

当存在不一致时，返回
<judge>no</judge>
             """
             },
            {
                "role": 'user',
                "content": f"""# 用户画像:
{question}

# 分析过程:
{think}

# 模型回复
{answer}
"""
            }
        ]
        return messages

    def rm_judge_score(self, rm_response) -> float:  # sample wise reward
        think, rm_response = filter_think(rm_response)
        result = re.findall(r'<judge>(.*?)</judge>', rm_response)
        # 后处理逻辑
        if len(result) > 0 and result[-1] == 'yes':
            return {"score": 0.2, "judge_valid": 1, "judge_llm_score": 1.}
        elif len(result) > 0 and result[-1] == 'no':
            return {"score": 0., "judge_valid": 1, "judge_llm_score": 0.}
        else:
            return {"score": 0., "judge_valid": 0, "judge_llm_score": 0.}

    def _build_grouped_messages(self, grouped_data: Dict[str, List]):  # group wise reward
        answers = []
        for uidr, param in grouped_data.items():
            question = param['question_str']
            ground_truth = param['ground_truth']
            think, solution_str = filter_think(param['solution_str'])
            answers.append(f"<answers_{uidr}>{solution_str}</answers_{uidr}>")
        gp_answer = '\n'.join(answers)
        messages = [
            {"role": "system", "content": """请根据给出的 <question> 和 不同$id对应的 <answer_$id>，对answers进行各自打分，打分格式为 <judge_$id>0~10</judge_$id>"""},
            {"role": "user", "content": f"<question>{question}</question>\n\n<grount_truth>{ground_truth}</ground_truth>\n\n{gp_answer}"}
        ]
        return messages

    def rm_judge_gp_score(self, uid, rm_response) -> Dict[int, float]:  # group wise reward
        think, rm_response = filter_think(rm_response)
        matches = re.findall(r'<judge_(\d+)>(.*?)</judge_\d+>', rm_response, flags=re.DOTALL)
        result_dict = {(uid, int(judge_id)): float(score) for judge_id, score in matches}
        return result_dict

    async def verify(self, data):
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        semaphore = asyncio.Semaphore(self.reward_kwargs.get("concurrency", 10))
        is_validate = data.meta_info.get('validate', False)
        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        tasks = []
        tasks_grouped = []
        scores = []
        gp_scores_dict = {}
        gp_rm_scores = []
        rm_scores = []
        rule_scores = []
        total_scores = []
        params = []
        already_printed = {}
        uids = []
        grouped_messages = defaultdict(dict)
        gp_score_rules = {}
        ce_weighted_avg = {}
        all_same_cnt = {}
        with simple_timer('build verify async params', {}):

            for i in range(len(data)):
                dataitem = data[i]
                valid_len = valid_response_lengths[i]
                valid_response_ids = response_ids[i][:valid_len]
                prompt_str = self.tokenizer.decode(prompt_ids[i], skip_special_tokens=True)
                extra_info = dataitem.non_tensor_batch.get("extra_info", None)
                data_source = dataitem.non_tensor_batch[self.reward_fn_key]
                uid = dataitem.non_tensor_batch['uid']
                uidr = dataitem.non_tensor_batch['uidr']
                system_str = prompt_str.split("user\n")[0].split("system\n")[-1]
                question_str = prompt_str.split("user\n")[-1].split("assistant\n")[0]
                ground_truth = dataitem.non_tensor_batch["reward_model"].get("ground_truth", None)
                answer_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

                param = dict(data_source=data_source,
                             solution_str=answer_str,
                             answer_str=answer_str,
                             ground_truth=ground_truth,
                             extra_info=extra_info,
                             uidr=uidr,
                             uid=uid,
                             question_str=question_str,
                             system_str=system_str)
                # system=system_str, question=question_str, answer=answer_str, ground_truth=ground_truth)
                messages = self.build_message(param)
                # 创建异步任务
                coro = self.call_llm(messages, semaphore, use_llm=self.use_single_judge and not is_validate, task='single')
                task = asyncio.create_task(coro)
                tasks.append(task)
                grouped_messages[uid][uidr] = param
                params.append(param)

        with simple_timer('calc gp reward', {}):
            for uid, grouped_data in sorted(grouped_messages.items(), key=lambda _: _[0]):
                _gp_score_rules, ce_weighted, all_same = self.group_rule_score(uid, grouped_data)
                gp_score_rules |= _gp_score_rules
                ce_weighted_avg[uid] = ce_weighted
                all_same_cnt[uid] = all_same
                _messages = self.build_grouped_messages(grouped_data)
                coro = self.call_llm(_messages, semaphore,
                                     use_llm=self.use_group_judge and not is_validate, task='group')
                task = asyncio.create_task(coro)
                tasks_grouped.append(task)
                uids.append(uid)

        backtrace = defaultdict(dict)
        # update rule_scores
        gp_rule_scores = []
        ce_weights = []
        all_sames = []
        for i in range(len(data)):
            param = params[i]
            compute_param = {k: v for k, v in param.items() if k in self.reward_meta}
            rule_score = self.compute_score(**compute_param)
            if isinstance(rule_score, float) or isinstance(rule_score, int):
                rule_score = {'score': rule_score}
            rule_scores.append(rule_score)
            rule_gp_rule = gp_score_rules[(param['uid'], int(param['uidr']))]
            gp_rule_scores.append(rule_gp_rule)
            ce_weights.append(ce_weighted_avg[param['uid']])
            all_sames.append(all_same_cnt[param['uid']])
            backtrace[i]['rule_score'] = rule_score
        if self.use_group_rule:
            total_scores.append(gp_rule_scores)
        total_scores.append(rule_scores)

        # 一起提交所有任务
        with simple_timer('verify gather', {}):
            all_tasks = tasks + tasks_grouped
            results = await asyncio.gather(*all_tasks)

        # 拆分结果
        rm_responses = results[:len(tasks)]
        rm_responses_gp = results[len(tasks):]

        # update rm_score
        # gather 并发执行所有任务，保持顺序一致
        # rm_responses = await asyncio.gather(*tasks)
        if rm_responses:
            for i in range(len(data)):
                param = params[i]
                rm_response = rm_responses[i]
                rm_score = self.rm_judge_score(rm_response)
                rm_scores.append(rm_score)
                backtrace[i]['rm_response'] = rm_response
        else:
            rm_scores = [{'score': 0} for i in range(len(data))]
        total_scores.append(rm_scores)

        # update gp_rm_score
        # gather 并发执行所有任务，保持顺序一致
        # rm_responses_gp = await asyncio.gather(*tasks_grouped)
        if rm_responses_gp:
            # uid -> map
            uid2gprm = {}
            for i in range(len(uids)):
                for uid_tuple, score_uidr in self.rm_judge_gp_score(uids[i], rm_responses_gp[i]).items():
                    gp_scores_dict[uid_tuple] = score_uidr
                    uid2gprm[uids[i]] = rm_responses_gp[i]
            # uid -> data_i
            for i in range(len(data)):
                param = params[i]

                rule_gp = gp_scores_dict.get((param['uid'], int(param['uidr'])), float('nan'))
                if rule_gp == rule_gp:
                    rule_gp = {'score': rule_gp, "gp_judge": rule_gp, "gp_judge_valid": 1}
                else:
                    rule_gp = {'score': 0, "gp_judge": rule_gp, "gp_judge_valid": 0}
                gp_rm_scores.append(rule_gp)
                backtrace[i]['rm_response_gp'] = uid2gprm.get(param['uid'])

        else:
            gp_rm_scores = [{'score': 0} for i in range(len(data))]
        total_scores.append(gp_rm_scores)

        # merge scores
        total_scores_for_accmu = list(zip(*total_scores))
        for i in range(len(data)):
            param = params[i]
            score = reduce(or_, total_scores_for_accmu[i])
            score['score'] = sum([x['score'] for x in total_scores_for_accmu[i]])

            scores.append(score)
            data_source = param['data_source']
            if already_printed.get(data_source, 0) < 0:
                prompt_str = self.tokenizer.decode(prompt_ids[i], skip_special_tokens=True)
                already_printed[data_source] = already_printed.get(data_source, 0)+1
                print("[judge prompt]", prompt_str)
                print("[response]", param['solution_str'])
                print("[judge_result]", backtrace[i].get('rm_response'))
                print("[judge_gp_result]", backtrace[i].get('rm_response_gp'))
                print("[ground_truth]", param['ground_truth'])
                print("[score]", scores[i])

        return scores, ce_weights, all_sames

    def __call__(self, data: DataProto, return_dict=False):
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]
        # print(data.non_tensor_batch['uid'])
        # print(data.non_tensor_batch['uidr'])
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch[self.reward_fn_key]

        # scores = await self.verify(data)

        with simple_timer('verify async', {}):
            scores, ce_weights, all_sames = asyncio.run(self.verify(data))
        rewards = []
        already_printed = {}

        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            score = scores[i]
            ce_weight = ce_weights[i]
            all_same = all_sames[i]
            if isinstance(score, dict):
                reward = score["score"] * ce_weight
                reward_extra_info['reward_info/ce_weight'].append(ce_weight)
                reward_extra_info['reward_info/all_same'].append(all_same)
                for key, value in score.items():
                    reward_extra_info[f"reward_info/{key}"].append(value)
            else:
                reward = score * ce_weight
            rewards.append(reward)
            reward_tensor[i, length - 1] = reward

            data_source = data_sources[i]
            if already_printed.get(data_source, 0) < self.num_examine:
                response_str = self.tokenizer.decode(data.batch["responses"][i][:length], skip_special_tokens=True)
                prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
                ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", scores[i])
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor
