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

import requests
import json
import json5
from multiprocessing import Pool
from functools import partial
from openai import OpenAI
import re
import difflib
import string
import os
import traceback
import random
import numpy as np
import time
from tools_server.http_client import get_sepc_llms, TaskType


def call_api(record, model_name, prompt_name='prompt',
             api_url="http://localhost:8000/v1/chat/completions",
             use_new_format=False, headers=None,
             n=1, stop_token=None, top_p=0.9, top_k=40, temperature=0.7):
    """
    处理单条记录，根据 use_new_format 参数决定使用哪种请求格式调用 API。

    参数:
      record (dict): 一条记录（例如包含 "prompt" 字段）
      model_name (str): 模型名称，作为 payload 的参数之一
      prompt_name (str): 记录中用于查找用户输入的键名，默认 'prompt'
      api_url (str): API 的 URL 地址
      use_new_format (bool): 是否使用新格式调用 API，默认 False
      headers (dict): 仅在 use_new_format 为 True 时使用，指定 HTTP 请求头
      n (int): 生成的个数，默认为 1
      stop_token (str or list): 生成时的停止标记，默认为 None
      top_p (float): nucleus sampling 参数，默认为 0.9
      top_k (int): top-k 采样参数，默认为 40
      temperature (float): 温度参数，默认为 0.7

    返回:
      dict: 包含输入 prompt 及 API 返回结果，或包含错误信息。
    """
    prompt = record.get(prompt_name, "")

    # 旧格式 payload
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        # "max_length": 8192,   # 生成文本的最大长度
        "top_p": top_p,
        "top_k": top_k,
        "temperature": temperature,
        "n": n,
        "stop": stop_token
    }

    headers = {}
#     headers["Authorization"] = f"Bearer 123"
    exception = None
    for i in range(3):
        try:
            response = requests.post(api_url, json=payload, headers=headers, timeout=60*60)
            response.raise_for_status()  # 非200状态码将抛出异常
            result = response.json()
            return {"prompt": prompt, "payload": payload, "result": result}
        except Exception as e:
            exception = e
        time.sleep((i + 1)*10)
    return {"prompt": prompt, "payload": payload, "error": str(exception)}


def get_model_response(prompt):
    model_name = "auto"  # 根据实际情况替换
#     api_url = "http://33.212.71.248:18083/v1/chat/completions"
    api_url = get_sepc_llms("/home/admin/Qwen3-235B-A22B-Instrcut-2507")
    if not api_url:
        # 回退到雨鹰的版本
        api_url = "https://aistudioproxy.alipay.com/proxy/workflow_47480056:3002/v1/chat/completions"
    use_new_format = False

    # 新增的参数，可以根据需求进行调整
    n = 1
    stop_token = None
    top_p = 0.95
    top_k = 50
    temperature = 0.8
    inputs = {}
    inp = prompt
    inputs['prompts'] = inp
    headers_new = {'Content-Type': 'application/json'}

    try:
        results = call_api(
            inputs,
            model_name,
            prompt_name="prompts",
            api_url=api_url,
            use_new_format=use_new_format,
            headers=headers_new,
            n=n,
            stop_token=stop_token,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature
        )
        output = results['result']['choices'][0]['message']['content']
    except Exception as e:
        # 处理异常情况：可以返回一个默认值或者抛出异常
        output = f"Error: Unable to fetch response due to {str(e)}。\n\nrawprompt {prompt}。\n\nrawcontent {results}"

    return output


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
                data = json5.loads(json_str)
            except Exception as e:
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


def check_tags_balance(solution_str: str) -> bool:
    """检查标签是否正确配对

    Args:
        solution_str: 需要检查的字符串

    Returns:
        bool: 标签是否都正确配对
    """
    # 需要检查的标签对
    tags_to_check = ['tool_call', 'think', 'answer']

    for tag in tags_to_check:
        # 计算开始和结束标签的数量
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"

        start_count = solution_str.count(start_tag)
        end_count = solution_str.count(end_tag)

        # 如果开始和结束标签数量不相等，返回False
        if start_count != end_count:
            return False

        # 检查标签的嵌套顺序（确保结束标签不会在开始标签之前出现）
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
    """预处理文本，用于数据集的评分

    处理步骤:
    1. 转换为小写
    2. 移除标点符号 (.,!?;:'"()[]{}...)
    3. 去除多余空格
    """
    # 将标点符号替换为空格
    for punct in string.punctuation:
        text = text.replace(punct, ' ')

    # 替换多个空格为单个空格
    text = re.sub(r'\s+', ' ', text)

    # 去除首尾空格
    text = text.strip()
    return text


def compute_f1(solution_str, ground_truth, val_type='f1') -> float:
    solution_str = solution_str.lower()
    ground_truth = ground_truth.lower()
    ground_truths = ground_truth.split("<|answer_split|>")
    # 首先检查标签是否配对正确(格式是否正确)
    if not check_tags_balance(solution_str):

        if val_type == 'noformatf1':
            return 0
        else:
            return -2.0
    # 使用正则提取第一个<answer>标签中的内容
    try:
        answer_match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            # 对答案进行预处理
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
        # 对ground truth进行预处理
        gt = preprocess_text(gt)

        if val_type == 'em':
            if gt == answer_content:
                return 1.0
        else:
            # 将答案和参考答案分词
            pred_tokens = set(answer_content.split())
            gt_tokens = set(gt.split())

            if not gt_tokens:  # 避免除零错误
                continue
            if not pred_tokens:
                continue

            # 计算共同的词数
            common_tokens = pred_tokens & gt_tokens

            # 计算精确率和召回率
            precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
            recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0

            # 计算F1分数
            if precision + recall > 0:  # 避免除零错误
                f1 = 2 * (precision * recall) / (precision + recall)
                max_score = max(max_score, f1)

    return max_score


def compute_score(prompts_str: str, predict_str: str, ground_truth, data_source: str, isvalid, tokenizer) -> float:

    key_path = []
    isreturnlist = False
    if type(ground_truth) != str and type(ground_truth) == list:
        key_path = ground_truth
        ground_truth = ground_truth[-1]
        isreturnlist = True

    turns = predict_str.count("assistant\n") - predict_str.count('Your last response was incomplete')  # 有效轮数
    format_reward_score = True
    for e in predict_str.split("user\n"):
        if "assistant\n" in e:
            if not check_tags_balance(e.split("assistant\n")[-1]):
                format_reward_score = False
    question_str = prompts_str.split("user\n")[1].split("assistant\n")[0]
    process_str = predict_str.split("<answer>")[0]

    try:
        answer_match = re.search(r'<answer>(.*?)</answer>', predict_str.split("assistant\n")[-1], re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            # 对答案进行预处理
            result_str = (answer_content)
        else:
            result_str = ''
    except Exception as e:
        result_str = ''
    if data_source == 'health_fact':
        prompts = f'''请检查以下 pred answer 输出是否符合要求，并根据评分细则给出一个打分。具体要求如下：
<question> {question_str} </question>
<ground truth answers> {ground_truth} </ground truth answers>
<pred answer> {result_str} </pred answer>

评分细则：
1. 整体结构：
   - 必须是一个完整的 JSON 对象，顶层有且只有这几个字段（键名必须完全一致）：
     - “cot”
     - “claim”
     - “result”
     - “search result”
     - “evidence”
   - 各字段顺序、标点等应与示例保持一致，不能多余字段，也不能缺少字段。

2. “cot” 字段（思考过程 / Chain of Thought）(20分)：
   - 内容需使用 Markdown 格式进行展示。
   - Markdown 语法要正确，比如标题用“#” 或“##”，列表项要统一使用“-” 或“*”。
   - 可以展示思考过程。

3. “claim” 字段（谣言或待验证论断）(20分)：
   - 必须是一个简短的字符串，与示例类似，如 “眼皮跳是疾病前兆”。
   - 不能省略，也不能写其他与主题无关的内容。

4. “result” 字段（可信度得分）(20分)：
   - 必须是一个介于 0 到 100 之间的整数（不要带小数）。
   - 要和ground truth answers 一致（100代表True/0代表False），得分 = 20分 * $result * $ground truth answers

5. “search result” 字段（检索到的网站域名列表）(20分)：
   - 必须是一个 JSON 数组，数组元素为字符串，代表已查看的网址域名（只保留域名部分，不要带 http://、路径等）。
   - 每个元素都要用双引号括起，元素之间用逗号分隔。

6. “evidence” 字段（证据链）(20分)：
   - 内容需使用 Markdown 格式进行展示。内容简明直接有条理。
   - 搜索到的材料里面正方反方结论都要列出来，并简单总结，然后再去判断。
   - 判断部分必须是一个简短的段落，说明判断依据，里面要引用具体的来源和相关信息。
   - 要包含对证据页面比如 “中华医学会”或“XX 医院”、“XX 政府网站”等可信来源的指引，并标注其域名，比如 “根据中华医学会的报道……（搜索到的具体的网址）。”、“兰陵县人民医院指出……（搜索到的具体的网址）。”、“建湖县人民政府提到……（搜索到的具体的网址）。”。
   - 语句要紧扣主题，以及引用哪些网站的哪些信息支撑该判断。
   - 展示网站必须要全部网址，不能只有域名。

7.pred answer为空直接得0分

请根据以上规则，给出得分。如果答案和问题的语言不一致，则得一半分。

The output should in the following json format:
{{
"rationale": "your rationale for the judgement, as a text",
"judgement”: "your judgement result, can only be 0～100 "
}}

Your output:'''
    else:
        prompts = f'''Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question_str} [/question]

[response]: {result_str} [/response]

[correct_answer]: {ground_truth} [/correct_answer]

Your task is to judge semantic equivalence, NOT strict string matching.

Follow these rules strictly:

1. Treat the answer as CORRECT if the extracted answer is semantically equivalent to the [correct_answer], even if:
   - It uses abbreviations, shortened forms, or alternative standard names
   - It paraphrases or rephrases the same concept
   - It provides a representative value within an explicitly stated acceptable range
   - It adds non-contradictory descriptive qualifiers that do not change the core meaning

2. Mark the answer as INCORRECT only if there is a meaningful scientific, mathematical,
   or conceptual difference, including but not limited to:
   - A different physical quantity 
   - A value outside the valid range
   - A change in definition, scope, or dimension
   - Ambiguity that makes the answer non-equivalent to the [correct_answer]

3. Do NOT penalize:
   - Missing honorifics, prefixes, or historical attributions if the concept is the same
   - Differences in units formatting 
   - Additional explanatory text that does not contradict the correct answer

Your output must be a JSON object in the following format:

{{
  "extracted_final_answer": <final exact answer extracted from the response, or "None">,
  "reasoning": "Explain only whether the extracted answer is semantically equivalent to the correct answer. Do not introduce new facts.",
  "judgement": "correct" or "incorrect",
  "confidence": "0–100"
}}

Your output:'''

    if '<summary>' in predict_str and '</summary>' in predict_str:
        result_accuracy = 0.0
    elif result_str.strip() == ground_truth.strip():
        result_accuracy = 1.0
    elif result_str.strip() == 'answer here':
        result_accuracy = 0
    elif result_str.strip() == '':
        result_accuracy = -0.2
    else:
        eval_response = get_model_response(prompts)
        if random.random() < 0.005:
            print('prompts', prompts)
            print('eval_response', eval_response)
        json_score = extract_json_from_text(eval_response)
        search_coverage = 0.0
        reasoning_rigor = 0.0
        result_accuracy = 0.0
        try:
            if json_score['judgement'] == 'correct':
                json_score['judgement'] = 100
            if json_score['judgement'] == 'incorrect':
                json_score['judgement'] = 0
            result_accuracy = float(json_score['judgement'])/100
        except:
            print(f"{str(traceback.format_exc())}")
            print(f"parse socre error is {eval_response}")

    total_score = result_accuracy
    if not isvalid:
        if total_score == 0:
            if turns <= 5:
                total_score -= 0.25
            elif turns <= 10:
                total_score -= 0.15
            elif turns <= 15:
                total_score -= 0.10
            elif turns <= 25:
                total_score -= 0.05
            elif turns <= 40:
                total_score -= 0.02

            if result_str == '':
                total_score -= 0.2
        if total_score == 1:
            if turns <= 5:
                total_score -= 0.1

    if isreturnlist:
        input_ids = tokenizer(predict_str, add_special_tokens=False)['input_ids']
        scores_list = [0.0] * len(input_ids)
        # 给过程打分
        bad_tool_call = [" unable to arrive at an answer, be sure to invoke a tool",
                         "The response is missing", "model tool call format"]
        assistant_ids = tokenizer("assistant\n", add_special_tokens=False)["input_ids"]
        user_ids = tokenizer("user\n", add_special_tokens=False)["input_ids"]

        # def match_at(pos, pattern):
        #     return input_ids[pos:pos + len(pattern)] == pattern

        # segments = []
        # segment_ranges = []
        # i = 0
        # while i < len(input_ids):
        #     if match_at(i, assistant_ids):
        #         start = i
        #         i += len(assistant_ids)
        #     elif match_at(i, user_ids):
        #         start = i
        #         i += len(user_ids)
        #     else:
        #         i += 1
        #         continue

        #     j = i
        #     while j < len(input_ids):
        #         if match_at(j, assistant_ids) or match_at(j, user_ids):
        #             break
        #         j += 1

        #     segments.append(input_ids[start:j])
        #     segment_ranges.append((start, j))
        #     scores_list[j - 1] = 1e-5
        #     i = j

        # seen_keys = set()
        # for idx, seg in enumerate(segments[:-1]):
        #     text = tokenizer.decode(seg, skip_special_tokens=True)
        #     for key in key_path:
        #         if key in seen_keys:
        #             continue
        #         if key.lower() in text.lower():
        #             seen_keys.add(key)
        #             start, end = segment_ranges[idx]
        #             if segments[idx][:len(assistant_ids)] == assistant_ids and scores_list[end - 1] == 1e-5:
        #                 if random.random() > 0.95:
        #                     print('good', key, tokenizer.decode(segments[idx], skip_special_tokens=True))
        #                 scores_list[end - 1] += key_path.index(key) * 0.5/len(key_path)
        #             elif idx - 1 >= 0:
        #                 prev_seg = segments[idx - 1]
        #                 prev_start, prev_end = segment_ranges[idx - 1]

        #                 if prev_seg[:len(assistant_ids)] == assistant_ids and scores_list[prev_end - 1] == 1e-5:
        #                     scores_list[prev_end - 1] += key_path.index(key) * 0.5/len(key_path)
        #                     if random.random() > 0.9:
        #                         print('good', key, tokenizer.decode(segments[idx - 1], skip_special_tokens=True))
        scores_list[-1] = total_score

        if max(scores_list) > 1:
            print('scores_list', scores_list[-1], np.argmax(scores_list), max(scores_list))
        if scores_list[-1] == 1.0:
            # 每多对一个要素 + 0.1分
            print('result_str', result_str, '||ground_truth', ground_truth, '|||')
            seen_keys = set()
            for key in key_path[:-1]:
                if key in predict_str:
                    scores_list[-1] += 0.1
        return scores_list[-1]
    else:
        return total_score


def compute_score_batch(prompts_strs: list, predict_strs: list, ground_truths: list, data_sources: list, default_batch_size=32, isvalid=True, tokenizer=None) -> list:

    if len(predict_strs) != len(ground_truths):
        raise ValueError("预测字符串和参考答案的数量不一致！")

    batch_size = min(default_batch_size, len(predict_strs))

    # 将输入数据封装为元组列表
    inputs = list(zip(prompts_strs, predict_strs, ground_truths, data_sources,
                      [isvalid] * len(data_sources), [tokenizer] * len(data_sources)))

    # 定义多进程池
    with Pool(processes=batch_size) as pool:
        # map 函数分发任务给多个进程
        scores = pool.starmap(compute_score, inputs)

    return scores
