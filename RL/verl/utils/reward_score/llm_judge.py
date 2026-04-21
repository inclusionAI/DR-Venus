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

from __future__ import annotations

import os
import random
import re
from multiprocessing import Pool

import requests


# ---------------------------------------------------------------------------
# Judge Prompt — aligned with evaluation pipeline (BrowseComp standard)
# ---------------------------------------------------------------------------

JUDGE_PROMPT_BC = """"Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}
[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.
[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there
is no confidence score available."""


# ---------------------------------------------------------------------------
# LLM Judge API
#
# Configuration is read LAZILY from environment variables inside
# ``_get_judge_config`` (NOT at module import time) to avoid the classic
# "env read too early and frozen to None" trap when this module happens to
# be imported before ``train_igpo.sh`` has sourced ``.env`` or before Ray
# has applied ``runtime_env.env_vars`` to a worker process.
#
# Required env vars (set in project-root ``.env`` and sourced by
# ``train_igpo.sh`` via ``set -a; source .env; set +a``):
#
#   JUDGE_API_BASE     - OpenAI-compatible endpoint base URL for the judge
#                        model. Falls back to API_BASE if unset.
#                        Example: https://api.example.com/v1
#   JUDGE_API_KEY      - Bearer token for JUDGE_API_BASE. Falls back to
#                        API_KEY if unset. Leave empty for unauthenticated
#                        endpoints (e.g. local vLLM without auth).
#   JUDGE_MODEL_NAME   - Model id on the judge endpoint. Default:
#                        "Qwen3-235B-A22B-Instruct-2507".
#   ENABLE_JUDGE_THINKING
#                      - If "true" (default), add Qwen/vLLM-specific
#                        ``chat_template_kwargs.enable_thinking`` to the
#                        request payload. Set to "false" for vanilla OpenAI
#                        or strict-mode endpoints that reject unknown fields.
#
# Ray worker note: the keys above MUST also appear in
# ``verl.trainer.constants_ppo._PASSTHROUGH_FROM_OS`` so that
# ``runtime_env.env_vars`` forwards them to remote Ray workers when
# ``reward_model.launch_reward_fn_async=true`` or on multi-node clusters.
# ---------------------------------------------------------------------------


def _get_judge_config():
    """Read judge API configuration from environment variables on every call.

    Returns a 4-tuple: (api_base, api_key, model_name, enable_thinking).
    """
    base = os.environ.get("JUDGE_API_BASE") or os.environ.get("API_BASE")
    key = os.environ.get("JUDGE_API_KEY") or os.environ.get("API_KEY", "")
    model = os.environ.get("JUDGE_MODEL_NAME", "Qwen3-235B-A22B-Instruct-2507")
    enable_thinking = os.environ.get("ENABLE_JUDGE_THINKING", "true").lower() == "true"
    return base, key, model, enable_thinking


def call_judge_api(prompt: str, api_url: str, model_name: str,
                   api_key: str = "", enable_thinking: bool = True,
                   temperature: float = 0.1, max_retries: int = 3) -> str:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": 0.95,
        "n": 1,
    }
    if enable_thinking:
        # Qwen / vLLM extension; vanilla OpenAI silently ignores unknown keys.
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, json=payload, headers=headers, timeout=300)
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            if content and content.strip():
                return content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                return f"Error: {e}"
    return "Error: all retries exhausted"


def parse_judge_result(raw: str) -> bool | None:
    """Extract correct: yes/no from judge response."""
    m = re.search(r"correct\s*:\s*(yes|no)\b", raw, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "yes"
    return None


def get_judge_response(prompt: str) -> str:
    base, api_key, model_name, enable_thinking = _get_judge_config()
    if not base:
        raise RuntimeError(
            "[LLM-Judge] No API base configured. Set JUDGE_API_BASE or API_BASE "
            "in the project-root .env file. `train_igpo.sh` must source it with "
            "`set -a; source .env; set +a` BEFORE launching Python.")
    base = base.rstrip("/")
    # Append /chat/completions unless user already supplied the full endpoint.
    if base.endswith("/chat/completions"):
        api_url = base
    else:
        api_url = base + "/chat/completions"
    return call_judge_api(prompt, api_url, model_name, api_key=api_key,
                          enable_thinking=enable_thinking)


# ---------------------------------------------------------------------------
# Main scoring entry point
# ---------------------------------------------------------------------------

def compute_score(prompts_str: str, predict_str: str, ground_truth, data_source: str, isvalid, tokenizer) -> float:

    key_path = []
    isreturnlist = False
    if type(ground_truth) != str and type(ground_truth) == list:
        key_path = ground_truth
        ground_truth = ground_truth[-1]
        isreturnlist = True

    question_str = prompts_str.split("user\n")[1].split("assistant\n")[0]

    try:
        answer_match = re.search(r'<answer>(.*?)</answer>', predict_str.split("assistant\n")[-1], re.DOTALL)
        if answer_match:
            result_str = answer_match.group(1).strip()
        else:
            result_str = ''
    except Exception:
        result_str = ''

    if result_str.strip() == ground_truth.strip():
        result_accuracy = 1.0
    elif result_str.strip() in ('', 'answer here'):
        result_accuracy = 0.0
    else:
        prompt = JUDGE_PROMPT_BC.format(
            question=question_str,
            correct_answer=ground_truth,
            response=result_str,
        )
        raw_response = get_judge_response(prompt)
        if random.random() < 0.005:
            print('[llm_judge prompt]', prompt)
            print('[llm_judge response]', raw_response)

        judge_result = parse_judge_result(raw_response)
        if judge_result is True:
            result_accuracy = 1.0
        elif judge_result is False:
            result_accuracy = 0.0
        else:
            print(f"[llm_judge] parse returned None, defaulting to 0.0. "
                  f"raw_response: {raw_response[:200]}")
            result_accuracy = 0.0

    total_score = result_accuracy

    if isreturnlist:
        total = total_score
        if total == 1.0:
            for key in key_path[:-1]:
                if key in predict_str:
                    total += 0.1
        return total
    else:
        return total_score


def compute_score_batch(prompts_strs: list, predict_strs: list, ground_truths: list, data_sources: list, default_batch_size=32, isvalid=True, tokenizer=None) -> list:

    if len(predict_strs) != len(ground_truths):
        raise ValueError("Number of predictions and ground truths do not match.")

    batch_size = min(default_batch_size, len(predict_strs))

    inputs = list(zip(prompts_strs, predict_strs, ground_truths, data_sources,
                      [isvalid] * len(data_sources), [tokenizer] * len(data_sources)))

    with Pool(processes=batch_size) as pool:
        scores = pool.starmap(compute_score, inputs)

    return scores
