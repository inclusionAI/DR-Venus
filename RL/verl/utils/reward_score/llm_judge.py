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
from pathlib import Path
import re
from multiprocessing import Pool
import sys
from urllib.parse import urlsplit, urlunsplit

import requests

try:
    from drvenus_repro.tool_cache import execute_with_tool_cache
except ModuleNotFoundError as exc:
    if exc.name not in {"drvenus_repro", "drvenus_repro.tool_cache"}:
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from drvenus_repro.tool_cache import execute_with_tool_cache


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
JUDGE_PROMPT_VERSION = "browsecomp-correctness-v1"


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
    seed = int(os.environ.get("JUDGE_SEED", "42"))
    timeout = float(os.environ.get("JUDGE_TIMEOUT_SECONDS", "300"))
    provider_id = os.environ.get("JUDGE_PROVIDER_ID", "").strip()
    return base, key, model, enable_thinking, seed, timeout, provider_id


def _endpoint_identity(base: str | None, provider_id: str = "") -> str:
    """Return a stable, credential-free provider identity for cache keys."""

    if provider_id:
        return provider_id
    if not base:
        raise RuntimeError(
            "JUDGE_API_BASE/API_BASE or JUDGE_PROVIDER_ID is required even in replay "
            "mode so Judge cache provenance cannot be mixed across providers"
        )
    parts = urlsplit(base.strip())
    if parts.scheme and parts.hostname:
        host = parts.hostname.casefold()
        if parts.port is not None:
            host += f":{parts.port}"
        path = parts.path.rstrip("/")
        if not path.endswith("/chat/completions"):
            path += "/chat/completions"
        # Drop userinfo, query, and fragment so credentials can never enter the
        # evidence cache. Provider deployments that share a URL should set an
        # explicit opaque JUDGE_PROVIDER_ID.
        return urlunsplit((parts.scheme.casefold(), host, path, "", ""))
    return base.rstrip("/") + "/chat/completions"


def call_judge_api(prompt: str, api_url: str, model_name: str,
                   api_key: str = "", enable_thinking: bool = True,
                   temperature: float = 0.0, max_retries: int = 3,
                   seed: int = 42, timeout: float = 300.0) -> str:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": 1.0,
        "n": 1,
        "seed": seed,
    }
    if enable_thinking:
        # Qwen / vLLM extension; vanilla OpenAI silently ignores unknown keys.
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            if content and content.strip():
                return content.strip()
        except Exception as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"judge request failed after {max_retries} attempts: {exc}"
                ) from exc
    raise RuntimeError("judge request failed without a response")


def parse_judge_result(raw: str) -> bool | None:
    """Extract correct: yes/no from judge response."""
    m = re.search(r"correct\s*:\s*(yes|no)\b", raw, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "yes"
    return None


def get_judge_response(prompt: str) -> str:
    base, api_key, model_name, enable_thinking, seed, timeout, provider_id = _get_judge_config()
    endpoint_identity = _endpoint_identity(base, provider_id)
    arguments = {
        "prompt": prompt,
        "model": model_name,
        "prompt_version": JUDGE_PROMPT_VERSION,
        "provider_protocol": "openai-compatible-chat-completions-v1",
        "endpoint_identity": endpoint_identity,
        "enable_thinking": enable_thinking,
        "seed": seed,
        "temperature": 0.0,
    }

    def call_online() -> str:
        if not base:
            raise RuntimeError(
                "[LLM-Judge] No API base configured. Set JUDGE_API_BASE or API_BASE "
                "before using online/record judge mode."
            )
        normalized_base = base.rstrip("/")
        api_url = (
            normalized_base
            if normalized_base.endswith("/chat/completions")
            else normalized_base + "/chat/completions"
        )
        return call_judge_api(
            prompt,
            api_url,
            model_name,
            api_key=api_key,
            enable_thinking=enable_thinking,
            temperature=0.0,
            seed=seed,
            timeout=timeout,
        )

    return execute_with_tool_cache(
        "outcome_judge",
        arguments,
        call_online,
        provider=f"openai-compatible:{endpoint_identity}:{model_name}",
        mode_env="JUDGE_CACHE_MODE",
        path_env="JUDGE_CACHE_PATH",
    )


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
        judge_result = parse_judge_result(raw_response)
        if judge_result is True:
            result_accuracy = 1.0
        elif judge_result is False:
            result_accuracy = 0.0
        else:
            raise RuntimeError(
                "[llm_judge] response did not contain a parseable `correct: yes|no`; "
                "refusing to convert infrastructure failure into reward=0"
            )

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
