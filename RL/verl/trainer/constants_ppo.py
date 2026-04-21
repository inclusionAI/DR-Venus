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

import json
import os

from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # To prevent hanging or crash during synchronization of weights between actor and rollout
        # in disaggregated mode. See:
        # https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
        # https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
        "NCCL_CUMEM_ENABLE": "0",
    },
}


# ---------------------------------------------------------------------------
# Open-source passthrough: environment variables that MUST be forwarded from
# the driver's ``os.environ`` into every Ray worker's ``runtime_env.env_vars``.
#
# Why this exists:
#   - On multi-node Ray clusters, workers on remote nodes do NOT inherit the
#     driver's OS env via normal process fork.
#   - ``@ray.remote compute_reward_async`` (enabled by
#     ``reward_model.launch_reward_fn_async=true``) runs in a separate Ray
#     worker process — same propagation concern.
#   - ``tool_server.tool_search`` reads SERPER_KEY_ID at MODULE IMPORT TIME
#     into a frozen module-level constant. If the var isn't in env when the
#     worker starts Python, the key is permanently ``None`` in that worker.
#
# If you add a new ``os.environ.get(...)`` read-point anywhere in the
# project, append the key here, or it will silently break multi-node
# training / async-reward paths.
# ---------------------------------------------------------------------------
_PASSTHROUGH_FROM_OS = [
    # LLM gateway (shared by summary extractor and LLM judge by default)
    "API_KEY", "API_BASE", "SUMMARY_MODEL_NAME",
    # LLM judge (optional separate gateway + model selector + thinking toggle)
    "JUDGE_API_KEY", "JUDGE_API_BASE", "JUDGE_MODEL_NAME", "ENABLE_JUDGE_THINKING",
    # Web search (Serper) — note: tool_search.py freezes this at import time
    "SERPER_KEY_ID",
    # Web visit (Jina)
    "JINA_API_KEYS", "PROXY",
    "VISIT_SERVER_TIMEOUT", "VISIT_SERVER_MAX_RETRIES", "WEBCONTENT_MAXLENGTH",
    # Distributed networking (Gloo/NCCL NIC selection)
    "GLOO_SOCKET_IFNAME",
]


def get_ppo_ray_runtime_env():
    """
    A filter function to return the PPO Ray runtime environment.
    To avoid repeat of some environment variables that are already set.
    """
    working_dir = (
        json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}")).get("runtime_env", {}).get("working_dir", None)
    )

    runtime_env = {
        "env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy(),
        **({"working_dir": None} if working_dir is None else {}),
    }
    for key in list(runtime_env["env_vars"].keys()):
        if os.environ.get(key) is not None:
            runtime_env["env_vars"].pop(key, None)

    # ★ ORDER-CRITICAL: the passthrough loop MUST run AFTER the filter above.
    # The filter pops any key already in ``os.environ``; our passthrough keys
    # are (by design) populated in ``os.environ`` by ``train_igpo.sh``'s
    # ``source .env``. If we added them BEFORE the filter, they'd be popped.
    for k in _PASSTHROUGH_FROM_OS:
        v = os.environ.get(k)
        if v is not None:
            runtime_env["env_vars"][k] = v
    return runtime_env
