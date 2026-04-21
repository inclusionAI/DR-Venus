#!/bin/bash

set -x

# Load project-local .env first (API_KEY / SERPER_KEY_ID / JUDGE_MODEL_NAME /
# GLOO_SOCKET_IFNAME, etc.). `set -a` auto-exports every variable assigned by
# the sourced file so that child processes inherit them.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
else
    echo "[train_igpo.sh] WARNING: $SCRIPT_DIR/.env not found; copy .env.example to .env and fill in credentials." >&2
fi

export GRPC_PYTHON_BUILD_WITH_CYTHON=1 

# ── Run identity (used by wandb / tensorboard / checkpoint dirs) ──────────
# Set these to whatever you want to see in your experiment tracker.
export project_name="PROJECT_NAME"
export RAY_memory_monitor_refresh_ms=0
export PYTHONFAULTHANDLER=1
export TORCH_DISABLE_ADDR2LINE=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ENABLE_MONITORING=0

export experiment_name="EXPERIMENT_NAME"
export PET_NODE_RANK=0

# ── Paths (EDIT THESE for your environment) ───────────────────────────────
# MODEL_PATH  : local path (or HF repo id) of the starting checkpoint.
# OUTPUT      : directory for checkpoints, rollout dumps, and training.log.
# EVAL_LOG_PATH: directory for validation traces.
# All three may be absolute or relative to this script's working dir.
export MODEL_PATH="/path/to/your/base_model"
export OUTPUT="./output"
export EVAL_LOG_PATH="./eval_log"
export _GPU_NUM=`nvidia-smi -L | wc -l`
mkdir -p "$OUTPUT"
mkdir -p "$EVAL_LOG_PATH"

# ── Training Control ──
TOTAL_TRAINING_STEPS=""                # e.g. "10" for quick verification; "" = derive from epochs

# ── Info-Gain Reward (IGPO) Configuration ──
USE_INFO_GAIN=true                    # master switch
INFO_GAIN_TYPE="log_prob_diff"        # "prob_diff" or "log_prob_diff"
INFO_GAIN_NORM_MODE="scaled_separate"           # "joint", "separate", "scaled_separate", or "raw_ig" (recommended: solves turn-shortening)
IG_COMPUTE_FREQ=1                     # info-gain compute frequency (1=every turn)
IG_TOOL_FILTER="visit"         # only compute IG on these tool turns (e.g. "search,visit"); empty=all turns
IG_WEIGHT=1.0                        # IG signal multiplier after normalization (>1 amplifies, <1 dampens, 1.0 = no change)
USE_FORMAT_PENALTY=true               # turn-level format penalty for malformed assistant outputs
FORMAT_PENALTY_SCALE=1.0              # penalty magnitude (applied as -1.0 * scale)
# Rollout-IG (EXPERIMENTAL): compute IG during rollout via vLLM prompt_logprobs
# instead of the post-rollout ig_kv_cache phase on the FSDP actor (~10-15%
# faster, but vLLM vs FSDP logprobs may diverge numerically). Keep false for
# paper-reproducibility; set IGPO_CROSSCHECK=N to sample-validate when enabled.
USE_ROLLOUT_IG=false
TRAIN_REWARD_TYPE="llm"               # "llm" (LLM judge) or "f1"
MASK_TOOL_RESPONSE=true               # mask tool_response tokens in policy loss

# ── Rollout Mode ──
USE_ASYNC_ROLLOUT=true                # true=async agent loop, false=sync generation
ASYNC_NUM_WORKERS=8                   # number of AgentLoopWorker Ray actors
ASYNC_MAX_CONCURRENT_SAMPLES=""       # "" = disabled; e.g. "16" to limit concurrency
ASYNC_COMPLETION_CUTOFF="0.9"         # "" = disabled; e.g. "0.9" = cut off stragglers when 90% done

# ── Rollout Robustness (optional; default OFF) ──
# Retries a turn on degenerate output / parse errors / duplicate tool calls.
# Recommended for long-horizon runs (max_turns >> 20); off by default for
# deterministic, paper-reproducible rollouts.
USE_ROBUST_ROLLOUT=false              # master switch
ROBUST_MAX_RETRIES=3                  # total attempts = this + 1
ROBUST_DEDUP=true                     # skip duplicate (name, args) tool calls
ROBUST_REPETITION=true                # retry on repeating tail (>5× of last 50 chars)

# ── Sequence Length Configuration ──
# MAX_MODEL_LEN      : vLLM context window (KV cache + per-turn truncation)
# MAX_PROMPT_LEN     : dataset filter + sync prompt padding
# MAX_RESPONSE_LEN   : per-turn generation cap (max_tokens)
# ULYSSES_SP_SIZE    : sequence parallel degree
# ASYNC_PROMPT_PAD   : (async only) left-pad for the initial question; in
#                      async mode "response" holds all turns' content.
# Training seq layout:
#   Sync:   [prompt padded to MAX_PROMPT_LEN   | single response]
#   Async:  [prompt padded to ASYNC_PROMPT_PAD | all turns' content]
# PPO_MAX_TOKEN_LEN is auto-computed so that
#   PPO_MAX_TOKEN_LEN × ULYSSES_SP_SIZE ≥ max training sequence length

MAX_MODEL_LEN=261000                  # model context window
MAX_PROMPT_LEN=246000                 # dataset filter + sync prompt padding
MAX_RESPONSE_LEN=8192                # per-turn generation limit
ULYSSES_SP_SIZE=8                     # sequence parallel size
ASYNC_PROMPT_PAD=1024                 # async prompt padding (≥ max initial prompt)

if [ "$USE_ASYNC_ROLLOUT" = "true" ]; then
    # Async training seq ≤ ASYNC_PROMPT_PAD + MAX_MODEL_LEN
    _max_seq=$((ASYNC_PROMPT_PAD + MAX_MODEL_LEN))
else
    # Sync training seq ≤ MAX_PROMPT_LEN + MAX_RESPONSE_LEN = MAX_MODEL_LEN
    _max_seq=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))
fi
PPO_MAX_TOKEN_LEN=$(( (_max_seq + ULYSSES_SP_SIZE - 1) / ULYSSES_SP_SIZE + 1000 ))

if [ "$USE_INFO_GAIN" = "true" ]; then
    ADV_ESTIMATOR="grpo_info_gain"
else
    ADV_ESTIMATOR="grpo"
fi

echo "Mode: $([ "$USE_ASYNC_ROLLOUT" = "true" ] && echo ASYNC || echo SYNC)"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}, MAX_PROMPT_LEN=${MAX_PROMPT_LEN}, MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN}"
[ "$USE_ASYNC_ROLLOUT" = "true" ] && echo "ASYNC_PROMPT_PAD=${ASYNC_PROMPT_PAD}"
echo "PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN} (effective: $((PPO_MAX_TOKEN_LEN * ULYSSES_SP_SIZE)))"

# Debug toggles (unset by default for best performance):
#   IGPO_VERIFY_ASYNC=1       -> enables dual-path async verification
#   IGPO_CROSSCHECK=N         -> cross-validates the first N samples between the
#                                vLLM rollout-logprob and FSDP actor-logprob paths
# Enable them only when diagnosing IG/numerical discrepancies.
HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 \
IGPO_ROLLOUT_IG=$([ "$USE_ROLLOUT_IG" = "true" ] && echo 1 || echo 0) \
IGPO_IG_COMPUTE_FREQ=${IG_COMPUTE_FREQ} \
IGPO_IG_TOOL_FILTER="${IG_TOOL_FILTER}" \
IGPO_INFO_GAIN_TYPE=${INFO_GAIN_TYPE} \
IGPO_PHASE1_CHUNK_SIZE=8192 \
ROLLOUT_ROBUST=$([ "$USE_ROBUST_ROLLOUT" = "true" ] && echo 1 || echo 0) \
ROLLOUT_ROBUST_MAX_RETRIES=${ROBUST_MAX_RETRIES} \
ROLLOUT_ROBUST_DEDUP=$([ "$ROBUST_DEDUP" = "true" ] && echo 1 || echo 0) \
ROLLOUT_ROBUST_REPETITION=$([ "$ROBUST_REPETITION" = "true" ] && echo 1 || echo 0) \
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${ADV_ESTIMATOR} \
    algorithm.use_info_gain=${USE_INFO_GAIN} \
    algorithm.info_gain_type=${INFO_GAIN_TYPE} \
    algorithm.info_gain_norm_mode=${INFO_GAIN_NORM_MODE} \
    algorithm.ig_compute_freq=${IG_COMPUTE_FREQ} \
    algorithm.ig_tool_filter="${IG_TOOL_FILTER}" \
    algorithm.ig_weight=${IG_WEIGHT} \
    algorithm.use_format_penalty=${USE_FORMAT_PENALTY} \
    algorithm.format_penalty_scale=${FORMAT_PENALTY_SCALE} \
    data.train_files=data/train.parquet \
    data.val_files=data/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESPONSE_LEN} \
    +data.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.disable_log_stats=false \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.mask_tool_response=${MASK_TOOL_RESPONSE} \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.model.enable_activation_offload=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ULYSSES_SP_SIZE} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.nccl_timeout=7200 \
    critic.optim.lr=1e-5 \
    critic.model.path=${MODEL_PATH} \
    critic.ppo_micro_batch_size_per_gpu=1 \
    algorithm.gamma=0.95 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$_GPU_NUM \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=-1 \
    trainer.validation_data_dir=${EVAL_LOG_PATH} \
    trainer.default_local_dir=${OUTPUT} \
    agent_grpo.n=8 \
    max_turns=200 \
    +tool_timeout=150 \
    +trace_save_interval=10 \
    +trace_max_samples=0 \
    search_engine=online_search \
    reward_model.train_reward_type=${TRAIN_REWARD_TYPE} \
    +reward_model.valid_reward_type=llm_em_noformatf1 \
    reward_model.reward_manager='naive_batch' \
    +reward_model.reward_kwargs.deepthink_disabled=true \
    data.return_raw_chat=true \
    trainer.total_epochs=1 \
    ${TOTAL_TRAINING_STEPS:+trainer.total_training_steps=${TOTAL_TRAINING_STEPS}} \
    $([ "$USE_ASYNC_ROLLOUT" = "true" ] && echo "actor_rollout_ref.rollout.mode=async" || true) \
    $([ "$USE_ASYNC_ROLLOUT" = "true" ] && echo "actor_rollout_ref.rollout.prompt_length=${ASYNC_PROMPT_PAD}" || true) \
    $([ "$USE_ASYNC_ROLLOUT" = "true" ] && echo "actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LEN}" || true) \
    $([ "$USE_ASYNC_ROLLOUT" = "true" ] && echo "actor_rollout_ref.rollout.agent.num_workers=${ASYNC_NUM_WORKERS}" || true) \
    $([ "$USE_ASYNC_ROLLOUT" = "true" ] && echo "actor_rollout_ref.rollout.agent.agent_loop_config_path=configs/dr_agent_loop.yaml" || true) \
    $([ "$USE_ASYNC_ROLLOUT" = "true" ] && [ -n "$ASYNC_MAX_CONCURRENT_SAMPLES" ] && [ "$ASYNC_MAX_CONCURRENT_SAMPLES" != "null" ] && echo "+actor_rollout_ref.rollout.agent.max_concurrent_samples=${ASYNC_MAX_CONCURRENT_SAMPLES}" || true) \
    $([ "$USE_ASYNC_ROLLOUT" = "true" ] && [ -n "$ASYNC_COMPLETION_CUTOFF" ] && [ "$ASYNC_COMPLETION_CUTOFF" != "null" ] && echo "+actor_rollout_ref.rollout.agent.completion_cutoff=${ASYNC_COMPLETION_CUTOFF}" || true) \
    > >(stdbuf -oL tee ${OUTPUT}/training.log) 2>&1

EXIT_CODE=${PIPESTATUS[0]:-$?}
echo "[$(date)] Training exited with code: $EXIT_CODE" | tee -a ${OUTPUT}/training.log
if [ $EXIT_CODE -ne 0 ]; then
    echo "=== dmesg (last 30 lines) ===" >> ${OUTPUT}/training.log
    dmesg -T 2>/dev/null | tail -30 >> ${OUTPUT}/training.log
    echo "=== nvidia-smi ===" >> ${OUTPUT}/training.log
    nvidia-smi >> ${OUTPUT}/training.log 2>&1
    echo "=== cgroup memory ===" >> ${OUTPUT}/training.log
    cat /sys/fs/cgroup/memory/memory.oom_control >> ${OUTPUT}/training.log 2>/dev/null
fi