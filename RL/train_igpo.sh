#!/usr/bin/env bash

set -euo pipefail

# Load project-local .env first (API_KEY / SERPER_KEY_ID / JUDGE_MODEL_NAME /
# GLOO_SOCKET_IFNAME, etc.). `set -a` auto-exports every variable assigned by
# the sourced file so that child processes inherit them.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOTENV_PATH=${DOTENV_PATH:-$SCRIPT_DIR/.env}
if [ -f "$DOTENV_PATH" ]; then
    # A launcher-provided environment is the experiment contract. Dotenv may
    # fill undefined defaults but cannot override any pre-existing variable.
    # shellcheck source=load_dotenv_defaults.sh
    source "$SCRIPT_DIR/load_dotenv_defaults.sh"
    load_dotenv_defaults "$DOTENV_PATH"
else
    echo "[train_igpo.sh] WARNING: $DOTENV_PATH not found; copy .env.example to .env and fill in credentials." >&2
fi

# Never xtrace while sourcing credentials. Opt in only for non-secret debugging.
TRACE_SHELL=${TRACE_SHELL:-false}
if [ "$TRACE_SHELL" = "true" ]; then
    set -x
fi

export GRPC_PYTHON_BUILD_WITH_CYTHON=1 

# ── Run identity (used by wandb / tensorboard / checkpoint dirs) ──────────
# Set these to whatever you want to see in your experiment tracker.
PROJECT_NAME=${PROJECT_NAME:-dr-venus-repro}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-igpo-run}
export project_name="$PROJECT_NAME"
export RAY_memory_monitor_refresh_ms=0
export PYTHONFAULTHANDLER=1
export TORCH_DISABLE_ADDR2LINE=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ENABLE_MONITORING=0

export experiment_name="$EXPERIMENT_NAME"
export PET_NODE_RANK=0

# ── Paths (EDIT THESE for your environment) ───────────────────────────────
# MODEL_PATH  : local path (or HF repo id) of the starting checkpoint.
# OUTPUT      : directory for checkpoints, rollout dumps, and training.log.
# EVAL_LOG_PATH: directory for validation traces.
# All three may be absolute or relative to this script's working dir.
export MODEL_PATH=${MODEL_PATH:-}
export OUTPUT=${OUTPUT:-./output}
export EVAL_LOG_PATH=${EVAL_LOG_PATH:-./eval_log}
TRAIN_FILES=${TRAIN_FILES:-data/train.parquet}
VAL_FILES=${VAL_FILES:-data/test.parquet}
GPU_NUM=${GPU_NUM:-}
if [ -z "$GPU_NUM" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_NUM=$(nvidia-smi -L | wc -l)
    else
        GPU_NUM=1
    fi
fi
export _GPU_NUM=$GPU_NUM
DRY_RUN=${DRY_RUN:-true}
CONFIRM_SPEND=${CONFIRM_SPEND:-NO}

if [ -z "$MODEL_PATH" ]; then
    echo "[train_igpo.sh] ERROR: MODEL_PATH must be set." >&2
    exit 2
fi

if [ "$DRY_RUN" != "true" ]; then
    if [ "$CONFIRM_SPEND" != "YES" ]; then
        echo "[train_igpo.sh] ERROR: refusing live RL. Review the dry run/pilot budget, then set CONFIRM_SPEND=YES." >&2
        exit 3
    fi
    mkdir -p "$OUTPUT"
    mkdir -p "$EVAL_LOG_PATH"
fi

# ── Training Control ──
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
GROUP_SIZE=${GROUP_SIZE:-8}
MAX_TURNS=${MAX_TURNS:-200}
TURN_JITTER=${TURN_JITTER:-5}
SEED=${SEED:-42}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}

# ── Info-Gain Reward (IGPO) Configuration ──
USE_INFO_GAIN=${USE_INFO_GAIN:-true}
INFO_GAIN_TYPE=${INFO_GAIN_TYPE:-log_prob_diff}
INFO_GAIN_NORM_MODE=${INFO_GAIN_NORM_MODE:-scaled_separate}
IG_COMPUTE_FREQ=${IG_COMPUTE_FREQ:-1}
IG_TOOL_FILTER=${IG_TOOL_FILTER:-visit}
IG_WEIGHT=${IG_WEIGHT:-1.0}
USE_FORMAT_PENALTY=${USE_FORMAT_PENALTY:-true}
FORMAT_PENALTY_SCALE=${FORMAT_PENALTY_SCALE:-1.0}
# Rollout-IG (EXPERIMENTAL): compute IG during rollout via vLLM prompt_logprobs
# instead of the post-rollout ig_kv_cache phase on the FSDP actor (~10-15%
# faster, but vLLM vs FSDP logprobs may diverge numerically). Keep false for
# paper-reproducibility; set IGPO_CROSSCHECK=N to sample-validate when enabled.
USE_ROLLOUT_IG=${USE_ROLLOUT_IG:-false}
TRAIN_REWARD_TYPE=${TRAIN_REWARD_TYPE:-llm}
MASK_TOOL_RESPONSE=${MASK_TOOL_RESPONSE:-true}

# ── Rollout Mode ──
USE_ASYNC_ROLLOUT=${USE_ASYNC_ROLLOUT:-true}
ASYNC_NUM_WORKERS=${ASYNC_NUM_WORKERS:-8}
ASYNC_MAX_CONCURRENT_SAMPLES=${ASYNC_MAX_CONCURRENT_SAMPLES:-}
# An explicitly empty value disables wall-clock completion cutoffs for formal
# ablations; use '-' rather than ':-' so dotenv/defaulting cannot restore 0.9.
ASYNC_COMPLETION_CUTOFF=${ASYNC_COMPLETION_CUTOFF-0.9}

# ── Rollout Robustness (optional; default OFF) ──
# Retries a turn on degenerate output / parse errors / duplicate tool calls.
# Recommended for long-horizon runs (max_turns >> 20); off by default for
# deterministic, paper-reproducible rollouts.
USE_ROBUST_ROLLOUT=${USE_ROBUST_ROLLOUT:-false}
ROBUST_MAX_RETRIES=${ROBUST_MAX_RETRIES:-3}
ROBUST_DEDUP=${ROBUST_DEDUP:-true}
ROBUST_REPETITION=${ROBUST_REPETITION:-true}

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

MAX_MODEL_LEN=${MAX_MODEL_LEN:-261000}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-246000}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-8192}
ULYSSES_SP_SIZE=${ULYSSES_SP_SIZE:-8}
ASYNC_PROMPT_PAD=${ASYNC_PROMPT_PAD:-1024}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-1}
ACTOR_LR=${ACTOR_LR:-1e-6}
CRITIC_LR=${CRITIC_LR:-1e-5}
GAMMA=${GAMMA:-0.95}
KL_COEF=${KL_COEF:-0.001}
TOOL_TIMEOUT=${TOOL_TIMEOUT:-150}
SAVE_FREQ=${SAVE_FREQ:-5}
TEST_FREQ=${TEST_FREQ:--1}
TRACE_SAVE_INTERVAL=${TRACE_SAVE_INTERVAL:-10}
TRACE_MAX_SAMPLES=${TRACE_MAX_SAMPLES:-0}
SEARCH_ENGINE=${SEARCH_ENGINE:-online_search}

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

PYTHON_BIN=${PYTHON_BIN:-python3}
REAL_PYTHON_BIN=$PYTHON_BIN
TRAIN_LOG=${TRAIN_LOG:-${OUTPUT}/training.log}

# Print a shell-copyable command in dry-run mode.  Using `echo` as the Python
# executable loses the environment prefix and makes the output unsuitable as
# experiment evidence.
print_training_command() {
    local variable
    local -a environment_names=(
        HYDRA_FULL_ERROR PYTHONUNBUFFERED IGPO_ROLLOUT_IG
        IGPO_IG_COMPUTE_FREQ IGPO_IG_TOOL_FILTER IGPO_INFO_GAIN_TYPE
        IGPO_PHASE1_CHUNK_SIZE ROLLOUT_ROBUST ROLLOUT_ROBUST_MAX_RETRIES
        ROLLOUT_ROBUST_DEDUP ROLLOUT_ROBUST_REPETITION ROLLOUT_TURN_JITTER
        TOOL_CACHE_MODE TOOL_CACHE_PATH JUDGE_CACHE_MODE JUDGE_CACHE_PATH
        JUDGE_SEED JUDGE_TIMEOUT_SECONDS ENABLE_JUDGE_THINKING JUDGE_PROVIDER_ID
    )
    printf 'env'
    for variable in "${environment_names[@]}"; do
        if [[ -v "$variable" ]]; then
            printf ' %q' "$variable=${!variable}"
        fi
    done
    printf ' %q' "$REAL_PYTHON_BIN" "$@"
    printf '\n'
}

if [ "$DRY_RUN" = "true" ]; then
    PYTHON_BIN=print_training_command
fi

# Debug toggles (unset by default for best performance):
#   IGPO_VERIFY_ASYNC=1       -> enables dual-path async verification
#   IGPO_CROSSCHECK=N         -> cross-validates the first N samples between the
#                                vLLM rollout-logprob and FSDP actor-logprob paths
# Enable them only when diagnosing IG/numerical discrepancies.
run_training() {
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
ROLLOUT_TURN_JITTER=${TURN_JITTER} \
$PYTHON_BIN -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${ADV_ESTIMATOR} \
    algorithm.use_info_gain=${USE_INFO_GAIN} \
    algorithm.info_gain_type=${INFO_GAIN_TYPE} \
    algorithm.info_gain_norm_mode=${INFO_GAIN_NORM_MODE} \
    algorithm.ig_compute_freq=${IG_COMPUTE_FREQ} \
    algorithm.ig_tool_filter="${IG_TOOL_FILTER}" \
    algorithm.ig_weight=${IG_WEIGHT} \
    algorithm.use_format_penalty=${USE_FORMAT_PENALTY} \
    algorithm.format_penalty_scale=${FORMAT_PENALTY_SCALE} \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    +data.seed=${SEED} \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESPONSE_LEN} \
    +data.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_PARALLEL_SIZE} \
    actor_rollout_ref.rollout.seed=${SEED} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} \
    actor_rollout_ref.rollout.disable_log_stats=false \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.mask_tool_response=${MASK_TOOL_RESPONSE} \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=${KL_COEF} \
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
    critic.optim.lr=${CRITIC_LR} \
    critic.model.path=${MODEL_PATH} \
    critic.ppo_micro_batch_size_per_gpu=1 \
    algorithm.gamma=${GAMMA} \
    algorithm.kl_ctrl.kl_coef=${KL_COEF} \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$_GPU_NUM \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.validation_data_dir=${EVAL_LOG_PATH} \
    trainer.default_local_dir=${OUTPUT} \
    agent_grpo.n=${GROUP_SIZE} \
    max_turns=${MAX_TURNS} \
    +tool_timeout=${TOOL_TIMEOUT} \
    +trace_save_interval=${TRACE_SAVE_INTERVAL} \
    +trace_max_samples=${TRACE_MAX_SAMPLES} \
    search_engine=${SEARCH_ENGINE} \
    reward_model.train_reward_type=${TRAIN_REWARD_TYPE} \
    +reward_model.valid_reward_type=llm_em_noformatf1 \
    reward_model.reward_manager='naive_batch' \
    +reward_model.reward_kwargs.deepthink_disabled=true \
    data.return_raw_chat=true \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    ${TOTAL_TRAINING_STEPS:+trainer.total_training_steps=${TOTAL_TRAINING_STEPS}} \
    $([ "$USE_ASYNC_ROLLOUT" = "true" ] && echo "actor_rollout_ref.rollout.mode=async" || true) \
    $([ "$USE_ASYNC_ROLLOUT" = "true" ] && echo "actor_rollout_ref.rollout.prompt_length=${ASYNC_PROMPT_PAD}" || true) \
    $([ "$USE_ASYNC_ROLLOUT" = "true" ] && echo "actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LEN}" || true) \
    $([ "$USE_ASYNC_ROLLOUT" = "true" ] && echo "actor_rollout_ref.rollout.agent.num_workers=${ASYNC_NUM_WORKERS}" || true) \
    $([ "$USE_ASYNC_ROLLOUT" = "true" ] && echo "actor_rollout_ref.rollout.agent.agent_loop_config_path=configs/dr_agent_loop.yaml" || true) \
    $([ "$USE_ASYNC_ROLLOUT" = "true" ] && [ -n "$ASYNC_MAX_CONCURRENT_SAMPLES" ] && [ "$ASYNC_MAX_CONCURRENT_SAMPLES" != "null" ] && echo "+actor_rollout_ref.rollout.agent.max_concurrent_samples=${ASYNC_MAX_CONCURRENT_SAMPLES}" || true) \
    $([ "$USE_ASYNC_ROLLOUT" = "true" ] && [ -n "$ASYNC_COMPLETION_CUTOFF" ] && [ "$ASYNC_COMPLETION_CUTOFF" != "null" ] && echo "+actor_rollout_ref.rollout.agent.completion_cutoff=${ASYNC_COMPLETION_CUTOFF}" || true)
}

set +e
if [ "$DRY_RUN" = "true" ]; then
    run_training
    EXIT_CODE=$?
else
    run_training > >(stdbuf -oL tee "${TRAIN_LOG}") 2>&1
    EXIT_CODE=$?
fi
set -e

if [ "$DRY_RUN" = "true" ]; then
    exit 0
fi

echo "[$(date)] Training exited with code: $EXIT_CODE" | tee -a "${TRAIN_LOG}"
if [ $EXIT_CODE -ne 0 ]; then
    echo "=== dmesg (last 30 lines) ===" >> "${TRAIN_LOG}"
    dmesg -T 2>/dev/null | tail -30 >> "${TRAIN_LOG}" || true
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "=== nvidia-smi ===" >> "${TRAIN_LOG}"
        nvidia-smi >> "${TRAIN_LOG}" 2>&1 || true
    fi
    echo "=== cgroup memory ===" >> "${TRAIN_LOG}"
    cat /sys/fs/cgroup/memory/memory.oom_control >> "${TRAIN_LOG}" 2>/dev/null || true
fi

exit "$EXIT_CODE"
