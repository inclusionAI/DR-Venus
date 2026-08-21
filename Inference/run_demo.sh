#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    # Treat .env as defaults. Preserve explicitly exported launch values,
    # including credentials, without ever printing them.
    preserve_names=(
        GLOO_SOCKET_IFNAME SERPER_KEY_ID JINA_API_KEYS API_KEY API_BASE
        SUMMARY_MODEL_NAME PROXY TOOL_CACHE_MODE TOOL_CACHE_PATH
        TOOL_TIMEOUT_SECONDS VISIT_SERVER_TIMEOUT MODEL_PATH MODEL_NAME
        MODEL_REVISION CHECKPOINT_REVISION QUESTION INPUT_FILE OUTPUT_FILE
        LOG_FILE PYTHON_BIN NUM_GPUS TP_SIZE BASE_PORT GPU_UTIL MAX_LEN
        BATCH_SIZE NUM_WORKERS MAX_STEPS SOLVER_MAX_LEN TIME_LIMIT LIMIT
        MAX_RETRIES
        SEED SAMPLES_PER_QUESTION TOOL_TIMEOUT TEMPERATURE TOP_P
        COST_CNY_PER_SAMPLE MAX_TOKENS_PER_TURN REUSE_SERVER
        SERVER_START_TIMEOUT
        DRY_RUN CONFIRM_SPEND
    )
    declare -A preserved_values=()
    declare -A preserved_presence=()
    for name in "${preserve_names[@]}"; do
        if [[ -v "$name" ]]; then
            preserved_presence["$name"]=1
            preserved_values["$name"]="${!name}"
        fi
    done
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
    for name in "${!preserved_presence[@]}"; do
        printf -v "$name" '%s' "${preserved_values[$name]}"
        export "$name"
    done
    unset preserve_names preserved_values preserved_presence name
fi

MODEL_PATH=${MODEL_PATH:-}
MODEL_NAME=${MODEL_NAME:-}
MODEL_REVISION=${MODEL_REVISION:-}
CHECKPOINT_REVISION=${CHECKPOINT_REVISION:-}
QUESTION=${QUESTION:-"In April of 1977, who was the Prime Minister of the first place mentioned by name in the Book of Esther (in the New International Version)?"}
INPUT_FILE=${INPUT_FILE:-}
OUTPUT_FILE=${OUTPUT_FILE:-./demo_trajectory.jsonl}
LOG_FILE=${LOG_FILE:-./demo.log}
PYTHON_BIN=${PYTHON_BIN:-python3}

NUM_GPUS=${NUM_GPUS:-2}
TP_SIZE=${TP_SIZE:-$NUM_GPUS}
BASE_PORT=${BASE_PORT:-6051}
GPU_UTIL=${GPU_UTIL:-0.95}
MAX_LEN=${MAX_LEN:-261000}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-1}
MAX_STEPS=${MAX_STEPS:-200}
SOLVER_MAX_LEN=${SOLVER_MAX_LEN:-230000}
TIME_LIMIT=${TIME_LIMIT:-18000}
MAX_RETRIES=${MAX_RETRIES:-5}
MAX_TOKENS_PER_TURN=${MAX_TOKENS_PER_TURN:-1024}
LIMIT=${LIMIT:-}
SEED=${SEED:-42}
SAMPLES_PER_QUESTION=${SAMPLES_PER_QUESTION:-1}
TOOL_CACHE_MODE=${TOOL_CACHE_MODE:-online}
TOOL_CACHE_PATH=${TOOL_CACHE_PATH:-}
TOOL_TIMEOUT=${TOOL_TIMEOUT:-${TOOL_TIMEOUT_SECONDS:-150}}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.95}
COST_CNY_PER_SAMPLE=${COST_CNY_PER_SAMPLE:-0}
REUSE_SERVER=${REUSE_SERVER:-false}
SERVER_START_TIMEOUT=${SERVER_START_TIMEOUT:-900}
DRY_RUN=${DRY_RUN:-true}
CONFIRM_SPEND=${CONFIRM_SPEND:-NO}

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: MODEL_PATH must be set to a local checkpoint or Hugging Face model ID." >&2
    exit 2
fi

if [ "$TOOL_CACHE_MODE" != "online" ] && [ -z "$TOOL_CACHE_PATH" ]; then
    echo "ERROR: TOOL_CACHE_PATH is required in record/replay mode." >&2
    exit 2
fi
if [ "$TOOL_CACHE_MODE" = "replay" ] && [ ! -f "$TOOL_CACHE_PATH" ]; then
    echo "ERROR: replay tool cache does not exist: $TOOL_CACHE_PATH" >&2
    exit 2
fi

input_args=(--question "$QUESTION")
input_label="Question: $QUESTION"
if [ -n "$INPUT_FILE" ]; then
    input_args=(--input_file "$INPUT_FILE")
    input_label="Input file: $INPUT_FILE"
fi

optional_args=()
if [ -n "$LIMIT" ]; then
    optional_args+=(--limit "$LIMIT")
fi
if [ -n "$MODEL_NAME" ]; then
    optional_args+=(--model_name "$MODEL_NAME")
fi
if [ -n "$MODEL_REVISION" ]; then
    optional_args+=(--model_revision "$MODEL_REVISION")
fi
if [ -n "$CHECKPOINT_REVISION" ]; then
    optional_args+=(--checkpoint_revision "$CHECKPOINT_REVISION")
fi
if [ -n "$TOOL_CACHE_PATH" ]; then
    optional_args+=(--tool_cache_path "$TOOL_CACHE_PATH")
fi
if [ "$REUSE_SERVER" = "true" ]; then
    optional_args+=(--reuse_server)
fi

# --- 执行 ---

echo "========================================="
echo "  Deploy Mode : tp"
echo "  Num GPUs    : $NUM_GPUS"
echo "  TP Size     : $TP_SIZE"
echo "  Port        : $BASE_PORT"
echo "  Model       : $MODEL_PATH"
echo "  Input       : $input_label"
echo "  Output      : $OUTPUT_FILE"
echo "  Log File    : $LOG_FILE"
echo "  Seed/Samples: $SEED / $SAMPLES_PER_QUESTION"
echo "  Tool Mode   : $TOOL_CACHE_MODE"
echo "========================================="

export TOOL_CACHE_MODE TOOL_CACHE_PATH

CMD=("$PYTHON_BIN" "$SCRIPT_DIR/run_demo.py" \
    --deploy_mode tp \
    --tp_size "$TP_SIZE" \
    --num_gpus "$NUM_GPUS" \
    --base_port "$BASE_PORT" \
    --server_start_timeout "$SERVER_START_TIMEOUT" \
    --model_path "$MODEL_PATH" \
    --gpu_util "$GPU_UTIL" \
    --max_model_len "$MAX_LEN" \
    "${input_args[@]}" \
    --output_file "$OUTPUT_FILE" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --max_steps "$MAX_STEPS" \
    --solver_max_len "$SOLVER_MAX_LEN" \
    --time_limit "$TIME_LIMIT" \
    --max_retries "$MAX_RETRIES" \
    --max_tokens_per_turn "$MAX_TOKENS_PER_TURN" \
    --seed "$SEED" \
    --samples_per_question "$SAMPLES_PER_QUESTION" \
    --tool_cache_mode "$TOOL_CACHE_MODE" \
    --tool_timeout "$TOOL_TIMEOUT" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --cost_cny_per_sample "$COST_CNY_PER_SAMPLE" \
    "${optional_args[@]}")

if [ "$DRY_RUN" = "true" ]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi
if [ "$CONFIRM_SPEND" != "YES" ]; then
    echo "ERROR: refusing live GPU/API inference. Review the dry run, then set CONFIRM_SPEND=YES." >&2
    exit 3
fi

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"

echo "Job Finished."
