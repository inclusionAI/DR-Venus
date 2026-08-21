#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python3}

MODEL_PATH=${MODEL_PATH:-}
MODEL_NAME=${MODEL_NAME:-}
CHECKPOINT_REVISION=${CHECKPOINT_REVISION:-}
EVAL_DATA=${EVAL_DATA:-"$REPO_ROOT/evaluation/data/eval-core-100.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/artifacts/evaluation/eval-core-100"}
TOOL_CACHE_MODE=${TOOL_CACHE_MODE:-replay}
TOOL_CACHE_PATH=${TOOL_CACHE_PATH:-"$REPO_ROOT/artifacts/tool_cache/eval.jsonl"}
SEED=${SEED:-42}
GPU_COUNT=${GPU_COUNT:-2}
GPU_TYPE=${GPU_TYPE:-A100-80G}
MAX_TURNS=${MAX_TURNS:-12}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-32768}
SOLVER_MAX_LENGTH=${SOLVER_MAX_LENGTH:-31744}
SAMPLES_PER_QUESTION=${SAMPLES_PER_QUESTION:-1}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.95}
TOOL_TIMEOUT=${TOOL_TIMEOUT:-150}
MAX_TOKENS_PER_TURN=${MAX_TOKENS_PER_TURN:-1024}
TIME_LIMIT=${TIME_LIMIT:-9000}
MAX_RETRIES=${MAX_RETRIES:-5}
JUDGE_MODE=${JUDGE_MODE:-offline}
JUDGE_MODEL=${JUDGE_MODEL:-}
JUDGE_CACHE_PATH=${JUDGE_CACHE_PATH:-"$REPO_ROOT/artifacts/evaluation/judge_cache.jsonl"}
JUDGE_BASE_URL=${JUDGE_BASE_URL:-}
JUDGE_TIMEOUT=${JUDGE_TIMEOUT:-60}
LIMIT=${LIMIT:-}
API_COST_CNY=${API_COST_CNY:-}
API_COST_PER_SAMPLE_CNY=${API_COST_PER_SAMPLE_CNY:-0}
DRY_RUN=${DRY_RUN:-true}
CONFIRM_SPEND=${CONFIRM_SPEND:-NO}
BUDGET_REPORT=${BUDGET_REPORT:-}
ALLOW_NON_REPLAY_EVAL=${ALLOW_NON_REPLAY_EVAL:-NO}
ALLOW_CUSTOM_PROTOCOL=${ALLOW_CUSTOM_PROTOCOL:-NO}

if [ -z "$MODEL_PATH" ]; then
    echo "MODEL_PATH is required." >&2
    exit 2
fi
if [ -z "$JUDGE_MODEL" ]; then
    echo "JUDGE_MODEL is required for score-complete final evidence." >&2
    exit 2
fi
if ! [[ "$SAMPLES_PER_QUESTION" =~ ^[1-9][0-9]*$ ]]; then
    echo "SAMPLES_PER_QUESTION must be a positive integer." >&2
    exit 2
fi
if ! [[ "$GPU_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPU_COUNT must be a positive integer." >&2
    exit 2
fi
if [ -n "$LIMIT" ] && ! [[ "$LIMIT" =~ ^[1-9][0-9]*$ ]]; then
    echo "LIMIT must be empty or a positive integer." >&2
    exit 2
fi
MODEL_NAME=${MODEL_NAME:-$(basename "$MODEL_PATH")}
DATASET_ROWS=$("$PYTHON_BIN" "$REPO_ROOT/experiments/count_dataset_rows.py" "$EVAL_DATA")
EVALUATED_ROWS=$DATASET_ROWS
if [ -n "$LIMIT" ] && [ "$LIMIT" -lt "$EVALUATED_ROWS" ]; then
    EVALUATED_ROWS=$LIMIT
fi
EXPECTED_GENERATIONS=$((EVALUATED_ROWS * SAMPLES_PER_QUESTION))
PROTOCOL_LABEL=budget-matched-non-official
if [ "$ALLOW_CUSTOM_PROTOCOL" != "YES" ]; then
    if [ -n "$LIMIT" ]; then
        echo "Formal fixed evaluation does not allow LIMIT." >&2
        exit 3
    fi
    "$PYTHON_BIN" -m evaluation.verify_fixed_protocol \
        --cohort "$EVAL_DATA" --max-turns "$MAX_TURNS" \
        --context-length "$CONTEXT_LENGTH" \
        --samples-per-question "$SAMPLES_PER_QUESTION" --seed "$SEED" \
        --max-tokens-per-turn "$MAX_TOKENS_PER_TURN"
    if [ "$JUDGE_MODE" != "offline" ]; then
        echo "Formal final evaluation requires offline Judge replay." >&2
        exit 3
    fi
else
    PROTOCOL_LABEL=custom-pilot-non-official
fi

if [ "$ALLOW_CUSTOM_PROTOCOL" != "YES" ]; then
    if [ -n "$API_COST_CNY" ] && [ "$API_COST_CNY" != "0" ] && [ "$API_COST_CNY" != "0.0" ]; then
        echo "Formal replay evaluation creates no new API spend; record cache-building cost separately." >&2
        exit 3
    fi
    EFFECTIVE_API_COST_CNY=0
    API_COST_SOURCE=replay-zero
    if [ "$API_COST_PER_SAMPLE_CNY" != "0" ] && [ "$API_COST_PER_SAMPLE_CNY" != "0.0" ]; then
        echo "Formal replay predictions must use API_COST_PER_SAMPLE_CNY=0." >&2
        exit 3
    fi
else
    EFFECTIVE_API_COST_CNY=${API_COST_CNY:-DECLARED_AFTER_RUN}
    API_COST_SOURCE=declared-cli
fi

if [ "$TOOL_CACHE_MODE" != "replay" ] && [ "$ALLOW_NON_REPLAY_EVAL" != "YES" ]; then
    echo "Final evaluation must use TOOL_CACHE_MODE=replay. Set ALLOW_NON_REPLAY_EVAL=YES only while recording." >&2
    exit 3
fi

RAW="$OUTPUT_DIR/raw_trajectories.jsonl"
UNSCORED_PREDICTIONS="$OUTPUT_DIR/predictions_unscored.jsonl"
PREDICTIONS="$OUTPUT_DIR/predictions.jsonl"
MANIFEST="$OUTPUT_DIR/experiment_manifest.yaml"
REPORT_JSON="$OUTPUT_DIR/report.json"
REPORT_MD="$OUTPUT_DIR/report.md"

infer=("$PYTHON_BIN" "$REPO_ROOT/Inference/run_demo.py"
    --deploy_mode tp --tp_size "$GPU_COUNT" --num_gpus "$GPU_COUNT"
    --model_path "$MODEL_PATH" --model_name "$MODEL_NAME"
    --checkpoint_revision "$CHECKPOINT_REVISION"
    --input_file "$EVAL_DATA" --output_file "$RAW"
    --max_steps "$MAX_TURNS" --max_model_len "$CONTEXT_LENGTH"
    --solver_max_len "$SOLVER_MAX_LENGTH" --samples_per_question "$SAMPLES_PER_QUESTION"
    --max_tokens_per_turn "$MAX_TOKENS_PER_TURN" --time_limit "$TIME_LIMIT"
    --max_retries "$MAX_RETRIES" --seed "$SEED"
    --cost_cny_per_sample "$API_COST_PER_SAMPLE_CNY")
infer+=(--temperature "$TEMPERATURE" --top_p "$TOP_P"
    --tool_timeout "$TOOL_TIMEOUT" --tool_cache_mode "$TOOL_CACHE_MODE"
    --tool_cache_path "$TOOL_CACHE_PATH")
if [ -n "$LIMIT" ]; then
    infer+=(--limit "$LIMIT")
fi

convert=("$PYTHON_BIN" -m evaluation.convert_inference "$RAW" "$UNSCORED_PREDICTIONS")
judge=("$PYTHON_BIN" -m evaluation.judge_predictions
    "$UNSCORED_PREDICTIONS" "$PREDICTIONS" --mode "$JUDGE_MODE"
    --model "$JUDGE_MODEL" --cache "$JUDGE_CACHE_PATH"
    --timeout-seconds "$JUDGE_TIMEOUT")
if [ -n "$JUDGE_BASE_URL" ]; then
    judge+=(--base-url "$JUDGE_BASE_URL")
fi
create_manifest_common=("$PYTHON_BIN" "$REPO_ROOT/experiments/create_manifest.py"
    --output "$MANIFEST" --experiment-id "$(basename "$OUTPUT_DIR")-seed$SEED"
    --checkpoint "$MODEL_PATH" --checkpoint-revision "$CHECKPOINT_REVISION" --model "$MODEL_NAME"
    --evaluation-dataset "$EVAL_DATA" --predictions "$PREDICTIONS"
    --seed "$SEED" --gpu-type "$GPU_TYPE" --gpu-count "$GPU_COUNT"
    --max-turns "$MAX_TURNS" --context-length "$CONTEXT_LENGTH"
    --tool-cache-mode "$TOOL_CACHE_MODE" --raw-trajectories "$RAW" --tool-cache "$TOOL_CACHE_PATH"
    --judge-cache "$JUDGE_CACHE_PATH" --judge-model "$JUDGE_MODEL"
    --protocol-label "$PROTOCOL_LABEL"
    --temperature "$TEMPERATURE" --top-p "$TOP_P" --samples-per-question "$SAMPLES_PER_QUESTION"
    --tool-timeout-seconds "$TOOL_TIMEOUT" --max-tokens-per-turn "$MAX_TOKENS_PER_TURN")

build_manifest_command() {
    local measured_gpu_hours=$1
    local api_cost=$2
    create_manifest=("${create_manifest_common[@]}"
        --gpu-hours "$measured_gpu_hours" --api-cost-cny "$api_cost"
        --gpu-hours-source measured-wall-clock --api-cost-source "$API_COST_SOURCE")
}
evaluate=("$PYTHON_BIN" -m evaluation.evaluate "$PREDICTIONS" --manifest "$MANIFEST"
    --output-json "$REPORT_JSON" --output-markdown "$REPORT_MD")

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

if [ "$DRY_RUN" = "true" ]; then
    printf 'TOOL_CACHE_MODE=%q TOOL_CACHE_PATH=%q ' "$TOOL_CACHE_MODE" "$TOOL_CACHE_PATH"
    print_command "${infer[@]}"
    print_command "${convert[@]}"
    print_command "${judge[@]}"
    build_manifest_command MEASURED_AFTER_INFERENCE "$EFFECTIVE_API_COST_CNY"
    print_command "${create_manifest[@]}"
    print_command "${evaluate[@]}"
    exit 0
fi

if [ "$CONFIRM_SPEND" != "YES" ] || [ -z "$BUDGET_REPORT" ]; then
    echo "Refusing paid evaluation. Set CONFIRM_SPEND=YES and a 3-5 step BUDGET_REPORT." >&2
    exit 4
fi
if [ -z "$CHECKPOINT_REVISION" ]; then
    echo "Live evidence requires CHECKPOINT_REVISION (immutable model revision or artifact hash)." >&2
    exit 5
fi
if [ "$ALLOW_CUSTOM_PROTOCOL" = "YES" ] && [ -z "$API_COST_CNY" ]; then
    echo "Custom live evaluation requires declared API_COST_CNY after recording." >&2
    exit 6
fi
"$PYTHON_BIN" "$REPO_ROOT/experiments/check_budget_report.py" \
    --report "$BUDGET_REPORT" --gpu-count "$GPU_COUNT" \
    --total-steps "$EXPECTED_GENERATIONS"
if [ "$TOOL_CACHE_MODE" = "replay" ] && [ ! -f "$TOOL_CACHE_PATH" ]; then
    echo "Replay cache does not exist: $TOOL_CACHE_PATH" >&2
    exit 7
fi
if [ "$JUDGE_MODE" = "offline" ] && [ ! -f "$JUDGE_CACHE_PATH" ]; then
    echo "Offline Judge cache does not exist: $JUDGE_CACHE_PATH" >&2
    exit 8
fi

mkdir -p "$OUTPUT_DIR"
export TOOL_CACHE_MODE TOOL_CACHE_PATH
cd "$REPO_ROOT"
INFERENCE_STARTED_AT=$(date +%s)
"${infer[@]}"
INFERENCE_FINISHED_AT=$(date +%s)
INFERENCE_ELAPSED_SECONDS=$((INFERENCE_FINISHED_AT - INFERENCE_STARTED_AT))
MEASURED_GPU_HOURS=$("$PYTHON_BIN" -c \
    'import sys; print(f"{float(sys.argv[1]) * int(sys.argv[2]) / 3600.0:.6f}")' \
    "$INFERENCE_ELAPSED_SECONDS" "$GPU_COUNT")
"${convert[@]}"
"${judge[@]}"
build_manifest_command "$MEASURED_GPU_HOURS" "$EFFECTIVE_API_COST_CNY"
"${create_manifest[@]}"
"${evaluate[@]}"

echo "Evaluation evidence written to $OUTPUT_DIR"
