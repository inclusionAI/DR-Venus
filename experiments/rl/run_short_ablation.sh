#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python3}

MODEL_PATH=${MODEL_PATH:-}
MODEL_ID=${MODEL_ID:-inclusionAI/DR-Venus-4B-SFT}
MODEL_REVISION=${MODEL_REVISION:-}
MODEL_SHA256=${MODEL_SHA256:-}
TRAIN_FILES=${TRAIN_FILES:-"$REPO_ROOT/experiments/data/rl-train-96.parquet"}
VAL_FILES=${VAL_FILES:-"$REPO_ROOT/experiments/data/rl-val-32.parquet"}
FINAL_EVAL_FILES=${FINAL_EVAL_FILES:-"$REPO_ROOT/evaluation/data/eval-core-100.parquet"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$REPO_ROOT/artifacts/rl"}
TOOL_CACHE_PATH=${TOOL_CACHE_PATH:-"$REPO_ROOT/artifacts/tool_cache/rl.jsonl"}
JUDGE_CACHE_PATH=${JUDGE_CACHE_PATH:-"$REPO_ROOT/artifacts/judge_cache/rl.jsonl"}
ARMS=${ARMS:-"r0,r1,r2"}
SEEDS=${SEEDS:-"42,43"}
STEPS=${STEPS:-20}
DRY_RUN=${DRY_RUN:-true}
CONFIRM_SPEND=${CONFIRM_SPEND:-NO}
BUDGET_REPORT=${BUDGET_REPORT:-}
ALLOW_CUSTOM_DATA=${ALLOW_CUSTOM_DATA:-NO}
ALLOW_CUSTOM_PROTOCOL=${ALLOW_CUSTOM_PROTOCOL:-NO}
GPU_TYPE=${GPU_TYPE:-A100-80G}
API_COST_CNY_PER_RUN=${API_COST_CNY_PER_RUN:-}
UPSTREAM_COMMIT=${UPSTREAM_COMMIT:-}
ENV_LOCK_FILE=${ENV_LOCK_FILE:-}

if [ -z "$MODEL_PATH" ]; then
    echo "MODEL_PATH is required." >&2
    exit 2
fi
if { [ "$ARMS" != "r0,r1,r2" ] || [ "$SEEDS" != "42,43" ] || [ "$STEPS" -ne 20 ]; } \
        && [ "$ALLOW_CUSTOM_PROTOCOL" != "YES" ]; then
    echo "Formal ablations require ARMS=r0,r1,r2, SEEDS=42,43, and STEPS=20. Set ALLOW_CUSTOM_PROTOCOL=YES only for a non-formal pilot." >&2
    exit 2
fi
if [ "$ALLOW_CUSTOM_PROTOCOL" != "YES" ]; then
    if [ ! -d "$MODEL_PATH" ]; then
        echo "Formal RL requires a downloaded local MODEL_PATH directory; remote IDs are custom-pilot only." >&2
        exit 2
    fi
    MODEL_PATH=$(realpath -- "$MODEL_PATH")
    if [ -z "$MODEL_REVISION" ]; then
        echo "Formal RL requires immutable MODEL_REVISION (and MODEL_SHA256 for a local checkpoint)." >&2
        exit 2
    fi
    if [[ ! "$MODEL_SHA256" =~ ^[0-9a-fA-F]{64}$ ]] \
            || [ "${MODEL_REVISION,,}" != "${MODEL_SHA256,,}" ]; then
        echo "Formal RL requires 64-hex MODEL_SHA256 and MODEL_REVISION equal to that tree hash." >&2
        exit 2
    fi
    checkpoint_args=(
        --checkpoint "$MODEL_PATH" --expected-model-id "inclusionAI/DR-Venus-4B-SFT"
        --revision "$MODEL_REVISION"
    )
    if [ -n "$MODEL_SHA256" ]; then
        checkpoint_args+=(--expected-sha256 "$MODEL_SHA256")
    fi
    if [ "$DRY_RUN" = "true" ]; then
        "$PYTHON_BIN" "$REPO_ROOT/experiments/verify_checkpoint.py" "${checkpoint_args[@]}" >/dev/null
    else
        "$PYTHON_BIN" "$REPO_ROOT/experiments/verify_checkpoint.py" \
            "${checkpoint_args[@]}" --output "$OUTPUT_ROOT/checkpoint_identity.json" >/dev/null
    fi
fi
TRAIN_FILES=$(realpath -- "$TRAIN_FILES")
VAL_FILES=$(realpath -- "$VAL_FILES")
fixed_train=$(realpath -- "$REPO_ROOT/experiments/data/rl-train-96.parquet")
fixed_val=$(realpath -- "$REPO_ROOT/experiments/data/rl-val-32.parquet")
FINAL_EVAL_FILES=$(realpath -- "$FINAL_EVAL_FILES")
if { [ "$TRAIN_FILES" != "$fixed_train" ] || [ "$VAL_FILES" != "$fixed_val" ]; } \
        && [ "$ALLOW_CUSTOM_DATA" != "YES" ]; then
    echo "Formal ablations require the committed 96-train/32-validation cohorts. Set ALLOW_CUSTOM_DATA=YES only for a non-formal pilot." >&2
    exit 2
fi
"$PYTHON_BIN" "$REPO_ROOT/experiments/verify_protocol_data.py" \
    --train "$TRAIN_FILES" --validation "$VAL_FILES" --eval "$FINAL_EVAL_FILES"
IFS=',' read -r -a ARM_LIST <<< "$ARMS"
IFS=',' read -r -a SEED_LIST <<< "$SEEDS"

if [ "$DRY_RUN" != "true" ]; then
    if [ "$CONFIRM_SPEND" != "YES" ] || [ -z "$BUDGET_REPORT" ]; then
        echo "Refusing paid training. Set CONFIRM_SPEND=YES and BUDGET_REPORT after a 3-5 step pilot." >&2
        exit 3
    fi
    expected_steps=$(( ${#ARM_LIST[@]} * ${#SEED_LIST[@]} * STEPS ))
    "$PYTHON_BIN" "$REPO_ROOT/experiments/check_budget_report.py" \
        --report "$BUDGET_REPORT" --gpu-count 4 --total-steps "$expected_steps"
    if ! "$PYTHON_BIN" -c \
        'import math, sys; value=float(sys.argv[1]); raise SystemExit(not (math.isfinite(value) and value >= 0))' \
        "$API_COST_CNY_PER_RUN"; then
        echo "Live RL requires explicit finite API_COST_CNY_PER_RUN>=0." >&2
        exit 2
    fi
    if [ "$ALLOW_CUSTOM_PROTOCOL" != "YES" ]; then
        if [ -z "$ENV_LOCK_FILE" ] || [ ! -f "$ENV_LOCK_FILE" ]; then
            echo "Formal live RL requires ENV_LOCK_FILE from the actual GPU runtime (for example: pip freeze --all)." >&2
            exit 2
        fi
        ENV_LOCK_FILE=$(realpath -- "$ENV_LOCK_FILE")
    elif [ -n "$ENV_LOCK_FILE" ]; then
        ENV_LOCK_FILE=$(realpath -- "$ENV_LOCK_FILE")
    fi
    if [ "$ALLOW_CUSTOM_PROTOCOL" = "YES" ]; then
        if [ -z "$MODEL_REVISION" ]; then
            echo "Live custom pilots still require immutable MODEL_REVISION (and MODEL_SHA256 for a local checkpoint)." >&2
            exit 2
        fi
        checkpoint_args=(
            --checkpoint "$MODEL_PATH" --expected-model-id "$MODEL_ID"
            --revision "$MODEL_REVISION" --output "$OUTPUT_ROOT/checkpoint_identity.json"
        )
        if [ -n "$MODEL_SHA256" ]; then
            checkpoint_args+=(--expected-sha256 "$MODEL_SHA256")
        fi
        "$PYTHON_BIN" "$REPO_ROOT/experiments/verify_checkpoint.py" "${checkpoint_args[@]}" >/dev/null
    fi
fi

invoke_rl() {
    local run_name=$1
    local seed=$2
    local use_ig=$3
    local use_format=$4
    local dry_run=$5
    local run_dir="$OUTPUT_ROOT/$run_name"
    (
        cd "$REPO_ROOT/RL"
        PROJECT_NAME=dr-venus-short-rl \
        EXPERIMENT_NAME="$run_name" \
        MODEL_PATH="$MODEL_PATH" \
        TRAIN_FILES="$TRAIN_FILES" \
        VAL_FILES="$VAL_FILES" \
        OUTPUT="$run_dir" \
        EVAL_LOG_PATH="$run_dir/eval" \
        TRAIN_LOG="$run_dir/training.log" \
        GPU_NUM=4 \
        TOTAL_TRAINING_STEPS="$STEPS" \
        TRAIN_BATCH_SIZE=4 \
        GROUP_SIZE=4 \
        MAX_TURNS=12 \
        TURN_JITTER=0 \
        SEED="$seed" \
        USE_INFO_GAIN="$use_ig" \
        INFO_GAIN_TYPE=log_prob_diff \
        INFO_GAIN_NORM_MODE=scaled_separate \
        IG_TOOL_FILTER=visit \
        USE_FORMAT_PENALTY="$use_format" \
        FORMAT_PENALTY_SCALE=1.0 \
        MAX_MODEL_LEN=32768 \
        MAX_PROMPT_LEN=2048 \
        MAX_RESPONSE_LEN=1024 \
        ASYNC_PROMPT_PAD=1024 \
        ULYSSES_SP_SIZE=4 \
        TENSOR_PARALLEL_SIZE=4 \
        PPO_MINI_BATCH_SIZE=4 \
        ASYNC_NUM_WORKERS=4 \
        ASYNC_COMPLETION_CUTOFF= \
        TOOL_CACHE_MODE=record \
        TOOL_CACHE_PATH="$TOOL_CACHE_PATH" \
        JUDGE_CACHE_MODE=record \
        JUDGE_CACHE_PATH="$JUDGE_CACHE_PATH" \
        ENABLE_JUDGE_THINKING=false \
        JUDGE_SEED=42 \
        JUDGE_TIMEOUT_SECONDS=300 \
        PYTHON_BIN="$PYTHON_BIN" \
        CONFIRM_SPEND="$CONFIRM_SPEND" \
        DRY_RUN="$dry_run" \
        bash train_igpo.sh
    )
}

latest_global_step() {
    local run_dir=$1
    local latest
    latest=$(find "$run_dir" -type d -name 'global_step_*' -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | head -n 1 || true)
    if [ -n "$latest" ]; then
        printf '%s\n' "${latest#* }"
    fi
}

write_rl_snapshot() {
    local run_name=$1
    local seed=$2
    local use_ig=$3
    local use_format=$4
    local run_dir="$OUTPUT_ROOT/$run_name"
    local -a args=(
        snapshot --output "$run_dir/run_snapshot.json"
        --stage rl --run-id "$run_name" --protocol-label 4b-short-rl-ablation
        --checkpoint-identity "$OUTPUT_ROOT/checkpoint_identity.json"
        --command-file "$run_dir/resolved_command.txt"
        --data "train=$TRAIN_FILES" --data "validation=$VAL_FILES"
        --data "final_eval=$FINAL_EVAL_FILES"
        --config "arm=${run_name%%-*}" --config "seed=$seed" --config "steps=$STEPS"
        --config 'context_length=32768' --config 'max_turns=12'
        --config 'max_generation_length_per_turn=1024' --config 'train_batch_size=4'
        --config 'group_size=4' --config "use_info_gain=$use_ig"
        --config 'info_gain_type=log_prob_diff'
        --config 'info_gain_norm_mode=scaled_separate' --config 'ig_tool_filter=visit'
        --config "use_format_penalty=$use_format" --config 'format_penalty_scale=1.0'
        --config 'sequence_parallel_size=4' --config 'tensor_parallel_size=4'
        --config 'async_workers=4' --config 'async_completion_cutoff=null'
        --config 'tool_cache_mode=record' --config "tool_cache_path=$TOOL_CACHE_PATH"
        --config 'judge_cache_mode=record' --config "judge_cache_path=$JUDGE_CACHE_PATH"
        --config 'judge_seed=42' --config 'judge_timeout_seconds=300'
        --config "starting_checkpoint_path=$MODEL_PATH"
        --config "starting_checkpoint_sha256=$MODEL_SHA256"
        --config "starting_checkpoint_revision=$MODEL_REVISION"
        --gpu-type "$GPU_TYPE" --gpu-count 4
    )
    if [ "$ALLOW_CUSTOM_PROTOCOL" = "YES" ]; then
        args+=(--protocol-label custom-pilot --allow-dirty)
    elif [ "$ALLOW_CUSTOM_DATA" = "YES" ]; then
        args+=(--protocol-label custom-data-pilot)
    fi
    if [ -n "$UPSTREAM_COMMIT" ]; then
        args+=(--upstream-commit "$UPSTREAM_COMMIT")
    fi
    if [ -n "$ENV_LOCK_FILE" ]; then
        args+=(--environment-lock "$ENV_LOCK_FILE")
    fi
    "$PYTHON_BIN" "$REPO_ROOT/experiments/training_evidence.py" "${args[@]}"
}

finalize_rl_run() {
    local run_name=$1
    local exit_code=$2
    local elapsed_seconds=$3
    local run_dir="$OUTPUT_ROOT/$run_name"
    local checkpoint
    local -a args=(
        finalize --snapshot "$run_dir/run_snapshot.json"
        --output "$run_dir/experiment_manifest.yaml"
        --exit-code "$exit_code" --elapsed-seconds "$elapsed_seconds"
        --api-cost-cny "$API_COST_CNY_PER_RUN" --log "$run_dir/training.log"
    )
    checkpoint=$(latest_global_step "$run_dir")
    if [ -n "$checkpoint" ]; then
        args+=(--checkpoint "$checkpoint")
    fi
    if [ -f "$run_dir/tool_cache.record.final.jsonl" ]; then
        args+=(--artifact "tool_cache_record_final=$run_dir/tool_cache.record.final.jsonl")
    fi
    if [ -f "$run_dir/judge_cache.record.final.jsonl" ]; then
        args+=(--artifact "judge_cache_record_final=$run_dir/judge_cache.record.final.jsonl")
    fi
    "$PYTHON_BIN" "$REPO_ROOT/experiments/training_evidence.py" "${args[@]}"
}

freeze_rl_caches() {
    local run_dir=$1
    local status=0
    if [ -f "$TOOL_CACHE_PATH" ]; then
        cp -- "$TOOL_CACHE_PATH" "$run_dir/tool_cache.record.final.jsonl" || status=$?
    fi
    if [ -f "$JUDGE_CACHE_PATH" ]; then
        cp -- "$JUDGE_CACHE_PATH" "$run_dir/judge_cache.record.final.jsonl" || status=$?
    fi
    return "$status"
}

for arm in "${ARM_LIST[@]}"; do
    case "$arm" in
        r0) use_ig=false; use_format=false ;;
        r1) use_ig=true;  use_format=false ;;
        r2) use_ig=true;  use_format=true ;;
        *) echo "Unknown arm: $arm (expected r0,r1,r2)" >&2; exit 4 ;;
    esac

    for seed in "${SEED_LIST[@]}"; do
        run_name="${arm}-seed${seed}-steps${STEPS}"
        echo "Preparing ${run_name} (DRY_RUN=${DRY_RUN})"
        if [ "$DRY_RUN" = "true" ]; then
            invoke_rl "$run_name" "$seed" "$use_ig" "$use_format" true
            continue
        fi

        run_dir="$OUTPUT_ROOT/$run_name"
        mkdir -p "$run_dir"
        invoke_rl "$run_name" "$seed" "$use_ig" "$use_format" true \
            | tail -n 1 >"$run_dir/resolved_command.txt"
        write_rl_snapshot "$run_name" "$seed" "$use_ig" "$use_format"

        started_at=$("$PYTHON_BIN" -c 'import time; print(time.time())')
        set +e
        invoke_rl "$run_name" "$seed" "$use_ig" "$use_format" false
        run_exit=$?
        set -e
        finished_at=$("$PYTHON_BIN" -c 'import time; print(time.time())')
        elapsed_seconds=$("$PYTHON_BIN" -c \
            'import sys; print(max(0.0, float(sys.argv[2]) - float(sys.argv[1])))' \
            "$started_at" "$finished_at")

        set +e
        freeze_rl_caches "$run_dir"
        freeze_exit=$?
        finalize_rl_run "$run_name" "$run_exit" "$elapsed_seconds"
        finalize_exit=$?
        set -e
        if [ "$finalize_exit" -ne 0 ]; then
            echo "Evidence finalization failed for $run_name (exit $finalize_exit)." >&2
            exit "$finalize_exit"
        fi
        if [ "$freeze_exit" -ne 0 ]; then
            echo "Cache freezing failed for $run_name after its training evidence was finalized." >&2
            exit "$freeze_exit"
        fi
        if [ "$run_exit" -ne 0 ]; then
            echo "Training failed for $run_name; evidence was preserved." >&2
            exit "$run_exit"
        fi
    done
done
