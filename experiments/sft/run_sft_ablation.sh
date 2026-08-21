#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python3}

MODEL_PATH=${MODEL_PATH:-}
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-4B-Thinking-2507}
MODEL_REVISION=${MODEL_REVISION:-}
MODEL_SHA256=${MODEL_SHA256:-}
S0_TRAIN_FILES=${S0_TRAIN_FILES:-}
S1_TRAIN_FILES=${S1_TRAIN_FILES:-}
VAL_FILES=${VAL_FILES:-}
TEST_FILES=${TEST_FILES:-}
SFT_ARMS_MANIFEST=${SFT_ARMS_MANIFEST:-}
DATA_MANIFEST=${DATA_MANIFEST:-}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$REPO_ROOT/artifacts/sft"}
SEED=${SEED:-42}
DRY_RUN=${DRY_RUN:-True}
CONFIRM_SPEND=${CONFIRM_SPEND:-NO}
BUDGET_REPORT=${BUDGET_REPORT:-}
ALLOW_CUSTOM_PROTOCOL=${ALLOW_CUSTOM_PROTOCOL:-NO}
GPU_TYPE=${GPU_TYPE:-A100-80G}
UPSTREAM_COMMIT=${UPSTREAM_COMMIT:-}
ENV_LOCK_FILE=${ENV_LOCK_FILE:-}

if [ -z "$MODEL_PATH" ] || [ -z "$S0_TRAIN_FILES" ] || [ -z "$S1_TRAIN_FILES" ] || [ -z "$VAL_FILES" ] || [ -z "$TEST_FILES" ] || [ -z "$SFT_ARMS_MANIFEST" ] || [ -z "$DATA_MANIFEST" ]; then
    echo "MODEL_PATH, S0_TRAIN_FILES, S1_TRAIN_FILES, VAL_FILES, TEST_FILES, SFT_ARMS_MANIFEST, and DATA_MANIFEST are required." >&2
    exit 2
fi
if [ "$SEED" -ne 42 ] && [ "$ALLOW_CUSTOM_PROTOCOL" != "YES" ]; then
    echo "Formal SFT ablations require SEED=42. Set ALLOW_CUSTOM_PROTOCOL=YES only for a non-formal pilot." >&2
    exit 2
fi
if [ "$ALLOW_CUSTOM_PROTOCOL" != "YES" ]; then
    if [ ! -d "$MODEL_PATH" ]; then
        echo "Formal SFT requires a downloaded local MODEL_PATH directory; remote IDs are custom-pilot only." >&2
        exit 2
    fi
    MODEL_PATH=$(realpath -- "$MODEL_PATH")
    if [ -z "$MODEL_REVISION" ]; then
        echo "Formal SFT requires immutable MODEL_REVISION (and MODEL_SHA256 for a local checkpoint)." >&2
        exit 2
    fi
    if [[ ! "$MODEL_SHA256" =~ ^[0-9a-fA-F]{64}$ ]] \
            || [ "${MODEL_REVISION,,}" != "${MODEL_SHA256,,}" ]; then
        echo "Formal SFT requires 64-hex MODEL_SHA256 and MODEL_REVISION equal to that tree hash." >&2
        exit 2
    fi
    checkpoint_args=(
        --checkpoint "$MODEL_PATH" --expected-model-id "Qwen/Qwen3-4B-Thinking-2507"
        --revision "$MODEL_REVISION"
    )
    if [ -n "$MODEL_SHA256" ]; then
        checkpoint_args+=(--expected-sha256 "$MODEL_SHA256")
    fi
    if [ "$DRY_RUN" = "True" ]; then
        "$PYTHON_BIN" "$REPO_ROOT/experiments/verify_checkpoint.py" "${checkpoint_args[@]}" >/dev/null
    else
        "$PYTHON_BIN" "$REPO_ROOT/experiments/verify_checkpoint.py" \
            "${checkpoint_args[@]}" --output "$OUTPUT_ROOT/checkpoint_identity.json" >/dev/null
    fi
fi
S0_TRAIN_FILES=$(realpath -- "$S0_TRAIN_FILES")
S1_TRAIN_FILES=$(realpath -- "$S1_TRAIN_FILES")
VAL_FILES=$(realpath -- "$VAL_FILES")
TEST_FILES=$(realpath -- "$TEST_FILES")
SFT_ARMS_MANIFEST=$(realpath -- "$SFT_ARMS_MANIFEST")
DATA_MANIFEST=$(realpath -- "$DATA_MANIFEST")
"$PYTHON_BIN" "$REPO_ROOT/experiments/verify_sft_protocol.py" \
    --s0 "$S0_TRAIN_FILES" --s1 "$S1_TRAIN_FILES" \
    --validation "$VAL_FILES" --test "$TEST_FILES" \
    --manifest "$SFT_ARMS_MANIFEST" --data-manifest "$DATA_MANIFEST" \
    --token-budget 20000000 --seed "$SEED" --max-length 32768 \
    --train-batch-size 8 --total-epochs 1
EXPECTED_TOTAL_STEPS=$("$PYTHON_BIN" "$REPO_ROOT/experiments/verify_sft_protocol.py" \
    --s0 "$S0_TRAIN_FILES" --s1 "$S1_TRAIN_FILES" \
    --validation "$VAL_FILES" --test "$TEST_FILES" \
    --manifest "$SFT_ARMS_MANIFEST" --data-manifest "$DATA_MANIFEST" \
    --token-budget 20000000 --seed "$SEED" --max-length 32768 \
    --train-batch-size 8 --total-epochs 1 --print-total-steps)
if [ "$DRY_RUN" != "True" ]; then
    if [ "$CONFIRM_SPEND" != "YES" ] || [ -z "$BUDGET_REPORT" ]; then
        echo "Refusing paid training. Set CONFIRM_SPEND=YES and BUDGET_REPORT after a 3-5 step pilot." >&2
        exit 3
    fi
    budget_args=(--report "$BUDGET_REPORT" --gpu-count 2 --total-steps "$EXPECTED_TOTAL_STEPS")
    "$PYTHON_BIN" "$REPO_ROOT/experiments/check_budget_report.py" "${budget_args[@]}"
    if [ "$ALLOW_CUSTOM_PROTOCOL" != "YES" ]; then
        if [ -z "$ENV_LOCK_FILE" ] || [ ! -f "$ENV_LOCK_FILE" ]; then
            echo "Formal live SFT requires ENV_LOCK_FILE from the actual GPU runtime (for example: pip freeze --all)." >&2
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

invoke_sft() {
    local run_name=$1
    local train_files=$2
    local dry_run=$3
    local run_dir="$OUTPUT_ROOT/$run_name"
    (
        cd "$REPO_ROOT/SFT"
        MODEL_PATH="$MODEL_PATH" \
        TRAIN_FILES="$train_files" \
        VAL_FILES="$VAL_FILES" \
        PROJECT_NAME=dr-venus-sft-ablation \
        EXP_NAME="$run_name" \
        OUTPUT_ROOT="$OUTPUT_ROOT" \
        CKPT_DIR="$run_dir" \
        LOG_FILE="$run_dir/training.log" \
        NUM_GPUS=2 \
        TRAIN_BATCH_SIZE=8 \
        MICRO_BATCH_SIZE_PER_GPU=1 \
        DATA_MAX_LENGTH=32768 \
        LORA_RANK=32 \
        LORA_ALPHA=64 \
        TARGET_MODULES=all-linear \
        LR=1e-4 \
        TOTAL_EPOCHS=1 \
        SEED="$SEED" \
        SP_SIZE=2 \
        CONFIRM_SPEND="$CONFIRM_SPEND" \
        DRY_RUN="$dry_run" \
        bash train_sft.sh
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

write_sft_snapshot() {
    local run_name=$1
    local train_files=$2
    local expected_steps=$3
    local run_dir="$OUTPUT_ROOT/$run_name"
    local -a args=(
        snapshot --output "$run_dir/run_snapshot.json"
        --stage sft --run-id "$run_name" --protocol-label 4b-sft-ablation
        --checkpoint-identity "$OUTPUT_ROOT/checkpoint_identity.json"
        --command-file "$run_dir/resolved_command.txt"
        --data "train=$train_files" --data "validation=$VAL_FILES"
        --data "test=$TEST_FILES" --data "sft_arms_manifest=$SFT_ARMS_MANIFEST"
        --data "data_manifest=$DATA_MANIFEST"
        --config "seed=$SEED" --config 'backbone=Qwen3-4B-Thinking-2507'
        --config 'precision=bf16' --config 'context_length=32768'
        --config 'train_batch_size=8' --config 'micro_batch_size_per_gpu=1'
        --config 'lora_rank=32' --config 'lora_alpha=64'
        --config 'target_modules=all-linear' --config 'learning_rate=0.0001'
        --config 'total_epochs=1' --config "expected_optimizer_steps=$expected_steps"
        --config 'sequence_parallel_size=2' --config 'gradient_checkpointing=true'
        --config "starting_checkpoint_path=$MODEL_PATH"
        --config "starting_checkpoint_sha256=$MODEL_SHA256"
        --config "starting_checkpoint_revision=$MODEL_REVISION"
        --gpu-type "$GPU_TYPE" --gpu-count 2
    )
    if [ "$ALLOW_CUSTOM_PROTOCOL" = "YES" ]; then
        args+=(--protocol-label custom-pilot --allow-dirty)
    fi
    if [ -n "$UPSTREAM_COMMIT" ]; then
        args+=(--upstream-commit "$UPSTREAM_COMMIT")
    fi
    if [ -n "$ENV_LOCK_FILE" ]; then
        args+=(--environment-lock "$ENV_LOCK_FILE")
    fi
    "$PYTHON_BIN" "$REPO_ROOT/experiments/training_evidence.py" "${args[@]}"
}

finalize_sft_run() {
    local run_name=$1
    local exit_code=$2
    local elapsed_seconds=$3
    local run_dir="$OUTPUT_ROOT/$run_name"
    local checkpoint
    local -a args=(
        finalize --snapshot "$run_dir/run_snapshot.json"
        --output "$run_dir/experiment_manifest.yaml"
        --exit-code "$exit_code" --elapsed-seconds "$elapsed_seconds"
        --api-cost-cny 0 --log "$run_dir/training.log"
    )
    checkpoint=$(latest_global_step "$run_dir")
    if [ -n "$checkpoint" ]; then
        args+=(--checkpoint "$checkpoint")
    fi
    "$PYTHON_BIN" "$REPO_ROOT/experiments/training_evidence.py" "${args[@]}"
}

for arm in s0 s1; do
    if [ "$arm" = "s0" ]; then
        train_files=$S0_TRAIN_FILES
    else
        train_files=$S1_TRAIN_FILES
    fi
    arm_rows=$("$PYTHON_BIN" "$REPO_ROOT/experiments/count_dataset_rows.py" "$train_files")
    expected_arm_steps=$((arm_rows / 8))
    run_name="${arm}-qwen3-4b-lora-seed${SEED}"
    echo "Preparing ${run_name} (DRY_RUN=${DRY_RUN})"
    if [ "$DRY_RUN" = "True" ]; then
        invoke_sft "$run_name" "$train_files" True
        continue
    fi

    run_dir="$OUTPUT_ROOT/$run_name"
    mkdir -p "$run_dir"
    invoke_sft "$run_name" "$train_files" True >"$run_dir/resolved_command.txt"
    write_sft_snapshot "$run_name" "$train_files" "$expected_arm_steps"

    started_at=$("$PYTHON_BIN" -c 'import time; print(time.time())')
    set +e
    invoke_sft "$run_name" "$train_files" False
    run_exit=$?
    set -e
    finished_at=$("$PYTHON_BIN" -c 'import time; print(time.time())')
    elapsed_seconds=$("$PYTHON_BIN" -c \
        'import sys; print(max(0.0, float(sys.argv[2]) - float(sys.argv[1])))' \
        "$started_at" "$finished_at")

    set +e
    finalize_sft_run "$run_name" "$run_exit" "$elapsed_seconds"
    finalize_exit=$?
    set -e
    if [ "$finalize_exit" -ne 0 ]; then
        echo "Evidence finalization failed for $run_name (exit $finalize_exit)." >&2
        exit "$finalize_exit"
    fi
    if [ "$run_exit" -ne 0 ]; then
        echo "Training failed for $run_name; evidence was preserved." >&2
        exit "$run_exit"
    fi
done
