#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)

: "${DATA_ROOT:=/path/to/sft_data}"

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Thinking-2507}"
export TRAIN_FILES="${TRAIN_FILES:-${DATA_ROOT}/train.parquet}"
export VAL_FILES="${VAL_FILES:-${DATA_ROOT}/val.parquet}"
export EXP_NAME="${EXP_NAME:-qwen3-4b-sft}"
export PROJECT_NAME="${PROJECT_NAME:-deepresearcher-sft}"

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
export DATA_MAX_LENGTH="${DATA_MAX_LENGTH:-200000}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export SAVE_FREQ="${SAVE_FREQ:-100}"
export TEST_FREQ="${TEST_FREQ:-100}"
export SP_SIZE="${SP_SIZE:-8}"
export MULTITURN="${MULTITURN:-True}"

exec "${ROOT_DIR}/train_sft.sh" "$@"
