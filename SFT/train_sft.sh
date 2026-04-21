#!/usr/bin/env bash
set -euo pipefail

show_help() {
    cat <<'EOF'
Usage:
  bash train_sft.sh [hydra_overrides...]

Required environment variables:
  MODEL_PATH        Model path or Hugging Face model ID.
  TRAIN_FILES       Training parquet file, directory, or hydra-compatible list.

Optional environment variables:
  VAL_FILES                 Validation parquet file(s). Defaults to TRAIN_FILES.
  ENTRYPOINT                Python entrypoint. Default: -m verl.trainer.fsdp_sft_trainer
  NNODES                    Number of nodes. Default: 1
  NODE_RANK                 Node rank. Default: 0
  NUM_GPUS                  GPUs per node. Default: 8
  MASTER_ADDR               Torch distributed master address. Default: 127.0.0.1
  MASTER_PORT               Torch distributed master port. Default: 29500
  PROJECT_NAME              Experiment group name. Default: deepresearcher-sft
  EXP_NAME                  Experiment name. Default: derived from model name + timestamp
  OUTPUT_ROOT               Root output directory. Default: ./outputs/sft
  CKPT_DIR                  Final checkpoint/log directory. Default: ${OUTPUT_ROOT}/${EXP_NAME}
  TRAIN_BATCH_SIZE          Global train batch size. Default: 32
  MICRO_BATCH_SIZE_PER_GPU  Per-GPU micro batch size. Default: 1
  DATA_MAX_LENGTH           Max sequence length. Default: 32768
  LR                        Learning rate. Default: 1e-5
  TOTAL_EPOCHS              Number of epochs. Default: 1
  TOTAL_TRAINING_STEPS      Total training steps. Default: null
  SAVE_FREQ                 Checkpoint save frequency. Default: -1
  TEST_FREQ                 Validation frequency. Default: -1
  LOGGER                    Hydra logger list. Default: ['console','tensorboard']
  SP_SIZE                   Ulysses sequence parallel size. Default: 1
  LORA_RANK                 LoRA rank. Default: 0
  LORA_ALPHA                LoRA alpha. Default: 16
  TARGET_MODULES            LoRA target modules. Default: all-linear
  LIGER                     Enable liger kernel. Default: False
  MULTITURN                 Enable multi-turn SFT dataset. Default: True
  MESSAGES_KEY              Multi-turn messages column. Default: messages
  TOOLS_KEY                 Multi-turn tools column. Default: tools
  ENABLE_THINKING_KEY       Multi-turn enable_thinking column. Default: enable_thinking
  PROMPT_KEY                Single-turn prompt column. Default: question
  RESPONSE_KEY              Single-turn response column. Default: gt
  PROMPT_DICT_KEYS          Optional prompt dict keys. Default: null
  RESPONSE_DICT_KEYS        Optional response dict keys. Default: null
  TRUNCATION                Truncation mode. Default: right
  RM_PAD                    Enable remove padding. Default: True
  RESUME_MODE               auto | disable | resume_path. Default: auto
  RESUME_FROM_PATH          Path used when RESUME_MODE=resume_path
  DRY_RUN                   Print the final torchrun command without executing it.
  EXTRA_TRAIN_ARGS          Extra hydra overrides appended verbatim.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    show_help
    exit 0
fi

ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.fsdp_sft_trainer"}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
NUM_GPUS=${NUM_GPUS:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

MODEL_PATH=${MODEL_PATH:-}
TRAIN_FILES=${TRAIN_FILES:-}
VAL_FILES=${VAL_FILES:-${TRAIN_FILES}}

PROJECT_NAME=${PROJECT_NAME:-deepresearcher-sft}
MODEL_NAME_FOR_EXP=$(basename "${MODEL_PATH:-model}")
EXP_NAME=${EXP_NAME:-"${MODEL_NAME_FOR_EXP,,}-$(date +'%Y%m%d-%H%M%S')"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"./outputs/sft"}
CKPT_DIR=${CKPT_DIR:-"${OUTPUT_ROOT}/${EXP_NAME}"}
LOG_FILE=${LOG_FILE:-"${CKPT_DIR}/train-$(date +'%Y%m%d-%H%M%S').log"}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
DATA_MAX_LENGTH=${DATA_MAX_LENGTH:-32768}
LR=${LR:-1e-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-null}
SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:--1}
LOGGER=${LOGGER:-"['console','tensorboard']"}

SP_SIZE=${SP_SIZE:-1}
LORA_RANK=${LORA_RANK:-0}
LORA_ALPHA=${LORA_ALPHA:-16}
TARGET_MODULES=${TARGET_MODULES:-all-linear}
LIGER=${LIGER:-False}
MULTITURN=${MULTITURN:-True}
MESSAGES_KEY=${MESSAGES_KEY:-messages}
TOOLS_KEY=${TOOLS_KEY:-tools}
ENABLE_THINKING_KEY=${ENABLE_THINKING_KEY:-enable_thinking}
PROMPT_KEY=${PROMPT_KEY:-question}
RESPONSE_KEY=${RESPONSE_KEY:-gt}
PROMPT_DICT_KEYS=${PROMPT_DICT_KEYS:-null}
RESPONSE_DICT_KEYS=${RESPONSE_DICT_KEYS:-null}
TRUNCATION=${TRUNCATION:-right}
RM_PAD=${RM_PAD:-True}
RESUME_MODE=${RESUME_MODE:-auto}
RESUME_FROM_PATH=${RESUME_FROM_PATH:-null}
DRY_RUN=${DRY_RUN:-False}

if [[ -z "${MODEL_PATH}" ]]; then
    echo "ERROR: MODEL_PATH must be set." >&2
    exit 1
fi

if [[ -z "${TRAIN_FILES}" ]]; then
    echo "ERROR: TRAIN_FILES must be set." >&2
    exit 1
fi

read -r -a ENTRYPOINT_ARGS <<< "${ENTRYPOINT}"

CMD=(
    torchrun
    --nnodes="${NNODES}"
    --node_rank="${NODE_RANK}"
    --nproc_per_node="${NUM_GPUS}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
    "${ENTRYPOINT_ARGS[@]}"
    "data.train_files=${TRAIN_FILES}"
    "data.val_files=${VAL_FILES}"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.max_length=${DATA_MAX_LENGTH}"
    "data.prompt_key=${PROMPT_KEY}"
    "data.response_key=${RESPONSE_KEY}"
    "data.prompt_dict_keys=${PROMPT_DICT_KEYS}"
    "data.response_dict_keys=${RESPONSE_DICT_KEYS}"
    "data.multiturn.enable=${MULTITURN}"
    "data.multiturn.messages_key=${MESSAGES_KEY}"
    "data.multiturn.tools_key=${TOOLS_KEY}"
    "data.multiturn.enable_thinking_key=${ENABLE_THINKING_KEY}"
    "data.truncation=${TRUNCATION}"
    "optim.lr=${LR}"
    "data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU}"
    "model.partial_pretrain=${MODEL_PATH}"
    "model.lora_rank=${LORA_RANK}"
    "model.lora_alpha=${LORA_ALPHA}"
    "model.target_modules=${TARGET_MODULES}"
    "model.use_liger=${LIGER}"
    "model.enable_gradient_checkpointing=true"
    "ulysses_sequence_parallel_size=${SP_SIZE}"
    "use_remove_padding=${RM_PAD}"
    "trainer.default_local_dir=${CKPT_DIR}"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${EXP_NAME}"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
    "trainer.logger=${LOGGER}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.resume_mode=${RESUME_MODE}"
    "trainer.resume_from_path=${RESUME_FROM_PATH}"
    "trainer.default_hdfs_dir=null"
)

if [[ -n "${EXTRA_TRAIN_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS_ARRAY=(${EXTRA_TRAIN_ARGS})
    CMD+=("${EXTRA_ARGS_ARRAY[@]}")
fi

CMD+=("$@")

if [[ "${DRY_RUN}" == "True" ]]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "${CKPT_DIR}"
exec > >(tee "${LOG_FILE}") 2>&1
set -x

"${CMD[@]}"
