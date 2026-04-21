#!/usr/bin/env bash
set -euo pipefail

show_help() {
    cat <<'EOF'
Usage:
  bash scripts/merge_checkpoint.sh --backend <fsdp|megatron> --local-dir <ckpt_dir> --target-dir <hf_dir> [extra_args...]

Examples:
  bash scripts/merge_checkpoint.sh \
    --backend fsdp \
    --local-dir /path/to/checkpoints/global_step_100/actor \
    --target-dir /path/to/merged_model

  bash scripts/merge_checkpoint.sh \
    --backend megatron \
    --local-dir /path/to/checkpoints/global_step_100/actor \
    --target-dir /path/to/merged_model \
    --tie-word-embedding
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    show_help
    exit 0
fi

BACKEND=""
LOCAL_DIR=""
TARGET_DIR=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --local-dir)
            LOCAL_DIR="$2"
            shift 2
            ;;
        --target-dir)
            TARGET_DIR="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "${BACKEND}" || -z "${LOCAL_DIR}" || -z "${TARGET_DIR}" ]]; then
    show_help
    exit 1
fi

python -m verl.model_merger merge \
    --backend "${BACKEND}" \
    --local_dir "${LOCAL_DIR}" \
    --target_dir "${TARGET_DIR}" \
    "${EXTRA_ARGS[@]}"
