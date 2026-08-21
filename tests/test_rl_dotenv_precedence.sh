#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT

dotenv_path="$tmp_dir/test.env"
cat >"$dotenv_path" <<'EOF'
MODEL_PATH=dotenv-model
DRY_RUN=false
CONFIRM_SPEND=NO
SEED=999
TOOL_CACHE_PATH=/dotenv/cache.jsonl
UNDEFINED_DEFAULT=from-dotenv
EOF

export MODEL_PATH=explicit-model
export DRY_RUN=true
export CONFIRM_SPEND=YES
export SEED=42
export TOOL_CACHE_PATH=/explicit/cache.jsonl

# shellcheck source=../RL/load_dotenv_defaults.sh
source "$REPO_ROOT/RL/load_dotenv_defaults.sh"
load_dotenv_defaults "$dotenv_path"

[ "$MODEL_PATH" = explicit-model ]
[ "$DRY_RUN" = true ]
[ "$CONFIRM_SPEND" = YES ]
[ "$SEED" = 42 ]
[ "$TOOL_CACHE_PATH" = /explicit/cache.jsonl ]
[ "$UNDEFINED_DEFAULT" = from-dotenv ]
