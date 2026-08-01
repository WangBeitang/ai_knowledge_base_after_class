#!/usr/bin/env bash
set -euo pipefail

# 正式 GRPO（群组相对策略优化）入口：固定 75 case × 4 rollout × 1 epoch。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

load_env() {
  local env_file="${CLOUD_GRPO_ENV_FILE:-}"
  if [[ -z "$env_file" ]]; then
    return
  fi
  if [[ ! -f "$env_file" && -f "$PROJECT_ROOT/$env_file" ]]; then
    env_file="$PROJECT_ROOT/$env_file"
  fi
  if [[ ! -f "$env_file" ]]; then
    echo "CLOUD_GRPO_ENV_FILE（云端环境文件）不存在：$env_file" >&2
    exit 2
  fi
  set -a
  source "$env_file"
  set +a
}

load_env
APP_ROOT="${APP_ROOT:-$PROJECT_ROOT}"
GRPO_PYTHON_BIN="${GRPO_PYTHON_BIN:-$APP_ROOT/.venv-sft/bin/python}"
GRPO_CONFIG="${GRPO_CONFIG:-evaluation/stage9/configs/planner_grpo_qwen3_5_4b_lora_formal.json}"
GRPO_LAUNCH_LOG_ROOT="${GRPO_LAUNCH_LOG_ROOT:-evaluation/stage9/artifacts/cloud_runs}"

cd "$APP_ROOT"
if [[ ! -x "$GRPO_PYTHON_BIN" ]]; then
  echo "GRPO_PYTHON_BIN（训练 Python）不可执行：$GRPO_PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$GRPO_CONFIG" ]]; then
  echo "GRPO_CONFIG（正式训练配置）不存在：$GRPO_CONFIG" >&2
  exit 2
fi

timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
mkdir -p "$GRPO_LAUNCH_LOG_ROOT"
launch_dir="$(mktemp -d "$GRPO_LAUNCH_LOG_ROOT/grpo_formal_launch_${timestamp}_XXXXXXXX")"
command=("$GRPO_PYTHON_BIN" -m app.rag.training.grpo.cli --config "$GRPO_CONFIG" "$@")
printf '%q ' "${command[@]}" > "$launch_dir/command.txt"
printf '\n' >> "$launch_dir/command.txt"

set +e
"${command[@]}" 2>&1 | tee "$launch_dir/training.log"
status="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$status" > "$launch_dir/exit_code.txt"
echo "launch_log_dir（启动日志目录）=$launch_dir"
exit "$status"
