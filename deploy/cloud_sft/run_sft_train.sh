#!/usr/bin/env bash
set -euo pipefail

# train（正式训练）入口只负责调用现有 SFT（监督微调）训练代码和收集审计报告。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

load_env() {
  local env_file="${CLOUD_SFT_ENV_FILE:-}"
  if [[ -z "$env_file" ]]; then
    return
  fi
  if [[ ! -f "$env_file" && -f "$PROJECT_ROOT/$env_file" ]]; then
    env_file="$PROJECT_ROOT/$env_file"
  fi
  if [[ ! -f "$env_file" ]]; then
    echo "CLOUD_SFT_ENV_FILE（云端环境文件）不存在：$env_file" >&2
    exit 2
  fi
  set -a
  source "$env_file"
  set +a
}

run_python() {
  read -r -a python_cmd <<< "${PYTHON_BIN:-uv run python}"
  "${python_cmd[@]}" "$@"
}

load_env
APP_ROOT="${APP_ROOT:-$PROJECT_ROOT}"
cd "$APP_ROOT"
SFT_VENV_PATH="${SFT_VENV_PATH:-$APP_ROOT/.venv-sft}"
PYTHON_BIN="${PYTHON_BIN:-$SFT_VENV_PATH/bin/python}"

timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
CLOUD_RUN_ROOT="${CLOUD_RUN_ROOT:-evaluation/stage9/artifacts/cloud_runs}"
RUN_DIR="$CLOUD_RUN_ROOT/sft_train_$timestamp"
mkdir -p "$RUN_DIR"

SFT_TRAIN_CONFIG="${SFT_TRAIN_CONFIG:-evaluation/stage9/configs/planner_sft_qwen3_5_4b_lora.json}"
COMMAND_TEXT="$PYTHON_BIN evaluation/stage9/model_planner/sft_train.py --config $SFT_TRAIN_CONFIG"
printf '%s\n' "$COMMAND_TEXT" > "$RUN_DIR/command.txt"

run_python evaluation/stage9/model_planner/sft_train.py --config "$SFT_TRAIN_CONFIG" 2>&1 | tee "$RUN_DIR/sft_train.log"

checkpoint_path="$(awk -F= '/^checkpoint=/{print $2}' "$RUN_DIR/sft_train.log" | tail -1)"
report_args=(
  scripts/cloud_sft/collect_cloud_run_report.py
  --output "$RUN_DIR/cloud_run_report.json"
  --training-config "$SFT_TRAIN_CONFIG"
  --command "$COMMAND_TEXT"
  --notes "stage9 cloud SFT train（阶段 9 云端监督微调训练）"
)
if [[ -n "$checkpoint_path" ]]; then
  report_args+=(--checkpoint-dir "$checkpoint_path")
fi
run_python "${report_args[@]}"

echo "run_dir（运行目录）=$RUN_DIR"
