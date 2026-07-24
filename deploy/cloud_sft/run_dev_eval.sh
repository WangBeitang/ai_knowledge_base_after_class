#!/usr/bin/env bash
set -euo pipefail

# dev eval（开发集评测）入口，用训练后的 checkpoint（检查点）跑固定 dev case（开发样本）。

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

if [[ -z "${SFT_CHECKPOINT_DIR:-}" ]]; then
  echo "必须设置 SFT_CHECKPOINT_DIR（监督微调检查点目录），例如 evaluation/stage9/artifacts/sft/checkpoints/<run_id>" >&2
  exit 2
fi

timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
CLOUD_RUN_ROOT="${CLOUD_RUN_ROOT:-evaluation/stage9/artifacts/cloud_runs}"
RUN_DIR="$CLOUD_RUN_ROOT/dev_eval_$timestamp"
mkdir -p "$RUN_DIR"

DEV_EVAL_CASES="${DEV_EVAL_CASES:-evaluation/stage8/cases/planner_cases.jsonl}"
DEV_EVAL_SNAPSHOT="${DEV_EVAL_SNAPSHOT:-evaluation/stage8/snapshots/environment_snapshot.json}"
DEV_EVAL_SPLIT="${DEV_EVAL_SPLIT:-dev}"
DEV_EVAL_PROVIDER="${DEV_EVAL_PROVIDER:-snapshot_expected_chunks}"
DEV_EVAL_OUTPUT="${DEV_EVAL_OUTPUT:-$RUN_DIR/sft_eval_dev.json}"
SFT_TRAIN_CONFIG="${SFT_TRAIN_CONFIG:-evaluation/stage9/configs/planner_sft_qwen3_5_4b_lora.json}"

eval_args=(
  evaluation/stage9/model_planner/eval_model_planner.py
  --checkpoint "$SFT_CHECKPOINT_DIR"
  --cases "$DEV_EVAL_CASES"
  --snapshot "$DEV_EVAL_SNAPSHOT"
  --split "$DEV_EVAL_SPLIT"
  --provider "$DEV_EVAL_PROVIDER"
  --output "$DEV_EVAL_OUTPUT"
)
if [[ -n "${DEV_EVAL_MAX_CASES:-}" ]]; then
  eval_args+=(--max-cases "$DEV_EVAL_MAX_CASES")
fi

COMMAND_TEXT="${PYTHON_BIN:-uv run python} ${eval_args[*]}"
printf '%s\n' "$COMMAND_TEXT" > "$RUN_DIR/command.txt"
run_python "${eval_args[@]}" 2>&1 | tee "$RUN_DIR/dev_eval.log"

run_python scripts/cloud_sft/collect_cloud_run_report.py \
  --output "$RUN_DIR/cloud_run_report.json" \
  --training-config "$SFT_TRAIN_CONFIG" \
  --checkpoint-dir "$SFT_CHECKPOINT_DIR" \
  --dev-eval-output "$DEV_EVAL_OUTPUT" \
  --command "$COMMAND_TEXT" \
  --notes "stage9 cloud dev eval（阶段 9 云端开发集评测）"

echo "run_dir（运行目录）=$RUN_DIR"
