#!/usr/bin/env bash
set -euo pipefail

# smoke（冒烟）训练默认只跑少量样本和极少 step（训练步），用于确认云端训练链路可用。

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
RUN_DIR="$CLOUD_RUN_ROOT/sft_smoke_$timestamp"
mkdir -p "$RUN_DIR"

SFT_SMOKE_BASE_CONFIG="${SFT_SMOKE_BASE_CONFIG:-${SFT_TRAIN_CONFIG:-evaluation/stage9/configs/planner_sft_qwen3_5_4b_lora.json}}"
SMOKE_CONFIG_PATH="$RUN_DIR/planner_sft_smoke_config.json"
export SFT_SMOKE_BASE_CONFIG SMOKE_CONFIG_PATH
export STAGE9_SFT_SMOKE_RUN_NAME="${STAGE9_SFT_SMOKE_RUN_NAME:-planner-sft-stage9-cloud-smoke}"
export STAGE9_SFT_SMOKE_MAX_SAMPLES="${STAGE9_SFT_SMOKE_MAX_SAMPLES:-4}"
export STAGE9_SFT_SMOKE_MAX_STEPS="${STAGE9_SFT_SMOKE_MAX_STEPS:-1}"
export STAGE9_SFT_SMOKE_NUM_EPOCHS="${STAGE9_SFT_SMOKE_NUM_EPOCHS:-1}"
export STAGE9_SFT_SMOKE_PREVIEW_COUNT="${STAGE9_SFT_SMOKE_PREVIEW_COUNT:-4}"
export SFT_OUTPUT_ROOT="${SFT_OUTPUT_ROOT:-evaluation/stage9/artifacts/sft/checkpoints}"

run_python - <<'PY'
import json
import os
from pathlib import Path

base_path = Path(os.environ["SFT_SMOKE_BASE_CONFIG"])
output_path = Path(os.environ["SMOKE_CONFIG_PATH"])
payload = json.loads(base_path.read_text(encoding="utf-8"))

payload["run_name"] = os.environ["STAGE9_SFT_SMOKE_RUN_NAME"]
payload["max_train_samples"] = int(os.environ["STAGE9_SFT_SMOKE_MAX_SAMPLES"])
payload["max_steps"] = int(os.environ["STAGE9_SFT_SMOKE_MAX_STEPS"])
payload["num_epochs"] = float(os.environ["STAGE9_SFT_SMOKE_NUM_EPOCHS"])
payload["save_training_preview_count"] = int(os.environ["STAGE9_SFT_SMOKE_PREVIEW_COUNT"])
payload["output_root"] = os.environ["SFT_OUTPUT_ROOT"]

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"smoke_config（冒烟配置）={output_path}")
PY

COMMAND_TEXT="$PYTHON_BIN evaluation/stage9/model_planner/sft_train.py --config $SMOKE_CONFIG_PATH"
printf '%s\n' "$COMMAND_TEXT" > "$RUN_DIR/command.txt"
run_python evaluation/stage9/model_planner/sft_train.py --config "$SMOKE_CONFIG_PATH" 2>&1 | tee "$RUN_DIR/sft_smoke.log"

checkpoint_path="$(awk -F= '/^checkpoint=/{print $2}' "$RUN_DIR/sft_smoke.log" | tail -1)"
report_args=(
  scripts/cloud_sft/collect_cloud_run_report.py
  --output "$RUN_DIR/cloud_run_report.json"
  --training-config "$SMOKE_CONFIG_PATH"
  --command "$COMMAND_TEXT"
  --notes "stage9 cloud SFT smoke（阶段 9 云端监督微调冒烟）"
)
if [[ -n "$checkpoint_path" ]]; then
  report_args+=(--checkpoint-dir "$checkpoint_path")
fi
run_python "${report_args[@]}"

echo "run_dir（运行目录）=$RUN_DIR"
