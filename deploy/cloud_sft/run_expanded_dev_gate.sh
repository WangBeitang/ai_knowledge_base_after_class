#!/usr/bin/env bash
set -euo pipefail

# 任务 9.3.16：用 direct checkpoint runtime（直接检查点运行时）跑完整 balanced dev，
# 冻结逐 case 证据和 9.4 准入结论；本脚本绝不运行 test/heldout case。

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

load_env
APP_ROOT="${APP_ROOT:-$PROJECT_ROOT}"
cd "$APP_ROOT"

SFT_VENV_PATH="${SFT_VENV_PATH:-$APP_ROOT/.venv-sft}"
PYTHON_BIN="${PYTHON_BIN:-$SFT_VENV_PATH/bin/python}"
SFT_TRAIN_CONFIG="${SFT_TRAIN_CONFIG:-evaluation/stage9/configs/planner_sft_qwen3_5_4b_lora.json}"
CLOUD_RUN_ROOT="${CLOUD_RUN_ROOT:-evaluation/stage9/artifacts/cloud_runs}"
EXPANDED_DEV_CASES="${EXPANDED_DEV_CASES:-evaluation/stage8/cases/planner_cases.jsonl}"
EXPANDED_DEV_SPLIT_MANIFEST="${EXPANDED_DEV_SPLIT_MANIFEST:-evaluation/stage8/cases/split_manifest.json}"
EXPANDED_DEV_SNAPSHOT="${EXPANDED_DEV_SNAPSHOT:-evaluation/stage9/artifacts/heldout_route_test/environment_snapshot.json}"
EXPANDED_DEV_ROUTE_MATRIX="${EXPANDED_DEV_ROUTE_MATRIX:-evaluation/stage9/configs/planner_eval_route_matrix_v1.json}"
EXPANDED_DEV_REWARD_PROFILE="${EXPANDED_DEV_REWARD_PROFILE:-evaluation/stage9/configs/reward_v1_1_training_profile.json}"
EXPANDED_DEV_REWARD_VALIDATION="${EXPANDED_DEV_REWARD_VALIDATION:-evaluation/stage9/artifacts/reward/reward_v1_1_balanced_dev_validation.json}"
EXPANDED_DEV_PROVIDER="${EXPANDED_DEV_PROVIDER:-snapshot_expected_chunks}"
EXPANDED_DEV_EVAL_OUTPUT="${EXPANDED_DEV_EVAL_OUTPUT:-evaluation/stage9/artifacts/sft/sft_expanded_dev_eval.json}"
EXPANDED_DEV_DECISION_OUTPUT="${EXPANDED_DEV_DECISION_OUTPUT:-evaluation/stage9/artifacts/sft/sft_9_4_admission_decision.json}"
EXPANDED_DEV_REPORT="${EXPANDED_DEV_REPORT:-evaluation/stage9/artifacts/reports/阶段9-SFT-9.4准入报告.md}"
EXPANDED_DEV_OVERWRITE="${EXPANDED_DEV_OVERWRITE:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "SFT Python 解释器不存在或不可执行：$PYTHON_BIN" >&2
  exit 2
fi
if [[ -z "${SFT_CHECKPOINT_DIR:-}" ]]; then
  echo "必须设置 SFT_CHECKPOINT_DIR（正式 SFT checkpoint 根目录）。" >&2
  exit 2
fi
if [[ "${REQUIRE_CUDA:-1}" != "1" ]]; then
  echo "9.3.16 是正式模型推理，REQUIRE_CUDA 必须为 1。" >&2
  exit 2
fi
if [[ "${HF_HUB_OFFLINE:-0}" != "1" || "${TRANSFORMERS_OFFLINE:-0}" != "1" ]]; then
  echo "GPU 计费期间禁止静默下载：HF_HUB_OFFLINE 和 TRANSFORMERS_OFFLINE 必须都为 1。" >&2
  exit 2
fi
if [[ "$EXPANDED_DEV_PROVIDER" != "snapshot_expected_chunks" ]]; then
  echo "9.3.16 冻结 provider 必须为 snapshot_expected_chunks。" >&2
  exit 2
fi
if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi

"$PYTHON_BIN" -c \
  'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("gpu=", torch.cuda.get_device_name(0))'

timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
RUN_DIR="$CLOUD_RUN_ROOT/expanded_dev_gate_$timestamp"
mkdir -p "$RUN_DIR"

gate_args=(
  -m evaluation.stage9.admission.run_sft_expanded_dev_gate
  --checkpoint "$SFT_CHECKPOINT_DIR"
  --cases "$EXPANDED_DEV_CASES"
  --split-manifest "$EXPANDED_DEV_SPLIT_MANIFEST"
  --snapshot "$EXPANDED_DEV_SNAPSHOT"
  --route-matrix "$EXPANDED_DEV_ROUTE_MATRIX"
  --reward-profile "$EXPANDED_DEV_REWARD_PROFILE"
  --reward-validation "$EXPANDED_DEV_REWARD_VALIDATION"
  --provider "$EXPANDED_DEV_PROVIDER"
  --eval-output "$EXPANDED_DEV_EVAL_OUTPUT"
  --decision-output "$EXPANDED_DEV_DECISION_OUTPUT"
  --report "$EXPANDED_DEV_REPORT"
)
if [[ "$EXPANDED_DEV_OVERWRITE" == "1" ]]; then
  gate_args+=(--overwrite)
fi

COMMAND_TEXT="$PYTHON_BIN ${gate_args[*]}"
printf '%s\n' "$COMMAND_TEXT" > "$RUN_DIR/command.txt"
"$PYTHON_BIN" "${gate_args[@]}" 2>&1 | tee "$RUN_DIR/expanded_dev_gate.log"

# 每个 run_dir 保留一份不可变副本；固定路径供阶段文档和后续 9.4 读取。
cp -p "$EXPANDED_DEV_EVAL_OUTPUT" "$RUN_DIR/sft_expanded_dev_eval.json"
cp -p "$EXPANDED_DEV_DECISION_OUTPUT" "$RUN_DIR/sft_9_4_admission_decision.json"
cp -p "$EXPANDED_DEV_REPORT" "$RUN_DIR/阶段9-SFT-9.4准入报告.md"

"$PYTHON_BIN" scripts/cloud_sft/collect_cloud_run_report.py \
  --output "$RUN_DIR/cloud_run_report.json" \
  --training-config "$SFT_TRAIN_CONFIG" \
  --checkpoint-dir "$SFT_CHECKPOINT_DIR" \
  --dev-eval-output "$EXPANDED_DEV_EVAL_OUTPUT" \
  --admission-decision-output "$EXPANDED_DEV_DECISION_OUTPUT" \
  --admission-report "$EXPANDED_DEV_REPORT" \
  --command "$COMMAND_TEXT" \
  --notes "stage9 9.3.16 expanded dev 与 9.4 准入门禁"

eligible="$("$PYTHON_BIN" -c \
  'import json,sys; print(str(bool(json.load(open(sys.argv[1]))["summary"]["eligible_for_stage9_4"])).lower())' \
  "$EXPANDED_DEV_DECISION_OUTPUT")"
printf 'run_dir（运行目录）=%s\n' "$RUN_DIR"
printf 'eligible_for_stage9_4（是否允许进入 9.4）=%s\n' "$eligible"
if [[ "$eligible" != "true" ]]; then
  echo "SFT checkpoint 未通过 9.3.16；产物已保留，禁止进入 9.4。" >&2
  exit 3
fi
