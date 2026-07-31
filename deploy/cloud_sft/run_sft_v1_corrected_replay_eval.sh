#!/usr/bin/env bash
set -euo pipefail

# 任务 9.3.20：同一 SFT v1（监督微调第一版）checkpoint（检查点）只替换为
# 9.3.18 冻结 Replay Provider（回放动作执行器），运行 25 条 reviewed dev（已审核开发集）。
# 旧 9.3.16 评测只作为明确引用的对比基线；不加载旧决定或递归历史哈希链。
# 新产物进入独立时间戳目录；绝不运行 heldout test（留出测试）。

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
CLOUD_RUN_ROOT="${CLOUD_RUN_ROOT:-evaluation/stage9/artifacts/cloud_runs}"
SFT_V1_CORRECTED_REPLAY_PREFLIGHT_ONLY="${SFT_V1_CORRECTED_REPLAY_PREFLIGHT_ONLY:-0}"
SFT_V1_CORRECTED_REPLAY_OLD_EVAL="${SFT_V1_CORRECTED_REPLAY_OLD_EVAL:-evaluation/stage9/artifacts/sft/sft_expanded_dev_eval.json}"
SFT_V1_CORRECTED_REPLAY_RECORDS="${SFT_V1_CORRECTED_REPLAY_RECORDS:-evaluation/stage9/artifacts/provider_records/expanded_dev_provider_observations.jsonl}"
SFT_V1_CORRECTED_REPLAY_CONTRACT="${SFT_V1_CORRECTED_REPLAY_CONTRACT:-evaluation/stage9/artifacts/provider_records/expanded_dev_replay_contract.json}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "SFT Python（监督微调 Python）解释器不存在或不可执行：$PYTHON_BIN" >&2
  exit 2
fi
if [[ -z "${SFT_CHECKPOINT_DIR:-}" ]]; then
  echo "必须设置 SFT_CHECKPOINT_DIR（正式监督微调检查点根目录）。" >&2
  exit 2
fi
if [[ "${HF_HUB_OFFLINE:-0}" != "1" || "${TRANSFORMERS_OFFLINE:-0}" != "1" ]]; then
  echo "禁止静默下载：HF_HUB_OFFLINE 和 TRANSFORMERS_OFFLINE 必须都为 1。" >&2
  exit 2
fi

timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
run_kind="sft_v1_corrected_replay"
if [[ "$SFT_V1_CORRECTED_REPLAY_PREFLIGHT_ONLY" == "1" ]]; then
  run_kind="sft_v1_corrected_replay_preflight"
fi
RUN_DIR="$CLOUD_RUN_ROOT/${run_kind}_$timestamp"
if [[ -e "$RUN_DIR" ]]; then
  echo "9.3.20 运行目录已存在，拒绝覆盖：$RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"

args=(
  -m evaluation.stage9.admission.run_sft_v1_corrected_replay_eval
  --checkpoint "$SFT_CHECKPOINT_DIR"
  --old-eval "$SFT_V1_CORRECTED_REPLAY_OLD_EVAL"
  --provider-records "$SFT_V1_CORRECTED_REPLAY_RECORDS"
  --replay-contract "$SFT_V1_CORRECTED_REPLAY_CONTRACT"
  --corrected-eval-output "$RUN_DIR/sft_v1_corrected_replay_eval.json"
  --comparison-output "$RUN_DIR/sft_v1_corrected_replay_comparison.json"
  --report "$RUN_DIR/阶段9-SFT-v1校正复评报告.md"
)

if [[ "$SFT_V1_CORRECTED_REPLAY_PREFLIGHT_ONLY" == "1" ]]; then
  args+=(--preflight-only)
else
  if [[ "${REQUIRE_CUDA:-0}" != "1" ]]; then
    echo "正式 9.3.20 复评要求 REQUIRE_CUDA=1。" >&2
    exit 2
  fi
  "$PYTHON_BIN" -c \
    'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("gpu=", torch.cuda.get_device_name(0))'
fi

COMMAND_TEXT="$PYTHON_BIN ${args[*]}"
printf '%s\n' "$COMMAND_TEXT" > "$RUN_DIR/command.txt"
"$PYTHON_BIN" "${args[@]}" 2>&1 | tee "$RUN_DIR/run.log"

if [[ "$SFT_V1_CORRECTED_REPLAY_PREFLIGHT_ONLY" == "1" ]]; then
  printf '9.3.20 preflight（运行前检查）status=completed\n'
else
  sha256sum \
    "$RUN_DIR/sft_v1_corrected_replay_eval.json" \
    "$RUN_DIR/sft_v1_corrected_replay_comparison.json" \
    "$RUN_DIR/阶段9-SFT-v1校正复评报告.md" \
    > "$RUN_DIR/SHA256SUMS"
  printf '9.3.20 corrected replay eval（校正回放复评）status=completed\n'
fi
printf 'run_dir（运行目录）=%s\n' "$RUN_DIR"
