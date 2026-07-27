#!/usr/bin/env bash
set -euo pipefail

# PlannerModelServer（规划器模型服务）云端启动入口，复用正式 vLLM（大模型推理服务框架）脚本。

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

export PLANNER_MODEL_PATH="${PLANNER_MODEL_PATH:-Qwen/Qwen3.5-4B}"
export PLANNER_BASE_MODEL_ID="${PLANNER_BASE_MODEL_ID:-qwen3_5_4b_base}"
export PLANNER_MODEL_ID="${PLANNER_MODEL_ID:-qwen3_5_4b_sft_stage9}"
export PLANNER_ADAPTER_PATH="${PLANNER_ADAPTER_PATH:-}"
export PLANNER_HOST="${PLANNER_HOST:-127.0.0.1}"
export PLANNER_PORT="${PLANNER_PORT:-8019}"
export PLANNER_MODEL_ENDPOINT="${PLANNER_MODEL_ENDPOINT:-http://$PLANNER_HOST:$PLANNER_PORT/v1/chat/completions}"
VLLM_VENV_PATH="${VLLM_VENV_PATH:-/root/.venv-vllm}"
export PATH="$VLLM_VENV_PATH/bin:$PATH"

echo "PLANNER_MODEL_PATH（模型路径）=$PLANNER_MODEL_PATH"
echo "PLANNER_MODEL_ID（模型身份）=$PLANNER_MODEL_ID"
echo "PLANNER_ADAPTER_PATH（适配器路径）=$PLANNER_ADAPTER_PATH"
echo "PLANNER_MODEL_ENDPOINT（模型服务地址）=$PLANNER_MODEL_ENDPOINT"

if [[ ! -x "$VLLM_VENV_PATH/bin/vllm" ]]; then
  echo "vLLM 可执行文件不存在：$VLLM_VENV_PATH/bin/vllm" >&2
  exit 2
fi

if [[ "${REQUIRE_GPU_PREFLIGHT:-1}" == "1" ]]; then
  # 每次云端启动 vLLM 前重跑只读门禁，避免复用上一次开机留下的过期 CUDA/端口结论。
  CLOUD_PREFLIGHT_MODE=gpu \
  CLOUD_SFT_ENV_FILE="${CLOUD_SFT_ENV_FILE:-}" \
    bash deploy/cloud_sft/run_runtime_preflight.sh
fi

exec bash deploy/planner_model_server/run_vllm_planner_server.sh
