#!/usr/bin/env bash
set -euo pipefail

# PlannerModelServer（规划器模型服务）正式入口。
# 这里使用 vLLM（大模型推理服务框架）把本地或云端模型暴露成
# OpenAI-compatible chat completions（兼容 OpenAI 的聊天补全接口）。

PLANNER_MODEL_PATH="${PLANNER_MODEL_PATH:-Qwen/Qwen3.5-4B}"
PLANNER_MODEL_ID="${PLANNER_MODEL_ID:-qwen3_5_4b_base}"
PLANNER_BASE_MODEL_ID="${PLANNER_BASE_MODEL_ID:-qwen3_5_4b_base}"
PLANNER_ADAPTER_PATH="${PLANNER_ADAPTER_PATH:-}"
PLANNER_HOST="${PLANNER_HOST:-127.0.0.1}"
PLANNER_PORT="${PLANNER_PORT:-8019}"
PLANNER_DTYPE="${PLANNER_DTYPE:-auto}"
PLANNER_MAX_MODEL_LEN="${PLANNER_MAX_MODEL_LEN:-4096}"
PLANNER_GPU_MEMORY_UTILIZATION="${PLANNER_GPU_MEMORY_UTILIZATION:-0.85}"
PLANNER_API_KEY="${PLANNER_API_KEY:-}"
PLANNER_EXTRA_ARGS="${PLANNER_EXTRA_ARGS:-}"

args=(
  vllm
  serve
  "$PLANNER_MODEL_PATH"
  --host "$PLANNER_HOST"
  --port "$PLANNER_PORT"
  --dtype "$PLANNER_DTYPE"
  --max-model-len "$PLANNER_MAX_MODEL_LEN"
  --gpu-memory-utilization "$PLANNER_GPU_MEMORY_UTILIZATION"
)

if [[ -n "$PLANNER_ADAPTER_PATH" ]]; then
  # adapter（适配器）存在时，请求侧使用 PLANNER_MODEL_ID（规划器模型身份）命中微调模型；
  # base model（基础模型）身份仍保留给审计和回滚。
  args+=(
    --served-model-name "$PLANNER_BASE_MODEL_ID"
    --enable-lora
    --lora-modules "$PLANNER_MODEL_ID=$PLANNER_ADAPTER_PATH"
  )
else
  args+=(--served-model-name "$PLANNER_MODEL_ID")
fi

if [[ -n "$PLANNER_API_KEY" ]]; then
  # api_key（接口密钥）由 vLLM 校验；业务侧 PlannerClient（规划器客户端）使用同值写入
  # Authorization Bearer（鉴权头），密钥不进入请求体或 Trace（轨迹记录）。
  args+=(--api-key "$PLANNER_API_KEY")
fi

if [[ -n "$PLANNER_EXTRA_ARGS" ]]; then
  # extra args（额外参数）用于云端临时调优，例如张量并行数；避免在脚本里预设所有硬件细节。
  # 含空格的路径请直接修改脚本或在云端包装脚本中传数组。
  read -r -a extra_args <<< "$PLANNER_EXTRA_ARGS"
  args+=("${extra_args[@]}")
fi

exec "${args[@]}"
