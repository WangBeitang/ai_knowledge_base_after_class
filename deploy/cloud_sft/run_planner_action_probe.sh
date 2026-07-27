#!/usr/bin/env bash
set -euo pipefail

# vLLM PlannerModelServer（规划器模型服务）启动后运行六类 Action 和 Web 禁用边界 HTTP 探针。

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
timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
RUN_DIR="$CLOUD_RUN_ROOT/planner_action_probe_$timestamp"
PROBE_OUTPUT="${CLOUD_PLANNER_HTTP_PROBE_OUTPUT:-evaluation/stage9/artifacts/sft/cloud_smoke_planner_http.json}"
mkdir -p "$RUN_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "训练 Python 解释器不存在：$PYTHON_BIN" >&2
  exit 2
fi

endpoint="${PLANNER_MODEL_ENDPOINT:-http://127.0.0.1:8019/v1/chat/completions}"
health_url="${endpoint%/v1/chat/completions}/health"
curl --fail --silent --show-error --max-time 10 "$health_url" >/dev/null

args=(
  scripts/cloud_sft/probe_planner_actions.py
  --endpoint "$endpoint"
  --model-id "${PLANNER_MODEL_ID:-qwen3_5_4b_sft_stage9}"
  --timeout-seconds "${PLANNER_TIMEOUT_SECONDS:-120}"
  --max-new-tokens "${PLANNER_MAX_NEW_TOKENS:-128}"
  --output "$PROBE_OUTPUT"
)
if [[ "${CLOUD_PROBE_STRICT_ACTION_MATCH:-0}" == "1" ]]; then
  args+=(--strict-action-match)
fi
if [[ "${CLOUD_PROBE_OVERWRITE:-0}" == "1" ]]; then
  args+=(--overwrite)
fi

COMMAND_TEXT="$PYTHON_BIN ${args[*]}"
printf '%s\n' "$COMMAND_TEXT" > "$RUN_DIR/command.txt"
printf 'planner_action_probe（规划器动作探针）status=running\n'
"$PYTHON_BIN" "${args[@]}" 2>&1 | tee "$RUN_DIR/planner_action_probe.log"
printf 'planner_action_probe（规划器动作探针）status=completed\n'
printf 'run_dir（运行目录）=%s\n' "$RUN_DIR"
printf 'output（报告）=%s\n' "$PROBE_OUTPUT"
