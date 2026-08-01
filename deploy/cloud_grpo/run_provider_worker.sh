#!/usr/bin/env bash
set -euo pipefail

# 在主业务 Python 中启动真实 Provider（动作执行器）；前台运行，便于 screen/tmux 审计。

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
BUSINESS_PYTHON_BIN="${BUSINESS_PYTHON_BIN:-$APP_ROOT/.venv/bin/python}"
GRPO_SNAPSHOT="${GRPO_SNAPSHOT:-evaluation/stage9/artifacts/sft_v2/frozen_reviewed_75_v1/sft_v2_environment_snapshot.json}"
GRPO_PROVIDER_HOST="${GRPO_PROVIDER_HOST:-127.0.0.1}"
GRPO_PROVIDER_PORT="${GRPO_PROVIDER_PORT:-8021}"

cd "$APP_ROOT"
if [[ ! -x "$BUSINESS_PYTHON_BIN" ]]; then
  echo "BUSINESS_PYTHON_BIN（业务 Python）不可执行：$BUSINESS_PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$GRPO_SNAPSHOT" ]]; then
  echo "GRPO_SNAPSHOT（冻结环境快照）不存在：$GRPO_SNAPSHOT" >&2
  exit 2
fi

exec "$BUSINESS_PYTHON_BIN" -m app.rag.evaluation.provider_worker \
  --snapshot "$GRPO_SNAPSHOT" \
  --host "$GRPO_PROVIDER_HOST" \
  --port "$GRPO_PROVIDER_PORT"
