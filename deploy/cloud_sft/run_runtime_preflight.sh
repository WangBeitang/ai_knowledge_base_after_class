#!/usr/bin/env bash
set -euo pipefail

# 阶段 9 cloud runtime preflight（云端运行时前置检查）。
# 本脚本只读检查环境、磁盘、checkpoint、模型缓存、vLLM/CUDA 和端口，不安装或下载任何内容。

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

CLOUD_PREFLIGHT_MODE="${CLOUD_PREFLIGHT_MODE:-gpu}"
SFT_VENV_PATH="${SFT_VENV_PATH:-$APP_ROOT/.venv-sft}"
PYTHON_BIN="${PYTHON_BIN:-$SFT_VENV_PATH/bin/python}"
SFT_REQUIREMENTS_LOCK="${SFT_REQUIREMENTS_LOCK:-deploy/cloud_sft/requirements-training.lock}"
VLLM_VENV_PATH="${VLLM_VENV_PATH:-/root/.venv-vllm}"
VLLM_VERSION="${VLLM_VERSION:-0.25.1}"
VLLM_TORCH_BACKEND="${VLLM_TORCH_BACKEND:-cu130}"
VLLM_ENV_FREEZE="${VLLM_ENV_FREEZE:-evaluation/stage9/artifacts/cloud_runs/vllm_environment_freeze.txt}"
CLOUD_RUN_ROOT="${CLOUD_RUN_ROOT:-evaluation/stage9/artifacts/cloud_runs}"
timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
RUN_DIR="$CLOUD_RUN_ROOT/runtime_preflight_${CLOUD_PREFLIGHT_MODE}_$timestamp"
CLOUD_PREFLIGHT_OUTPUT="${CLOUD_PREFLIGHT_OUTPUT:-$RUN_DIR/preflight.json}"
mkdir -p "$RUN_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "训练 Python 解释器不存在：$PYTHON_BIN" >&2
  exit 2
fi

if [[ -z "${SFT_CHECKPOINT_DIR:-}" || -z "${PLANNER_ADAPTER_PATH:-}" ]]; then
  echo "必须设置 SFT_CHECKPOINT_DIR 和 PLANNER_ADAPTER_PATH。" >&2
  exit 2
fi

args=(
  scripts/cloud_sft/preflight_cloud_runtime.py
  --mode "$CLOUD_PREFLIGHT_MODE"
  --checkpoint-dir "$SFT_CHECKPOINT_DIR"
  --adapter-path "$PLANNER_ADAPTER_PATH"
  --model-path "${PLANNER_MODEL_PATH:-Qwen/Qwen3.5-4B}"
  --sft-python "$PYTHON_BIN"
  --expected-sft-python "$SFT_VENV_PATH/bin/python"
  --sft-requirements-lock "$SFT_REQUIREMENTS_LOCK"
  --vllm-python "$VLLM_VENV_PATH/bin/python"
  --vllm-freeze "$VLLM_ENV_FREEZE"
  --expected-vllm-version "$VLLM_VERSION"
  --expected-torch-backend "$VLLM_TORCH_BACKEND"
  --host "${PLANNER_HOST:-127.0.0.1}"
  --port "${PLANNER_PORT:-8019}"
  --system-path "${CLOUD_SYSTEM_DISK_PATH:-/}"
  --data-path "${CLOUD_DATA_DISK_PATH:-/root/autodl-tmp}"
  --min-system-free-gb "${CLOUD_MIN_SYSTEM_FREE_GB:-2}"
  --min-data-free-gb "${CLOUD_MIN_DATA_FREE_GB:-5}"
  --output "$CLOUD_PREFLIGHT_OUTPUT"
)

COMMAND_TEXT="$PYTHON_BIN ${args[*]}"
printf '%s\n' "$COMMAND_TEXT" > "$RUN_DIR/command.txt"
printf 'PYTHON_BIN（配置解释器）=%s\n' "$PYTHON_BIN"
printf 'SFT_VENV_PATH（正式训练环境）=%s\n' "$SFT_VENV_PATH"
"$PYTHON_BIN" - <<'PY'
import importlib.metadata
import platform
import sys

print(f"sys.executable（实际解释器）={sys.executable}")
print(f"python（解释器版本）={platform.python_version()}")
for package in ("torch", "transformers", "peft", "loguru", "pydantic"):
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        version = "unavailable"
    print(f"{package}（SFT 关键依赖）={version}")
PY
printf 'preflight（前置检查）mode=%s status=running\n' "$CLOUD_PREFLIGHT_MODE"
"$PYTHON_BIN" "${args[@]}" 2>&1 | tee "$RUN_DIR/preflight.log"
printf 'preflight（前置检查）mode=%s status=completed\n' "$CLOUD_PREFLIGHT_MODE"
printf 'run_dir（运行目录）=%s\n' "$RUN_DIR"
printf 'output（报告）=%s\n' "$CLOUD_PREFLIGHT_OUTPUT"
