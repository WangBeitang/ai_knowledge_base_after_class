#!/usr/bin/env bash
set -euo pipefail

# bootstrap（初始化脚本）只准备 GPU（显卡算力）服务器环境，不启动正式 SFT（监督微调）训练。

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

echo "APP_ROOT（项目根目录）=$APP_ROOT"
echo "SFT_VENV_PATH（监督微调专用虚拟环境）=$SFT_VENV_PATH"
echo "PYTHON_BIN（训练 Python 解释器）=$PYTHON_BIN"
echo "BOOTSTRAP_INSTALL_DEPS（是否安装依赖）=${BOOTSTRAP_INSTALL_DEPS:-1}"
echo "REQUIRE_CUDA（是否强制要求 CUDA）=${REQUIRE_CUDA:-1}"

if command -v uv >/dev/null 2>&1; then
  uv --version
else
  echo "缺少 uv（Python 包管理器），请先安装 uv。" >&2
  exit 2
fi

if [[ "${BOOTSTRAP_INSTALL_DEPS:-1}" == "1" ]]; then
  requirements_file="$SCRIPT_DIR/requirements-training.lock"
  if [[ ! -f "$requirements_file" ]]; then
    echo "缺少 requirements-training.lock（监督微调依赖锁定文件）：$requirements_file" >&2
    exit 2
  fi
  if [[ ! -x "$PYTHON_BIN" ]]; then
    mkdir -p "$(dirname "$SFT_VENV_PATH")"
    uv venv --python "${SFT_PYTHON_VERSION:-3.12}" "$SFT_VENV_PATH"
  fi
  uv pip install --python "$PYTHON_BIN" --requirement "$requirements_file"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "训练 Python 解释器不存在：$PYTHON_BIN；请先运行 bootstrap（初始化脚本）。" >&2
  exit 2
fi

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi; then
  :
else
  echo "nvidia-smi（英伟达 GPU 管理工具）当前不可用。" >&2
  if [[ "${REQUIRE_CUDA:-1}" == "1" ]]; then
    exit 2
  fi
  echo "REQUIRE_CUDA=0：当前只做无卡环境准备，跳过 CUDA（英伟达 GPU 计算平台）硬门禁。"
fi

run_python - <<'PY'
import importlib.metadata
import platform

print(f"python（解释器）={platform.python_version()}")
for package in ("torch", "transformers", "datasets", "peft", "bitsandbytes", "loguru", "pydantic"):
    try:
        print(f"{package}（训练依赖）={importlib.metadata.version(package)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{package}（训练依赖）=unavailable")

try:
    import torch

    print(f"torch.cuda.is_available（CUDA 是否可用）={torch.cuda.is_available()}")
    print(f"torch.cuda.device_count（CUDA 设备数量）={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            print(f"cuda:{index}（显卡名称）={torch.cuda.get_device_name(index)}")
except Exception as exc:
    print(f"torch_cuda_check_error（CUDA 检查错误）={exc}")
PY

echo "bootstrap（初始化）完成。未启动正式训练。"
