# AutoDL SFT（监督微调）上云指引

本文只覆盖阶段 9 的 SFT（监督微调）上云流程，不包含 GRPO（组相对策略优化强化训练）。
当前目标是用 AutoDL（云 GPU 平台）跑通：

```text
9.3.8 cloud smoke（云端冒烟）
-> 9.3.9 正式 SFT（监督微调）训练与 checkpoint（检查点）冻结
-> 为 9.4 baseline compare（基线对比）准备 SFT policy（监督微调策略）
```

## 关键结论

- 9.3.7 已经准备好训练包和启动脚本。
- 9.3.8 和 9.3.9 使用同一套代码；区别是运行脚本、样本规模和验收目标。
- 9.3.8 跑 `run_sft_smoke.sh（监督微调冒烟脚本）`，只验证链路。
- 9.3.9 跑 `run_sft_train.sh（监督微调训练脚本）`，产出正式 checkpoint（检查点）。
- 正常情况下，9.3.9 不再改核心代码，只改 `env.local（本机环境变量文件）`和训练 config（配置）。
- SFT（监督微调）训练依赖和 vLLM（大模型推理服务框架）依赖都优先在 no-card mode（无卡模式）安装；
  vLLM 必须使用独立虚拟环境和可复用缓存，避免在 GPU（显卡）计费时等待大依赖下载。

## AutoDL 平台边界

以下信息来自 AutoDL（云 GPU 平台）官方文档，平台界面和价格以后可能变化，最终以控制台显示为准。

- 创建实例时需要选择计费方式、地区、GPU（显卡）型号、GPU 数量、空闲主机和镜像；实例进入运行中后开始计费。参考 [AutoDL 快速开始](https://www.autodl.com/docs/quick_start/)。
- 镜像优先选平台内置 PyTorch/CUDA（深度学习框架/英伟达 GPU 计算平台）镜像；如果内置镜像不满足，再用 Miniconda/CUDA 镜像自行安装。参考 [AutoDL 环境配置](https://www.autodl.com/docs/base_config/)。
- 实例关机后数据通常保留，但本地数据盘无冗余保证，连续关机 15 天会触发释放风险；重要 checkpoint（检查点）必须备份。参考 [AutoDL 实例数据](https://www.autodl.com/docs/instance_data/)。
- 长时间训练要用 JupyterLab（浏览器开发环境）终端、`screen（会话守护工具）`或 `tmux（会话守护工具）`，并保存日志，避免 SSH（安全远程登录）断开导致训练中断。参考 [AutoDL 守护进程](https://www.autodl.com/docs/daemon/)。
- AutoDL 实例通常没有独立公网 IP（公网地址），任意端口访问建议用 SSH tunnel（SSH 隧道）；平台默认只对 `6006/6008` 提供自定义服务映射。参考 [AutoDL 开放端口](https://www.autodl.com/docs/port/) 和 [AutoDL SSH 隧道](https://www.autodl.com/docs/ssh_proxy/)。
- 不需要 GPU（显卡）时可以用 no-card mode（无卡模式）同步代码、安装依赖、整理文件和下载产物；
  无卡模式会释放 GPU，重新有卡开机时可能遇到空闲 GPU 不足。依赖预装最好在正式占用目标 GPU 前完成，
  已经拿到稀缺 GPU 后是否切换要权衡重新拿不到同型号 GPU 的风险。参考
  [AutoDL 省钱绝招](https://www.autodl.com/docs/save_money/)。

## 推荐实例配置

第一版正式 SFT（监督微调）建议优先：

```text
GPU：A800 单卡
训练方法：LoRA（低秩适配）
训练配置：evaluation/stage9/configs/planner_sft_qwen3_5_4b_lora.json
```

为什么这么选：

- 当前模型是 `Qwen/Qwen3.5-4B（通义千问 3.5 4B）`，不是大参数模型，A800 单卡足够稳。
- LoRA（低秩适配）不走 4bit（4 位量化）加载，依赖边界比 QLoRA（4 位量化低秩适配）更简单。
- 第一版目标是冻结可对比的 SFT policy（监督微调策略），先降低环境兼容风险。

如果优先省钱，可以用 5090：

```text
GPU：RTX 5090 单卡
训练方法：优先 QLoRA（4 位量化低秩适配）
训练配置：evaluation/stage9/configs/planner_sft_qwen3_5_4b_qlora.json
```

为什么 5090 更适合先跑 QLoRA：

- 5090 显存通常小于 A800，QLoRA（4 位量化低秩适配）能降低基础模型显存占用。
- 5090 属于更新架构，CUDA/PyTorch/bitsandbytes（显卡计算平台/深度学习框架/量化库）兼容性更容易踩坑。
- 如果 QLoRA 在 5090 上报 `bitsandbytes（量化库）`或 CUDA（英伟达 GPU 计算平台）错误，直接换 A800 + LoRA 更省时间。

## 上云前本地检查

### 1. 确认本地代码状态

```bash
git status --short
uv run pytest tests/test_cloud_sft_package.py
```

这步做什么：

- `git status（代码状态）`确认你要同步到云端的代码是否包含未提交改动。
- `test_cloud_sft_package（云端监督微调包测试）`确认 9.3.7 的脚本仍然可用。

为什么要做：

- 云端训练产生的 checkpoint（检查点）会记录 `code_version（代码版本）`。
- 如果本地代码不清楚，后面 9.4 baseline compare（基线对比）很难解释这个模型到底由哪版代码训练出来。

### 2. 确认训练入口已经存在

```bash
ls deploy/cloud_sft
ls scripts/cloud_sft
```

应该能看到：

```text
deploy/cloud_sft/
  README.md
  AUTODL_SFT_GUIDE.md
  env.example
  bootstrap_gpu_server.sh
  run_sft_smoke.sh
  run_sft_train.sh
  run_planner_server.sh
  run_dev_eval.sh

scripts/cloud_sft/
  collect_cloud_run_report.py
```

这步做什么：

- 确认上云所需脚本已经随项目代码一起存在。

为什么要做：

- 9.3.9 不应该在 AutoDL（云 GPU 平台）上临时写训练脚本。
- 云端只做配置、运行和产物冻结，不做临时开发。

## 创建 AutoDL 实例

### 1. 选择 GPU 与镜像

推荐第一版：

```text
GPU：A800
镜像：PyTorch 2.7/2.8 + Python 3.12 + CUDA 12.8 或 AutoDL 当前可用的较新 PyTorch/CUDA 镜像
磁盘：数据盘尽量放大，至少要能容纳模型缓存、checkpoint 和 cloud_runs
```

这步做什么：

- 选择云端训练的硬件和基础系统环境。

为什么要做：

- SFT（监督微调）真正消耗的是 GPU 显存、模型下载缓存和 checkpoint（检查点）落盘空间。
- 项目要求 Python `>=3.11`，所以不要选 Python 3.8/3.10 的老镜像。
- AutoDL 官方文档列出的新 PyTorch 镜像包含 Python 3.12 和 CUDA 12.8，更接近当前项目依赖。

如果选 5090：

```text
优先选择更新的 PyTorch/CUDA 镜像。
优先使用 QLoRA 配置。
必须先跑 9.3.8 cloud smoke（云端冒烟），不要直接正式训练。
```

### 2. 进入实例

可以用两种方式：

```text
JupyterLab（浏览器开发环境）：适合直接在网页终端跑命令。
SSH（安全远程登录）：适合从本地终端或 VSCode/PyCharm 远程连接。
```

SSH（安全远程登录）示例：

```bash
ssh -p <AutoDL显示的端口> root@<AutoDL显示的主机>
```

这步做什么：

- 进入云端 Linux 环境。

为什么要做：

- 后续所有训练、模型服务和评测脚本都在 AutoDL 实例内执行。
- 如果用 SSH 直接跑长任务，必须用 `screen/tmux（会话守护工具）`，避免网络断开导致进程退出。

## 准备云端目录

### 1. 使用数据盘作为项目目录

```bash
cd /root/autodl-tmp
```

这步做什么：

- 把项目、模型缓存、训练产物放到 AutoDL 数据盘。

为什么要做：

- 系统盘更容易被 Python 依赖、模型缓存和 checkpoint（检查点）挤满。
- 数据盘更适合保存训练中间产物，但重要结果仍要额外备份。

### 2. 同步项目代码

如果仓库能从 Git 访问：

```bash
cd /root/autodl-tmp
git clone https://gitee.com/wangyanning1995/ai_knowledge_base_after_class.git ai_knowledge_base_after_class
cd ai_knowledge_base_after_class
```

如果不方便用 Git，可以在本地用 `rsync（增量同步工具）`同步：

```bash
rsync -av \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  /Users/beitang/PycharmProjects/ai_knowledge_base_after_class/ \
  root@<AutoDL主机>:/root/autodl-tmp/ai_knowledge_base_after_class/
```

这步做什么：

- 把本地项目完整放到 AutoDL 实例。

为什么要做：

- SFT（监督微调）需要训练脚本、训练数据、Reward profile（奖励函数配置）、model profile（模型配置档案）和报告脚本同时存在。
- 只传单个训练脚本不够，后续 checkpoint manifest（检查点清单）无法记录完整来源。

## 安装依赖

推荐顺序：

```text
no-card mode（无卡模式）
-> 同步项目和配置 env.local
-> 安装 .venv-sft（监督微调环境）
-> 安装 .venv-vllm（模型服务环境）
-> 记录依赖版本
-> 正常有卡开机
-> CUDA（英伟达 GPU 计算平台）最终门禁
-> cloud smoke（云端冒烟）
```

这条顺序把耗时较长、但不需要 GPU 的下载放在低价无卡实例完成。不要等 SFT（监督微调）训练结束后
才第一次安装 vLLM；vLLM 会解析大量 PyTorch/CUDA（深度学习框架/显卡计算平台）wheel（预编译包），
网络较慢时可能耗时很久。

### 1. 安装 uv（Python 包管理器）

如果镜像里已经有 `uv`，跳过这步。

```bash
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

这步做什么：

- 准备项目使用的 Python 包管理器。

为什么要做：

- 当前云端训练入口用 `uv（Python 包管理器）`创建独立 `.venv-sft（监督微调虚拟环境）`。
- 主业务的 `magic-pdf（PDF 解析框架）`要求 `Transformers < 5`，Qwen3.5 训练需要
  `Transformers 5.9.0`；隔离环境可以复用同一套项目代码和数据，同时避免依赖冲突。

### 2. 配置缓存目录

```bash
mkdir -p /root/autodl-tmp/cache/huggingface
mkdir -p /root/autodl-tmp/cache/modelscope
mkdir -p /root/autodl-tmp/cache/uv

export HF_HOME=/root/autodl-tmp/cache/huggingface
export MODELSCOPE_CACHE=/root/autodl-tmp/cache/modelscope
export UV_CACHE_DIR=/root/autodl-tmp/cache/uv
```

也建议把这几行写入 `deploy/cloud_sft/env.local（本机环境变量文件）`，这样后续
`run_sft_smoke.sh（监督微调冒烟脚本）`和 `run_sft_train.sh（监督微调训练脚本）`
都会继承同一套缓存路径。

这步做什么：

- 把模型和依赖缓存放到数据盘。

为什么要做：

- `Qwen/Qwen3.5-4B（通义千问 3.5 4B）`模型权重会占用明显磁盘空间。
- 放系统盘容易导致系统盘空间不足。

### 3. 创建 env.local（本机环境变量文件）

```bash
cd /root/autodl-tmp/ai_knowledge_base_after_class
cp deploy/cloud_sft/env.example deploy/cloud_sft/env.local
```

这步做什么：

- 创建当前 AutoDL 实例专用的环境文件。

为什么要做：

- no-card mode（无卡模式）和正常有卡模式需要切换 `REQUIRE_CUDA（是否强制要求 CUDA）`，
  同时要固定 SFT、vLLM、模型缓存和产物路径。
- 复制后必须按下一节修改，不能直接使用模板中的 `/workspace`示例路径。

## 配置 env.local

编辑：

```bash
vim deploy/cloud_sft/env.local
```

A800 + LoRA（低秩适配）推荐配置：

```dotenv
APP_ROOT=/root/autodl-tmp/ai_knowledge_base_after_class
SFT_VENV_PATH=/root/autodl-tmp/ai_knowledge_base_after_class/.venv-sft
PYTHON_BIN=/root/autodl-tmp/ai_knowledge_base_after_class/.venv-sft/bin/python
SFT_PYTHON_VERSION=3.12
BOOTSTRAP_INSTALL_DEPS=1
REQUIRE_CUDA=1
OMP_NUM_THREADS=1
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1

# vLLM（大模型推理服务框架）必须与训练环境隔离。
# 下面示例适合系统盘剩余空间更大时；如果数据盘更大，venv 和 cache 要一起改到数据盘。
VLLM_VERSION=0.25.1
VLLM_VENV_PATH=/root/.venv-vllm
VLLM_UV_CACHE_DIR=/root/.cache/uv-vllm
VLLM_TORCH_BACKEND=cu130
VLLM_ENV_FREEZE=evaluation/stage9/artifacts/cloud_runs/vllm_environment_freeze.txt
REQUIRE_GPU_PREFLIGHT=1

CLOUD_RUN_ROOT=evaluation/stage9/artifacts/cloud_runs
SFT_OUTPUT_ROOT=evaluation/stage9/artifacts/sft/checkpoints
SFT_TRAIN_CONFIG=evaluation/stage9/configs/planner_sft_qwen3_5_4b_lora.json
SFT_SMOKE_BASE_CONFIG=evaluation/stage9/configs/planner_sft_qwen3_5_4b_lora.json
STAGE9_SFT_SMOKE_MAX_SAMPLES=4
STAGE9_SFT_SMOKE_MAX_STEPS=1

SFT_CHECKPOINT_DIR=
DEV_EVAL_PROVIDER=snapshot_expected_chunks
DEV_EVAL_MAX_CASES=7

PLANNER_MODEL_PATH=Qwen/Qwen3.5-4B
PLANNER_BASE_MODEL_ID=qwen3_5_4b_base
PLANNER_MODEL_ID=qwen3_5_4b_sft_stage9
PLANNER_ADAPTER_PATH=
PLANNER_HOST=127.0.0.1
PLANNER_PORT=8019
PLANNER_MODEL_ENDPOINT=http://127.0.0.1:8019/v1/chat/completions
PLANNER_API_KEY=

CLOUD_PLANNER_HTTP_PROBE_OUTPUT=evaluation/stage9/artifacts/sft/cloud_smoke_planner_http.json
CLOUD_PROVIDER_PROBE_OUTPUT=evaluation/stage9/artifacts/provider_records/cloud_smoke_provider_observations.jsonl
CLOUD_PROBE_STRICT_ACTION_MATCH=0
CLOUD_PROBE_OVERWRITE=0

HF_HOME=/root/autodl-tmp/cache/huggingface
MODELSCOPE_CACHE=/root/autodl-tmp/cache/modelscope
UV_CACHE_DIR=/root/autodl-tmp/cache/uv
```

如果当前使用 AutoDL 的 no-card mode（无卡模式）准备依赖，把 `REQUIRE_CUDA`临时改成：

```dotenv
REQUIRE_CUDA=0
```

此时 `bootstrap（初始化脚本）`允许 `nvidia-smi（显卡状态工具）`不可用，但仍会完成独立
训练环境安装和版本检查。切换到正常有卡实例后，必须恢复为 `REQUIRE_CUDA=1`，
让 CUDA（英伟达 GPU 计算平台）不可用时立即停止。

这里的 `.venv-sft（监督微调虚拟环境）`只安装
`deploy/cloud_sft/requirements-training.lock（训练依赖锁定文件）`中的依赖；该锁文件由
`requirements-training.txt（直接训练依赖清单）`解析生成。不要执行
`uv add transformers==5.9.0`去修改主业务环境，也不要使用 `--frozen`跳过依赖检查；
前者会与 `magic-pdf`发生版本冲突，后者会留下无法复现的损坏环境。
训练锁文件还必须包含项目正式入口直接使用的 `loguru==0.7.3`，不能只列
Torch、Transformers 和 PEFT 等模型训练包。

5090 + QLoRA（4 位量化低秩适配）只改这两行：

```dotenv
SFT_TRAIN_CONFIG=evaluation/stage9/configs/planner_sft_qwen3_5_4b_qlora.json
SFT_SMOKE_BASE_CONFIG=evaluation/stage9/configs/planner_sft_qwen3_5_4b_qlora.json
```

这步做什么：

- 把云端路径、训练配置、模型服务配置写到本机私有环境文件。

为什么要做：

- `env.example（环境变量模板）`不能写真实密钥或机器路径。
- `env.local（本机环境变量文件）`不提交到仓库，只服务当前 AutoDL 实例。

## no-card mode（无卡模式）预装全部依赖

这一节推荐在正式占用 GPU（显卡）前完成。如果已经拿到稀缺 GPU，切换无卡模式前要知道：
实例文件会保留，但当前 GPU 会被释放，重新正常开机时可能没有同型号空闲 GPU。

### 1. 检查磁盘并选择 vLLM 安装位置

```bash
cd /root/autodl-tmp/ai_knowledge_base_after_class

set -a
source deploy/cloud_sft/env.local
set +a

df -h / /root/autodl-tmp
```

选择规则：

- `VLLM_VENV_PATH（vLLM 虚拟环境路径）`和
  `VLLM_UV_CACHE_DIR（vLLM 的 uv 下载缓存路径）`放在同一个文件系统，避免跨盘复制大 wheel。
- 优先选择剩余空间更大的盘；第一次安装会同时存在环境文件和下载缓存，建议所选盘至少保留
  25 GB 可用空间。
- 模型缓存、训练 checkpoint（检查点）和 cloud run（云端运行记录）仍放数据盘。
- 如果两个盘都不满足空间门禁，先扩容或清理可再生成缓存，不要让安装把系统盘写满。

### 2. 安装 SFT（监督微调）训练依赖

确认 `env.local（本机环境变量文件）`中：

```dotenv
BOOTSTRAP_INSTALL_DEPS=1
REQUIRE_CUDA=0
```

然后执行：

```bash
cd /root/autodl-tmp/ai_knowledge_base_after_class
CLOUD_SFT_ENV_FILE=deploy/cloud_sft/env.local bash deploy/cloud_sft/bootstrap_gpu_server.sh
```

通过标准：

- `.venv-sft（监督微调虚拟环境）`创建成功。
- `torch/transformers/peft/bitsandbytes（训练框架/参数高效微调库/量化库）`版本可以输出。
- no-card mode 下 `torch.cuda.is_available（CUDA 是否可用）=False`是预期结果，不是失败。

### 3. 安装独立 vLLM（大模型推理服务框架）环境

vLLM 不能安装进 `.venv-sft（监督微调虚拟环境）`。两套环境的 PyTorch/Transformers
（深度学习框架/模型框架）依赖由各自流程管理，避免为了部署推理服务破坏已经通过 smoke（冒烟）的训练环境。

本文当前固定：

```text
vLLM 版本：0.25.1
目标 torch backend（PyTorch 后端）：cu130
```

`VLLM_TORCH_BACKEND（vLLM 的 PyTorch 后端）`必须根据目标 GPU 实例的驱动兼容范围预先确定。
当前 RTX 4090 + driver 580.76.05 的有卡实测中，`--torch-backend=auto`解析为 CUDA 13
相关 wheel，因此对应无卡预装固定为 `cu130`。无卡模式没有 GPU 供 `auto`再次探测，更新 vLLM
或 CUDA 路线前，先查 [vLLM GPU 安装文档](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)；
未经验证不要改成 nightly（每日构建版）或不固定版本的 latest（最新版）。

打开守护会话：

```bash
screen -S stage9_vllm_install
```

进入会话后执行：

```bash
cd /root/autodl-tmp/ai_knowledge_base_after_class

set -a
source deploy/cloud_sft/env.local
set +a

mkdir -p "$VLLM_UV_CACHE_DIR"

if [[ ! -x "$VLLM_VENV_PATH/bin/python" ]]; then
  uv venv --python 3.12 "$VLLM_VENV_PATH"
fi

UV_CACHE_DIR="$VLLM_UV_CACHE_DIR" \
UV_CONCURRENT_DOWNLOADS=4 \
UV_HTTP_TIMEOUT=600 \
UV_HTTP_RETRIES=10 \
uv pip install \
  --python "$VLLM_VENV_PATH/bin/python" \
  "vllm==$VLLM_VERSION" \
  --torch-backend="$VLLM_TORCH_BACKEND"
```

这里不要加 `--no-cache（不保留缓存）`：

- vLLM 会下载大量 NVIDIA CUDA/PyTorch wheel（英伟达显卡运行库/深度学习框架预编译包）。
- 使用持久化 `VLLM_UV_CACHE_DIR`后，安装失败或网络中断时，已完成的包可以复用。
- `--no-cache`使用临时缓存；长时间下载中断后更容易从头重来，不适合 AutoDL 慢网络。
- `UV_CONCURRENT_DOWNLOADS=4（最大并发下载数）`降低大量 CUDA 大包同时抢连接的概率；
  `UV_HTTP_TIMEOUT=600（读取超时秒数）`和 `UV_HTTP_RETRIES=10（请求重试次数）`
  避免慢连接频繁从头重试。参数说明参考
  [uv 环境变量文档](https://docs.astral.sh/uv/configuration/environment/)。
- uv（Python 包管理器）建议 cache（缓存）和虚拟环境位于同一文件系统。参考
  [uv cache 文档](https://docs.astral.sh/uv/concepts/cache/)。

如果出现以下任一情况，停止安装并检查，不要反复重跑：

- `No space left on device（磁盘空间不足）`。
- `No solution found（依赖无法解析）`。
- 找不到目标 CUDA wheel（预编译包）。
- 开始从源码编译 vLLM，而不是下载预编译 wheel。

### 4. 无卡模式下记录版本

```bash
"$VLLM_VENV_PATH/bin/python" -c \
  'from importlib.metadata import version; import torch; print("vllm=", version("vllm")); print("torch=", torch.__version__); print("torch_cuda=", torch.version.cuda); print("cuda_available=", torch.cuda.is_available())'

mkdir -p evaluation/stage9/artifacts/cloud_runs
UV_CACHE_DIR="$VLLM_UV_CACHE_DIR" \
uv pip freeze --python "$VLLM_VENV_PATH/bin/python" \
  > evaluation/stage9/artifacts/cloud_runs/vllm_environment_freeze.txt
```

无卡模式不要使用 `vllm --version`作为安装门禁。vLLM 0.25.1 的 CLI（命令行入口）
在解析参数时会尝试推断 device type（运行设备类型）；没有 GPU 时可能报
`Failed to infer device type`。这不代表安装失败，应使用上面的 package metadata
（包元数据）读取版本，并把 CLI 检查留到恢复有卡模式后执行。

无卡模式通过标准：

- `vllm=0.25.1`。
- `torch_cuda（PyTorch 编译对应的 CUDA 版本）`与 `VLLM_TORCH_BACKEND`一致。
- `cuda_available=False`是预期结果；真正的 CUDA 可用性要在正常有卡开机后确认。
- `vllm_environment_freeze.txt（vLLM 环境冻结清单）`存在。

### 5. 无卡模式运行结构化 runtime preflight（运行时前置检查）

已有正式 checkpoint 时，先在无卡模式执行：

```bash
cd /root/autodl-tmp/ai_knowledge_base_after_class

sed -i 's/^REQUIRE_CUDA=.*/REQUIRE_CUDA=0/' deploy/cloud_sft/env.local

CLOUD_PREFLIGHT_MODE=no-card \
CLOUD_SFT_ENV_FILE=deploy/cloud_sft/env.local \
bash deploy/cloud_sft/run_runtime_preflight.sh
```

该命令只读检查，不会安装依赖、下载模型或启动服务。它会生成
`runtime_preflight_no-card_<UTC时间>/preflight.json（无卡前置检查报告）`和同目录下的
`sft_environment_freeze.txt（SFT 环境冻结文件）`。冻结文件只在所有 preflight check
通过后生成，并明确使用 `$PYTHON_BIN`，避免再次记录成主项目 `.venv`。报告按以下层级给出失败：

- `environment（环境）`：`OMP_NUM_THREADS`、离线开关、expanded dev 固定配置和
  `REQUIRE_CUDA`。
- `storage（存储）`：系统盘、数据盘剩余空间。
- `artifact（产物）`：checkpoint、adapter、manifest 和 vLLM freeze。
- `dependency（依赖）`：实际解释器是否为 `.venv-sft`、正式训练与 expanded dev 入口能否
  完整导入，以及 runtime、checkpoint manifest、训练锁文件三方版本是否一致。
- `dependency/model_cache（依赖/模型缓存）`：vLLM/PyTorch 版本、CUDA backend 和基础模型离线缓存。
- `network（网络端口）`：启动端口是否空闲。

停止条件：任何 check（检查）为 `failed`时都不要开 GPU。尤其是
`model_cache`失败，表示基础模型无法通过 `local_files_only（只读本地缓存）`解析；必须留在无卡模式补齐。
`sft_python_identity`、`sft_entrypoint_imports`、`sft_lock_versions` 或
`sft_checkpoint_versions`失败时也必须停在无卡模式修复，禁止临时换解释器或手工补包后直接开卡。

### 6. 切回正常有卡模式后的最终门禁

先把 `env.local（本机环境变量文件）`恢复为：

```dotenv
REQUIRE_CUDA=1
```

恢复有卡后，不再安装依赖。当前 SFT v1 的正式验收直接在 `screen`中执行一条命令：

```bash
cd /root/autodl-tmp/ai_knowledge_base_after_class

screen -S stage9_gpu_gate
CLOUD_SFT_ENV_FILE=deploy/cloud_sft/env.local \
bash deploy/cloud_sft/run_gpu_acceptance_gate.sh
```

该入口严格串联：

```text
GPU preflight（GPU 前置检查）
-> 启动 vLLM
-> 等待 /health
-> 7 个 HTTP probe（六类 Action + Web 禁用边界）
-> 停止本次 vLLM
-> 记录停服后的 GPU 进程
```

默认最多等待服务 600 秒。服务提前退出或等待超时，脚本会打印
`planner_server.log（规划器服务日志）`末尾并停止；不要另开一条安装命令救场。

有卡模式通过标准：

- `preflight.json`的 `ok=true`，CUDA、vLLM、checkpoint identity（检查点身份）和模型缓存均通过。
- `cloud_smoke_planner_http.json`存在，六类目标 Action 输入齐全，HTTP/解析/model_id 和 Web 禁用边界通过。
- `planner_server.log`出现 application startup complete，结束后本次服务进程已停止。

`CLOUD_PROBE_STRICT_ACTION_MATCH=0`是刻意设置：9.3.10 把 Action 不匹配记录下来，但只把
HTTP 协议、结构化解析、model_id 和 Web 权限当作工程门禁。模型是否选择了期望 Action 属于
9.3.11 quality gate（质量门禁），不能把工程连通性和模型泛化混为一个结论。

如果 vLLM 安装和 GPU 门禁都已通过、且磁盘空间紧张，可以清理仅属于 vLLM 的下载缓存：

```bash
UV_CACHE_DIR="$VLLM_UV_CACHE_DIR" uv cache clean
```

这只删除可重新下载的 vLLM uv cache（下载缓存），不删除 `.venv-vllm（模型服务虚拟环境）`、
模型缓存或训练 checkpoint（检查点）。

## 9.3.8：cloud smoke（云端冒烟）

### 1. 开守护会话

```bash
screen -S stage9_sft_smoke
```

如果没有 `screen（会话守护工具）`：

```bash
apt-get update && apt-get install -y screen
screen -S stage9_sft_smoke
```

这步做什么：

- 创建一个不会因本地网络断开而立即退出的训练会话。

为什么要做：

- AutoDL 官方也建议长时间任务使用守护进程或 JupyterLab 终端并保存日志。
- 训练中断会浪费 GPU（显卡）时间，也会留下半截 checkpoint（检查点）。

### 2. 跑 smoke（冒烟）训练

```bash
cd /root/autodl-tmp/ai_knowledge_base_after_class
CLOUD_SFT_ENV_FILE=deploy/cloud_sft/env.local bash deploy/cloud_sft/run_sft_smoke.sh
```

这步做什么：

- 用正式训练配置派生一个临时小配置。
- 默认只训练 4 条样本、1 个 step（训练步）。
- 生成一个 smoke checkpoint（冒烟检查点）和 cloud run report（云端运行报告）。

为什么要做：

- 用最小成本确认模型下载、tokenizer（分词器）、Trainer（训练器）、LoRA/QLoRA（低秩适配/4 位量化低秩适配）、checkpoint 写入和报告收集都可用。
- 不在 smoke（冒烟）阶段消耗大量余额。

### 3. 检查 smoke 结果

脚本最后会输出：

```text
run_dir（运行目录）=evaluation/stage9/artifacts/cloud_runs/sft_smoke_<timestamp>
```

查看日志：

```bash
tail -n 80 evaluation/stage9/artifacts/cloud_runs/sft_smoke_<timestamp>/sft_smoke.log
```

查看报告：

```bash
uv run python -m json.tool \
  evaluation/stage9/artifacts/cloud_runs/sft_smoke_<timestamp>/cloud_run_report.json | head -120
```

通过标准：

- 日志里有 `checkpoint=...`。
- `cloud_run_report.json（云端运行报告）`存在。
- 报告里能看到 `training_config（训练配置）`、`model_profile（模型配置档案）`、
  `train_manifest（训练清单）`、`reward_profile（奖励函数配置）`和 `checkpoint_manifest（检查点清单）`。
- 没有 CUDA OOM（显存不足）、模型加载失败、tokenizer（分词器）加载失败或 `bitsandbytes（量化库）`错误。

如果失败：

- CUDA 不可用：先看 `nvidia-smi`，再看 bootstrap（初始化）输出里的 `torch.cuda.is_available（CUDA 是否可用）`。
- 5090 + QLoRA 报 bitsandbytes（量化库）错误：优先换更新 PyTorch/CUDA 镜像；还不行就切 A800 + LoRA。
- 显存不足：先切 QLoRA（4 位量化低秩适配），再考虑 A800。
- 磁盘不足：确认模型缓存和项目都在 `/root/autodl-tmp`。

## 9.3.9：正式 SFT（监督微调）

只有 9.3.8 通过后才执行本节。

### 1. 确认训练配置

```bash
set -a
source deploy/cloud_sft/env.local
set +a
uv run python -m json.tool "$SFT_TRAIN_CONFIG" | head -120
```

重点确认：

```text
training_backend（训练后端）= transformers_causal_lm
base_model_id（基础模型身份）= Qwen/Qwen3.5-4B
model_profile_id（模型配置档案身份）= qwen3_5_4b
snapshot_id（快照身份）= stage85-env-20260721-v2
tuning_method（训练方法）= lora 或 qlora
max_train_samples（最多训练样本）= null
max_steps（最多训练步）= null
```

这步做什么：

- 确认正式训练不是 smoke（冒烟）配置。

为什么要做：

- `max_train_samples=null` 和 `max_steps=null` 表示使用正式训练配置，而不是只跑 4 条样本、1 个 step。
- 如果误用 smoke config（冒烟配置），训练出来的 checkpoint（检查点）不能进入 9.4 baseline compare（基线对比）。

### 2. 跑正式训练

```bash
screen -S stage9_sft_train
cd /root/autodl-tmp/ai_knowledge_base_after_class
CLOUD_SFT_ENV_FILE=deploy/cloud_sft/env.local bash deploy/cloud_sft/run_sft_train.sh
```

这步做什么：

- 运行正式 SFT（监督微调）训练。
- 训练完成后生成 checkpoint（检查点）、adapter（适配器）、tokenizer（分词器）、train_metrics（训练指标）和 cloud run report（云端运行报告）。

为什么要做：

- 这是 9.3.9 的核心动作。
- 后续 PlannerModelServer（规划器模型服务）和 9.4 baseline compare（基线对比）都依赖这个 checkpoint（检查点）。

### 3. 记录正式 checkpoint

训练日志会输出：

```text
checkpoint=evaluation/stage9/artifacts/sft/checkpoints/<run_id>
```

把它写回 `env.local（本机环境变量文件）`：

```dotenv
SFT_CHECKPOINT_DIR=evaluation/stage9/artifacts/sft/checkpoints/<run_id>
```

检查 manifest（清单）：

```bash
uv run python -m json.tool \
  evaluation/stage9/artifacts/sft/checkpoints/<run_id>/checkpoint_manifest.json | head -160
```

这步做什么：

- 固定本次正式训练产物的路径和身份。

为什么要做：

- 9.4 baseline compare（基线对比）需要明确比较哪个 SFT policy（监督微调策略）。
- 不能用“最新目录”或“刚训练那个”这种口头描述。

### 4. 冻结产物

至少保留这些文件：

```text
evaluation/stage9/artifacts/sft/checkpoints/<run_id>/checkpoint_manifest.json
evaluation/stage9/artifacts/sft/checkpoints/<run_id>/train_metrics.json
evaluation/stage9/artifacts/sft/checkpoints/<run_id>/training_config.json
evaluation/stage9/artifacts/sft/checkpoints/<run_id>/model/adapter/
evaluation/stage9/artifacts/sft/checkpoints/<run_id>/tokenizer/
evaluation/stage9/artifacts/cloud_runs/sft_train_<timestamp>/cloud_run_report.json
```

使用冻结脚本打包。`--checkpoint-dir` 必须传 checkpoint 根目录，不能传 `model` 或
`model/adapter` 子目录：

```bash
uv run --frozen --no-sync python scripts/cloud_sft/freeze_sft_artifacts.py \
  --checkpoint-dir evaluation/stage9/artifacts/sft/checkpoints/<run_id> \
  --train-run-dir evaluation/stage9/artifacts/cloud_runs/sft_train_<timestamp> \
  --dev-run-dir evaluation/stage9/artifacts/cloud_runs/dev_eval_<timestamp> \
  --output-dir /root/autodl-tmp/stage9_backups \
  --label sft-v1
```

这步做什么：

- 校验 checkpoint、正式训练报告与 dev eval 的 `run_id（运行身份）`是否一致。
- 把 checkpoint、训练输入、dev eval、环境快照和 vLLM 环境冻结清单打成一个可下载包。
- 在归档内写入 `_freeze/freeze_manifest.json（冻结清单）`和
  `_freeze/SHA256SUMS.txt（逐文件哈希）`，并在归档旁生成整体 `.sha256` 文件。

为什么要做：

- AutoDL 本地数据盘不是长期可靠归档。
- 训练完成后必须把正式 checkpoint（检查点）和报告备份到本地或长期存储。
- 仅有压缩包但没有 run_id 交叉校验和文件哈希，不能证明下载后的文件仍属于本次正式训练。

预期输出：

```text
{"ok": true, "run_id": "...", "archive": "...tar.gz", "archive_sha256": "...", ...}
```

出现以下任意情况立即停止，不要用手工 `tar` 绕过：

- checkpoint 目录名与 manifest 的 `run_id` 不一致。
- train/dev cloud run report 指向其他 checkpoint。
- dev eval 不是由本次 checkpoint 生成。
- adapter、训练配置、dev 日志或 vLLM 环境冻结清单缺失。

## 启动 SFT PlannerModelServer（规划器模型服务）

### 1. 找到 adapter_path（适配器路径）

```bash
uv run python - <<'PY'
import json
from pathlib import Path

checkpoint = Path("evaluation/stage9/artifacts/sft/checkpoints/<run_id>")
manifest = json.loads((checkpoint / "checkpoint_manifest.json").read_text(encoding="utf-8"))
print(manifest["adapter_path"])
PY
```

把输出写到 `env.local（本机环境变量文件）`：

```dotenv
PLANNER_ADAPTER_PATH=evaluation/stage9/artifacts/sft/checkpoints/<run_id>/model/adapter
PLANNER_MODEL_ID=qwen3_5_4b_sft_stage9
PLANNER_BASE_MODEL_ID=qwen3_5_4b_base
PLANNER_MODEL_PATH=Qwen/Qwen3.5-4B
PLANNER_HOST=127.0.0.1
PLANNER_PORT=8019
PLANNER_MODEL_ENDPOINT=http://127.0.0.1:8019/v1/chat/completions
```

这步做什么：

- 告诉 vLLM（大模型推理服务框架）加载哪个 LoRA adapter（低秩适配器）。

为什么要做：

- 训练后的模型不是直接覆盖基础模型，而是基础模型 + adapter（适配器）。
- PlannerClient（规划器客户端）请求时要用 `PLANNER_MODEL_ID（规划器模型身份）`命中微调后的 served model（服务模型）。

### 2. 启动模型服务

```bash
screen -S stage9_planner_server
cd /root/autodl-tmp/ai_knowledge_base_after_class

set -a
source deploy/cloud_sft/env.local
set +a

"$VLLM_VENV_PATH/bin/vllm" --version

PATH="$VLLM_VENV_PATH/bin:$PATH" \
CLOUD_SFT_ENV_FILE=deploy/cloud_sft/env.local \
bash deploy/cloud_sft/run_planner_server.sh
```

这步做什么：

- 在 AutoDL 实例内启动 PlannerModelServer（规划器模型服务）。
- 显式把 `.venv-vllm/bin（vLLM 可执行文件目录）`加入当前启动命令的 `PATH（命令搜索路径）`；
  训练专用 `.venv-sft`中没有 vLLM，直接运行脚本会报 `vllm: command not found`。

为什么要做：

- 业务 Planner（规划器）链路通过 HTTP（超文本传输协议）调用模型服务。
- 不应该在每个评测进程里重复加载大模型。

### 3. 健康检查

另开一个终端：

```bash
cd /root/autodl-tmp/ai_knowledge_base_after_class
uv run python scripts/planner_model_server/healthcheck_planner_server.py \
  --endpoint http://127.0.0.1:8019/v1/chat/completions \
  --model-id qwen3_5_4b_sft_stage9
```

这步做什么：

- 验证 HTTP（超文本传输协议）接口能访问，模型输出能被 `decision_codec（决策编解码器）`解析。

为什么要做：

- 训练成功不代表模型服务成功。
- 9.4 baseline compare（基线对比）需要业务链路能真正调用 SFT Planner（监督微调规划器）。

如果要从本地 Mac 访问 AutoDL 上的模型服务：

```bash
ssh -CNg -L 8019:127.0.0.1:8019 root@<AutoDL SSH 主机> -p <AutoDL SSH 端口>
```

然后本地访问：

```text
http://127.0.0.1:8019/v1/chat/completions
```

为什么默认不直接开放公网端口：

- AutoDL 通常不能任意开放端口。
- 训练评测阶段优先在同一台 AutoDL 实例内调用模型服务，减少网络变量。

### 4. 真实 Web Provider（网页执行器）最小观察记录

HTTP probe（接口探针）只验证 Planner 输出；它不会真的调用网页检索。真实 Provider 记录不依赖
GPU，建议在无卡模式、业务环境的 Web/Milvus 配置就绪后单独执行：

```bash
cd /root/autodl-tmp/ai_knowledge_base_after_class

CLOUD_SFT_ENV_FILE=deploy/cloud_sft/env.local \
bash deploy/cloud_sft/run_provider_probe.sh
```

默认只执行已审核的 `stage9-route-web-refuse-001`，输出
`cloud_smoke_provider_observations.jsonl（云端执行器观察记录）`。每条记录包含
`case_id（样本身份）`、`action（动作）`、候选数量、Observation（观察）、耗时和结构化错误。

Web 关闭时，`web_search`在 Environment/Planner 的 allowed_actions（允许动作）边界就应被拒绝，
因此不会伪造一次“真实 Provider 调用”。这一关闭边界记录在
`cloud_smoke_planner_http.json`的 `policy-web-disabled`探针中；Web 开启后的真实执行结果记录在
Provider JSONL。两份产物合起来构成 Web 开/关证据。

默认拒绝覆盖旧结果。只有明确废弃旧探针产物并准备重新生成时，才临时设置
`CLOUD_PROBE_OVERWRITE=1`；正式产物生成后应恢复为 `0`。

## 跑 dev eval（开发集评测）

`run_dev_eval.sh`当前使用 `ModelPlanner.from_checkpoint()`直接加载基础模型和 LoRA adapter，
不会调用已经启动的 vLLM HTTP 服务。因此完成上一节 healthcheck（健康检查）后，先停止
vLLM 并用 `nvidia-smi`确认显存释放，再运行 dev eval；否则两个运行时会重复加载模型并
产生 OOM（显存不足）风险。

checkpoint runtime（检查点运行时）必须使用包含 CUDA device placement（显卡设备放置）
修复的代码版本。本次根因是代码缺少以下两次设备迁移：

- 模型加载后没有执行 `.to("cuda")`；当前实现使用等价的
  `self._model.to(self._device)`，其中 CUDA 可用时 `self._device=cuda`。
- tokenizer（分词器）生成的输入 tensor（张量）仍留在 CPU；当前实现逐项执行
  `value.to(self._device)`，保证模型和输入位于同一设备。

CUDA 可用时，模型同时以 BF16/FP16 加载，避免默认 float32（32 位浮点）带来的额外显存。
旧代码会让模型和输入都留在 CPU，表现为权重加载完成后长时间没有输出，并非模型下载或
GPU 性能问题。

评测脚本还必须输出逐 case 日志。当前每条 case 会打印：

```text
[dev_eval] case=1/7 case_id=<case_id> status=running
[dev_eval] case=1/7 case_id=<case_id> status=completed duration_ms=<耗时> action_path=<动作路线>
```

这样可以区分模型加载、单条生成、程序卡死和正常长耗时，不能再只在全部评测结束后打印汇总。

```bash
cd /root/autodl-tmp/ai_knowledge_base_after_class
CLOUD_SFT_ENV_FILE=deploy/cloud_sft/env.local bash deploy/cloud_sft/run_dev_eval.sh
```

这步做什么：

- 用正式 checkpoint（检查点）跑小规模 dev case（开发样本）。
- 输出 SFT Planner（监督微调规划器）的评测 JSON（结构化数据）和 cloud run report（云端运行报告）。

为什么要做：

- 验证训练后的 Planner（规划器）不是只能保存，还能被 `OfflineRagEnvironment（离线 RAG 环境）`调用。
- 这一步不是最终模型质量结论，只是进入 9.4 baseline compare（基线对比）前的工程验收。

第一轮建议：

```dotenv
DEV_EVAL_PROVIDER=snapshot_expected_chunks
DEV_EVAL_MAX_CASES=7
```

原因：

- 这组配置用于复现第一轮只有 7 条 dev 时的历史评测；当前 registry 已扩充为 25 条，
  因而 `DEV_EVAL_MAX_CASES=7`现在只能做运行时回归，不能作为正式质量结论。
- 先验证 checkpoint runtime（检查点运行时）和 Planner（规划器）调用链。
- 等真实 Milvus/Web（向量数据库/网页检索）配置稳定后，再切：

```dotenv
DEV_EVAL_PROVIDER=milvus
```

如果控制台出现缺少 `flash-linear-attention/causal-conv1d`和安装网址，那只是 Transformers
提示可选 fast path（加速实现）不可用，并不代表正在下载。当前 PyTorch fallback（回退实现）
可以继续使用；在 `HF_HUB_OFFLINE=1`时，缺少本地模型缓存会直接报离线错误。

## 任务 9.3.16：跑完整 expanded dev 与 9.4 准入门禁

上面的 `run_dev_eval.sh`保留为首轮 7 条历史开发集运行入口，不能代替 9.3.16。
9.3.16 必须使用新增的 25 条 reviewed balanced dev（已审核均衡开发集），五条路线各 5 条，
并按 9.3.12 在看结果前冻结的阈值判定。

运行前先确认 vLLM 已停止、`nvidia-smi`没有残留模型进程，并保持：

```dotenv
REQUIRE_CUDA=1
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
EXPANDED_DEV_PROVIDER=snapshot_expected_chunks
EXPANDED_DEV_OVERWRITE=0
```

正式运行：

```bash
cd /root/autodl-tmp/ai_knowledge_base_after_class

# 推荐先在无卡模式执行；只校验输入，不加载模型、不写推理结果。
set -a
source deploy/cloud_sft/env.local
set +a
"$PYTHON_BIN" -m evaluation.stage9.admission.run_sft_expanded_dev_gate \
  --checkpoint "$SFT_CHECKPOINT_DIR" \
  --preflight-only

# preflight 通过后再切回有卡模式执行正式 25 条推理。
CLOUD_SFT_ENV_FILE=deploy/cloud_sft/env.local \
bash deploy/cloud_sft/run_expanded_dev_gate.sh
```

脚本会在加载模型、开始 GPU 推理前检查：

- checkpoint 必须是 `transformers_causal_lm` 的 LoRA/QLoRA 正式产物，禁止 smoke；
- Reward v1.1、9.3.15A 验证、路线矩阵、25 条 dev 和 snapshot 的 SHA256 必须一致；
- 只能运行 25 条 reviewed dev，不能设置 `max_cases`，不能混入 test/heldout；
- 固定输出默认不存在，防止静默覆盖上一轮准入记录；
- Hugging Face 与 Transformers 必须为离线模式，避免 GPU 计费期间重新下载模型。

执行结束后固定生成：

```text
evaluation/stage9/artifacts/sft/sft_expanded_dev_eval.json
evaluation/stage9/artifacts/sft/sft_9_4_admission_decision.json
evaluation/stage9/artifacts/reports/阶段9-SFT-9.4准入报告.md
evaluation/stage9/artifacts/cloud_runs/expanded_dev_gate_<timestamp>/
```

若全部门禁通过，脚本输出
`eligible_for_stage9_4=true`并正常退出。若未通过，逐 case 产物和报告仍会保存，
随后以退出码 `3`停止，明确禁止进入 9.4；此时只能根据 dev 失败补独立 train-only 数据，
不能把 balanced dev 或 heldout 原题放入训练集。

9.3.16 完成后再次运行 `freeze_sft_artifacts.py`时，把 `--dev-run-dir`指向
`expanded_dev_gate_<timestamp>`。冻结器会识别 `stage9_3_16_expanded_dev_gate`布局，
把原始 eval、准入决定、Markdown 报告、日志、命令和 cloud run report 一并收入归档。

## 任务 9.3.20：使用真实回放校正复评 SFT v1（监督微调第一版）

9.3.20 不复用 `run_expanded_dev_gate.sh`，因为该脚本属于 9.3.16 冻结入口并强制使用
`snapshot_expected_chunks`（按期望文本块构造快照）。本任务必须运行
`run_sft_v1_corrected_replay_eval.sh`，绑定 9.3.18 的 Replay Provider（回放动作执行器）
和 9.3.19 的 Reward（奖励函数）回归身份。

先在无卡模式执行 `preflight`（运行前检查）；它只读验证旧评测、旧准入决定、当前 25 条
reviewed dev（已审核开发集）、checkpoint（检查点）、Replay（回放）记录和全部
SHA256（文件内容哈希），不加载模型、不写推理结果：

```bash
cd /root/autodl-tmp/ai_knowledge_base_after_class

sed -i 's/^REQUIRE_CUDA=.*/REQUIRE_CUDA=0/' deploy/cloud_sft/env.local

SFT_V1_CORRECTED_REPLAY_PREFLIGHT_ONLY=1 \
CLOUD_SFT_ENV_FILE=deploy/cloud_sft/env.local \
bash deploy/cloud_sft/run_sft_v1_corrected_replay_eval.sh
```

预期输出包含 `ok=true`（检查通过）、`preflight_only=true`（仅运行前检查）、
`case_count=25`（样本数为 25）、`model_execution_performed=false`（未执行模型）和
`heldout_inference_result_count=0`（留出集推理数为零）。任一检查失败时不要开启
GPU（图形处理器）。

无卡检查通过后再恢复 GPU（图形处理器），把 `REQUIRE_CUDA` 改回 `1`，确认
`nvidia-smi`（英伟达显卡状态工具）没有残留模型进程，然后运行：

```bash
sed -i 's/^REQUIRE_CUDA=.*/REQUIRE_CUDA=1/' deploy/cloud_sft/env.local

SFT_V1_CORRECTED_REPLAY_PREFLIGHT_ONLY=0 \
CLOUD_SFT_ENV_FILE=deploy/cloud_sft/env.local \
bash deploy/cloud_sft/run_sft_v1_corrected_replay_eval.sh
```

正式入口只运行同一 SFT v1（监督微调第一版）checkpoint（检查点）和 25 条
dev（开发集）。它会保存结构化 Trace（执行轨迹）中的 Decision（决策）与
Observation（观察结果），但不保存聊天历史、Prompt（提示词）或模型私有思维链。
每次运行生成独立目录：

```text
evaluation/stage9/artifacts/cloud_runs/sft_v1_corrected_replay_<UTC时间>/
├── command.txt
├── run.log
├── sft_v1_corrected_replay_eval.json
├── sft_v1_corrected_replay_comparison.json
├── 阶段9-SFT-v1校正复评报告.md
└── SHA256SUMS
```

旧 9.3.16 产物不会被覆盖。9.3.20 输出固定
`eligible_for_stage9_4=false`（不直接允许进入 9.4）；只能在人工确认逐 case（逐样本）
归因后进入 9.3.21，且 heldout test（留出测试）仍不得运行。

## 完成后必须下载和备份

训练完成后至少下载：

```text
stage9_sft_artifacts_<run_id>.tar.gz
cloud_run_report.json
checkpoint_manifest.json
train_metrics.json
dev_eval 输出 JSON
9.3.16 admission decision JSON 与准入报告
```

这步做什么：

- 把关键训练产物带离 AutoDL 实例。

为什么要做：

- AutoDL 数据盘适合训练运行，不适合作为唯一长期备份。
- 后续 9.4 baseline compare（基线对比）和论文/面试表达都需要可追溯证据。

## 省钱和关机

训练完成并备份后：

```bash
nvidia-smi
du -sh evaluation/stage9/artifacts/sft/checkpoints
du -sh evaluation/stage9/artifacts/cloud_runs
```

确认没有仍在跑的训练或模型服务后，可以在控制台关机。

如果只是整理文件、下载产物或改文档，可以使用 AutoDL no-card mode（无卡模式）。

不要在第一次正式训练时自动关机，除非你已经确认日志和产物路径都正确。AutoDL 官方文档提醒，自动关机后标准输出日志不再可见，所以要先保存日志。

## 失败处理速查

| 问题 | 优先检查 | 处理建议 |
|---|---|---|
| `torch.cuda.is_available=false（CUDA 不可用）` | `nvidia-smi`、PyTorch 版本是否带 CUDA | 换 PyTorch/CUDA 镜像，重新 bootstrap（初始化） |
| CUDA OOM（显存不足） | `nvidia-smi` 显存占用 | 切 QLoRA（4 位量化低秩适配）或换 A800 |
| `bitsandbytes（量化库）`报错 | CUDA/PyTorch/bitsandbytes 版本 | 5090 优先换更新镜像；仍失败则用 A800 + LoRA |
| 模型下载慢或失败 | 网络、缓存目录、磁盘空间 | 预下载模型到 `/root/autodl-tmp`，或重试；避免写系统盘 |
| vLLM 安装超过预期时间 | 是否在 GPU 计费模式、下载进度、`VLLM_UV_CACHE_DIR`、磁盘空间 | 优先在无卡模式安装；保留持久化 cache，不要使用 `--no-cache` |
| dev eval 权重加载后无进度 | `nvidia-smi`显存/利用率、checkpoint runtime 是否执行模型 `.to("cuda")`和输入 tensor `.to(device)` | 使用已修复运行时；确认 CUDA 推理，并查看逐 case `running/completed`日志 |
| `vllm: command not found` | `VLLM_VENV_PATH`、启动命令的 `PATH` | 不要装进 `.venv-sft`；用 `PATH="$VLLM_VENV_PATH/bin:$PATH"`启动模型服务 |
| 无卡安装选错 CUDA wheel | `VLLM_TORCH_BACKEND`、`torch.version.cuda`、目标 GPU 驱动 | 无卡模式固定已验证 backend；切回有卡后必须验证 `torch.cuda.is_available=True` |
| 系统盘满 | `df -h`、缓存路径 | 把 HF/UV/ModelScope 缓存迁到 `/root/autodl-tmp/cache` |
| SSH 断开训练停了 | 是否用了 screen/tmux | 用 `screen/tmux` 重新跑，脚本会重新生成 run_dir（运行目录） |
| 模型服务外部访问失败 | AutoDL 端口策略 | 同机评测优先；本地访问用 SSH tunnel（SSH 隧道） |
| dev eval（开发集评测）找不到 checkpoint | `SFT_CHECKPOINT_DIR` 是否设置 | 把正式训练日志里的 checkpoint 路径写入 `env.local` |

## 9.3.10 闭环与进入 9.3.11 的验收清单

- 9.3.8 cloud smoke（云端冒烟）已通过。
- 9.3.9 正式 SFT（监督微调）已完成。
- `checkpoint_manifest.json（检查点清单）`存在，并记录正确的 `base_model_id（基础模型身份）`、`model_profile_id（模型配置档案身份）`、`snapshot_id（快照身份）`、`reward_profile（奖励函数配置）`。
- `train_metrics.json（训练指标）`存在。
- `model/adapter（适配器目录）`存在。
- `tokenizer（分词器目录）`存在。
- `cloud_run_report.json（云端运行报告）`存在。
- vLLM 独立环境已固定版本，`vllm_environment_freeze.txt（vLLM 环境冻结清单）`存在。
- vLLM 环境在正常有卡模式下 `torch.cuda.is_available=True`。
- no-card 和 gpu runtime preflight（无卡/有卡运行时前置检查）均生成报告，且 GPU 报告 `ok=true`。
- PlannerModelServer（规划器模型服务）可启动。
- healthcheck（健康检查）通过。
- 六类 Action 与 Web 禁用边界 HTTP 探针报告已生成；工程错误与 Action 质量偏差分开记录。
- Web 开启的真实 Provider observation 已生成；Web 关闭边界记录为“不允许调用”，没有伪造检索结果。
- dev eval（开发集评测）至少跑通 `snapshot_expected_chunks（快照期望文本块执行器）`。
- 训练产物已经从 AutoDL 下载或备份。

达到以上条件后，9.3.10 才算闭环，随后进入 9.3.11 做现有 dev 结果分析。9.4 baseline compare
（基线对比）仍需等待 9.3.11～9.3.16 的评测集补强和准入结论，本节不提前放行。
