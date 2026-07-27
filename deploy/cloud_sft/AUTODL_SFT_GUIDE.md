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

## AutoDL 平台边界

以下信息来自 AutoDL（云 GPU 平台）官方文档，平台界面和价格以后可能变化，最终以控制台显示为准。

- 创建实例时需要选择计费方式、地区、GPU（显卡）型号、GPU 数量、空闲主机和镜像；实例进入运行中后开始计费。参考 [AutoDL 快速开始](https://www.autodl.com/docs/quick_start/)。
- 镜像优先选平台内置 PyTorch/CUDA（深度学习框架/英伟达 GPU 计算平台）镜像；如果内置镜像不满足，再用 Miniconda/CUDA 镜像自行安装。参考 [AutoDL 环境配置](https://www.autodl.com/docs/base_config/)。
- 实例关机后数据通常保留，但本地数据盘无冗余保证，连续关机 15 天会触发释放风险；重要 checkpoint（检查点）必须备份。参考 [AutoDL 实例数据](https://www.autodl.com/docs/instance_data/)。
- 长时间训练要用 JupyterLab（浏览器开发环境）终端、`screen（会话守护工具）`或 `tmux（会话守护工具）`，并保存日志，避免 SSH（安全远程登录）断开导致训练中断。参考 [AutoDL 守护进程](https://www.autodl.com/docs/daemon/)。
- AutoDL 实例通常没有独立公网 IP（公网地址），任意端口访问建议用 SSH tunnel（SSH 隧道）；平台默认只对 `6006/6008` 提供自定义服务映射。参考 [AutoDL 开放端口](https://www.autodl.com/docs/port/) 和 [AutoDL SSH 隧道](https://www.autodl.com/docs/ssh_proxy/)。
- 不需要 GPU（显卡）时可以用 no-card mode（无卡模式）做文件管理和轻量调试，但无卡模式会释放 GPU，重新有卡开机时可能遇到空闲 GPU 不足。参考 [AutoDL 省钱绝招](https://www.autodl.com/docs/save_money/)。

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

### 3. 运行 bootstrap（初始化）

```bash
cd /root/autodl-tmp/ai_knowledge_base_after_class
CLOUD_SFT_ENV_FILE=deploy/cloud_sft/env.local bash deploy/cloud_sft/bootstrap_gpu_server.sh
```

如果还没有 `env.local（本机环境变量文件）`，先复制：

```bash
cp deploy/cloud_sft/env.example deploy/cloud_sft/env.local
```

这步做什么：

- 安装训练依赖。
- 检查 `nvidia-smi（显卡状态工具）`。
- 检查 PyTorch（深度学习框架）是否能看到 CUDA（英伟达 GPU 计算平台）。
- 打印 `torch/transformers/peft/bitsandbytes（训练框架/参数高效微调库/量化库）`版本。

为什么要做：

- 这是正式训练前的环境门禁。
- 如果这里都不能识别 GPU（显卡），正式 SFT（监督微调）一定不可靠。

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

HF_HOME=/root/autodl-tmp/cache/huggingface
MODELSCOPE_CACHE=/root/autodl-tmp/cache/modelscope
UV_CACHE_DIR=/root/autodl-tmp/cache/uv
```

如果当前使用 AutoDL 的 no-card mode（无卡模式）只准备依赖，把 `REQUIRE_CUDA`临时改成：

```dotenv
REQUIRE_CUDA=0
```

此时 `bootstrap（初始化脚本）`允许 `nvidia-smi（显卡状态工具）`不可用，但仍会完成独立
训练环境安装和版本检查。切换到 5090 正式实例后，必须恢复为 `REQUIRE_CUDA=1`，
让 CUDA（英伟达 GPU 计算平台）不可用时立即停止。

这里的 `.venv-sft（监督微调虚拟环境）`只安装
`deploy/cloud_sft/requirements-training.lock（训练依赖锁定文件）`中的依赖；该锁文件由
`requirements-training.txt（直接训练依赖清单）`解析生成。不要执行
`uv add transformers==5.9.0`去修改主业务环境，也不要使用 `--frozen`跳过依赖检查；
前者会与 `magic-pdf`发生版本冲突，后者会留下无法复现的损坏环境。

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

建议打包：

```bash
tar -czf /root/autodl-tmp/stage9_sft_artifacts_<run_id>.tar.gz \
  evaluation/stage9/artifacts/sft/checkpoints/<run_id> \
  evaluation/stage9/artifacts/cloud_runs/sft_train_<timestamp>
```

这步做什么：

- 把训练产物和审计报告打成一个可下载包。

为什么要做：

- AutoDL 本地数据盘不是长期可靠归档。
- 训练完成后必须把正式 checkpoint（检查点）和报告备份到本地或长期存储。

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
CLOUD_SFT_ENV_FILE=deploy/cloud_sft/env.local bash deploy/cloud_sft/run_planner_server.sh
```

这步做什么：

- 在 AutoDL 实例内启动 PlannerModelServer（规划器模型服务）。

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

## 跑 dev eval（开发集评测）

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

- 先验证 checkpoint runtime（检查点运行时）和 Planner（规划器）调用链。
- 等真实 Milvus/Web（向量数据库/网页检索）配置稳定后，再切：

```dotenv
DEV_EVAL_PROVIDER=milvus
```

## 完成后必须下载和备份

训练完成后至少下载：

```text
stage9_sft_artifacts_<run_id>.tar.gz
cloud_run_report.json
checkpoint_manifest.json
train_metrics.json
dev_eval 输出 JSON
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
| 系统盘满 | `df -h`、缓存路径 | 把 HF/UV/ModelScope 缓存迁到 `/root/autodl-tmp/cache` |
| SSH 断开训练停了 | 是否用了 screen/tmux | 用 `screen/tmux` 重新跑，脚本会重新生成 run_dir（运行目录） |
| 模型服务外部访问失败 | AutoDL 端口策略 | 同机评测优先；本地访问用 SSH tunnel（SSH 隧道） |
| dev eval（开发集评测）找不到 checkpoint | `SFT_CHECKPOINT_DIR` 是否设置 | 把正式训练日志里的 checkpoint 路径写入 `env.local` |

## 进入 9.4 前的验收清单

- 9.3.8 cloud smoke（云端冒烟）已通过。
- 9.3.9 正式 SFT（监督微调）已完成。
- `checkpoint_manifest.json（检查点清单）`存在，并记录正确的 `base_model_id（基础模型身份）`、`model_profile_id（模型配置档案身份）`、`snapshot_id（快照身份）`、`reward_profile（奖励函数配置）`。
- `train_metrics.json（训练指标）`存在。
- `model/adapter（适配器目录）`存在。
- `tokenizer（分词器目录）`存在。
- `cloud_run_report.json（云端运行报告）`存在。
- PlannerModelServer（规划器模型服务）可启动。
- healthcheck（健康检查）通过。
- dev eval（开发集评测）至少跑通 `snapshot_expected_chunks（快照期望文本块执行器）`。
- 训练产物已经从 AutoDL 下载或备份。

达到以上条件后，才进入 9.4 baseline compare（基线对比）。
