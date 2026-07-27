# AutoDL SFT 上云问题记录

用于阶段 9 SFT（监督微调）首次上云后的对话续接。

## 当前状态

- `Qwen/Qwen3.5-4B` 已完整下载到 `/root/autodl-tmp/cache/huggingface`。
- 独立 `.venv-sft（监督微调虚拟环境）`已建立：
  `torch=2.10.0`、`transformers=5.9.0`、`peft=0.18.1`。
- config（模型配置）和 tokenizer（分词器）已通过离线加载验证。
- 4 条样本、1 个 step（训练步）的 LoRA smoke（低秩微调冒烟训练）成功，
  耗时约 4 秒，已生成 checkpoint（检查点）和 cloud run report（云端运行报告）。
- smoke 只证明训练链路可用，不代表模型质量或正式 SFT 已完成。
- 当前底层 GPU 显示 RTX 4090，vGPU 实际显存容量仍待确认。

## 已踩问题

| 问题 | 原因 | 处理 |
|---|---|---|
| 无卡模式 `nvidia-smi: Permission denied` | no-card mode（无卡模式）没有 GPU 权限 | 准备阶段 `REQUIRE_CUDA=0`；正式训练恢复为 `1`；脚本已兼容 |
| Hugging Face `Network is unreachable` | AutoDL 到 Hugging Face 网络不通 | 仅下载模型时执行 `source /etc/network_turbo` |
| Xet 下载 `401 Unauthorized` | CAS/Xet 链路认证失败 | 设置 `HF_HUB_DISABLE_XET=1`，改走普通 HTTP |
| 下载完成但离线缓存找不到 | `TRANSFORMERS_CACHE`与 `HF_HOME`造成缓存路径不一致 | 删除已废弃的 `TRANSFORMERS_CACHE`，只使用 `HF_HOME` |
| Transformers 不认识 `qwen3_5` | 原版本 4.57.6 太旧 | 独立训练环境固定 `transformers==5.9.0` |
| `magic-pdf`与 Transformers 5 冲突 | `magic-pdf`要求 `Transformers < 5` | 主业务环境不动；训练使用独立 `.venv-sft`和 `requirements-training.lock` |
| smoke 加载 tokenizer 时再次联网失败 | 训练仍发起 Hugging Face 在线检查 | 训练阶段设置 `HF_HUB_OFFLINE=1` |
| 终端看似卡住 | 模型加载阶段缺少进度输出，或网络检查正在等待 | 用 `nvidia-smi`和进程状态判断；本次实际失败原因已由离线模式修复 |

不要使用 `uv add transformers==5.9.0 --frozen`强行修改主业务环境。

## 当前可忽略的警告

- `torch_dtype is deprecated`：后续改为 `dtype`，当前不阻塞。
- 缺少 `fla/causal-conv1d`：当前回退到 PyTorch 实现，现有训练规模速度足够。
- PEFT 找不到远程 config：离线检查导致；当前没有修改词表，可以接受。

## 云端关键配置

```dotenv
APP_ROOT=/root/autodl-tmp/ai_knowledge_base_after_class
SFT_VENV_PATH=/root/autodl-tmp/ai_knowledge_base_after_class/.venv-sft
PYTHON_BIN=/root/autodl-tmp/ai_knowledge_base_after_class/.venv-sft/bin/python
SFT_PYTHON_VERSION=3.12
BOOTSTRAP_INSTALL_DEPS=0
REQUIRE_CUDA=1
HF_HOME=/root/autodl-tmp/cache/huggingface
MODELSCOPE_CACHE=/root/autodl-tmp/cache/modelscope
UV_CACHE_DIR=/root/autodl-tmp/cache/uv
HF_HUB_DISABLE_XET=1
HF_HUB_OFFLINE=1
```

## 下一步

1. 用以下命令确认购买的 vGPU 显存是否匹配：

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version \
  --format=csv,noheader
```

2. 核对 smoke 的 checkpoint 和 cloud run report。
3. 确认无误后运行 `run_sft_train.sh`进行正式 SFT；暂不进入 GRPO（组相对策略优化）。
