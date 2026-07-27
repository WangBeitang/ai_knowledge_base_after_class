# AutoDL SFT 上云问题记录

用于阶段 9 SFT（监督微调）首次上云后的对话续接。

## 当前状态

- `Qwen/Qwen3.5-4B` 已完整下载到 `/root/autodl-tmp/cache/huggingface`。
- 独立 `.venv-sft（监督微调虚拟环境）`已建立：
  `torch=2.10.0`、`transformers=5.9.0`、`peft=0.18.1`。
- config（模型配置）和 tokenizer（分词器）已通过离线加载验证。
- 4 条样本、1 个 step（训练步）的 LoRA smoke（低秩微调冒烟训练）成功，
  耗时约 4 秒，已生成 checkpoint（检查点）和 cloud run report（云端运行报告）。
- 正式 LoRA SFT 已完成：155 条样本、20 个 step、1 个 epoch（训练轮次），
  `train_loss=0.2804`。这只说明训练过程成功，不代表 held-out（留出集）质量已经验证。
- 正式 checkpoint 根目录约 101MB，LoRA adapter（低秩适配器）和训练报告均已核对：
  `evaluation/stage9/artifacts/sft/checkpoints/planner-sft-stage9-qwen3-5-4b-lora_20260727T085537Z_94a77563`。
- 当前实例的 `nvidia-smi`实测为 RTX 4090、49140MiB 显存、driver 580.76.05。
- vLLM（大模型推理服务框架）0.25.1 已在 no-card mode（无卡模式）完成安装，
  `torch=2.11.0+cu130`、依赖冻结清单已生成；恢复 GPU 后仍需执行 CUDA 运行门禁。

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
| 正式训练结束后 `vllm: command not found` | `.venv-sft`只包含训练依赖，vLLM 不属于该环境 | 使用独立 `.venv-vllm`；不要把 vLLM 装进训练环境 |
| 有卡模式安装 vLLM 一小时仍在缓慢下载 | vLLM 会解析并下载大量 PyTorch/CUDA wheel（预编译包）；本次有卡模式实测网络很慢 | 停止付费下载，切到无卡模式安装；依赖准备完成后再申请 GPU |
| 重启安装后看起来重新下载 | 安装命令使用了 `--no-cache`，中断后无法充分复用已下载内容 | 去掉 `--no-cache`，固定持久化 `VLLM_UV_CACHE_DIR`，允许断点重试 |
| 无卡模式执行 `nvidia-smi`显示 `No devices were found` | 无卡模式已经释放 GPU，不是显卡或驱动损坏 | 无卡阶段只安装依赖和检查文件；恢复有卡后再做 CUDA 门禁 |
| 无卡模式执行 `vllm --version`报 `Failed to infer device type` | vLLM 0.25.1 CLI 在解析参数时会尝试推断运行设备，没有 GPU 时可能失败 | 不重装；无卡模式用 Python package metadata（包元数据）读取版本，CLI 检查移到恢复有卡后 |
| 释放 GPU 后担心微调权重丢失 | GPU 显存与磁盘文件不是同一生命周期 | 用 `checkpoint_manifest.json`和 adapter 文件确认；本次无卡重启后 checkpoint 仍存在 |
| cloud run report 找不到 manifest | 首次收集报告时传入了 `model`子目录，而不是 checkpoint 根目录 | 重新以 checkpoint 根目录生成报告，并保留修正前报告副本 |
| 数据盘只剩约 15GB，系统盘约 25GB 可用 | 模型缓存、项目、checkpoint 和大依赖同时占空间 | checkpoint/模型缓存留数据盘；vLLM 环境与其 uv cache 放同一块空间充足的盘；安装前执行 `df -h` |
| dev eval 权重加载后十多分钟没有输出 | 代码问题：旧版 checkpoint runtime 加载模型后缺少 `.to("cuda")`，tokenizer 生成的输入 tensor 也缺少 `.to(device)`，因此实际在 CPU 上做 4B 推理；评测脚本还没有逐 case 日志 | 模型执行 `self._model.to(self._device)`、输入逐项执行 `value.to(self._device)`，CUDA 使用 BF16/FP16；每条 case 打印 `running/completed`、耗时和 Action 路线 |
| dev eval 显示加速库安装网址，误以为正在下载 | Transformers 只是在提示缺少可选 fast path，并已回退 PyTorch 实现 | 不安装、不等待下载；`HF_HUB_OFFLINE=1`下缺少本地模型缓存会直接报错 |
| vLLM 健康检查后直接跑 checkpoint dev eval | 两条链路会分别加载模型，vLLM 已占约 41GB 时再加载 checkpoint 容易 OOM | healthcheck 通过后停止 vLLM并确认显存释放，再运行当前 direct-checkpoint dev eval |

不要使用 `uv add transformers==5.9.0 --frozen`强行修改主业务环境。

## 有卡与无卡模式下载速度差异

### 已确认事实

- 本次 vLLM 安装在有卡计费模式下长时间下载极慢，继续等待的预计费用不可接受。
- 停止有卡实例并进入无卡模式后，依赖下载速度明显改善。
- 这足以支持“以后不要在付费 GPU 模式首次安装大依赖”的操作决策。

### 尚未证实的判断

目前没有证据证明 AutoDL 在有卡模式下“故意限速”。有卡和无卡实例可能落到不同宿主机、
共享出口或网络路由，也可能受到节点负载、镜像/CDN 命中率和时段波动影响。

如果需要把问题归因为平台策略，至少要在相同地区、相近时段、相同下载地址和相同并发参数下，
分别重复测试有卡与无卡模式，并记录开始/结束时间、下载字节数、错误重试和出口路由。
在完成该 A/B test（对照测试）前，文档只记录“存在显著速度差异”，不写成已证实的故意限速。

### 当前操作规则

1. 项目同步、Python 环境、SFT/vLLM 依赖和模型下载优先在无卡模式完成。
2. vLLM 使用独立 `.venv-vllm`和持久化 `VLLM_UV_CACHE_DIR`，不得使用 `--no-cache`。
3. 有卡模式只执行 CUDA 验证、smoke、正式训练、模型服务 healthcheck（健康检查）和 dev eval（开发集评测）。
4. 如果下载速度导致预计等待成本不可接受，立即停止，不因“已经下了一部分”继续付费。
5. 训练结束后先核对并备份 checkpoint，再释放 GPU；无卡模式下看不到 GPU 不影响已落盘权重。

完整的无卡预装命令和恢复有卡后的验证门禁见
[`AUTODL_SFT_GUIDE.md`](./AUTODL_SFT_GUIDE.md#no-card-mode无卡模式预装全部依赖)。

## 替代 GPU 平台的选择

切换平台是合理的成本控制方案，但不要在当前正式 checkpoint 已成功落盘后立即推倒重来。
本轮优先继续完成 AutoDL 的无卡依赖准备和有卡评测；下一轮训练前再做平台对照测试。

选择新平台时按以下顺序核对：

1. 是否支持无卡创建或停止 GPU 后继续准备环境。
2. 是否支持 SSH、持久化磁盘和停止计算计费。
3. 依赖、模型和 checkpoint 在关机、释放实例后的保留规则。
4. PyPI/Hugging Face 的实际下载速度，而不是只看宣传带宽。
5. GPU 实际显存。当前实例是约 48GB 显存，普通 RTX 4090 通常只有 24GB，不能默认复用相同训练参数。

候选平台：

| 优先级 | 平台 | 适用判断 | 迁移风险 |
|---|---|---|---|
| 1 | [起源云](https://docs.origincloud.com.cn/getting-started/quickstart) | 国内优先测试；官方文档支持无卡创建、SSH、按秒计费、关机保留系统盘和数据盘，并提供跨实例云盘 | 网络速度仍需实测；连续关机 15 天后实例会自动释放；需另选 48GB 级 GPU 才能接近当前显存余量 |
| 2 | [RunPod](https://www.runpod.io/pricing) | 可选 48GB GPU，并支持独立的 [network volume（网络持久卷）](https://docs.runpod.io/storage/network-volumes) | 国内到海外节点的 SSH、付款和跨境网络稳定性需先测试；存储单独计费 |
| 3 | [阿里云 PAI DSW](https://help.aliyun.com/en/pai/create-and-manage-dsw-instances) | 更重视企业稳定性时考虑；支持 SSH，停止实例后可释放计算资源 | 配置和计费更复杂；必须区分临时存储与持久化云盘 |

[恒源云本地存储文档](https://gpushare.com/docs/data/storage/)说明实例关机超过 24 小时后，
`/hy-tmp`数据会被删除；除非把 checkpoint 主动复制到个人云存储，否则不作为本项目首选。

迁移前只做一次小额验证：在候选平台使用相同 vLLM 版本、相同 CUDA backend（后端）、
相同并发和独立空缓存，记录固定时间窗口内的下载量。网络没有明显改善，或 48GB GPU
的综合价格更高，就不迁移。

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

1. 将 checkpoint runtime 的模型 `.to("cuda")`等价修复、输入 tensor `.to(device)`和逐 case 日志同步到云端。
2. 确认 vLLM 已停止、GPU 显存已释放后，运行完整 7 条 dev eval。
3. 打包并下载 checkpoint、cloud run report 和评测结果；暂不进入 GRPO（组相对策略优化）。
