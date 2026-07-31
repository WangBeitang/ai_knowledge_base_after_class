# AutoDL SFT 上云问题记录

用于阶段 9 SFT（监督微调）首次上云后的对话续接。

## 当前状态

- `Qwen/Qwen3.5-4B` 已完整下载到 `/root/autodl-tmp/cache/huggingface`。
- 独立 `.venv-sft（监督微调虚拟环境）`已建立：
  `torch=2.10.0`、`transformers=5.9.0`、`peft=0.18.1`。
- config（模型配置）和 tokenizer（分词器）已通过离线加载验证。
- 4 条样本、1 个 step（训练步）的 LoRA smoke（低秩微调冒烟训练）成功，
  耗时约 4 秒，已生成 checkpoint（检查点）和 cloud run report（云端运行报告）。
- 正式 LoRA SFT 已完成：155 个 Action step（对应 70 个来源轨迹）、20 个 optimizer step、
  1 个 epoch（训练轮次），
  `train_loss=0.2804`。这只说明训练过程成功，不代表 held-out（留出集）质量已经验证。
- 正式 checkpoint 根目录约 101MB，LoRA adapter（低秩适配器）和训练报告均已核对：
  `evaluation/stage9/artifacts/sft/checkpoints/planner-sft-stage9-qwen3-5-4b-lora_20260727T085537Z_94a77563`。
- 当前实例的 `nvidia-smi`实测为 RTX 4090、49140MiB 显存、driver 580.76.05。
- vLLM（大模型推理服务框架）0.25.1 已在 no-card mode（无卡模式）完成安装，
  `torch=2.11.0+cu130`、依赖冻结清单已生成；有卡 CUDA 门禁、LoRA 加载和 HTTP
  healthcheck（健康检查）已经通过。
- checkpoint runtime（检查点运行时）的 CUDA device placement（显卡设备放置）代码问题
  已修复；完整 7 条 dev eval 已成功，平均 Reward 为 0.8444。该结果只属于小规模工程验收，
  不能作为正式泛化结论。
- 2026-07-29 已完成正式 25 条 balanced dev GPU 复评：路线正确 8/25、macro accuracy=0.32，
  决定 `reject_sft_v1_train_sft_v2`；heldout test 未运行。
- expanded dev 的 29 文件 v2 归档已下载到本地长期备份并通过归档整体、manifest 和内部逐文件
  SHA256 校验；归档 SHA256 为
  `0f58738d92fb81f8f7bb3010c39cb7be95436947ed04ae4886b57cc82a106b44`。
- 当前已经切回无卡模式，正式 checkpoint、训练报告和 expanded dev 结果仍保存在数据盘。

## 2026-07-29 expanded dev 上云复评补充复盘

### 本轮结论

- GPU 运行链路正常：25 条全部完成，格式合法率 1.0，无 execution failure（执行失败）和
  forbidden action（禁止动作）。
- SFT v1 未通过 9.4 准入，但 17 条失败不能全部归因给模型。
- 5 条 Web 和 2 条澄清属于明确的训练覆盖问题；5 条 HyDE 存在 evaluator false negative
  （评测器误判）；5 条安全拒绝因 Provider 没有返回安全证据，暂时无法准确归因。
- 评测契约、训练数据和完整经验见
  [`阶段9-SFT-v1训练与复评复盘.md`](../../重构方案/阶段9-SFT-v1训练与复评复盘.md)。

### 本轮新发现的云端问题

| 问题 | 现象 | 根因 | 后续处理 |
|---|---|---|---|
| 正式入口缺少 `loguru` | 无卡 preflight 导入 expanded dev 入口时报 `ModuleNotFoundError: No module named 'loguru'` | `.venv-sft` 锁文件只覆盖训练包，没有覆盖正式评测入口导入的项目日志依赖 | 9.3.17 已把 `loguru==0.7.3`补入 source requirements、lock 和 import 门禁；云端锁文件 audit 与完整入口 import 已通过 |
| 新 shell 没有加载 `env.local` | `$PYTHON_BIN` 为空时执行 `uv pip freeze`，第一次得到主项目 `.venv` 的 `transformers=4.57.6` | 无卡重启后环境变量不会自动继承；uv 在解释器参数无效时选择了项目默认环境 | freeze 前强制 `source env.local`并打印 `$PYTHON_BIN`、`sys.executable`和核心包版本 |
| 离线变量不完整 | `env.local`只有 `HF_HUB_OFFLINE=1`，缺少脚本硬门禁要求的 `TRANSFORMERS_OFFLINE=1` | 旧实例配置没有随新入口模板自动补齐 | 无卡阶段显式补齐并 grep 复核；正式脚本继续拒绝不完整离线配置 |
| 只校验依赖列表不足 | 文件、checkpoint 和 SHA256 preflight 通过后，正式 Python 入口仍可能缺运行依赖 | 过去只检查若干包版本，没有导入最终执行模块 | 9.3.17 preflight 已使用正式 `$PYTHON_BIN`完整 import 训练与 expanded dev 入口，并校验解释器和版本身份 |
| `screen`退出容易被误判为崩溃 | 25 条结束后 `screen -r`显示 `[screen is terminating]` | 未准入时脚本保存产物后按设计以退出码 3 结束，后台会话随子进程关闭 | 不重复启动；先定位唯一 `expanded_dev_gate_<timestamp>`并读取 log、decision 和 report |
| 环境 freeze 可能记录错解释器 | 错误 freeze 看起来格式正常，版本却和 checkpoint manifest 不一致 | 没有在写文件前冻结解释器绝对路径和包身份 | 把 checkpoint manifest、lock、runtime freeze 三方一致性设为归档门禁 |

### 本轮正确做法

1. 先在无卡模式拉代码、备份 `env.local`、执行 checkpoint/data/hash preflight。
2. 缺少 `loguru`时没有开 GPU，而是在无卡模式补包并重新通过完整入口检查。
3. 有卡前显式检查离线变量、输出不存在、磁盘空间和 GPU 空进程。
4. 正式评测运行在独立 `screen` 会话，并有逐 case 进度和 Action path。
5. 门禁拒绝后立即切回无卡模式，没有继续占用 GPU 做分析和打包。
6. 环境、checkpoint、训练 run、expanded dev run、准入决定和报告统一进入 v2 冻结归档。
7. 下载后完成外层 SHA256、内外 manifest 对比和 29 文件逐文件 SHA256。

### 本轮仍需整改

- [x] 已把 `loguru==0.7.3`写入训练依赖源文件和锁文件。
- [x] 无卡 preflight（运行前检查）已增加正式入口 import（导入）、解释器身份、
  锁文件/checkpoint（检查点）版本三方一致性硬门禁；2026-07-31 云端使用最新脚本复跑返回
  `ok=true`（检查通过）、`failed_check_count=0`（失败检查数为零），并用同一正式解释器自动生成
  SFT environment freeze（监督微调环境冻结）。运行目录与两个证据文件的 SHA256（文件内容哈希）
  已归档回本地项目，9.3.17 正式完成。
- [x] 已完成 9.3.18：修复 Snapshot Provider（快照动作执行器）的 HyDE（假设文档嵌入）和
  安全拒绝 Observation（观察结果）契约，并冻结真实检索 Replay（回放）记录。
- [x] 已完成 9.3.19：基于冻结 Replay（回放）重跑 Reward v1.1（奖励函数第一点一版）回归，
  保留原权重和实现。
- [ ] 只根据修正后仍成立的失败补独立 train-only 数据；不复制 balanced dev 或 heldout。

## 2026-07-27 首次正式上云复盘

### 总结判断

本次正式 SFT 训练本身成功，主要问题发生在训练前后的 cloud readiness（云端就绪度）：
我们验证了云端训练 smoke（冒烟链路），但没有在占用付费 GPU 前一次性验证依赖、推理服务、
checkpoint runtime、评测数据覆盖和产物路径。因此多个本应在本地或无卡模式暴露的问题，
集中到有卡计费阶段才被发现。

这次成本浪费不能只归因于 AutoDL：

- 平台侧确实观测到有卡模式下载极慢、无卡模式不到 10 分钟完成同一批 vLLM 依赖的显著差异。
- 项目侧也存在上云前门禁不足、依赖未预装、运行时代码缺陷、日志不足和评测集设计不完整。

### 问题分类

| 类别 | 上云后才发现的问题 | 直接影响 | 根因 |
|---|---|---|---|
| 数据与评测 | dev 只有 7 条；35 条 test 全部偏向 `local_search -> answer`，缺少独立的追问、HyDE、Web、拒答路线测试 | 7 条只能做工程验收，35 条即使全跑也不能证明完整 Planner 泛化 | 正式训练前只核对了训练样本数量和 Action 计数，没有把 dev/test 的独立性、数量和路线分布设为硬门禁 |
| 依赖准备 | 正式训练后才发现没有 vLLM；训练环境与服务环境依赖冲突；大批 CUDA wheel 需要临时下载 | GPU 开机后等待依赖，产生无效计费 | 只准备 `.venv-sft`，没有在无卡阶段同时冻结 `.venv-vllm`和完整 cache |
| 网络与缓存 | 有卡模式下载极慢；首次命令使用 `--no-cache`，中断后复用能力差 | vLLM 安装等待超过一小时，下载进度和费用不可控 | 没有提前规定“所有大依赖只能无卡下载”，也没有持久化独立 vLLM uv cache |
| 运行时代码 | checkpoint runtime 缺少模型 `.to("cuda")`等价迁移和输入 tensor `.to(device)` | 4B 模型实际在 CPU 推理，GPU 付费但闲置，十多分钟没有结果 | 训练 smoke 只覆盖 Trainer，没有覆盖正式 checkpoint 的 GPU 推理入口 |
| 可观测性 | dev eval 只在全部完成后打印汇总，没有逐 case 状态 | 正常慢、CPU 卡住、下载和死锁无法快速区分 | 评测入口缺少 `running/completed`、耗时和 Action 路线日志 |
| 运行方式边界 | 误以为 vLLM 服务和 direct-checkpoint dev eval 是同一条链路 | vLLM 占约 41GB 时再加载 checkpoint，存在 OOM 风险 | 文档没有明确“HTTP 服务健康检查”和“Transformers 直接 checkpoint 评测”分别加载模型 |
| 环境变量 | 平台环境遗留 `OMP_NUM_THREADS=0`；无卡运行 `vllm --version`尝试推断 GPU | 出现误导性 warning/error，增加排障时间 | 没有对继承环境变量和无卡 CLI 行为做预检 |
| 产物与路径 | cloud run report 首次传入 `model`子目录而不是 checkpoint 根目录 | 报告找不到 manifest，审计信息不完整 | 没有在脚本结束后自动校验报告中的 checkpoint root、run_id 和 adapter path |
| 磁盘规划 | 数据盘仅余约 15GB，系统盘约 25GB；环境、缓存和产物的归属临时决定 | 安装过程中存在写满磁盘和重复下载风险 | 上云前没有按模型、两套虚拟环境、下载 cache、checkpoint 计算空间预算 |
| 会话管理 | 服务虽然运行，但 `screen -r`找不到对应会话，只能通过 PID 定位和停止 | 长任务管理和停止流程不稳定 | 启动前没有验证任务确实运行在可恢复的 screen/tmux 会话中 |

### 主要成本浪费点

1. 在有卡计费模式首次安装 vLLM 和 CUDA 大依赖。
2. 下载缓慢时继续等待，直到确认无卡模式速度显著更快。
3. vLLM、训练环境和服务启动没有在无卡阶段提前准备并冻结。
4. checkpoint runtime 实际使用 CPU，GPU 处于空闲状态但仍在计费。
5. 训练、HTTP 服务检查和 direct-checkpoint 评测的模型生命周期没有提前拆开。
6. 数据覆盖问题在正式训练后才审计，导致已经得到 checkpoint 才发现独立路线评测不足。

本次没有精确记录每段 GPU 占用时长和金额，因此不能给出可靠的浪费费用数字。下一次必须记录
实例开卡/关卡时间、每个阶段开始/结束时间和异常等待时间，避免事后凭感觉估算。

### 为什么原 smoke 没拦住

原 smoke 证明了以下内容：

- 模型和 tokenizer 能离线加载。
- 4 条训练样本可以完成 1 个 step。
- LoRA checkpoint 和 cloud run report 可以生成。

但它没有覆盖：

- vLLM 服务环境是否已经安装并能加载正式 LoRA。
- checkpoint runtime 是否真的把模型和输入放到 GPU。
- dev/test 是否具备足够数量和独立路线覆盖。
- HTTP 服务与 direct-checkpoint eval 的显存生命周期。
- 逐 case 进度、超时、GPU 空闲和停止条件。
- 两套环境、模型 cache、依赖 cache 和训练产物的完整磁盘预算。

所以 smoke 成功不等于“已经可以无风险地开始完整云端流程”。

### 下次上云前的硬门禁

#### 无卡或本地阶段

1. 统计并冻结 train/dev/test 数量、leakage group（泄漏组）和 Action 路线分布。
2. 训练数据只证明可训练；必须另有覆盖 local、HyDE、Web、追问和拒答的独立 dev/test。
3. 同时建立 `.venv-sft`和 `.venv-vllm`，冻结版本并完成离线 import（导入）检查。
4. 模型、tokenizer、PyTorch/CUDA wheel 和 uv cache 全部提前下载到持久化磁盘。
5. 禁止 `--no-cache`；确认断线后可以复用已完成下载。
6. 执行磁盘预算，环境与 cache 所在盘至少保留安装峰值空间。
7. 检查 `OMP_NUM_THREADS`等继承环境变量，拒绝空值、0 或非法字符串。
8. checkpoint runtime 单元测试必须覆盖 CUDA device、BF16/FP16 和输入 tensor 设备一致性。
9. 评测脚本必须有逐 case 进度、耗时和 Action 路线日志。

#### 有卡阶段

1. 开卡后第一步记录 GPU、驱动、CUDA、显存和开始计费时间。
2. 只做必须依赖 CUDA 的工作：GPU smoke、正式训练、vLLM healthcheck 和模型评测。
3. 禁止在有卡阶段首次下载大模型或安装 vLLM；发现缺依赖立即停卡回无卡处理。
4. vLLM healthcheck 通过后，如果下一步是 direct-checkpoint eval，先停止 vLLM并确认显存释放。
5. 权重加载完成后若连续无进度且 GPU 利用率为 0，立即检查 device placement，不继续盲等。
6. 每一步结束都校验 checkpoint、报告、run_id、adapter path 和输出文件，再决定是否释放 GPU。
7. test 只在模型、配置和评测规则冻结后运行一次；不能把看过结果的 test 继续当最终留出集。

### 已完成整改

- [x] 正式 LoRA SFT、checkpoint 和 cloud run report 已完成并核对。
- [x] vLLM 改为独立环境，并在无卡模式完成安装与版本冻结。
- [x] 去掉 `--no-cache`，增加持久化 cache、超时和重试配置。
- [x] vLLM CUDA 门禁、LoRA 加载、HTTP `/health`和 PlannerClient healthcheck 已通过。
- [x] 修复 checkpoint runtime 的模型/输入 CUDA 设备迁移和 BF16/FP16 加载。
- [x] dev eval 增加逐 case 日志，完整 7 条 dev 已成功运行。
- [x] 区分 vLLM HTTP 服务验证和 direct-checkpoint dev eval。
- [x] 新增分层 runtime preflight（运行时前置检查），自动检查 `OMP_NUM_THREADS`、离线开关、
  磁盘、checkpoint/adapter 身份、vLLM/PyTorch/CUDA 版本、模型本地缓存和端口。
- [x] vLLM 启动入口强制先过 GPU preflight，缺依赖、缺缓存或身份不一致时不再启动长任务。
- [x] 新增六类 Action HTTP probe（动作接口探针）和 Web 禁用边界；工程连通性与
  9.3.11 模型质量判断分开。
- [x] 新增一键 GPU 验收入口，自动启服、等待 `/health`、运行探针并停止本次服务释放显存。
- [ ] 在下一次有卡窗口实际生成 `cloud_smoke_planner_http.json`；本地代码和测试通过不替代真实 GPU 结果。
- [ ] 在无卡业务环境配置就绪后生成真实 Web Provider observation（网页执行器观察记录）。
- [ ] 补充独立且路线均衡的 dev/test；当前 35 条 test 不能代表完整 Planner 路线。
- [ ] 冻结新的评测边界后再做正式 test 和 9.4 baseline compare（基线对比）。
- [x] checkpoint、cloud run report、dev eval 和环境冻结清单已归档到本地长期目录；
  归档整体 SHA256、manifest 和 26 个内部文件逐文件 SHA256 均通过。

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
| 开卡后才发现 checkpoint、缓存或端口问题 | 过去只有人工命令，没有同一份结构化门禁 | 开卡后第一条业务命令运行 `run_gpu_acceptance_gate.sh`；preflight 任一层失败立即停卡 |
| 单条 healthcheck 被误当成完整 smoke | healthcheck 只证明一个输入能走通 HTTP，不能覆盖全部 Action 和 Web 权限 | 固定运行 7 个 HTTP 探针：六类 Action 各一条，外加 Web 禁用边界 |
| Action 不匹配导致工程验证和模型质量混在一起 | 协议故障与模型路由错误使用了同一个成败口径 | 9.3.10 默认只卡 HTTP/解析/model_id/Web 权限；Action 命中率留给 9.3.11 分析 |
| expanded dev 无卡入口缺少 `loguru` | 训练锁文件没有覆盖项目日志模块依赖 | 在无卡模式临时补装；9.3.17 更新 requirements/lock，并让 preflight import 正式入口 |
| 重启后 freeze 出现 `transformers=4.57.6` | 未重新 `source env.local`，`$PYTHON_BIN`为空，uv 选择主项目 `.venv` | freeze 前打印解释器路径和版本；与 checkpoint manifest、lock 三方交叉校验 |
| HyDE 第一次检索已有 0.95 分目标证据却仍被要求 HyDE | Snapshot Provider 用 expected label 直接构造高分候选，路线前置条件与评分要求冲突 | 修复 Provider/Observation 契约或使用真实检索 observation replay，不能按该结果直接补模型数据 |
| 安全拒绝 case 本地检索为空 | Provider 把 `should_answer=false`错误等价为“不返回本地证据” | 返回手册安全警告后再判断模型是否拒绝，当前 5 条不能准确归因 |

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

1. [x] 9.3.17 已完成：`.venv-sft` 依赖、正式入口导入、解释器身份和
   SFT environment freeze（监督微调环境冻结）均已闭环。
2. [x] 9.3.18 已完成：HyDE（假设文档嵌入）与安全拒绝的 Provider/Observation
   （动作执行器/观察结果）回放契约已修复并冻结。
3. [x] 9.3.19 已完成：Reward v1.1（奖励函数第一点一版）回归通过，原权重和实现保持不变。
4. [ ] 9.3.20 的 SFT v1（监督微调第一版）独立 Replay（回放）入口已在本地补齐并通过
   43 项相关测试；待云端先完成无卡 preflight（运行前检查），再由用户开启
   GPU（图形处理器）执行正式校正复评。
5. 根据校正后真实失败执行 9.3.21～9.3.23；在准入通过前不运行 heldout、不进入 9.4。
6. 已完成 SFT v1 checkpoint 和首次 expanded dev 归档；后续运行不得覆盖原始 run_dir 和归档。
