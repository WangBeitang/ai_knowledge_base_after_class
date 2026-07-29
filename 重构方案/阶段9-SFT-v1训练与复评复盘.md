# 阶段 9 SFT v1 训练与 expanded dev 复评复盘

## 文档定位

本文记录阶段 9 第一轮正式 SFT（监督微调）、云端运行、25 条 balanced dev（均衡开发集）
复评、失败归因和后续整改原则。

它不是新的训练配置，也不替代：

- [`阶段9.md`](./阶段9.md)：阶段任务、执行顺序和验收标准。
- [`AUTODL_SFT_GUIDE.md`](../deploy/cloud_sft/AUTODL_SFT_GUIDE.md)：AutoDL 可复制操作步骤。
- [`AUTODL_SFT_CLOUD_ISSUES.md`](../deploy/cloud_sft/AUTODL_SFT_CLOUD_ISSUES.md)：云端环境和平台坑点。
- 云端生成的 `阶段9-SFT-9.4准入报告.md`：本次 25 条逐 case 正式结果。

本文只使用已经冻结和校验的证据，不把 snapshot（快照）、离线占位回答、dev 结果或训练 loss
扩大解释为真实检索质量、真实回答质量、heldout 泛化或上线能力。

## 一、已验证事实

### 1. 正式 SFT v1

- 基础模型：`Qwen/Qwen3.5-4B`。
- 训练方法：LoRA（低秩适配）。
- `run_id`：
  `planner-sft-stage9-qwen3-5-4b-lora_20260727T085537Z_94a77563`。
- 训练记录：155 个 Action step（动作步骤）、70 个 source trajectory（来源轨迹）、
  1 个 epoch（训练轮次）、20 个 optimizer step（优化器步数）。
- `train_loss=0.2804`；该值只证明训练数据拟合过程完成，不证明独立质量。
- checkpoint、adapter、tokenizer、训练配置、训练指标和 manifest 均已生成并完成首次异地备份。
- checkpoint manifest 冻结的主要环境版本为：
  `python=3.12.3`、`torch=2.10.0`、`transformers=5.9.0`、
  `peft=0.18.1`、`datasets=4.8.4`、`bitsandbytes=0.49.2`。

### 2. 2026-07-29 expanded dev 正式复评

- 评测集：25 条 reviewed balanced dev，五个路线桶各 5 条。
- heldout test 推理数量：0。
- Provider：`snapshot_expected_chunks`。
- Reward：v1.1。
- 格式合法率：1.0。
- execution failure（执行失败）：0。
- forbidden action（禁止动作）：0。
- 路线正确：8/25。
- overall route accuracy（总体路线准确率）：0.32。
- macro accuracy（路线宏平均准确率）：0.32。
- 准入决定：`reject_sft_v1_train_sft_v2`。

五路线结果：

| route bucket（路线桶） | 正确/总数 | 准确率 | 当前归因可信度 |
|---|---:|---:|---|
| `local_answer` | 5/5 | 1.00 | 可归因：模型路线正确 |
| `ask_clarification` | 3/5 | 0.60 | 可归因：2 条模型路线偏差 |
| `hyde_fallback` | 0/5 | 0.00 | 不可直接归因：Provider 前置条件矛盾 |
| `web_required` | 0/5 | 0.00 | 可归因：训练路线零覆盖 |
| `safe_refuse` | 0/5 | 0.00 | 暂不能准确归因：安全证据未返回 |

### 3. 归档身份

- freeze version：`stage9-sft-artifact-freeze-v2`。
- 文件数：29。
- 归档大小：82,112,604 bytes。
- 归档整体 SHA256：
  `0f58738d92fb81f8f7bb3010c39cb7be95436947ed04ae4886b57cc82a106b44`。
- 云端归档校验、本地下载校验、外部 manifest 与归档内 manifest 对比、内部 29 文件逐文件
  SHA256 均已通过。

因此，SFT v1 checkpoint 和本次复评结果都应继续保留，不能覆盖或删除；它们是后续修复前后的
对照基线。

## 二、为什么 17 条失败不能全部算成模型失败

一次离线 Planner（规划器）评测的实际链路是：

```text
Planner 选择 Action
-> Provider 执行检索 Action
-> Environment 生成 Observation
-> Planner 根据 Observation 继续决策
-> Evaluator 对完整 Action path 和终态评分
```

只有 Provider 给出的前置事实真实、充分且与 case 路线一致，Evaluator 的错误归因才成立。

### 1. HyDE 的 evaluator false negative

冻结 case 想验证的是：

```text
local_search 未命中或证据低分
-> hyde_search 扩展查询后命中
-> answer
```

但 `SnapshotExpectedChunkActionProvider.local_search` 对所有
`expected_behavior.should_answer=true` 的 case，第一次本地检索就直接返回
`expected_chunks`。候选构造还会给予首个目标 chunk：

```text
retrieval_score=0.95
rerank_score=0.95
content=expected_answer_points
```

也就是说，模型实际看到的是高分目标证据和答案摘要。它选择
`local_search -> answer` 符合当前 Observation，却因为没有继续 HyDE 被评分器判错。

结论：5 条 HyDE 是明确的评测前置条件冲突，不能作为“模型不会 HyDE”的充分证据。

### 2. 安全拒绝的 Provider limitation

安全拒绝 case 想验证的是：

```text
local_search 找到手册安全警告
-> Planner 识别禁止操作或风险边界
-> refuse
```

但当前 Snapshot Provider 的本地检索实现把
`expected_behavior.should_answer=false` 直接解释为“返回空候选”。安全拒绝 case 因此看不到：

- 带电蜂鸣测试的触电风险；
- COM 对地超过 500V 的禁止条件；
- 10A 档持续时间上限；
- 打印过程中强行拉纸的设备损坏风险；
- 定影区域高温和冷却要求。

模型在空检索结果后输出 `ask_clarification` 仍然不是期望路线，但当前运行不能区分：

- 模型拿到正确安全警告后仍不会拒绝；
- 模型只是因为没有得到证据而选择澄清。

结论：5 条安全拒绝暂时无法准确归因，必须修 Provider 后再测。

### 3. 为什么淘汰 SFT v1 的结论仍然成立

即使把 HyDE 和安全拒绝两个有争议路线桶都按满分计算：

```text
local_answer       = 1.0
hyde_fallback      = 1.0
safe_refuse        = 1.0
ask_clarification  = 0.6
web_required       = 0.0

macro = (1.0 + 1.0 + 1.0 + 0.6 + 0.0) / 5 = 0.72
```

0.72 仍低于冻结门槛 0.80，且 Web 路线桶仍未达到每桶最低 0.60。因此：

- “SFT v1 不允许进入 9.4”保持有效；
- “SFT v1 有 17 条真实模型错误”不成立；
- 后续补数据前必须先修正失败归因。

## 三、训练数据暴露的问题

### 1. 155 条不是 155 个独立问题

155 是 Action step 数，不是独立 query 数。它们来自 70 个 source case，其中 50 个 route seed
只有 26 个唯一 query。大量不同 `case_id` 实际复用了相同问题模板。

以后训练数据必须同时报告：

- Action step 数；
- 独立 trajectory 数；
- 唯一 query 数；
- 唯一 leakage group 数；
- 设备族和来源文档数；
- 每条路线的语义类型数。

不能再单独使用“样本总行数”表达数据规模或覆盖能力。

### 2. Web 评测要求与训练覆盖不一致

本次训练的 10 个 `web_search` target 只有 2 种唯一 query，轨迹都以 `refuse` 收口。
训练集中没有一条 `web_search -> answer`，而 expanded dev 的 5 条 Web case 全部要求：

```text
web_search -> answer
```

这是可以在训练前通过静态路线审计发现的零覆盖，不需要等 GPU 评测后才确认。

### 3. 澄清样本重复且语义单一

10 个 `ask_clarification` target 只有 2 种唯一 query，主要覆盖：

- 完全泛化的“这个报警怎么处理”；
- E020/E021 是否能按同一故障处理。

它没有充分覆盖缺少设备型号、缺少部件、缺少操作阶段、未知按键布局等自然场景，所以
“未知华为打印机型号”和“未知身份证复印面板”两条 case 出现绕行或错误终态。

### 4. HyDE route seed 泄漏了路线答案

现有 HyDE 训练 query 常直接包含：

```text
本地初次检索证据不足时应怎样继续确认？
```

模型容易把这句话当作选择 HyDE 的文本暗号，而不是根据
`top_rerank_score`、候选内容、identifier match（标识命中）和 evidence threshold
（证据阈值）判断是否需要 fallback（回退）。

同时，历史 route seed 的 Observation 形状与直接 answer 样本高度接近；若缺少真实分数和内容差异，
模型无法从可观察事实学习正确切换条件。

### 5. 安全拒绝语义覆盖偏离正式 dev

训练中的拒绝样本大量集中于：

- 隐藏维修密码；
- 未授权私有维修记录；
- Web/HyDE 空结果后的安全收口。

它缺少不同设备、不同风险等级、不同手册警告驱动的物理安全操作。因此即使修复 Provider 后，
安全拒绝仍可能需要补独立 train-only 数据，但必须以修正后的复评结果为依据。

## 四、云端与工程流程经验

### 1. GPU 开机前必须导入正式入口

仅检查 `torch/transformers/peft` 已安装不够。本轮无卡 preflight 第一次执行正式 expanded dev
入口时，发现 `.venv-sft` 缺少 `loguru`；原因是项目评测模块会导入统一日志模块，而训练依赖锁
没有包含它。

正确门禁应是：

```text
正式 PYTHON_BIN 身份正确
-> 正式入口完整 import
-> checkpoint/data/hash/offline 配置通过
-> 才申请 GPU
```

### 2. 每次重启都要重新 source env.local

无卡模式重新开 shell 后，`$PYTHON_BIN` 曾为空。随后执行
`uv pip freeze --python "$PYTHON_BIN"` 时，uv 自动选用了主项目 `.venv`，第一次错误记录为
`transformers=4.57.6`。

重新 `source deploy/cloud_sft/env.local` 后，正式 `.venv-sft` 的实际版本为
`transformers=5.9.0`，与 checkpoint manifest 和锁文件一致。

以后生成环境 freeze 前必须先打印：

```text
PYTHON_BIN
sys.executable
python/torch/transformers/peft/loguru 版本
```

变量为空或解释器路径不符时立即停止。

### 3. 依赖安装和模型缓存属于无卡工作

vLLM、CUDA wheel、模型、tokenizer 和 uv cache 必须在无卡模式准备并持久化。有卡模式只做：

- CUDA 门禁；
- 必须依赖 GPU 的 smoke；
- 正式训练；
- 模型服务 healthcheck；
- 正式模型推理。

发现缺包、缺缓存、路径或哈希错误时应立即释放 GPU，不能边计费边修环境。

### 4. direct checkpoint runtime 与 vLLM 不是同一条链路

- 训练和本次 25 条 expanded dev 使用 Transformers direct checkpoint runtime。
- vLLM 用于 HTTP 模型服务和上线形态验证。
- 两者都会加载模型，不能同时占用同一张 48GB GPU。

训练或 direct eval 不依赖 vLLM；vLLM healthcheck 通过后必须停服并确认显存释放，再运行
direct checkpoint eval。

### 5. 长任务必须有后台会话和逐 case 日志

本轮正式评测通过独立 `screen` 会话运行，逐 case 输出 running/completed、耗时和 Action path。
screen 结束只表示子进程退出；门禁不通过时脚本会保存产物并以退出码 3 结束，不能看到
`[screen is terminating]` 就重复启动。

应先读取唯一 run directory 的日志和 decision，再决定是否重跑。

### 6. 产物必须在释放云端前完成异地校验

本轮先在云端生成：

- 归档；
- 外部 manifest；
- 归档 SHA256；
- 归档内 `_freeze/SHA256SUMS.txt`。

下载到本地后又完成：

- 外层 SHA256 校验；
- 外部/内部 manifest diff；
- 内部 29 文件逐文件 SHA256。

只有以上检查全部通过，云端数据盘才不再是唯一副本。

## 五、最终经验教训

1. 训练行数不等于独立问题数，更不等于路线语义覆盖。
2. 评测 Provider 必须忠实提供路线成立所需的前置事实。
3. 不得用 `should_answer` 直接决定“检索是否返回证据”；终态标签和检索事实是两个维度。
4. 先区分模型错误、训练覆盖不足、Provider 限制、Evaluator 误判和 label 问题。
5. 训练前必须静态检查每条正式评测路线是否存在对应训练轨迹。
6. Reward 正确暴露问题时不能为了通过门槛而调权重。
7. balanced dev 用于诊断和模型选择，heldout test 必须继续密封。
8. GPU 前置门禁必须使用正式解释器导入正式入口，而不是只检查若干包存在。
9. 环境 freeze 前必须输出解释器身份，避免 uv 静默选择错误虚拟环境。
10. 失败的 SFT v1 仍然有价值：它证明训练、checkpoint、结构化输出、GPU 推理和归档链路可用，
    并把真正的问题集中到训练覆盖和评测契约，而不是证明整条技术路线失败。

## 六、后续决策

9.4 前的固定执行顺序为：

1. 9.3.17：修复 SFT 云端依赖与无卡 preflight。
2. 9.3.18：修复 Provider/Observation 契约并冻结真实或可审计回放记录。
3. 9.3.19：重跑 evaluator 与 Reward v1.1 回归，默认不改权重。
4. 9.3.20：保持 checkpoint/dev/Reward 不变，复评 SFT v1 并校正真实失败数。
5. 9.3.21：只针对修正后仍成立的失败补独立 train-only 数据。
6. 9.3.22：完成 SFT v2 数据独立审核、去重、泄漏门禁和冻结。
7. 9.3.23：训练 SFT v2 并重新执行 expanded dev 准入门禁。

以下行为明确禁止：

- 直接把 25 条 balanced dev 或 heldout test 原题、近义改写、query variant 写入训练集。
- 在修 Provider 前按 17 条失败直接补训练数据。
- 为让 SFT v1 通过而降低 0.80 门槛或调整 Reward 权重。
- 用 snapshot provider 的结果宣称真实 Milvus/Web 召回质量或真实答案质量。
- 在 `eligible_for_stage9_4=true` 前执行 9.4 或运行 heldout test。
