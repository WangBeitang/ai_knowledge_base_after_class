---
name: efficient-data-workflow
description: Handle this repository's dataset construction, JSONL case repair, evidence alignment, split and manifest updates, Provider replay, evaluation artifacts, blind review, fingerprints, SFT data, and other data-heavy tasks with an evidence-first minimum-change workflow. Use when work touches evaluation/stage8, evaluation/stage9, generated cases, chunks, snapshots, review bundles, retrieval probes, SHA256, or when a data task risks repeated regeneration, broad testing, documentation churn, or slow external-service calls.
---

# 高效数据处理工作流

## 目标

用尽可能少的读取、改动、重建和验证，得到可信的结论或产物。

把“高质量”理解为更早定位正确责任层、减少无关改动并在门禁处及时停止，而不是增加流程、报告和测试数量。

## 先确定本轮唯一目标

开始前用一句话写清：

- 本轮要回答的一个问题或交付的一个产物。
- 本轮明确不做的后续事项。
- 完成条件。
- 停止条件，例如需要独立审核、真实服务、GPU 或用户决策。

一轮只推进一个目标。不要顺手进入下一阶段，也不要为尚未批准的未来状态修改下游代码。

## 先做一次只读诊断

先检查工作树，保护用户已有修改。然后一次性找出：

- `source of truth（事实源）`：人工维护的 case、recipe、配置、原始文档或审核决定。
- `generator（生成器）`：负责构造 JSONL、manifest、snapshot、review bundle 或报告的脚本。
- `generated artifact（生成产物）`：可以由事实源和生成器重建的文件。
- `gate（门禁）`：独立审核、外部服务、GPU、正式冻结或用户决策。
- 已有测试、历史失败证据和当前脏文件。

不要看到某个 JSONL 错误就直接编辑。先判断它是事实源还是生成物；生成物默认修生成器或上游事实源，再统一重建一次。

## 先输出失败矩阵

涉及多条 case 时，在修改前先汇总成一张紧凑表：

| case_id | 期望 | 实际 | 责任层 | 直接证据 | 最小下一步 |
|---|---|---|---|---|---|

矩阵必须覆盖全部当前失败，不能逐条边读边修。相同根因合并处理。

若两轮只读检查后仍不能归因，停止扩散读取，设计一个能区分主要假设的最小 probe（探针）。

## 按责任层归因

使用以下顺序定位，不要一看到评测失败就补训练数据：

1. `Case（评测题）`：query、目标路线、答案要点、证据或安全边界是否自洽。
2. `Provider（动作执行器/环境结果提供器）`：是否执行或回放了正确动作。
3. `Observation（观察结果）`：候选、分数、错误码和证据正文是否忠实反映执行结果。
4. `Evaluator（评测器/裁判）`：是否按 case 契约、动作路径、终态和证据正确判分。
5. `retrieval/runtime（检索与运行环境）`：Milvus、Mongo、Web、索引版本和配置是否一致。
6. `training/model（训练数据或模型）`：只有前五层成立后，才判断训练覆盖或模型决策问题。

调用顺序是：

`Planner（规划器）选择 Action（动作） -> Provider 执行/回放 -> 返回 Observation -> Planner 继续决策 -> Evaluator 判分`

明确错误发生在哪一层。禁止通过调 Reward（奖励分数）、降低阈值或给 query 塞答案关键词来掩盖上游契约错误。

## 只做最小修复

每类根因只选择一个最小修复点：

- Case 错：修事实源 case，并说明旧 fingerprint（内容指纹）为何失效。
- 生成器错：修生成器和一个针对性测试，不手改生成物。
- Provider/Observation 错：修执行或回放契约，保留旧记录作为失败证据。
- Evaluator 错：修判分语义，并用正例和反例各证明一次。
- 检索环境错：修配置、索引或录制输入，不改模型标签。
- 训练覆盖不足：只在独立证据成立后补训练样本，并保持来源和证据可追溯。

不在核心修复尚未证明时同步更新所有文档、报告、清单和下游测试。

## 使用递进验证

按以下层级执行，到足以证明当前目标时停止：

1. 静态校验：JSON/JSONL、schema、路径、ID、hash 和导入。
2. 单条 probe：每类根因选一条代表 case。
3. 受影响子集：只跑本轮改变涉及的 case。
4. 全量重建：前三级通过后只执行一次。
5. 阶段测试：正式完成前只执行一次必要的 broader test（较广测试）。

同一输入、命令和环境已经得到确定结果时，不重复运行。只有结果矛盾、环境改变或需要复现证据时才重跑。

调用 Milvus、Mongo、Web、LLM、GPU 或云端服务前，先完成本地静态校验和最小子集验证。外部服务一旦不连通，记录阻塞点并停止，不用替代数据伪造通过。

## 控制重建和产物

- 只在事实源和生成器稳定后重建一次。
- 保留本轮前的历史失败证据，不覆盖原始运行。
- 历史产物只封存，不进入当前门禁；只有任务明确引用它时，才按最小范围重新激活和校验。哈希在冻结、传输和重新激活时检查，不在每轮任务重复检查。
- 不因旧产物仍在仓库中，就把旧报告、旧决定、旧脚本或它们的递归哈希链加入当前 preflight（运行前检查）、测试或输入清单。只有新旧对比、审计、复现、回滚或当前结论直接继承旧决定时，历史产物才是当前输入。
- 重新激活历史产物时，只校验本轮结论实际依赖的最小身份；例如对比评测只确认旧评测、模型 checkpoint（检查点）和必要配置，不递归验证更早的无关产物。
- 只生成任务验收明确要求的产物。
- 不为一次性诊断新增通用脚本；同一操作确认会复用至少三次或属于正式入口时再抽取。
- 不创建重复 README、总结或临时报告；优先更新唯一正式文档。
- 不用目录名、时间戳或相似文件名推断身份，读取 manifest 和内容确认。

只对本轮实际使用的当前输入，或任务明确重新激活的历史输入，按结论所需核对以下必要身份；
不要因为字段存在就默认全部校验：

- `case_id`
- `document_id`
- `chunk_id`
- `snapshot_id`
- `index/config version（索引/配置版本）`
- `case_fingerprint`
- `SHA256`

## 审核和 fingerprint 门禁

query、证据、答案要点、目标路线或 Observation 契约发生变化时：

1. 判定旧审核决定对新 fingerprint 无效。
2. 保留旧决定和旧 fingerprint，不静默覆盖。
3. 只导出字段白名单的 clean blind review bundle（干净盲审包）。
4. 做污染扫描、fingerprint 复算和包 SHA256 校验。
5. 到独立审核门禁立即停止，等待外部结果。

审核 pending（待审）期间，不为了让旧全量测试继续通过而放宽门禁或改写下游断言。报告当前“多少 reviewed（已审核）+ 多少 pending”，并停在这里。

## 区分诊断、冻结和能力结论

- probe 只用于归因，不等于正式数据冻结。
- snapshot/replay 只证明固定环境下可复现，不等于真实线上检索质量。
- train-only 数据和开发集结果不等于 heldout（留出集）泛化。
- Provider 或 Evaluator 缺陷不等于模型失败。
- SFT（监督微调）拟合固定路线；新路线探索或策略权衡属于后续策略优化问题，不能靠改测试解释。

## 进度沟通

工具执行期间只汇报四类信息：

- 已确认的事实。
- 当前唯一假设。
- 正在运行的最小验证。
- 明确停止条件。

避免重复叙述背景、长篇预告和无变化的状态更新。

## 完成交付

最终只交付：

- 结论：问题属于哪一层，是否已闭环。
- 改动：事实源、生成器或契约改了什么。
- 验证：实际运行的最小验证及结果。
- 边界：哪些结论不能由本轮证据推出。
- 下一步：只给一个最近的动作；若已到门禁，明确由谁继续。

若发现任务开始扩大，回到“本轮唯一目标”和失败矩阵，删除与完成条件无关的工作。
