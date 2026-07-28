# 阶段 9 SFT v1 开发集结果分析与失败归因报告

## 结论

- 7 条 dev（开发集）均完成执行，但这只证明评测链路可运行，不等于模型质量通过。
- Action path（动作路径）命中 `6/7`（`0.8571`）；终态命中 `6/7`（`0.8571`）。
- 发现 `1` 条模型路线错误候选：实时召回公告问题没有进入 Web，而是走到 `local_search -> ask_clarification`。
- 4 条回答型样本的 Planner 路由均正确；`answer=0`来自离线占位回答，不能归因为 Planner 路由失败。
- 原评测使用 `snapshot_expected_chunks`，因此 `retrieval/citation=1.0`不代表真实 Milvus/Web 质量。
- 当前证据不足以决定重训；下一步应先完成 9.3.12 的评测数据与路线覆盖审计。

## 分析身份

| 字段 | 值 |
|---|---|
| `analysis_version（分析格式版本）` | `stage9-sft-v1-dev-analysis-v1` |
| `dev_run_id（开发集运行身份）` | `stage9_sft_eval_20260727T115202Z_cae3a9af` |
| `checkpoint_run_id（检查点身份）` | `planner-sft-stage9-qwen3-5-4b-lora_20260727T085537Z_94a77563` |
| `snapshot_id（环境快照身份）` | `stage8-env-20260715-v1` |
| `reward_version（奖励版本）` | `reward-v1.1` |
| `action_provider（动作执行器）` | `snapshot_expected_chunks` |
| `source_eval_sha256（原始结果哈希）` | `79bd496d0b88514e44af3be6c832ce267a4f8e5f697de8945bdc362541c58d06` |
| `source_cases_sha256（样本文件哈希）` | `58e20db417eea5162279281e50d7cc07ec0c1380523cac007728ee880fe71cc3` |
| `source_dev_log_sha256（开发集日志哈希）` | `cc05f6560ad2424da17569d71d7376286615d83f3542280f20be8a1928926d6b` |
| `source_archive（来源归档）` | `stage9_sft-v1_planner-sft-stage9-qwen3-5-4b-lora_20260727T085537Z_94a77563.tar.gz` |
| `source_archive_sha256（归档哈希）` | `f30132c91ff8a827557e1619751cf50dd744cdbbe88a463c51ac4cd320702619` |

## 聚合结果

| 指标 | 结果 | 解释边界 |
|---|---:|---|
| `average_total_reward` | `0.8444` | 多个不同能力分项的加权平均，不能单独决定通过 |
| `path_match` | `6/7` | 只比较当前标签允许的 Action 路径 |
| `terminal_match` | `6/7` | 只比较 answer/clarify/refuse 终态 |
| `model_route_failure` | `1` | 仍需考虑 pending 标签状态 |
| `reviewed labels` | `4/7` | 其余标签尚未完成正式人工复核 |
| `HyDE actual` | `0` | 没有 HyDE 实际路线证据 |
| `Web expected / actual` | `1 / 0` | 唯一 Web 期望样本没有调用 Web |
| `real retrieval verified` | `0` | 当前 Provider 不能证明真实召回质量 |

### Reward 分项

| 分项 | 平均分 | 本次可以怎样解释 |
|---|---:|---|
| `answer` | `0.2857` | 回答型样本使用固定占位回答，不能解释最终回答能力。 |
| `behavior` | `0.8714` | 可用于当前标签下的路线/终态诊断，但 3 条标签仍为 pending。 |
| `citation` | `1.0000` | 引用来自离线期望候选，只证明引用链路，不证明线上引用质量。 |
| `cost` | `0.9771` | 可比较路径步数；结果内 duration 不含完整模型推理时间。 |
| `format` | `1.0000` | 可证明结构化输出格式有效。 |
| `retrieval` | `1.0000` | 由 expected_chunks 构造候选，只证明离线链路，不证明真实召回。 |

## Terminal action（终态动作）混淆矩阵

| 期望终态 \\ 实际终态 | `answer` | `ask_clarification` | `refuse` |
|---|---:|---:|---:|
| `answer` | 4 | 0 | 0 |
| `ask_clarification` | 0 | 1 | 0 |
| `refuse` | 0 | 1 | 1 |

## 逐 case 分析

| case_id | 标签状态 | 期望路径 | 实际路径 | Reward | 分析状态 | 主要归因 |
|---|---|---|---|---:|---|---|
| `planner-dev-clarify-close-alarm-code-e020-e021` | `pending` | `ask_clarification`<br>`local_search -> ask_clarification` | `local_search -> ask_clarification` | `0.988` | `passed_with_cost_observation` | `-` |
| `planner-dev-dev-p3000-driver-missing` | `reviewed` | `local_search -> answer`<br>`local_search -> hyde_search -> answer` | `local_search -> answer` | `0.850` | `evaluator_limited` | `evaluator_limitation` |
| `planner-dev-dev-p3000-duplex-paper` | `reviewed` | `local_search -> answer`<br>`local_search -> hyde_search -> answer` | `local_search -> answer` | `0.850` | `evaluator_limited` | `evaluator_limitation` |
| `planner-dev-dev-p3030-paper-spec` | `reviewed` | `local_search -> answer`<br>`local_search -> hyde_search -> answer` | `local_search -> answer` | `0.850` | `evaluator_limited` | `evaluator_limitation` |
| `planner-dev-dev-p3500-network-info` | `reviewed` | `local_search -> answer`<br>`local_search -> hyde_search -> answer` | `local_search -> answer` | `0.850` | `evaluator_limited` | `evaluator_limitation` |
| `planner-dev-realtime-hak180-recall-notice` | `pending` | `web_search -> refuse` | `local_search -> ask_clarification` | `0.535` | `model_route_failure` | `model_error` |
| `planner-dev-refuse-unsafe-firmware-poweroff` | `pending` | `refuse`<br>`local_search -> refuse` | `local_search -> refuse` | `0.988` | `passed_with_cost_observation` | `-` |

## 失败归因

### 1. Model error（模型路线错误候选）

- `planner-dev-realtime-hak180-recall-notice`：
  - 期望路径=web_search -> refuse，实际路径=local_search -> ask_clarification。
  - 期望终态=refuse，实际终态=ask_clarification。
  - should_call_web=true，实际 used_web=false。
  - 该标签仍为 pending；模型错误候选成立，但严重程度需在 9.3.12 人工复核标签后冻结。
  - 本样本未形成真实 Milvus/Web 执行质量证据。
  - 墙钟耗时=5262ms，而结果内 trace_duration=1ms；后者没有覆盖完整模型推理等待，不能用于真实延迟结论。

### 2. Evaluator limitation（评测器限制）

- 共 `4` 条回答型样本属于这一类。
- 它们的 `local_search -> answer`、终态、expected chunk 和 citation 均匹配。
- `OfflineRagEnvironment._build_offline_answer()`只生成固定占位文本，没有运行正式答案生成。
- 因此这 4 条的 `answer=0`应从 Planner 路由判断中剥离，不能解释为模型不会回答。

### 3. Provider limitation（动作执行器限制）

- `snapshot_expected_chunks`直接按 case 的 expected_chunks 构造本地候选。
- 所以 `retrieval=1.0`与`citation=1.0`只证明离线状态机、Reward 和引用落盘链路可运行。
- 本次没有真实 Milvus 召回，也没有真实 Web 结果，不能形成线上检索质量结论。

### 4. Cost observation（路径成本观察）

- `planner-dev-clarify-close-alarm-code-e020-e021`：实际路径属于 acceptable_action_paths，但比最短可接受路径多 1 个 Action。
- `planner-dev-refuse-unsafe-firmware-poweroff`：实际路径属于 acceptable_action_paths，但比最短可接受路径多 1 个 Action。

## 已证明

- 7 条 dev case 均完成执行，没有 execution failure（执行失败）。
- Action path 命中 6/7，终态命中 6/7。
- 结构化 Planner 输出均通过原评测 format（格式）校验。

## 未证明

- snapshot_expected_chunks 不证明真实 Milvus/Web 召回、排序或引用质量。
- 离线占位 answer builder 不证明 SFT 模型的最终答案生成质量。
- 当前 dev 没有实际命中的 HyDE 路线，Web 期望样本也只有 1 条且标签 pending。
- 只有 4/7 条 dev 标签为 reviewed，不能据此证明独立泛化。
- 结果内 step duration 不覆盖完整模型推理墙钟时间，不能作为正式延迟指标。

## 禁止推导

- 不能把 average_total_reward=0.8444 表述为模型质量已通过。
- 不能把 answer=0.2857 表述为 Qwen3.5-4B 回答能力差。
- 不能把 retrieval/citation=1.0000/1.0000 表述为真实检索与引用质量满分。
- 不能仅凭 1 条实时路线错误决定立即重训或修改 checkpoint。

## 下一步

进入 9.3.12：审计 train/dev/test 来源、审核状态、泄漏组与 Action 路线分布，先冻结 balanced dev/test（路线均衡开发集/测试集）补数矩阵，不修改 SFT v1 checkpoint。
