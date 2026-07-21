# 阶段 8.5 高置信 gold 样本复审提示词

你是一个独立 RAG 数据审核 agent。你的主要任务是复核已经重写的 20 条 source-grounded gold，确认每个标准答案要点都被 UCI 官方来源直接支持。不要只检查 JSON schema，也不要默认同意上一轮 agent 的结论。

## 背景

项目先生成了 52 条公开数据候选，随后发现这些候选混入了来源未直接支撑的维修动作和根因。当前已基于 UCI 官方规则与 profile 标签重写 20 条独立 gold。注意：`schema_approved_cases.jsonl` 只代表旧流程的基础门禁结果；本轮主要复核目标是 `gold_cases_authoring.jsonl`。

## 需要阅读的文件

- `evaluation/stage8_5/artifacts/intermediate/public_candidate/sources/source_manifest.jsonl`：公开来源清单，每行一个 source，包含 `source_id`、标题、URL、许可证状态和备注。
- `evaluation/stage8_5/artifacts/intermediate/public_candidate/sources/license_manifest.jsonl`：许可证清单，用来判断来源是否允许训练或再分发派生数据。
- `evaluation/stage8_5/artifacts/intermediate/public_candidate/fault_scenario_cards.jsonl`：故障场景卡片。它是原始公开资料和 `PlannerEvalCase` 之间的中间层。
- `evaluation/stage8_5/artifacts/intermediate/public_candidate/public_documents_manifest.jsonl`：本地卡片化摘要被视作哪些文档，每条 document 对应哪个 source。
- `evaluation/stage8_5/artifacts/intermediate/public_candidate/chunk_source_map.jsonl`：每个 expected chunk 对应哪个 source/card/document。
- `evaluation/stage8_5/artifacts/intermediate/public_candidate/planner_case_candidates.jsonl`：完整 52 条候选 case。
- `evaluation/stage8_5/artifacts/review/public_candidate/schema_approved_cases.jsonl`：通过基础门禁的 reviewed 样本，但不等于 Gold。
- `evaluation/stage8_5/artifacts/review/public_candidate/review_queue.jsonl`：当前流程中仍待审核的样本。
- `evaluation/stage8_5/artifacts/review/curated_gold/gold_review_decisions.jsonl`：已有一轮高置信 Gold 审核结果，可作为参考，但你需要独立判断。
- `evaluation/stage8_5/artifacts/intermediate/curated_gold/gold_evidence_documents.jsonl`：2 个 Gold 证据文档清单。
- `evaluation/stage8_5/artifacts/intermediate/curated_gold/gold_evidence_chunks.jsonl`：10 个来源证据 chunk，包含 UCI URL、页面定位、中文证据摘要和原子事实 `fact_id`。
- `evaluation/stage8_5/artifacts/intermediate/curated_gold/gold_cases_authoring.jsonl`：本轮主要审核对象，共 20 条重写后的 `PlannerEvalCase`。
- `evaluation/stage8_5/artifacts/review/curated_gold/gold_case_audit.jsonl`：20 条逐点审计记录，包含旧 case 映射、答案点到 fact ID 的映射和已删除内容。
- `evaluation/stage8_5/artifacts/review/curated_gold/gold_rewrite_report.md`：本轮重写数量、覆盖范围和运行边界摘要。

## 术语解释

- `PlannerEvalCase`：用于评测 Planner 的一条样本。Planner 是负责决定检索、追问、拒答、回答时机的策略模块。
- `expected_answer_points`：标准答案要点。高置信 gold 要求这里的每个要点都有来源证据支持。
- `expected_chunks`：期望命中的证据片段。它只能说明证据定位，不能自动证明答案正确。
- `acceptable_action_paths`：允许的检索/回答路线，用来约束 Planner 行为。它不是业务证据。
- `human_review_status=reviewed`：当前流程里的审核状态，不等于领域专家确认。
- `label_source=synthetic`：说明样本由脚本或模型式规则生成，要提高警惕。
- `gold`：每个答案点都能被公开来源、官方变量说明、profile 标签说明或本地可读证据直接支撑的样本。
- `source-grounded`：问题和答案只使用来源明确给出的事实，不加入“合理但无出处”的维修经验。
- `fact_id`：证据 chunk 中一条原子事实的稳定 ID；审计文件用它证明每个答案点来自哪里。
- `second_review_status=pending`：只表示尚未完成独立二审，不影响你重新判断；复核后应输出自己的结论。

## 公开来源

你必须优先使用以下公开来源，而不是泛泛凭行业经验判断：

- MetroPT-3 Dataset: https://archive.ics.uci.edu/dataset/791/metropt%2B3%2Bdataset
- AI4I 2020 Predictive Maintenance Dataset: https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset
- Condition Monitoring of Hydraulic Systems: https://archive.ics.uci.edu/dataset/447/condition%2Bmonitoring%2Bof%2Bhydraulic%2Bsystems

## 审核规则

1. 逐条读取 `gold_cases_authoring.jsonl`；不要因为文件名叫 Gold 就直接判定通过。
2. 不要因为样本看起来像合理维修建议就判为 gold。
3. 如果来源只说明了字段或状态标签，答案就只能围绕字段或状态标签，不要扩展到维修动作。
4. 如果答案点包含“检查轴承”“清洁风道”“更换阀件”“校准传感器”“维修后基线”等来源未直接说明的内容，原样不能判为 gold。
5. 通用安全提示如果来源未说明，也不能作为 gold 的核心答案点。
6. 如果样本可以通过删除不受支持的答案点改成高置信样本，判为 `needs_rewrite`，不要直接判 `gold`。
7. 只有当 query、expected_answer_points、expected_chunks/source 信息都被来源直接支撑时，才能判 `gold`。
8. 对 AI4I，优先接受可直接引用 TWF/HDF/PWF/OSF/RNF 规则的重写样本。
9. 对 Hydraulic，优先接受可直接引用 cooler/valve/pump leakage/accumulator/stable flag profile 标签含义的重写样本。
10. 对 MetroPT，优先接受 Air Leak failure report、变量含义、传感器信号解释样本；不要接受复杂维修根因样本。
11. 对每个 gold case，核对 `expected_answer_points` 是否与 `gold_case_audit.answer_evidence` 完全一致，并继续核对其中的 `evidence_fact_ids` 是否存在于对应 evidence chunk。
12. 打开 `source_url`，在 `source_locator` 指向的位置独立核对数字、单位、逻辑关系和标签含义；不能只相信中文摘要。
13. `label_source=api_assisted` 是有意保留的生成来源；不要把它改成 `manual`。如果事实正确，它不妨碍样本成为 source-grounded gold。
14. `human_review_status=reviewed` 只表示通过当前门禁；你仍需在输出里给出独立的 `gold/needs_fix/reject` 结论。
15. 检查被列入 `excluded_content` 的维修动作或根因是否重新出现在 query/答案点中；若回流，判为 `needs_fix`。
16. 证据文档尚未导入检索库不是内容错误。内容审核通过后仍要保留“导入并生成新环境快照”的运行前置条件。

## 输出格式

请输出 JSONL，每行一个 case 的审核结果：

```json
{"case_id":"...","decision":"gold|needs_fix|reject","confidence":"high|medium|low","source_support_summary":"...","unsupported_answer_points":["..."],"incorrect_fact_ids":["..."],"suggested_fix":"...","reviewer_notes":"..."}
```

最后再给一个 Markdown 摘要，包含：

- 总 case 数。
- gold 数。
- needs_fix 数。
- reject 数。
- 按 AI4I/Hydraulic 分开的通过率。
- 是否同意 `gold_case_audit.jsonl` 的逐点映射，以及不同意的具体 case_id/fact_id。
- 是否发现数字、单位、逻辑连接词或标签含义错误。
