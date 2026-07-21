# 阶段 8.5 source-grounded gold 重写报告

## 结果

- 共生成 20 条 source-grounded gold：AI4I 10 条，Hydraulic 10 条。
- AI4I 覆盖 TWF/HDF/PWF/OSF/RNF 五类官方规则；Hydraulic 覆盖 profile.txt 五列标签。
- 所有答案要点均由 gold evidence chunk 中的原子事实直接生成，没有维修动作、设备根因或经验性建议。
- 原 52 条候选及其第一轮审核结论未修改；每条新 case 都保存 rewritten_from_case_id。

## 审核状态

- `gold_status=source_verified`：主审核 agent 已逐答案点核对 UCI 官方说明。
- `label_source=api_assisted`：明确保留 agent 辅助生成来源，不标为 manual。
- `human_review_status=reviewed`：表示已通过阶段 8.5 当前审核门禁，不等同于领域专家背书。
- `second_review_status=passed`：Claude 已作为独立 agent 复核 20 条，全部为高置信 gold，无不受支持答案点、错误 fact ID 或建议修复项。

## 运行边界

- 本批 20 条明确归类为 `curated_seed_gold`：保留作为 train 种子和回归集，不作为独立 dev/test 或生产文档切分能力的证明。
- `artifacts/intermediate/curated_gold/gold_evidence_documents.jsonl` 是作者层逻辑文档定义；2 份 evidence document 和 10 个定界 chunk 已经导入 Mongo/Milvus。
- `gold_cases_indexed.jsonl` 已使用真实 Milvus 整数 `chunk_id`，`gold_case_runtime_bindings.jsonl` 保留逻辑证据键到运行时 ID 的映射。
- 导入快照为 `environment_snapshot_import_v1.json`（内部 `snapshot_id=stage85-env-20260721-v1`）；8.5.4 后续使用基于训练就绪 case 构建的 v2 快照，不与阶段 8 旧快照混用。
- 20 条目前全部放入 train；同一 UCI 官方说明不拆到 dev/test，后续 held-out 应使用独立来源文档。
- 后续新增 Gold 必须先让原始文档经过生产 parser/chunker/indexer 入库并冻结快照，再基于真实 `document_id + chunk_id + index_version` 生成问题、答案和审计映射。

## 逐条映射

| 新 gold case | 原候选 case | 证据 chunk | 答案点数 |
|---|---|---|---:|
| `stage85-gold-ai4i-twf-rule-001` | `stage85-ai4i-2020-tool-wear-failure-001` | `chunk_ai4i_twf_rule` | 2 |
| `stage85-gold-ai4i-twf-rule-002` | `stage85-ai4i-2020-tool-wear-failure-002` | `chunk_ai4i_twf_rule` | 2 |
| `stage85-gold-ai4i-hdf-rule-001` | `stage85-ai4i-2020-heat-dissipation-failure-001` | `chunk_ai4i_hdf_rule` | 3 |
| `stage85-gold-ai4i-hdf-rule-002` | `stage85-ai4i-2020-heat-dissipation-failure-002` | `chunk_ai4i_hdf_rule` | 2 |
| `stage85-gold-ai4i-pwf-rule-001` | `stage85-ai4i-2020-power-failure-high-load-001` | `chunk_ai4i_pwf_rule` | 3 |
| `stage85-gold-ai4i-pwf-rule-002` | `stage85-ai4i-2020-power-failure-high-load-002` | `chunk_ai4i_pwf_rule` | 3 |
| `stage85-gold-ai4i-osf-rule-001` | `stage85-ai4i-2020-overstrain-failure-001` | `chunk_ai4i_osf_rule` | 4 |
| `stage85-gold-ai4i-osf-rule-002` | `stage85-ai4i-2020-overstrain-failure-002` | `chunk_ai4i_osf_rule` | 2 |
| `stage85-gold-ai4i-rnf-rule-001` | `stage85-ai4i-2020-random-failure-review-001` | `chunk_ai4i_rnf_rule` | 2 |
| `stage85-gold-ai4i-rnf-rule-002` | `stage85-ai4i-2020-random-failure-review-002` | `chunk_ai4i_rnf_rule` | 2 |
| `stage85-gold-hydraulic-cooler-profile-001` | `stage85-hydraulic-condition-cooler-efficiency-low-001` | `chunk_hydraulic_profile_cooler` | 4 |
| `stage85-gold-hydraulic-cooler-profile-002` | `stage85-hydraulic-condition-cooler-efficiency-low-002` | `chunk_hydraulic_profile_cooler` | 2 |
| `stage85-gold-hydraulic-valve-profile-001` | `stage85-hydraulic-condition-valve-switching-delay-001` | `chunk_hydraulic_profile_valve` | 5 |
| `stage85-gold-hydraulic-valve-profile-002` | `stage85-hydraulic-condition-valve-switching-delay-002` | `chunk_hydraulic_profile_valve` | 2 |
| `stage85-gold-hydraulic-pump-profile-001` | `stage85-hydraulic-condition-internal-pump-leakage-001` | `chunk_hydraulic_profile_pump_leakage` | 4 |
| `stage85-gold-hydraulic-pump-profile-002` | `stage85-hydraulic-condition-internal-pump-leakage-002` | `chunk_hydraulic_profile_pump_leakage` | 2 |
| `stage85-gold-hydraulic-accumulator-profile-001` | `stage85-hydraulic-condition-accumulator-pressure-low-001` | `chunk_hydraulic_profile_accumulator` | 5 |
| `stage85-gold-hydraulic-accumulator-profile-002` | `stage85-hydraulic-condition-accumulator-pressure-low-002` | `chunk_hydraulic_profile_accumulator` | 2 |
| `stage85-gold-hydraulic-stable-profile-001` | `stage85-hydraulic-condition-stable-flag-mismatch-001` | `chunk_hydraulic_profile_stable_flag` | 3 |
| `stage85-gold-hydraulic-stable-profile-002` | `stage85-hydraulic-condition-stable-flag-mismatch-002` | `chunk_hydraulic_profile_stable_flag` | 2 |
