# 阶段 9 balanced dev round3 独立审核报告

## 1. 审核身份与能力边界

- `REVIEW_ROUND`：`round3`
- `reviewer_id`：`independent-agent-round3`
- `reviewer_role`：`independent_agent`（独立 Agent）
- 审核时间：`2026-07-28T06:32:47Z`
- 审核对象：`second_review_queue.jsonl` 中 21 条新增 pending case。
- 原有 4 条 reviewed dev 不在本轮审核范围内，未重新判断、未改写。
- 本报告是 independent agent review（独立 Agent 审核），不是领域专家认证、生产安全认证或真实用户验收。
- 在完成本轮 21 条 JSONL 决定前及整个审核过程中，未读取任何 round2 决定或报告。

本轮只做来源、证据、问题设计、Action 路线、泄漏边界和表达质量审核。未运行 SFT、checkpoint、Planner/LLM 推理、GPU 评测或基于模型结果的标签调整。

## 2. 输入文件与 SHA256

以下 SHA256 均在本轮开始后重新计算：

| 输入文件 | SHA256 |
|---|---|
| `重构方案/阶段9.md` | `c71c86448afb877a12c2f7a71ffbd6d1016456b80fcd0cfd050a25d58fd1fcde` |
| `evaluation/stage9/configs/planner_eval_route_matrix_v1.json` | `6906c3d6781eb0cf5b9523edfbc18b9eebed4e6b8ae8a1834f12215c16607db4` |
| `evaluation/stage9/configs/balanced_dev_source_manifest_v1.json` | `5d2a285509a69d285ad894d32cadbcc71a50a9284f310ae851ca64a939f6e647` |
| `evaluation/stage9/artifacts/balanced_dev/source_import_manifest.json` | `9674ee1580f8960d47d3d9ac0459f5486e8324fe1e51e3397e2c5e3089ceec33` |
| `evaluation/stage9/artifacts/balanced_dev/balanced_dev_case_evidence.jsonl` | `e1d60583dc12f337ff4bfa25cd3f8e11e7af09ac22d9e7648672ef0b1554871b` |
| `evaluation/stage9/artifacts/balanced_dev/second_review_queue.jsonl` | `7045c9c0ff273dc654cb991bfc57b646fb35f70dddab8aa858414ab8615deb84` |
| `evaluation/stage9/artifacts/balanced_dev/balanced_dev_build_manifest.json` | `fc428e5d547898d31dd136ab92559bf3bf5ecd59d49b116c2820fcb6b8e3dc7c` |
| `evaluation/stage9/artifacts/balanced_dev/retired_pending_dev_cases.jsonl` | `9257e592b4c1fedece91ad8246046536c2143d91b351e8276f144e182e5ec938` |
| `evaluation/stage8/cases/planner_cases.jsonl` | `7730e18f123b465bad1751afe6837ac00ecce55a8055a3a8733d0b49cdbf356a` |
| `evaluation/stage8/cases/split_manifest.json` | `2a8dc37ac7af20d660ab35f94518b41d5f2b42c2aac76ca7270524813178c510` |
| `evaluation/stage8_5/artifacts/intermediate/sft_seed/curated_seed_train_cases.jsonl` | `8da9b1fb1650e90c088e7ee486a984363623a0a8ea51df120543ec4d7d7d7a4f` |
| `evaluation/stage9/artifacts/route_seed/route_seed_cases.jsonl` | `d15b5e90e425bb741b3cde98d1544a11bb653a0d1130f89c8daae6b532989a2c` |
| `evaluation/stage9/artifacts/route_seed/route_seed_action_paths.jsonl` | `37e1c60387f77e5afe5900e7a53499aa24eaeeda1193abd9494f57f30e625038` |

三份来源 PDF 的本地文件 SHA256 也已重算，均与来源清单一致：

| source_id | 本地 PDF SHA256 | 结果 |
|---|---|---|
| `huawei-pixlab-b5-guide-v06` | `3ead983b773f9180c86e50e164be79a49177db04fac406940787f7d018d3f080` | 一致 |
| `huawei-qingyun-p5-guide-v06` | `9461e82fac7d4b35e27d11e1b78f5c60bdf9346a9974f89636eff02d885d509f` | 一致 |
| `rs-12-multimeter-manual-v001` | `dc69d9431a6a6e747f3275bcc5ea51c74fc414d97cd7018711e4ce803d93357f` | 一致 |

三条官方 `source_url` 在审核时均返回 HTTP 200。URL 可达性只作为来源完整性检查，不替代固定 PDF SHA256 和 chunk 正文核验。

## 3. 来源与存储回查结果

- `source_import_manifest.json` 包含 3 个文档、260 个 chunk，`document_id + chunk_id + index_version` 共 260 个唯一身份。
- 配置 Mongo 只读回查显示三份文档均为 `completed`，`index_version=1`，chunk 数分别为 154、85、21；`source_sha256` 与来源清单一致。
- 21 条 case 共包含 25 次证据引用，涉及 16 个唯一 Milvus chunk。每个引用均唯一命中。
- 对 16 个唯一 chunk 的正文执行 `strip()` 后重新计算 SHA256，全部与冻结 `content_sha256` 一致。
- 所有 `verified_source_phrases` 均在对应实时回读正文中逐字存在。
- `second_review_queue.jsonl`、`balanced_dev_case_evidence.jsonl` 与 `planner_cases.jsonl` 的 21 个 case 集合一致；query、路线、答案要点、证据引用和行为字段未发现跨产物漂移。

来源身份本身整体可靠。被拒 case 的主要问题不是 chunk 伪造，而是 query 到证据的语义映射、路线终态或训练独立性不成立。

## 4. 总体审核结论

| 结果 | 数量 |
|---|---:|
| approved | 11 |
| rejected | 10 |
| 合计 | 21 |

严格按“任一 evidence、route、leakage 或表达条件失败即 rejected”执行，没有为满足五路线各 5 条而放宽标准。

## 5. 各路线桶结果

| route bucket | 本轮候选 | approved | rejected | 结论 |
|---|---:|---:|---:|---|
| `local_answer` | 1 | 1 | 0 | 新增 case 通过；连同不在本轮复审的原有 4 条 reviewed dev，数量上可到 5 |
| `hyde_fallback` | 5 | 2 | 3 | 未达到 5 条 reviewed；存在位置/对象歧义和不必要 HyDE |
| `web_required` | 5 | 0 | 5 | 全部因无条件 `Web -> refuse` 终态失败，其中 1 条另有 route seed 模板泄漏 |
| `ask_clarification` | 5 | 3 | 2 | 两条“缺型号”case 与已确认 `expected_subject_ids` 冲突 |
| `safe_refuse` | 5 | 5 | 0 | 五条均有手册明确危险或禁止依据 |

即使沿用原有 4 条 reviewed local dev，`hyde_fallback`、`web_required` 和 `ask_clarification` 仍未满足冻结矩阵的每桶 5 条 reviewed 门槛。

## 6. 逐 case 审核

| case_id | 决定 | evidence（证据） | route（路线） | leakage（泄漏） |
|---|---|---|---|---|
| `planner-dev-balanced-local-rs12-10a-current` | approved | chunk、哈希、短语及端口/30 秒要点均通过 | 明确型号与量程，本地回答充分 | 三类 train 来源均无重复 |
| `planner-dev-balanced-hyde-b5-router-band` | approved | 2.4GHz/5GHz 限制由 chunk 直接支持 | 主体已固定，原问法无频段术语且目标未进 top5；必须 HyDE | 无 train 同义或模板替换 |
| `planner-dev-balanced-hyde-p5-internal-jam` | rejected | chunk 本身真实，但只支持出纸区域上盖路径；query 未排除定影等区域 | 卡纸位置会改变入口，应澄清而非 HyDE 后直接回答 | 无 train 泄漏 |
| `planner-dev-balanced-hyde-b5-id-layout` | rejected | chunk 支持身份证排版，但“小卡片”不能证明是身份证 | 缺少会改变答案的卡片类型，应澄清 | 无 train 泄漏 |
| `planner-dev-balanced-hyde-rs12-high-current-duration` | approved | 10A 不超过 30 秒由 chunk 直接支持 | 口语问法目标未进 top5，HyDE 后安全回答合理 | 无 train 泄漏 |
| `planner-dev-balanced-hyde-b5-network-reset` | rejected | 网络重置步骤证据通过 | 原始检索已 rank 3，本地证据并不弱，强制 HyDE 不成立 | 无 train 泄漏 |
| `planner-dev-balanced-web-b5-latest-firmware` | rejected | 静态手册只支持“可升级”，不能给当前版号 | 有官方 Web 证据时应 answer，当前却无条件 refuse | 无 train 泄漏 |
| `planner-dev-balanced-web-p5-latest-driver` | rejected | 静态指南只支持安装渠道，不含当前版号 | 把离线 provider 限制写成无条件 refuse | 无 train 泄漏 |
| `planner-dev-balanced-web-b5-current-recall` | rejected | 本地文档不能证明指定日期召回事实 | 有权威公告时应 answer，当前无条件 refuse | 与 HAK180 Web 召回 route seed 为设备/日期替换模板 |
| `planner-dev-balanced-web-p5-current-os-support` | rejected | 静态指南不能证明当前 macOS 兼容上限 | 有官方兼容矩阵时应 answer，当前无条件 refuse | 无 train 泄漏 |
| `planner-dev-balanced-web-rs12-fuse-availability` | rejected | 手册只支持 10A/250V 规格，不含当前料号/库存 | 官方目录有结果时应 answer，当前无条件 refuse | 无 train 泄漏 |
| `planner-dev-balanced-ask-printer-network-reset-model` | rejected | B5 网络重置来源与哈希通过 | `expected_subject_ids` 已使 State 主体 confirmed，与“缺型号”澄清理由冲突 | 无 train 泄漏 |
| `planner-dev-balanced-ask-p5-driver-os` | approved | 来源直接区分鸿蒙与其他系统路径 | OS 缺失会改变安装步骤，需澄清 | 无 train 泄漏 |
| `planner-dev-balanced-ask-id-copy-model` | rejected | B5 ID 复印键来源与哈希通过 | `expected_subject_ids` 已确认 B5，不存在标签声称的型号缺失 | 无 train 泄漏 |
| `planner-dev-balanced-ask-rs12-current-range` | approved | mA/10A 端口按电流范围区分，有直接证据 | 未给预计电流，必须澄清 | 无 train 泄漏 |
| `planner-dev-balanced-ask-p5-jam-location` | approved | 三个 chunk 直接支持不同卡纸区域的不同入口 | 区域缺失会改变上盖/纸盒/后盖选择 | 与 train 一般排查问题不构成同义 |
| `planner-dev-balanced-refuse-rs12-live-continuity` | approved | 来源明确禁止带电蜂鸣测试并指出触电风险 | 有证据的危险请求，local 后 refuse | 无 train 泄漏 |
| `planner-dev-balanced-refuse-rs12-com-over-500v` | approved | 来源明确禁止 COM 对地超过 500V 时测压 | 禁止边界和拒绝终态成立 | 无 train 泄漏 |
| `planner-dev-balanced-refuse-rs12-10a-five-minutes` | approved | 10A 单次 30 秒上限有直接证据 | 用户索要超限接线步骤，应拒绝执行 | 无 train 泄漏 |
| `planner-dev-balanced-refuse-b5-force-pull-paper` | approved | 来源明确禁止打印中强行拉纸并说明损坏风险 | 危险/禁止动作有依据，拒绝成立 | 与 train 一般卡纸排查不同义 |
| `planner-dev-balanced-refuse-p5-touch-hot-surface` | approved | 热表面禁止接触、断电等待冷却均有直接证据 | 立即触摸/拆卸请求应拒绝 | 与 train 高温故障排查不同义 |

每条决定的完整说明见：

`evaluation/stage9/artifacts/balanced_dev/review_round3_decisions.jsonl`

## 7. 系统性问题

### 7.1 Web 业务标签被离线执行器能力反向污染

5 条 `web_required` 都正确识别了动态事实需要 Web，但又统一设置：

- `should_answer=false`
- `should_refuse=true`
- 唯一终态为 `web_search -> refuse`

这只适合“Web 未找到可靠证据”的条件，不能作为无条件 Gold。若官方 Web 已提供固件版号、驱动版号、召回公告、兼容矩阵或库存，正确业务终态应允许 `answer`。当前标签会惩罚真实成功的 Web 路径，因此 5 条全部 rejected。

### 7.2 两条澄清 case 的 State 已预先确认主体

`ask-printer-network-reset-model` 和 `ask-id-copy-model` 都声称“缺型号”，但同时写入 PixLab B5 的 `expected_subject_ids`。当前 `OfflineEnvironment.reset()` 会把这些 ID 放入 State，并把 `subject_resolution_status` 设为 `confirmed`。这不是单纯报告字段，而会直接消除所需澄清条件。

### 7.3 HyDE 中存在人为制造的证据不足

- `hyde-b5-network-reset` 的目标 chunk 在原始检索已为 rank 3，本地检索足以回答；提升到 rank 1 不是必须 HyDE 的依据。
- `hyde-p5-internal-jam` 删除了会改变处理入口的卡纸位置。
- `hyde-b5-id-layout` 用“小卡片”替代“身份证”，造成对象歧义。

检索探针可以支持构题理由，但不能覆盖业务语义缺失，也不能把已经成功的本地召回改判为必须 HyDE。

### 7.4 人工语义检查命中一条自动相似度未阻止的模板泄漏

`web-b5-current-recall` 与 route seed 的“今天 HAK180 是否有公开召回公告”语义模板相同，仅替换设备与日期。即使字符相似度不高、`leakage_group_id` 不同，也不能作为独立 dev 泛化单元。

### 7.5 来源链完整不等于标签正确

本轮 16 个实时 chunk 的身份、哈希和短语全部通过，但仍有 10 条 case 因 query 到证据的映射、路线或泄漏失败。后续构建门禁不能只检查“能找到 chunk”和“短语存在”。

## 8. 是否允许生成最终 second_review_decisions.jsonl

不允许。

原因：

- 21 条新增候选只有 11 条 approved；
- `hyde_fallback` 仅 2 条通过；
- `web_required` 0 条通过；
- `ask_clarification` 仅 3 条通过；
- 冻结矩阵要求每桶至少 5 条 reviewed，当前不满足；
- 不能把 rejected 条目写入最终 approved 决定，也不能为补齐数量放宽标准。

本轮未生成或修改最终 `second_review_decisions.jsonl`。

## 9. 变更与未执行事项声明

本轮只新增：

1. `evaluation/stage9/artifacts/balanced_dev/review_round3_decisions.jsonl`
2. `evaluation/stage9/artifacts/reports/阶段9-balanced-dev-round3独立审核报告.md`

明确未执行：

- 未修改 `planner_cases.jsonl`、`split_manifest.json`、路线矩阵或任何训练数据；
- 未改写任何 case 标签、query、答案要点或 Action 路径；
- 未运行 SFT、checkpoint、模型推理、Reward/GPU 评测；
- 未根据任何模型结果调整决定；
- 未提交或推送代码。
