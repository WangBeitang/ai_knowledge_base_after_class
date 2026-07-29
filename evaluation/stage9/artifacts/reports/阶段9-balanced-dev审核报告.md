# 阶段 9 balanced dev 审核报告

- 构建版本：`stage9-balanced-dev-build-v2`
- 构建时间：`2026-07-28T07:40:00+00:00`
- 当前验收状态：**通过：25 条均有独立二审记录**
- 来源导入清单：`evaluation/stage9/artifacts/balanced_dev/source_import_manifest.json`
- 来源导入清单 SHA256：`9674ee1580f8960d47d3d9ac0459f5486e8324fe1e51e3397e2c5e3089ceec33`
- Web 证据清单：`evaluation/stage9/artifacts/balanced_dev/web_evidence_manifest.json`
- Web 证据清单 SHA256：`6f823cd870e73690f1a23f325b8480d718911ca6229e218a18303f690609bc72`
- case 证据台账：`evaluation/stage9/artifacts/balanced_dev/balanced_dev_case_evidence.jsonl`
- case 证据台账 SHA256：`d07716bf7801b9f35fff23c4b4731ed79fc7ad73773661bbba978a41650ee194`

## 结论

- balanced dev 候选共 25 条，五个路线桶均为 5 条，且每桶 5 个独立 `leakage_group_id`。
- 保留原有 4 条 reviewed local-answer；本地/HyDE/追问/安全候选由生产 chunk 反向构造，Web 候选由冻结的官方页面事实构造。
- 退役 3 条旧 pending dev；原始记录保存在退役清单，未删除证据。
- 三审 rejected 后有 7 条旧题义改用新 case_id，原始行保存在 superseded 清单。
- 新候选已通过 primary source review（主构建者来源核验），但这不等于独立二审。
- 未提供 `second_review_decisions.jsonl` 时，新增 case 保持 `pending`，不会为了凑数自动改成 `reviewed`。
- 本任务未导出 SFT、未重训、未运行 SFT v1；数据只能在独立二审和新 snapshot 冻结后进入 9.3.15。

## 路线分布与审核状态

| route bucket | 候选数 | reviewed | pending/rejected | 唯一 leakage group |
|---|---:|---:|---:|---:|
| `local_answer` | 5 | 5 | 0 | 5 |
| `hyde_fallback` | 5 | 5 | 0 | 5 |
| `web_required` | 5 | 5 | 0 | 5 |
| `ask_clarification` | 5 | 5 | 0 | 5 |
| `safe_refuse` | 5 | 5 | 0 | 5 |

## 来源与构题方法

- 三份独立来源 PDF 均记录 publisher、官方 URL、文件 SHA256 和版本。
- PDF 已经过实际生产导入图：解析、图片增强、生产切分、主题识别、BGE embedding 和 Milvus 索引；本任务使用回读后的真实 chunk 身份。
- 每条新 case 的证据台账记录 `source_url -> source_sha256 -> document_id -> chunk_id -> index_version -> content_sha256`。
- `hyde_fallback` 额外绑定检索探针：原始口语问法目标证据弱，来源约束的 hypothetical query 后目标 chunk 升至 rank 1；探针只证明构题理由，不替代运行时真实 HyDE 评测。
- Web case 绑定官方 URL、抓取时间、原始响应 SHA256、事实 SHA256 和 fact_id；终态改为 `web_search -> answer`。这只证明冻结快照可评分，真实运行仍需真实 Web provider。

## train/dev 泄漏审计

- `case_id`、标准化 query、query variant、leakage group 及保守近重复规则均未发现 train/dev 交叉；独立二审仍需做语义检查。

## 尚未满足的门禁

- 修订后仍为 pending 的候选需要由非主构建者逐条完成 evidence、route 和 leakage 二审。
- 二审通过后重新运行本脚本，只有有明确 approved 决定的 case 才会改为 reviewed。
- balanced dev 执行前必须生成包含三份新文档的 EnvironmentSnapshot（环境快照），不能继续复用 stage8 旧 snapshot。
- 当前不允许进入 9.3.15，更不允许根据这些 pending case 的模型结果改标签。
