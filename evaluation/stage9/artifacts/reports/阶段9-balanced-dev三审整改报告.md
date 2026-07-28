# 阶段 9 balanced dev 三审整改报告

## 1. 结论

- round3（三审）审核的 21 条新增候选中，11 条 approved（通过）、10 条 rejected（拒绝）。
- 本次只沿用 11 条内容指纹未变化的审核决定；10 条拒绝项修订后全部保持
  `pending（待审核）`，没有把“已修复”冒充成“已通过”。
- 当前 balanced dev 共 25 条、五路线各 5 条；15 条 reviewed、10 条 pending。
- Web 评测契约和 10 条 case 的工程整改已经完成，但 9.3.13 的独立审核门禁尚未通过，
  不能进入 9.3.15，也没有运行 SFT、checkpoint 或 GPU 推理。

## 2. Web 评测契约整改

旧契约只允许回答型 case 绑定本地 `document_id + chunk_id + index_version`，导致真实
Web 检索即使找到权威证据，也无法按 URL 计算 retrieval（检索）和 citation（引用），
五条 Web Gold 因而被错误设计成无条件 `web_search -> refuse`。

本次新增 `expected_web_evidence（冻结网页标准证据）`：

- `source_id`、publisher、page title 和规范化 URL；
- `captured_at（抓取时间）`；
- `response_sha256（原始 HTTP 响应哈希）`；
- `evidence_content_sha256（审核事实列表哈希）`；
- `fact_ids（事实 ID）` 与 `answer_point_ids（答案要点 ID）`。

Reward v1.1 的权重和版本号未变，只扩展证据身份：

- retrieval 同时支持本地 chunk 身份与 Web URL；
- citation 接受指向冻结规范化 URL 的 Web 引用；
- baseline 的快照执行器可用冻结 Web 证据重放 `web_search -> answer`，仅用于契约和
  Reward 离线测试，不冒充真实 Web provider 质量。

5 个官方页面均在 2026-07-28 抓取成功，并通过配置中的必需短语校验。完整 URL、响应哈希、
事实哈希和事实内容见 `web_evidence_manifest.json`。

## 3. 10 条 rejected case 的修订

### 3.1 HyDE：3 条

| 修订后 case | 处理 |
|---|---|
| `planner-dev-balanced-hyde-p5-internal-jam` | 明确卡纸位于出纸区域，消除会改变处理入口的位置歧义；重新执行检索探针，原问法目标 chunk 未进 top5，hypothetical query 升至 rank 1。 |
| `planner-dev-balanced-hyde-p5-print-serial-page` | 替换“小卡片即身份证”的对象歧义题，改为根据手册查询如何打印序列号信息页；使用新 ID。 |
| `planner-dev-balanced-hyde-b5-scan-quality` | 替换原始检索已 rank 3 的网络重置题，改为扫描质量设置题；原问法目标 chunk 未进 top5，hypothetical query 升至 rank 1；使用新 ID。 |

三条探针均在修订标签前使用生产检索链路执行；探针只证明 HyDE 构题理由，不证明模型运行时
一定会生成同样的 hypothetical query。

### 3.2 Web：5 条

旧的动态版号、召回、兼容上限和库存题全部退出正式 dev。新题不再要求无条件拒答，而是只询问
抓取时官方页面明确存在的事实：

| 修订后 case | 冻结官方事实 |
|---|---|
| `planner-dev-balanced-web-b5-firmware-upgrade-guidance` | 固件升级供电要求与完成标志 |
| `planner-dev-balanced-web-b5-shared-client-systems` | B5 共享客户端系统与局域网前提 |
| `planner-dev-balanced-web-p5-drum-replacement-guidance` | P5 硒鼓更换信号、粉盒处理与冷却要求 |
| `planner-dev-balanced-web-p5-product-os-list` | P5 当前产品页列出的操作系统 |
| `planner-dev-balanced-web-p5-official-print-specs` | P5 官方打印速度与预装耗材印量 |

五条新 case 均为 `should_answer=true`、`should_call_web=true`，唯一接受路径为
`web_search -> answer`，并绑定对应官方 URL、事实 ID 和哈希。

### 3.3 Ask clarification：2 条

- `planner-dev-balanced-ask-printer-network-reset-model`
- `planner-dev-balanced-ask-id-copy-model`

两条问题的正确澄清条件都是“用户没有提供具体打印机型号”。修订后不再写入
`expected_subject_ids` 或 `expected_subject_names`，避免环境 reset 时提前把主体状态设为
`confirmed（已确认）`，从而消除本应出现的追问条件。

## 4. 审核决定继承边界

- `second_review_decisions.jsonl` 只保存 round3 已通过且内容未变化的 11 条决定。
- 每条决定绑定 `case_fingerprint`；影响审核结论的 case 规格变化后，构建脚本会拒绝旧决定。
- 7 条题义或证据发生实质变化的 rejected 旧 case 使用新 ID；原始行保存在
  `superseded_round3_rejected_cases.jsonl`，没有静默覆盖历史。
- 3 条沿用 ID 的修订项同样因 fingerprint 变化失去旧审核资格。
- `second_review_queue.jsonl` 当前只包含这 10 条修订项，但 reviewer 不再直接读取该文件
  以外的原始 case 台账；盲审统一使用隔离导出的 clean bundle。

## 5. 当前审核分布

| route bucket | case 数 | reviewed | pending |
|---|---:|---:|---:|
| `local_answer` | 5 | 5 | 0 |
| `hyde_fallback` | 5 | 2 | 3 |
| `web_required` | 5 | 0 | 5 |
| `ask_clarification` | 5 | 3 | 2 |
| `safe_refuse` | 5 | 5 | 0 |
| **合计** | **25** | **15** | **10** |

自动 train/dev 泄漏检查为 0；这不替代独立 reviewer 的语义泄漏检查。

## 6. 关键产物与 SHA256

| 产物 | SHA256 |
|---|---|
| `evaluation/stage9/artifacts/balanced_dev/web_evidence_manifest.json` | `6f823cd870e73690f1a23f325b8480d718911ca6229e218a1834f12215c16607db4` |
| `evaluation/stage9/artifacts/balanced_dev/second_review_decisions.jsonl` | `34c314da580a8e94a971da5d7bf9b0f7b16fcef3e8de0882004bbd87cfa3fdc4` |
| `evaluation/stage9/artifacts/balanced_dev/second_review_queue.jsonl` | `9a23b4617a985b165f28b9d1d2c2fd6984bcceb298f6d2b0383215190186b9bb` |
| `evaluation/stage8/cases/planner_cases.jsonl` | `c816bb46bec6bab3992816e97f29f967d9157036c7f005c36953792936b79d9a` |

## 7. 下一门禁

首次重审发现原白名单包含历史审核字段，因此该轮不计入 clean blind review。当前新增：

- `evaluation/stage9/balanced_dev/export_blind_review_bundle.py`
- `evaluation/stage9/artifacts/balanced_dev/blind_review_bundle_v1/`

bundle 只保留 10 条待审 case、可复算 fingerprint、冻结证据、脱敏 leakage reference
和路线定义；不含 reviewer、decision、审核状态或 notes。由未参与本次整改的独立 reviewer
只读取该 bundle、本地来源 PDF 和 bundle 冻结的官方 URL，逐条检查 evidence、route、
leakage 与表达质量。10 条全部形成新的显式决定并重新构建后，五路线每桶达到 5 条
reviewed，9.3.13 才算完成。
