# Stage 9 heldout route test clean blind review - Round 1

## Blind review status

- `bundle_id`: `stage9-heldout-route-test-clean-c52eea2519a1a065`
- Blind review contamination（盲审污染）: **否**
- `bundle_manifest.json` 中 `contamination_scan`: `passed`
- `historical_decision_count`: `0`
- 本轮没有读取历史 decisions、review report、Planner 输出、Reward 分数或 heldout 推理结果。
- 本轮没有运行 Planner、SFT、vLLM、任何模型或任何 heldout 推理。
- 本审核属于 `independent_agent`（独立 Agent）审核，不等于领域专家认证。

## Input integrity

- `bundle_manifest.json` SHA256: `1d8fbd58d29b0e20dc4490f09a51d12d81dee61892a670895c08d483dcb39c29`
- Manifest 声明的 8 个包内输出文件均重新计算 SHA256 和字节数，全部一致。
- `review_cases.jsonl` 实际 25 行，与 `case_count=25` 一致。
- 实际 route 计数与 manifest 一致：`local_answer=5`、`hyde_fallback=5`、`web_required=5`、`ask_clarification=5`、`safe_refuse=5`。
- 25 条 `fingerprint_payload` 均按指定 canonical JSON 方法重算，全部等于同行 `case_fingerprint`。
- 输入包 hash/fingerprint 漂移：**无**。

## Source verification

- 5 份 `local_source_manifest.json` 指向的原始 PDF 均重新计算 SHA256，全部与 manifest 一致。
- 20 个本地 evidence ref 的 `source_id`、`document_id`、`chunk_id`、`chunk_index`、`index_version`、`source_sha256` 和 `content_sha256` 均与本地两个 manifest 一致。
- 所有本地 case 的 `verified_source_phrases` 均在对应原始 PDF 中定位，并按型号、设备变体、操作条件和安全边界核对。
- 5 个官方 Web URL 均可访问，当前页面均能支持冻结事实。
- 5 个 Web case 的 URL、冻结 `response_sha256`、`extracted_text_sha256`、`evidence_content_sha256`、`fact_id`、statement 和 verified phrases 在 `review_cases.jsonl`、`web_source_manifest.json`、`web_evidence_manifest.json` 之间一一对应。
- 当前实时抓取中 4 个官方页面响应 SHA256 与冻结值完全一致。MateBook B3-520 Windows 11 支持页当前响应 SHA256 为 `a74f9b723b303bc8ca0e41a4b3632a9bc92faaac227a4941da7167951bb3975a`，与冻结值 `b05fb659e987b630d2a776b1968ba694c8c62a8ad81087c80e04eb79a0b89d44` 不同；但当前官方表格仍逐项支持冻结的型号、六个代码和“支持”结论，因此这属于 live page response（实时页面响应）变化，不是盲审包输入漂移，也不导致该 case 因事实失效而拒绝。

## Decision summary

- Total cases: **25**
- Approved: **20**
- Rejected: **5**

| route_bucket | approved | rejected |
|---|---:|---:|
| `local_answer` | 5 | 0 |
| `hyde_fallback` | 1 | 4 |
| `web_required` | 5 | 0 |
| `ask_clarification` | 4 | 1 |
| `safe_refuse` | 5 | 0 |

## Rejected cases

1. `planner-test-heldout-hyde-display-black`
   - 主 query 没有 B5-341W 型号或可审计上下文，“这块屏”不足以绑定设备。
   - 即使 HyDE rank 改善，也应先追问设备型号，不能直接 answer。
   - 修订建议：主 query 明确型号，或改标 `ask_clarification`。

2. `planner-test-heldout-hyde-matestation-fingerprint-port`
   - 原检索已经在 top5 的 rank 2 命中目标证据，本地证据并不不足。
   - 强制 HyDE 只是把 rank 2 提到 rank 1，属于多余回退。
   - 修订建议：改为 `local_answer`；若保留 HyDE，需提供原检索没有足够证据的自然 query 和探针。

3. `planner-test-heldout-hyde-matebook-fan-heat`
   - 原检索已经在 top5 的 rank 2 命中目标证据。
   - 手册的“正常现象”有“开启高能模式后”前提，主 query 仅说“跑重活”，条件没有对齐。
   - 修订建议：明确 query 的高能模式条件并改为 `local_answer`，或先追问是否开启高能模式。

4. `planner-test-heldout-hyde-tablet-force-restart`
   - 原检索已在 top5 的 rank 3 命中目标证据，足以本地回答。
   - 强制 HyDE 没有“本地不足”依据。
   - 修订建议：改为 `local_answer`；若保留 HyDE，需要未获取足够证据的探针结果。

5. `planner-test-heldout-ask-tablet-multiscreen-prerequisites`
   - query 明确问“先检查什么”，手册已直接给出 WLAN、蓝牙和电脑管家 11.1+ 三项。
   - 不缺少回答所需信息，`ask_clarification` 把本可直接回答的问题误标为追问。
   - 修订建议：改为 `local_answer`，补齐三个 `expected_answer_points`。

## Issue classification

- Evidence/model-condition mismatch（证据/型号条件错配）: 2
  - `planner-test-heldout-hyde-display-black`
  - `planner-test-heldout-hyde-matebook-fan-heat`
- Route overreach（路线过度回退或误追问）: 4
  - `planner-test-heldout-hyde-matestation-fingerprint-port`
  - `planner-test-heldout-hyde-matebook-fan-heat`
  - `planner-test-heldout-hyde-tablet-force-restart`
  - `planner-test-heldout-ask-tablet-multiscreen-prerequisites`
- Leakage（跨 split 泄漏）: 0
- Expression/subject completeness（表达或主体完整性）: 1
  - `planner-test-heldout-hyde-display-black`
- Frozen Web fact invalidation（冻结 Web 事实失效）: 0

## Leakage review

- 仅使用包内 `leakage_reference.jsonl`。
- 25 条 case 均未发现与 train/dev 相同 `leakage_group_id`、规范化 query/variant 完全重复，或仅替换型号、数字、少量词语的同义近重复。
- Web case 与 dev 中其他“截至 2026-07-28 官网”问题共享时间表达模板，但设备、官方来源、事实对象和答案均不同，不判为泄漏。
- Safe refuse case 与参考集共享一般危险请求语气，但危险对象、动作、设备来源和安全边界不同，不判为泄漏。
