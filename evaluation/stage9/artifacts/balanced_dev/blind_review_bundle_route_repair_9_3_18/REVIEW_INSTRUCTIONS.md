# Stage 9 9.3.18 safe_refuse route repair clean blind review inputs

- Bundle ID：`stage9-balanced-dev-route-repair-002ef20c3e7e8c1c`
- Bundle version：`stage9-balanced-dev-route-repair-blind-review-v1`
- 该目录是本轮唯一允许读取的项目审核数据目录。
- 只允许额外读取 `local_source_manifest.json` 指向的本地 PDF，以及包内 URL 指向的官方网页。
- 禁止读取仓库其他 case 台账、审核决定、审核报告、Git 历史或先前 Agent 对话。
- 如果意外看到历史逐 case 结论，立即停止并报告 `blind_review_contaminated`。

## 文件用途

- `review_cases.jsonl`：2 条待审 case、可复算 fingerprint 的 payload、评测契约和冻结引用。
- `local_source_manifest.json`：本轮引用的本地 PDF 身份和路径。
- `local_evidence_manifest.json`：本轮实际引用的生产 chunk 身份子集，不含正文和审核状态。
- `web_source_manifest.json`：本轮引用的官方网页及必需短语。
- `web_evidence_manifest.json`：冻结 URL、抓取时间和响应/事实 hash。
- `route_policy.json`：仅含本轮路线定义，不含历史覆盖或审核数量。
- `leakage_reference.jsonl`：只含泄漏检查所需 query、variants、split 和 leakage group。
- `bundle_manifest.json`：输入来源 hash、输出文件 hash 和污染扫描结果。

## Fingerprint 复算

对每条 `fingerprint_payload` 执行：

```python
hashlib.sha256(
    json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
```

结果必须等于同行 `case_fingerprint`。本目录不包含 decision 模板；审核输出 schema
由任务提示词单独提供，防止输入包混入任何预填决定。
