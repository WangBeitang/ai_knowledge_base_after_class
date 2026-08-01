# SFT V2 clean blind review bundle

- Bundle ID：`sft-v2-clean-2d55a91841d2fa87`
- 本目录是二审、三审共同使用的只读输入包。
- 两位审核者必须使用不同输出目录，且不得读取对方结果。
- `review_cases.jsonl` 含 125 条脱敏候选；已删除备用身份、生成自评和审核状态。
- `provider_observations.jsonl` 含真实 Provider（动作执行器/环境结果提供器）记录。
- `web_evidence_manifest.json`、`environment_snapshot.json` 用于来源与索引身份核对。
- `leakage_reference.jsonl` 仅用于 dev/test 的问题和 chunk（文本块）泄漏检查。
- `content_fingerprint` 绑定原候选；`blind_case_fingerprint` 可由同条 `blind_fingerprint_payload` 复算。
- 不允许修改本目录、候选源文件或 37 条旧轨迹；不得在本轮冻结正式集或运行训练。
