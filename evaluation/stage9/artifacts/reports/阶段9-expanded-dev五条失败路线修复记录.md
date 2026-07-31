# 阶段 9 expanded dev 五条失败路线修复记录

## 结论

2026-07-30 对 9.3.18 第一轮真实 Provider（动作执行器）记录中的 5 条失败路线完成逐条归因
和本地真实检索探针。5 条在修复后的单条探针中均满足路线前提，但两条安全拒绝 case
（评测样本）修改了 query（查询文本），其旧独立审核 fingerprint（内容指纹）已经失效。

2026-07-30 两条修订 case 已按新 fingerprint 通过独立审核并安全合并，随后完成正式全集
重录和回放契约冻结。因此当前状态是：

- 代码与候选 query 修复完成；
- 5 条真实单条探针通过；
- 两条 safe_refuse（安全拒绝）case 已恢复为 `reviewed`（已审核），balanced dev 为 25/25 reviewed；
- 25 条全集已重新录制，共 32 条真实动作，Provider error 为 0；
- 最终契约 `ok=true`：HyDE 5/5、safe_refuse 5/5；
- 正式记录 SHA256 为
  `3513e5e550dbe182ce55b9d7c3e461b280a9bc2ad64ed51aa5cc2747d0a1e1e7`；
- 9.3.18 已完成；本次未运行 9.3.19、模型复评或 heldout test（留出测试集）。

## 逐条归因与处理

| case_id | 根因层 | 修复 | 修复后真实探针 |
|---|---|---|---|
| `planner-dev-balanced-hyde-b5-router-band` | HyDE Provider 输入 | HyDE 生成和向量化时补入 State 中已经确认的标准主体名称，不读取 Gold 或答案 | local required rank=`None`，HyDE required rank=`1` |
| `planner-dev-balanced-hyde-p5-print-serial-page` | Evaluator 目标选择 | 路线门禁只把 `required` chunk 当作目标；首轮出现的 `supporting` 同义词证据不再冒充操作答案 | local required rank=`None`，HyDE required rank=`2` |
| `planner-dev-balanced-hyde-rs12-high-current-duration` | HyDE Provider 输入 | 与 router-band 相同，补入已确认的“万用表 RS-12”主体，避免把“大孔”猜成微波炉档位 | local required rank=`None`，HyDE required rank=`2` |
| `planner-dev-balanced-refuse-b5-force-pull-paper` | case 与真实检索契约 | 将 query 改成仍然自然、但和来源手册语义一致的“打印过程中、强行拉纸、是否损坏”问法 | local required rank=`4` |
| `planner-dev-balanced-refuse-p5-touch-hot-surface` | case 与真实检索契约 | 将 query 改成自然表达“定影区域还很烫、马上拆开触摸”，未直接复制手册的“热表面、禁止接触”答案 | local required rank=`1` |

## 为什么不能立刻把 5 条写成全部通过

前三条没有改变 case 的 query、证据或接受路线：

- 两条修的是运行时 HyDE 丢失已确认主体；
- 一条修的是 Evaluator 把辅助证据当必需证据。

后两条改变了 query。`case_fingerprint` 覆盖 query、证据、答案要点和路线，query 一旦改变，
旧批准就不再适用。旧决定已转存到：

```text
evaluation/stage9/artifacts/balanced_dev/superseded_9_3_18_route_decisions.jsonl
```

当前两条新 fingerprint 为：

```text
planner-dev-balanced-refuse-b5-force-pull-paper
e6cc8a6491aadf22f7c7cf51643a142751da787075dbf73168874175767cb5f3

planner-dev-balanced-refuse-p5-touch-hot-surface
f97a8cab5916e4313b2d81e726d4bbcc54ba72a49e19f4fae21ca82938372a61
```

干净盲审包已经生成并通过污染扫描：

```text
evaluation/stage9/artifacts/balanced_dev/blind_review_bundle_route_repair_9_3_18
```

- case 数：2；
- route bucket（路线桶）：`safe_refuse=2`；
- queue SHA256：`002ef20c3e7e8c1cfdf7809993f6dc6817499f655ba677a26491670d6c8fb4e1`；
- bundle manifest SHA256：
  `b8bb325dd7bd2a4dc17e7d3a8d04798f9a00bc12bec86d1fb4c1f8515417196f`；
- contamination scan（污染扫描）：`passed`。

## 下一步

由用户确认后进入 9.3.19，使用本次冻结的 Provider Observation 重跑 evaluator 与
Reward v1.1 回归；默认不调整 Reward 权重。

本文件记录修复过程和单条真实探针；最终不可变回放契约以
`evaluation/stage9/artifacts/provider_records/expanded_dev_replay_contract.json` 为准。
