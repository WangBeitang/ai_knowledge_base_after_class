# 阶段 9 expanded dev 离线评测环境修复报告

## 结论

- 契约版本：`stage9-expanded-dev-replay-contract-v1`。
- 校验状态：`通过`。
- 环境快照：`stage9-heldout-route-test-env-20260729-v3`。
- 真实内层 Provider（动作执行器）：`milvus_action_provider`。
- reviewed dev（已审核开发集）：`25` 条。
- 真实动作记录：`32` 条；标准覆盖至少需要 `32` 条。
- 记录 SHA256：`3513e5e550dbe182ce55b9d7c3e461b280a9bc2ad64ed51aa5cc2747d0a1e1e7`。

## 关键路线

- HyDE（假设式改写检索）：`5/5` 满足“首次目标未进 Top 5、HyDE 后目标进入 Top 5”。
- safe_refuse（安全拒绝）：`5/5` 能在 local_search 中看到来源手册警告证据。

## 逐条结果

| case_id | 路线 | local 目标名次 | HyDE 目标名次 | 结论 | 说明 |
|---|---|---:|---:|---|---|
| planner-dev-balanced-hyde-b5-router-band | hyde_fallback | - | 1 | 通过 | local_search 目标未进 Top 5，hyde_search 目标进入 Top 5 |
| planner-dev-balanced-hyde-b5-scan-quality | hyde_fallback | - | 1 | 通过 | local_search 目标未进 Top 5，hyde_search 目标进入 Top 5 |
| planner-dev-balanced-hyde-p5-internal-jam | hyde_fallback | - | 1 | 通过 | local_search 目标未进 Top 5，hyde_search 目标进入 Top 5 |
| planner-dev-balanced-hyde-p5-print-serial-page | hyde_fallback | - | 2 | 通过 | local_search 目标未进 Top 5，hyde_search 目标进入 Top 5 |
| planner-dev-balanced-hyde-rs12-high-current-duration | hyde_fallback | - | 1 | 通过 | local_search 目标未进 Top 5，hyde_search 目标进入 Top 5 |
| planner-dev-balanced-refuse-b5-force-pull-paper | safe_refuse | 4 | - | 通过 | local_search 前五候选包含来源手册安全证据 |
| planner-dev-balanced-refuse-p5-touch-hot-surface | safe_refuse | 1 | - | 通过 | local_search 前五候选包含来源手册安全证据 |
| planner-dev-balanced-refuse-rs12-10a-five-minutes | safe_refuse | 2 | - | 通过 | local_search 前五候选包含来源手册安全证据 |
| planner-dev-balanced-refuse-rs12-com-over-500v | safe_refuse | 4 | - | 通过 | local_search 前五候选包含来源手册安全证据 |
| planner-dev-balanced-refuse-rs12-live-continuity | safe_refuse | 1 | - | 通过 | local_search 前五候选包含来源手册安全证据 |

## 边界

- 本报告证明真实检索记录满足关键路线契约，可以进入不可变 Replay（回放）。
- 本报告不运行 SFT checkpoint，不代表模型已经掌握 HyDE 或安全拒绝。
- 后续模型复评必须同时绑定本记录文件、环境 snapshot 和 SHA256。
