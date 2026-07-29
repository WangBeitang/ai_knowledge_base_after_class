# 阶段 9 heldout route test 冻结报告

- 构建版本：`stage9-heldout-route-test-build-v3`
- 冻结版本：`stage9-heldout-route-test-freeze-v3`
- 构建时间：`2026-07-29T04:50:00+00:00`
- 当前状态：**通过：25 条均有独立审核记录，可冻结；仍禁止在 9.3.16 前执行**
- snapshot_id：`stage9-heldout-route-test-env-20260729-v3`

## 数据边界

- 原有 `core_answer_test`：35 条，规范化内容 SHA256 为 `74412bcc5a91c363e5de339bd2ac3976bfb5feebdbbb04f30d1521613fb4eee3`；构建前后保持一致。
- 新增 `route_heldout_test`：25 条，五个路线桶各 5 条、每条独立 leakage group。
- 与 train/dev 的来源文档重叠：0。
- 与 train/dev 的 leakage group 重叠：0。
- 新增本地证据来自 5 份独立来源文档的生产 chunk；Web 路线绑定冻结的官方页面事实。

## 路线与审核状态

| route bucket | 候选数 | reviewed | pending/rejected |
|---|---:|---:|---:|
| `local_answer` | 5 | 5 | 0 |
| `hyde_fallback` | 5 | 5 | 0 |
| `web_required` | 5 | 5 | 0 |
| `ask_clarification` | 5 | 5 | 0 |
| `safe_refuse` | 5 | 5 | 0 |

## 硬门禁

- 当前待独立审核：0 条；主构建者的来源核验不等于独立审核。
- `allowed_for_model_selection=false`：本测试集不能用于选 checkpoint、调 Prompt 或修标签。
- 在任务 9.3.16 完成模型选择和 checkpoint 冻结前，不允许运行 heldout test。
- 本任务没有生成推理结果、Reward 分数或模型对比报告；当前产物只证明数据建设和审计边界。
