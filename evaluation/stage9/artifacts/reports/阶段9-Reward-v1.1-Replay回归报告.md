# 阶段 9 Reward v1.1 Replay 回归报告

## 结论

- 9.3.19 决定：`pass_keep_v1_1`。
- Reward：`reward-v1.1`；权重和实现均未修改。
- balanced dev：25 条；旧路线框架 231 条。
- 使用冻结 Replay 实际评分：179 条；排除缺少非必要 Observation 的负路线：52 条。
- 错误反超：0；关键反模式：`{}`。
- 最小 total Reward margin：`0.0700`；最小 Planner route margin：`0.1077`。
- Provider records SHA256：`3513e5e550dbe182ce55b9d7c3e461b280a9bc2ad64ed51aa5cc2747d0a1e1e7`。

- 25 条 reviewed balanced dev 的最低正确轨迹均严格高于最高错误轨迹。
- 五个路线桶均无错误反超，保留 Reward v1.1；不需要执行 9.3.15B。

## Replay 与输入边界

- ActionProvider：`replay_action_provider`；snapshot：`stage9-heldout-route-test-env-20260729-v3`。
- 仅使用 9.3.18 冻结的 32 条真实 Observation；本任务没有调用 Milvus、Web 或 LLM。
- 52 条未评分路线缺少的是非必要负路线 Observation；没有用空候选或 `action_provider_failed` 伪造低分。
- 每个 case 都保留至少一条正确轨迹和一条错误轨迹。
- 模型执行：`false`；heldout 推理：0。

| 输入 | 路径 | SHA256 |
|---|---|---|
| `planner_cases` | `evaluation/stage8/cases/planner_cases.jsonl` | `1ab1c169ce1a4bd9cdb4be9a868494755012c008e432c2aa03c2b4ff198dc19b` |
| `environment_snapshot` | `evaluation/stage9/artifacts/heldout_route_test/environment_snapshot.json` | `fb2e1fbc858ee72eabddf190ccde2bd96945b36dd5626cb4a84b575cbbbb0160` |
| `reward_profile_v1_1` | `evaluation/stage9/configs/reward_v1_1_training_profile.json` | `2d4fc0f92c51beaf655fed57d2f5d3b43abaa50eeb4117ddb66b694f901ccf4e` |
| `route_matrix` | `evaluation/stage9/configs/planner_eval_route_matrix_v1.json` | `e9345a2e04d76ee825eacae82aa26bf867528e3eea513c134a545dbb76948383` |
| `reward_implementation` | `app/rag/evaluation/reward.py` | `1413877cd57a5000c23516359467f67c6170fbce141128a8ed0a1b60a3ed1f72` |
| `provider_records_9_3_18` | `evaluation/stage9/artifacts/provider_records/expanded_dev_provider_observations.jsonl` | `3513e5e550dbe182ce55b9d7c3e461b280a9bc2ad64ed51aa5cc2747d0a1e1e7` |
| `replay_contract_9_3_18` | `evaluation/stage9/artifacts/provider_records/expanded_dev_replay_contract.json` | `559ab751eb1ff85b50cd1421f5884e44fbd32307cea3ef329548064490211e7b` |
| `baseline_validation_9_3_15a` | `evaluation/stage9/artifacts/reward/reward_v1_1_balanced_dev_validation.json` | `df52eb3869c31b30729a2885d834f06460bc33fece3986f033aa55adf33b3a37` |

## 五路线桶 separation margin

| route bucket | case | 正轨迹 | 负轨迹 | 正轨迹均分 | 负轨迹均分 | 最小 margin | route margin | 反超 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `local_answer` | 5 | 9 | 41 | 0.8369 | 0.3198 | 0.3480 | 0.4123 | 0 |
| `hyde_fallback` | 5 | 5 | 50 | 0.7415 | 0.3175 | 0.1001 | 0.1077 | 0 |
| `web_required` | 5 | 5 | 15 | 0.7348 | 0.3643 | 0.2800 | 0.4308 | 0 |
| `ask_clarification` | 5 | 5 | 19 | 1.0000 | 0.5219 | 0.0700 | 0.1077 | 0 |
| `safe_refuse` | 5 | 5 | 25 | 1.0000 | 0.5584 | 0.0700 | 0.1077 | 0 |

## 未冻结负路线 Observation

| path_id | 排除数量 |
|---|---:|
| `local_answer` | 7 |
| `local_ask` | 7 |
| `local_hyde_answer` | 16 |
| `local_hyde_refuse` | 5 |
| `local_hyde_web_answer` | 5 |
| `local_refuse` | 7 |
| `local_web_answer` | 5 |

完整逐 case 排除记录保存在机器可读 JSON 的 `skipped_paths`。

## 边界与下一步

- 回答型 case 仍使用占位 answer executor；本结论只说明冻结 Replay 下 Reward v1.1 没有错误鼓励错误路线。
- 本报告不代表 SFT v1 已通过，也不代表 heldout 泛化。
- 当前结论允许保留 Reward v1.1；待 9.3.17 环境 freeze 闭环后，由用户确认是否进入 9.3.20 重跑 SFT v1。
