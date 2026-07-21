# 阶段 9 SFT 路线覆盖报告

## 结论

- 9.1 route seed 已生成、执行、导出并与阶段 8.5 curated seed 合并。
- route seed case 数：`50`。
- route seed SFT 样本数：`115`。
- 合并后 SFT 样本数：`155`。
- 合并后来源 case 数：`70`。
- baseline run：`stage9_route_seed_20260721T124909Z_58007af6`。
- snapshot：`stage85-env-20260721-v2`。
- Reward：`reward-v1.1`。

## Route Seed 分布

| 路线家族 | 数量 |
|---|---:|
| `ask_clarification` | 10 |
| `hyde_fallback` | 10 |
| `multi_step_fallback` | 10 |
| `refuse` | 10 |
| `web_search` | 10 |

| 目标终态 | 数量 |
|---|---:|
| `answer` | 10 |
| `ask_clarification` | 10 |
| `refuse` | 30 |

## 合并后 Action 分布

| Action | 数量 |
|---|---:|
| `answer` | 30 |
| `ask_clarification` | 10 |
| `hyde_search` | 20 |
| `local_search` | 55 |
| `refuse` | 30 |
| `web_search` | 10 |

## 合并后路线家族分布

| 路线家族 | 数量 |
|---|---:|
| `ask_clarification` | 15 |
| `hyde_fallback` | 30 |
| `multi_step_fallback` | 30 |
| `refuse` | 15 |
| `stop_when_enough` | 40 |
| `web_search` | 25 |

## 导出边界

| Gold 来源 | 数量 |
|---|---:|
| `curated_seed_gold` | 40 |
| `route_seed_gold` | 115 |

| Label 来源 | 数量 |
|---|---:|
| `manual_route_seed` | 115 |
| `rule` | 40 |

| Review 状态 | 数量 |
|---|---:|
| `reviewed` | 155 |

## Route Seed 明细

| case_id | route_family | action_path |
|---|---|---|
| `stage9-route-ask-001` | `ask_clarification` | `ask_clarification` |
| `stage9-route-ask-002` | `ask_clarification` | `ask_clarification` |
| `stage9-route-ask-003` | `ask_clarification` | `ask_clarification` |
| `stage9-route-ask-004` | `ask_clarification` | `ask_clarification` |
| `stage9-route-ask-005` | `ask_clarification` | `ask_clarification` |
| `stage9-route-ask-006` | `ask_clarification` | `local_search -> ask_clarification` |
| `stage9-route-ask-007` | `ask_clarification` | `local_search -> ask_clarification` |
| `stage9-route-ask-008` | `ask_clarification` | `local_search -> ask_clarification` |
| `stage9-route-ask-009` | `ask_clarification` | `local_search -> ask_clarification` |
| `stage9-route-ask-010` | `ask_clarification` | `local_search -> ask_clarification` |
| `stage9-route-hyde-answer-001` | `hyde_fallback` | `local_search -> hyde_search -> answer` |
| `stage9-route-hyde-answer-002` | `hyde_fallback` | `local_search -> hyde_search -> answer` |
| `stage9-route-hyde-answer-003` | `hyde_fallback` | `local_search -> hyde_search -> answer` |
| `stage9-route-hyde-answer-004` | `hyde_fallback` | `local_search -> hyde_search -> answer` |
| `stage9-route-hyde-answer-005` | `hyde_fallback` | `local_search -> hyde_search -> answer` |
| `stage9-route-hyde-answer-006` | `hyde_fallback` | `local_search -> hyde_search -> answer` |
| `stage9-route-hyde-answer-007` | `hyde_fallback` | `local_search -> hyde_search -> answer` |
| `stage9-route-hyde-answer-008` | `hyde_fallback` | `local_search -> hyde_search -> answer` |
| `stage9-route-hyde-answer-009` | `hyde_fallback` | `local_search -> hyde_search -> answer` |
| `stage9-route-hyde-answer-010` | `hyde_fallback` | `local_search -> hyde_search -> answer` |
| `stage9-route-multi-fallback-001` | `multi_step_fallback` | `local_search -> hyde_search -> refuse` |
| `stage9-route-multi-fallback-002` | `multi_step_fallback` | `local_search -> hyde_search -> refuse` |
| `stage9-route-multi-fallback-003` | `multi_step_fallback` | `local_search -> hyde_search -> refuse` |
| `stage9-route-multi-fallback-004` | `multi_step_fallback` | `local_search -> hyde_search -> refuse` |
| `stage9-route-multi-fallback-005` | `multi_step_fallback` | `local_search -> hyde_search -> refuse` |
| `stage9-route-multi-fallback-006` | `multi_step_fallback` | `local_search -> hyde_search -> refuse` |
| `stage9-route-multi-fallback-007` | `multi_step_fallback` | `local_search -> hyde_search -> refuse` |
| `stage9-route-multi-fallback-008` | `multi_step_fallback` | `local_search -> hyde_search -> refuse` |
| `stage9-route-multi-fallback-009` | `multi_step_fallback` | `local_search -> hyde_search -> refuse` |
| `stage9-route-multi-fallback-010` | `multi_step_fallback` | `local_search -> hyde_search -> refuse` |
| `stage9-route-refuse-001` | `refuse` | `refuse` |
| `stage9-route-refuse-002` | `refuse` | `refuse` |
| `stage9-route-refuse-003` | `refuse` | `refuse` |
| `stage9-route-refuse-004` | `refuse` | `refuse` |
| `stage9-route-refuse-005` | `refuse` | `refuse` |
| `stage9-route-refuse-006` | `refuse` | `local_search -> refuse` |
| `stage9-route-refuse-007` | `refuse` | `local_search -> refuse` |
| `stage9-route-refuse-008` | `refuse` | `local_search -> refuse` |
| `stage9-route-refuse-009` | `refuse` | `local_search -> refuse` |
| `stage9-route-refuse-010` | `refuse` | `local_search -> refuse` |
| `stage9-route-web-refuse-001` | `web_search` | `web_search -> refuse` |
| `stage9-route-web-refuse-002` | `web_search` | `web_search -> refuse` |
| `stage9-route-web-refuse-003` | `web_search` | `web_search -> refuse` |
| `stage9-route-web-refuse-004` | `web_search` | `web_search -> refuse` |
| `stage9-route-web-refuse-005` | `web_search` | `web_search -> refuse` |
| `stage9-route-web-refuse-006` | `web_search` | `local_search -> web_search -> refuse` |
| `stage9-route-web-refuse-007` | `web_search` | `local_search -> web_search -> refuse` |
| `stage9-route-web-refuse-008` | `web_search` | `local_search -> web_search -> refuse` |
| `stage9-route-web-refuse-009` | `web_search` | `local_search -> web_search -> refuse` |
| `stage9-route-web-refuse-010` | `web_search` | `local_search -> web_search -> refuse` |

## 使用边界

- route seed 全部为 `train + reviewed + route_seed_gold + approved_training_seed`。
- `route_seed_gold` 只表示阶段 9 人工路线种子，不是独立 held-out test。
- 当前 Web 路线在离线 provider 下以 `web_search -> refuse` 或 `local_search -> web_search -> refuse` 训练 Web 识别和安全收口；Web answer 需要真实 Web/replay provider 与 Web 证据 schema 后再补。
- 当前 route seed 使用 `snapshot_expected_chunks` provider 验证流程和 Reward，不代表真实 Milvus/Web 质量。
