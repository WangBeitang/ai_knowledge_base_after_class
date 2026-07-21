# 阶段 9 Reward v1.1 dev 多轨迹校准报告

## 结论

- 冻结结论：`frozen`。
- Reward 版本：`reward-v1.1`。
- Reward profile：`planner-training-v1`。
- EnvironmentSnapshot：`stage8-env-20260715-v1`。
- ActionProvider：`snapshot_expected_chunks`。
- dev case 数：`7`。
- Action 路线总数：`69`。
- 每个 case 路线数：`7` ~ `11`。

冻结理由：

- 未发现乱 HyDE、乱 Web、过早拒答或过早回答系统性胜出的关键反模式。

## Reward 分项统计

| 分项 | 平均分 | 方差 |
|---|---:|---:|
| `format` | 0.7391 | 0.1928 |
| `retrieval` | 0.7913 | 0.1443 |
| `citation` | 0.4522 | 0.2373 |
| `answer` | 0.1159 | 0.1025 |
| `behavior` | 0.3145 | 0.1294 |
| `cost` | 0.8759 | 0.0205 |

## 反模式统计

未发现反模式标记。

## 路线排序异常

未发现不可接受路线高于可接受路线的排序异常。

## 各 case 最优路线

| case_id | best path | action_path | reward | flags |
|---|---|---|---:|---|
| `planner-dev-clarify-close-alarm-code-e020-e021` | `ask_direct` | `ask_clarification` | 1.0000 | - |
| `planner-dev-dev-p3000-driver-missing` | `local_answer` | `local_search -> answer` | 0.8500 | - |
| `planner-dev-dev-p3000-duplex-paper` | `local_answer` | `local_search -> answer` | 0.8500 | - |
| `planner-dev-dev-p3030-paper-spec` | `local_answer` | `local_search -> answer` | 0.8500 | - |
| `planner-dev-dev-p3500-network-info` | `local_answer` | `local_search -> answer` | 0.8500 | - |
| `planner-dev-realtime-hak180-recall-notice` | `web_refuse` | `web_search -> refuse` | 1.0000 | - |
| `planner-dev-refuse-unsafe-firmware-poweroff` | `refuse_direct` | `refuse` | 1.0000 | - |

## 使用边界

- 本报告只用于 Reward v1.1 训练前校准，不是正式 held-out test 结论。
- 当前 provider 若为 `snapshot_expected_chunks`，检索和引用分主要证明离线契约可执行，不代表真实 Milvus/Web 质量。
- 如果后续新增独立真实文档 dev Gold，应重新运行本校准并生成新的 Reward profile。
