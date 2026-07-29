# 阶段 9 Reward v1.1 balanced dev 独立验证报告

## 结论

- 9.3.15A 决定：`pass_keep_v1_1`。
- Reward 版本：`reward-v1.1`；profile：`planner-training-v1`。
- balanced dev：25 条；固定 Action 轨迹：231 条。
- 最小 total Reward margin：`0.0700`；最小 Planner route margin：`0.1077`。
- 错误反超数量：`0`。

- 25 条 reviewed balanced dev 的最低正确轨迹均严格高于最高错误轨迹。
- 五个路线桶均无错误反超，保留 Reward v1.1；不需要执行 9.3.15B。

## 不可变边界

- Reward profile mutation（配置修改）：`false`。
- Reward implementation mutation（实现修改）：`false`。
- 模型执行：`false`。
- heldout 推理结果数：`0`。
- ActionProvider：`snapshot_expected_chunks`；只证明离线契约，不证明真实检索。
- 实际评分范围：25 条 dev（SHA256 `6058ff8f17570ceea62163b3504f660163a1ecd53457ea7a927b16999423f5d1`）；同 registry 中其余 71 条非 dev case 未进入评分循环。
- 回答型 case 使用占位 answer executor；原始 answer 分仍保留在 v1.1 总分中，同时单独报告 Planner route score，未调整权重。

| 输入 | 路径 | SHA256 |
|---|---|---|
| `planner_cases` | `evaluation/stage8/cases/planner_cases.jsonl` | `00fe0cb5299a30c36cb10f27197dcfb43e09cf3ec6027604fd40b305a78f82c7` |
| `environment_snapshot` | `evaluation/stage9/artifacts/heldout_route_test/environment_snapshot.json` | `fb2e1fbc858ee72eabddf190ccde2bd96945b36dd5626cb4a84b575cbbbb0160` |
| `reward_profile_v1_1` | `evaluation/stage9/configs/reward_v1_1_training_profile.json` | `2d4fc0f92c51beaf655fed57d2f5d3b43abaa50eeb4117ddb66b694f901ccf4e` |
| `route_matrix` | `evaluation/stage9/configs/planner_eval_route_matrix_v1.json` | `e9345a2e04d76ee825eacae82aa26bf867528e3eea513c134a545dbb76948383` |
| `reward_implementation` | `app/rag/evaluation/reward.py` | `1413877cd57a5000c23516359467f67c6170fbce141128a8ed0a1b60a3ed1f72` |

## 五路线桶 separation margin

| route bucket | case | 正轨迹 | 负轨迹 | 正轨迹均分 | 负轨迹均分 | 最小 margin | route margin | 反超 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `local_answer` | 5 | 9 | 42 | 0.8447 | 0.3330 | 0.1120 | 0.1723 | 0 |
| `hyde_fallback` | 5 | 5 | 50 | 0.8500 | 0.3648 | 0.0700 | 0.1077 | 0 |
| `web_required` | 5 | 5 | 50 | 0.8500 | 0.4665 | 0.0820 | 0.1262 | 0 |
| `ask_clarification` | 5 | 5 | 30 | 1.0000 | 0.5581 | 0.0700 | 0.1077 | 0 |
| `safe_refuse` | 5 | 5 | 30 | 1.0000 | 0.5613 | 0.0700 | 0.1077 | 0 |

## Reward component（奖励分项）均值

| component | 正确轨迹 | 错误轨迹 |
|---|---:|---:|
| `answer` | 0.3448 | 0.0495 |
| `behavior` | 1.0000 | 0.2431 |
| `citation` | 1.0000 | 0.3762 |
| `cost` | 0.9890 | 0.8839 |
| `format` | 1.0000 | 0.7673 |
| `retrieval` | 1.0000 | 0.6515 |

## 逐 case 最难负样本

| case_id | route bucket | 最低正轨迹 | 最高负轨迹 | margin | route margin | attribution |
|---|---|---:|---:|---:|---:|---|
| `planner-dev-balanced-ask-id-copy-model` | `ask_clarification` | 1.0000 | 0.9180 | 0.0820 | 0.1262 | `no_issue` |
| `planner-dev-balanced-ask-p5-driver-os` | `ask_clarification` | 1.0000 | 0.9300 | 0.0700 | 0.1077 | `no_issue` |
| `planner-dev-balanced-ask-p5-jam-location` | `ask_clarification` | 1.0000 | 0.9300 | 0.0700 | 0.1077 | `no_issue` |
| `planner-dev-balanced-ask-printer-network-reset-model` | `ask_clarification` | 1.0000 | 0.9180 | 0.0820 | 0.1262 | `no_issue` |
| `planner-dev-balanced-ask-rs12-current-range` | `ask_clarification` | 1.0000 | 0.9300 | 0.0700 | 0.1077 | `no_issue` |
| `planner-dev-balanced-hyde-b5-router-band` | `hyde_fallback` | 0.8500 | 0.7800 | 0.0700 | 0.1077 | `no_issue` |
| `planner-dev-balanced-hyde-b5-scan-quality` | `hyde_fallback` | 0.8500 | 0.7800 | 0.0700 | 0.1077 | `no_issue` |
| `planner-dev-balanced-hyde-p5-internal-jam` | `hyde_fallback` | 0.8500 | 0.7800 | 0.0700 | 0.1077 | `no_issue` |
| `planner-dev-balanced-hyde-p5-print-serial-page` | `hyde_fallback` | 0.8500 | 0.7800 | 0.0700 | 0.1077 | `no_issue` |
| `planner-dev-balanced-hyde-rs12-high-current-duration` | `hyde_fallback` | 0.8500 | 0.7800 | 0.0700 | 0.1077 | `no_issue` |
| `planner-dev-balanced-local-rs12-10a-current` | `local_answer` | 0.8500 | 0.7380 | 0.1120 | 0.1723 | `no_issue` |
| `planner-dev-balanced-refuse-b5-force-pull-paper` | `safe_refuse` | 1.0000 | 0.9300 | 0.0700 | 0.1077 | `no_issue` |
| `planner-dev-balanced-refuse-p5-touch-hot-surface` | `safe_refuse` | 1.0000 | 0.9300 | 0.0700 | 0.1077 | `no_issue` |
| `planner-dev-balanced-refuse-rs12-10a-five-minutes` | `safe_refuse` | 1.0000 | 0.9300 | 0.0700 | 0.1077 | `no_issue` |
| `planner-dev-balanced-refuse-rs12-com-over-500v` | `safe_refuse` | 1.0000 | 0.9300 | 0.0700 | 0.1077 | `no_issue` |
| `planner-dev-balanced-refuse-rs12-live-continuity` | `safe_refuse` | 1.0000 | 0.9300 | 0.0700 | 0.1077 | `no_issue` |
| `planner-dev-balanced-web-b5-firmware-upgrade-guidance` | `web_required` | 0.8500 | 0.7680 | 0.0820 | 0.1262 | `no_issue` |
| `planner-dev-balanced-web-b5-shared-client-systems` | `web_required` | 0.8500 | 0.7680 | 0.0820 | 0.1262 | `no_issue` |
| `planner-dev-balanced-web-p5-drum-replacement-guidance` | `web_required` | 0.8500 | 0.7680 | 0.0820 | 0.1262 | `no_issue` |
| `planner-dev-balanced-web-p5-official-print-specs` | `web_required` | 0.8500 | 0.7680 | 0.0820 | 0.1262 | `no_issue` |
| `planner-dev-balanced-web-p5-product-os-list` | `web_required` | 0.8500 | 0.7680 | 0.0820 | 0.1262 | `no_issue` |
| `planner-dev-dev-p3000-driver-missing` | `local_answer` | 0.8380 | 0.4900 | 0.3480 | 0.4123 | `no_issue` |
| `planner-dev-dev-p3000-duplex-paper` | `local_answer` | 0.8380 | 0.4900 | 0.3480 | 0.4123 | `no_issue` |
| `planner-dev-dev-p3030-paper-spec` | `local_answer` | 0.8380 | 0.4900 | 0.3480 | 0.4123 | `no_issue` |
| `planner-dev-dev-p3500-network-info` | `local_answer` | 0.8380 | 0.4900 | 0.3480 | 0.4123 | `no_issue` |

## Answer 与证据边界

- 占位 answer case：15 条。
- `planner_route_score` 只重汇总 v1.1 已有的 format、behavior、cost 分项，仅用于诊断，不进入 Reward 总分。
- `evidence_contract_score` 只重汇总 retrieval、citation；由于 provider 为 `snapshot_expected_chunks`，不能解释为真实 Milvus/Web 质量。
- 9.3.15A 没有写入或重建 Reward profile，也没有生成 v1.2。

## 下一步门禁

- 当前结论为 `pass_keep_v1_1`：保留 v1.1，跳过 9.3.15B；在人工确认本报告后才进入 9.3.16。
