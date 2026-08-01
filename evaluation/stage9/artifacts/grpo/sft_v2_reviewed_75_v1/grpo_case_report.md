# GRPO train-only case 导出报告

- 数据版本：`grpo-cases-from-sft-v2-reviewed-75-v1`。
- 来源：冻结的 `sft-v2-reviewed-75-v1`。
- case（案例）数量：75，唯一 query（问题）数量：75。
- 全部属于 train-only（仅训练）、全部 reviewed（已审核）。
- 每条参考轨迹均在 `acceptable_action_paths`（可接受动作路线）中。
- 绑定 Reward profile（奖励配置）：`evaluation/stage9/configs/reward_v1_1_training_profile.json`。
- 本轮没有生成 rollout（采样轨迹），没有调用 Provider（动作执行器/环境结果提供器），没有执行 GRPO（组相对策略优化）训练。

该文件只为训练阶段提供问题、事实、证据、行为边界和参考轨迹。训练阶段仍须由当前策略实时采样多条轨迹，并在真实或冻结回放环境中评分。
