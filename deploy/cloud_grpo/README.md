# AutoDL 正式 GRPO 运行入口

本目录只提供正式 GRPO（群组相对策略优化）部署入口，不提供 smoke（冒烟验证）参数。训练代码位于 `app/rag/training/grpo/`，配置固定为 75 个 case（训练案例）、每个 4 条 rollout（模型交互轨迹）、1 个 epoch（训练轮次）。

## 运行前身份

- `policy model`（策略模型）：`Qwen/Qwen3.5-4B` + 已完成的 SFT V2 LoRA（监督微调第二版低秩适配）。
- `reference model`（参考模型）：从同一 SFT V2 adapter（适配器）独立加载并冻结。
- `Provider`（动作执行器）：主业务 Python 内的 `MilvusActionProvider`，接收模型实际生成的 `query/action`（查询/动作）。
- `Reward`（奖励分数）：冻结的 `reward-v1.1` / `planner-training-v1`。
- 输出：全新的 `evaluation/stage9/artifacts/grpo/runs/<run_id>/`，不会覆盖 SFT 或历史 GRPO 产物。

## 原子命令

先复制并填写环境文件；真实密钥只能放在 `env.local`，不能提交：

```bash
cp deploy/cloud_grpo/env.example deploy/cloud_grpo/env.local
```

终端 A 启动真实 Provider Worker（动作执行器工作进程）：

```bash
CLOUD_GRPO_ENV_FILE=deploy/cloud_grpo/env.local bash deploy/cloud_grpo/run_provider_worker.sh
```

预期第一行 JSON 中 `ready=true`、`strict_errors=true`，且 `snapshot_id` 与正式配置一致。任一环境变量、Milvus collection（向量集合）或 BGE-M3 embedding（嵌入模型）不可用时立即停止。

得到用户上云确认后，终端 B 执行正式训练：

```bash
CLOUD_GRPO_ENV_FILE=deploy/cloud_grpo/env.local bash deploy/cloud_grpo/run_formal_grpo.sh
```

不要追加 `max_cases`、`max_train_samples` 或训练步截断参数。启动脚本会新建 `grpo_formal_launch_*` 日志目录，保存完整命令、`training.log` 和 `exit_code.txt`。

如训练在已保存 checkpoint（检查点）之后中断，只能从同一 run 的最新 checkpoint 恢复：

```bash
CLOUD_GRPO_ENV_FILE=deploy/cloud_grpo/env.local bash deploy/cloud_grpo/run_formal_grpo.sh \
  --resume-from evaluation/stage9/artifacts/grpo/runs/<run_id>/checkpoints/step_000025
```

恢复入口会校验配置、输入 SHA256、adapter（适配器）、optimizer state（优化器状态）、scheduler state（学习率调度状态）、随机状态、case 顺序和所有 checkpoint 文件；发现漂移或目标 run 已完成时拒绝运行。

## 正式停止条件

- SFT V2 adapter 无法读取或字节数不等于 `84968408`。
- GRPO case 数不是 75，或任一冻结输入 SHA256 漂移。
- CUDA、Qwen3.5-4B 本地缓存或依赖不可用。
- 真实 Provider 无法执行模型实际 Action（动作），或返回快照外候选。
- Reward/advantage/policy loss/KL/total loss（奖励/相对优势/策略损失/策略偏移/总损失）出现 NaN/Inf（非数值/无穷大）。
- reference model（参考模型）参数发生变化，或 LoRA（低秩适配）参数最终没有更新。
- 需要修改冻结数据或覆盖已有产物才能继续。
