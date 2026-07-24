# Cloud SFT（云端监督微调）运行包

本目录保存阶段 9 上云所需的可执行脚本和环境模板。它是部署资产，不是一次性
`evaluation/stage9（阶段实验目录）`产物；训练配置、训练数据和报告仍放在 `evaluation/stage9/`。

AutoDL（云 GPU 平台）执行细节见 `AUTODL_SFT_GUIDE.md（AutoDL 监督微调上云指引）`。

## 运行顺序

1. 把完整项目代码同步到 GPU（显卡算力）服务器。
2. 复制环境模板：

```bash
cp deploy/cloud_sft/env.example deploy/cloud_sft/env.local
```

3. 修改 `deploy/cloud_sft/env.local`，至少确认 `APP_ROOT（项目根目录）`、
   `SFT_TRAIN_CONFIG（监督微调训练配置）`、`PLANNER_MODEL_PATH（模型路径）`和
   `PLANNER_ADAPTER_PATH（适配器路径）`。
4. 初始化服务器：

```bash
CLOUD_SFT_ENV_FILE=deploy/cloud_sft/env.local bash deploy/cloud_sft/bootstrap_gpu_server.sh
```

5. 先跑 smoke（冒烟）训练：

```bash
CLOUD_SFT_ENV_FILE=deploy/cloud_sft/env.local bash deploy/cloud_sft/run_sft_smoke.sh
```

6. smoke（冒烟）通过后再跑正式 SFT（监督微调）：

```bash
CLOUD_SFT_ENV_FILE=deploy/cloud_sft/env.local bash deploy/cloud_sft/run_sft_train.sh
```

7. 训练完成后，把输出日志里的 `checkpoint（检查点）`填到 `SFT_CHECKPOINT_DIR`。
   如果要启动微调后的 PlannerModelServer（规划器模型服务），还要把
   `checkpoint_manifest.json（检查点清单）`里的 `adapter_path（适配器路径）`
   填到 `PLANNER_ADAPTER_PATH`。

```bash
CLOUD_SFT_ENV_FILE=deploy/cloud_sft/env.local bash deploy/cloud_sft/run_planner_server.sh
```

8. 运行 dev eval（开发集评测）：

```bash
CLOUD_SFT_ENV_FILE=deploy/cloud_sft/env.local bash deploy/cloud_sft/run_dev_eval.sh
```

## 脚本边界

- `bootstrap_gpu_server.sh`：只安装依赖、检查 CUDA（英伟达 GPU 计算平台）并打印版本，不启动训练。
- `run_sft_smoke.sh`：从正式训练配置派生临时小样本配置，默认只跑 4 条样本和 1 个 step（训练步）。
- `run_sft_train.sh`：调用 `evaluation/stage9/model_planner/sft_train.py` 执行正式 SFT（监督微调）。
- `run_planner_server.sh`：复用 `deploy/planner_model_server/run_vllm_planner_server.sh` 启动 vLLM（大模型推理服务框架）。
- `run_dev_eval.sh`：用指定 checkpoint（检查点）跑 dev case（开发样本）并生成报告。
- `scripts/cloud_sft/collect_cloud_run_report.py`：收集 code version（代码版本）、config hash（配置哈希）、
  model profile（模型配置档案）、train manifest（训练清单）、Reward profile（奖励函数配置）、
  snapshot_id（快照身份）和运行命令。

## 密钥要求

`env.example（环境变量模板）`不包含真实密钥。云端如需 `PLANNER_API_KEY（规划器接口密钥）`，
只写入本机 `env.local`，不要提交到仓库。
