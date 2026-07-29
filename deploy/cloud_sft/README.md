# Cloud SFT（云端监督微调）运行包

本目录保存阶段 9 上云所需的可执行脚本和环境模板。它是部署资产，不是一次性
`evaluation/stage9（阶段实验目录）`产物；训练配置、训练数据和报告仍放在 `evaluation/stage9/`。

AutoDL（云 GPU 平台）执行细节见 `AUTODL_SFT_GUIDE.md（AutoDL 监督微调上云指引）`。
首次上云的已知问题和当前续接状态见
`AUTODL_SFT_CLOUD_ISSUES.md（AutoDL SFT 上云问题记录）`。

## 运行顺序

1. 把完整项目代码同步到 GPU（显卡算力）服务器。
2. 复制环境模板：

```bash
cp deploy/cloud_sft/env.example deploy/cloud_sft/env.local
```

3. 修改 `deploy/cloud_sft/env.local`，至少确认 `APP_ROOT（项目根目录）`、
   `SFT_VENV_PATH（监督微调专用虚拟环境）`、
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

8. 服务启动后运行六类 Action（动作）和 Web 禁用边界 HTTP probe（探针）：

```bash
CLOUD_SFT_ENV_FILE=deploy/cloud_sft/env.local bash deploy/cloud_sft/run_planner_action_probe.sh
```

也可以在 `screen（终端持久化工具）`中用一条命令完成 preflight、启服、探针和停服：

```bash
CLOUD_SFT_ENV_FILE=deploy/cloud_sft/env.local bash deploy/cloud_sft/run_gpu_acceptance_gate.sh
```

9. 停止 vLLM、确认显存释放后运行 dev eval（开发集评测）：

```bash
CLOUD_SFT_ENV_FILE=deploy/cloud_sft/env.local bash deploy/cloud_sft/run_dev_eval.sh
```

10. 9.3.12～9.3.15A 的数据、阈值和 Reward 门禁全部冻结后，运行完整 25 条
    expanded dev 与 9.4 准入门禁：

```bash
CLOUD_SFT_ENV_FILE=deploy/cloud_sft/env.local \
bash deploy/cloud_sft/run_expanded_dev_gate.sh
```

## 脚本边界

- `requirements-training.txt`：固定 SFT（监督微调）专用依赖，不包含 `magic-pdf（PDF 解析框架）`
  等与训练无关且会限制 Transformers（模型训练框架）版本的业务依赖。
- `requirements-training.lock`：锁定 Linux 云端训练环境的全部直接和间接依赖版本，
  `bootstrap（初始化脚本）`实际按该文件安装。
- `bootstrap_gpu_server.sh`：创建独立 `.venv-sft（监督微调虚拟环境）`、安装训练依赖、
  检查 CUDA（英伟达 GPU 计算平台）并打印版本，不启动训练。
- `run_sft_smoke.sh`：从正式训练配置派生临时小样本配置，默认只跑 4 条样本和 1 个 step（训练步）。
- `run_sft_train.sh`：调用 `evaluation/stage9/model_planner/sft_train.py` 执行正式 SFT（监督微调）。
- `run_planner_server.sh`：复用 `deploy/planner_model_server/run_vllm_planner_server.sh` 启动 vLLM（大模型推理服务框架）。
- `run_runtime_preflight.sh`：只读检查 `.venv-sft` 解释器身份、正式训练/expanded dev
  入口导入、训练锁文件与 checkpoint 框架版本、环境、磁盘、离线模型缓存、
  vLLM/PyTorch/CUDA 版本和端口；通过后用同一 `$PYTHON_BIN`生成 SFT 环境冻结文件，
  检查失败时禁止启动 vLLM。
- `run_planner_action_probe.sh`：通过真实 vLLM HTTP 服务覆盖六类 Action，并额外验证 Web 禁用边界。
- `run_gpu_acceptance_gate.sh`：串联 GPU preflight、vLLM 启停、健康等待和 HTTP 探针，
  结束时主动停止本次启动的服务并记录显存进程。
- `run_provider_probe.sh`：通过真实业务 Provider 执行一条已审核 Web 路线并写入 JSONL 审计记录；
  不依赖 GPU，但依赖业务侧 Web/Milvus 配置。
- `run_dev_eval.sh`：用指定 checkpoint（检查点）跑 dev case（开发样本）并生成报告。
- `run_expanded_dev_gate.sh`：任务 9.3.16 正式入口；模型加载前校验 25 条 reviewed dev、
  checkpoint、snapshot、Reward v1.1 和冻结阈值，运行后生成逐 case 结果与 9.4 准入决定。
- `scripts/cloud_sft/collect_cloud_run_report.py`：收集 code version（代码版本）、config hash（配置哈希）、
  model profile（模型配置档案）、train manifest（训练清单）、Reward profile（奖励函数配置）、
  snapshot_id（快照身份）和运行命令。
- `scripts/cloud_sft/freeze_sft_artifacts.py`：校验 checkpoint、正式训练报告和历史 dev eval
  或 9.3.16 expanded dev run 的 `run_id（运行身份）`一致性，生成逐文件 SHA256（文件哈希）、
  冻结 manifest（清单）和可下载归档。

## 密钥要求

`env.example（环境变量模板）`不包含真实密钥。云端如需 `PLANNER_API_KEY（规划器接口密钥）`，
只写入本机 `env.local`，不要提交到仓库。

## 为什么使用独立训练环境

主业务环境中的 `magic-pdf（PDF 解析框架）`要求 `Transformers < 5`，而
`Qwen3.5（通义千问 3.5）`需要能够识别 `qwen3_5（模型架构标识）`的 Transformers 5。
两者不能在同一个 Python 虚拟环境中同时满足。云端脚本因此只复用项目代码、训练数据和配置，
不复用主业务依赖环境；默认训练解释器为 `$APP_ROOT/.venv-sft/bin/python`。
