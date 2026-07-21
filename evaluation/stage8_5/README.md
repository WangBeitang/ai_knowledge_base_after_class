# 阶段 8.5 数据工程目录

本目录只服务阶段 8.5 的离线数据处理、审核、评测和训练导出，不进入线上 RAG 请求路径。
目录按“流水线代码”和“产物生命周期”组织，不再把历史候选、Gold、快照和最终训练数据
混放在 `candidates/processed/results` 中。

## 先看这里

阶段 9 直接读取的文件只有：

- `artifacts/final/sft_curated_seed_train.jsonl`：40 条单步 Planner SFT 训练样本。
- `artifacts/final/sft_curated_seed_manifest.json`：训练数据来源、Reward、快照和审批清单。

需要追溯这 40 条数据时，按下面顺序查看：

```text
artifacts/intermediate/curated_gold/gold_cases_authoring.jsonl
  -> artifacts/review/curated_gold/gold_case_audit.jsonl
  -> artifacts/review/curated_gold/gold_case_second_review.jsonl
  -> artifacts/intermediate/curated_gold/gold_cases_indexed.jsonl
  -> artifacts/intermediate/sft_seed/curated_seed_train_cases.jsonl
  -> artifacts/intermediate/sft_seed/reward_v1_1_baseline_train.json
  -> artifacts/final/sft_curated_seed_train.jsonl
```

## 目录结构

```text
evaluation/stage8_5/
├── pipelines/
│   ├── common/             # 三条流水线共用的 schema 和目录契约
│   ├── public_candidate/   # 最初 52 条公开候选的历史生成与基础门禁
│   ├── curated_gold/       # 20 条高置信 Gold 的生成、审核衔接和知识库导入
│   └── sft_seed/           # 8.5.4 训练 case 准备与正式 SFT seed 校验
├── artifacts/
│   ├── intermediate/       # 可重放中间产物；阶段 9 不直接读取
│   ├── review/             # 人工或 agent 审核证据与分流结果
│   └── final/              # 阶段 9 直接消费的正式训练输入
├── reports/                # 面向人的阶段报告
└── README.md               # 本导航文件
```

`__pycache__` 是 Python 自动生成的缓存，不属于数据资产，可以忽略或删除。

## 三条流水线

### 1. public_candidate

用途：保留最初 52 条公开数据候选实验，证明来源、许可证、故障卡片和基础 schema 门禁
能够串通。它不是当前正式训练数据来源。

主要产物：

- `artifacts/intermediate/public_candidate/planner_case_candidates.jsonl`：52 条原始候选。
- `artifacts/review/public_candidate/schema_approved_cases.jsonl`：24 条通过基础门禁的样本。
- `artifacts/review/public_candidate/review_queue.jsonl`：28 条待审核样本。

`schema_approved` 只表示 schema、来源和基础字段检查通过，不表示 Gold，也不表示允许训练。

重放命令：

```bash
uv run python evaluation/stage8_5/pipelines/public_candidate/seed_public_candidate_pool.py
```

### 2. curated_gold

用途：把 UCI 官方规则和 profile 标签重写成 20 条逐答案点可追溯的 Gold，经独立二审后
导入 Mongo/Milvus，并把逻辑证据键绑定到真实整数 `chunk_id`。

主要产物：

- `gold_cases_authoring.jsonl`：作者态 Gold，引用逻辑证据键，便于审核。
- `gold_cases_indexed.jsonl`：运行态 Gold，引用真实 Milvus `chunk_id`。
- `gold_case_audit.jsonl`：答案点到来源事实的逐条映射。
- `environment_snapshot_import_v1.json`：证据导入完成后的冻结环境。

生成 Gold 会把二审状态初始化为 pending，不能在已有审核产物上随意重跑。知识库导入命令
会访问 Mongo、Milvus 和 embedding 服务，必须在对应基础设施可用时执行。

### 3. sft_seed

用途：把二审通过、已绑定真实 chunk 的 Gold 转为 Planner 可运行 case，执行 Reward v1.1
路由冒烟，并导出正式训练种子。

重放顺序：

```bash
uv run python evaluation/stage8_5/pipelines/sft_seed/prepare_curated_seed_training.py

uv run python evaluation/stage8/run_planner_eval.py \
  --cases evaluation/stage8_5/artifacts/intermediate/sft_seed/curated_seed_train_cases.jsonl \
  --snapshot evaluation/stage8_5/artifacts/intermediate/sft_seed/environment_snapshot_training_v2.json \
  --split train \
  --planners rule \
  --output evaluation/stage8_5/artifacts/intermediate/sft_seed/reward_v1_1_baseline_train.json

uv run python evaluation/stage8/export_sft_data.py \
  --eval-result evaluation/stage8_5/artifacts/intermediate/sft_seed/reward_v1_1_baseline_train.json \
  --cases evaluation/stage8_5/artifacts/intermediate/sft_seed/curated_seed_train_cases.jsonl \
  --reward-threshold 0.80 \
  --allowed-splits train \
  --artifact-status approved_training_seed \
  --output evaluation/stage8_5/artifacts/final/sft_curated_seed_train.jsonl \
  --manifest evaluation/stage8_5/artifacts/final/sft_curated_seed_manifest.json

uv run python evaluation/stage8_5/pipelines/sft_seed/generate_stage85_4_report.py
```

当前 baseline 使用 `snapshot_expected_chunks` 离线 provider，只验证 Planner 路由、Reward 和
导出链路，不代表真实 Milvus 召回率或答案生成质量。

## 产物生命周期

| 目录 | 谁读取 | 是否允许直接训练 | 说明 |
|---|---|---:|---|
| `artifacts/intermediate/` | 流水线、测试、审计人员 | 否 | 可以重建最终产物，但不是阶段 9 稳定输入。 |
| `artifacts/review/` | 审核人员和训练门禁 | 否 | 保存审核结论、拒绝原因和证据映射。 |
| `artifacts/final/` | 阶段 9 Planner SFT | 是 | 只允许放经过审批且 manifest 完整的训练输入。 |
| `reports/` | 开发、产品、审计人员 | 否 | 面向人的解释，不作为机器训练契约。 |

## 当前边界

- 20 条 `curated_seed_gold` 可以作为 Planner SFT 的小规模高置信启动集。
- 这 20 条全部属于 train，不能伪装成独立 dev/test。
- 后续新增 Gold 必须遵循“真实语料先生产入库并冻结 chunk，再生成问题和答案”的流程。
- `artifacts/final/` 当前只有训练 seed 和 manifest；快照、baseline、审核文件不得混入。
