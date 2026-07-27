# 阶段 9 SFT v1（监督微调第一版）冻结报告

## 结论

任务 9.3.9 已于 2026-07-27 完成。正式 LoRA（低秩适配器）checkpoint、训练报告、
dev eval（开发集评测）、训练复现输入、环境快照和 vLLM（大模型推理服务框架）依赖清单
已经从 AutoDL 数据盘归档到本地，并通过归档整体 SHA256（文件哈希）、manifest（一致性清单）
和 26 个内部文件的逐文件 SHA256 校验。

该 checkpoint 冻结为 `SFT v1 baseline（监督微调第一版基线）`。冻结只证明产物完整、身份一致、
能够被后续分析复用，不证明独立泛化、上线质量或真实回答质量。

## 训练身份

| 字段 | 值 |
|---|---|
| `run_id（运行身份）` | `planner-sft-stage9-qwen3-5-4b-lora_20260727T085537Z_94a77563` |
| `base_model_id（基础模型身份）` | `Qwen/Qwen3.5-4B` |
| `model_profile_id（模型配置档案身份）` | `qwen3_5_4b` |
| `snapshot_id（环境快照身份）` | `stage85-env-20260721-v2` |
| `reward_profile（奖励函数配置）` | `evaluation/stage9/configs/reward_v1_1_training_profile.json` |
| `sample_count（训练样本数）` | 155 |
| `train_loss（训练损失）` | 0.2804 |
| `checkpoint code_version（训练检查点代码版本）` | `df45525-dirty` |
| `dev eval code_version（开发集评测代码版本）` | `b83ca7b`，报告记录为 dirty |

`dirty` 不能静默省略。训练和 dev 报告中的 `status_short（工作树状态）`显示 dirty 主要来自
未跟踪的 `deploy/cloud_sft/env.local`、cloud run 和 checkpoint 产物目录；归档已保留原始
`cloud_run_report.json（云端运行报告）`，后续审计应以报告中的完整状态为准。

## 归档身份

| 字段 | 值 |
|---|---|
| `freeze_version（冻结格式版本）` | `stage9-sft-artifact-freeze-v1` |
| 归档文件 | `stage9_sft-v1_planner-sft-stage9-qwen3-5-4b-lora_20260727T085537Z_94a77563.tar.gz` |
| 归档大小 | 82,080,762 bytes |
| 归档 SHA256 | `f30132c91ff8a827557e1619751cf50dd744cdbbe88a463c51ac4cd320702619` |
| 内部文件数 | 26 |
| 云端归档目录 | `/root/autodl-tmp/stage9_backups` |
| 本地长期备份目录 | `/Users/beitang/Backups/ai_knowledge_base_after_class/stage9/sft_v1_20260727` |

本地长期备份目录位于 Git 工作区之外，避免误把约 79MB 的模型归档提交到代码仓库。

## 校验结果

- 云端执行归档整体 `sha256sum -c`：通过。
- 本地执行归档整体 `shasum -a 256 -c`：通过。
- 外部 manifest 与归档内 `_freeze/freeze_manifest.json` 执行 `diff`：无差异。
- 归档内 `_freeze/SHA256SUMS.txt` 共校验 26 个文件：全部通过，无 `FAILED`。
- checkpoint 目录名、checkpoint manifest、正式训练 cloud run report 和 dev eval cloud run report
  的 `run_id`：一致。
- `sft_eval_dev.json` 中 `planner_summaries[0].config.checkpoint`：指向本次正式 checkpoint。

## 已冻结内容

- 正式 checkpoint manifest、训练配置、训练指标和训练样本预览。
- LoRA adapter 配置、权重与说明文件。
- tokenizer（分词器）配置、chat template（对话模板）和 tokenizer 数据。
- 正式训练 cloud run report、命令和日志，包括修正前报告副本。
- 7 条 dev eval 的结果、命令、日志和 cloud run report。
- SFT 训练数据、训练 manifest、Reward profile 和 model profile。
- Planner cases（规划器样本）、环境 snapshot（快照）和 vLLM 环境依赖清单。

## 后续边界

- 原 checkpoint、归档和本地备份均不得原地覆盖。
- 后续如补训练数据并重训，必须生成新的 `run_id` 和 `SFT v2` 归档。
- 当前 7 条 dev 只证明评测链路跑通，不能替代 balanced dev（均衡开发集）和 heldout test
  （留出测试集）。
- 进入 9.4 前仍须完成 9.3.10～9.3.16 的工程门禁、结果归因、数据补齐和重新校准。
