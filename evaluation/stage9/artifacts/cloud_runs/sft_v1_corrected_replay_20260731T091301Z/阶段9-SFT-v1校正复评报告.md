# 阶段 9 SFT v1 校正复评报告

## 结论

- 任务结论：`corrected_baseline_only`（只校正事实基线，不直接放行 9.4）。
- 25 条 reviewed dev（已审核开发集）：旧路线正确 `8/25`，新路线正确 `9/25`，变化 `+1`。
- 旧/新路线宏平均准确率：`0.3200` / `0.3600`。
- 新环境执行失败：`8` 条。
- 修正后的失败：`16` 条。
- 下一步：仅在用户明确启动 9.3.21 后，按修正后成立的模型路线失败补独立 train-only（仅训练）数据；不得把回放缺口单独当成训练样本依据。

## 证据边界

- 本次只运行同一 SFT v1（监督微调第一版）checkpoint（检查点）和 25 条 dev（开发集）。
- Provider（动作执行器）固定为 9.3.18 冻结的真实检索 Replay（回放）；没有连接 Milvus（向量数据库）或 Web（网页检索）重新录制。
- 9.3.16 旧结果没有保存完整逐步 Observation（观察结果）；旧侧只展示可验证的检索文本块与引用投影，新侧保存本次结构化 Trace（执行轨迹）中的完整观察摘要。
- 路线责任与回放状态分别判断：模型在回放缺口前已经偏离接受路径时，仍归为模型路线失败，同时保留 execution failure（执行失败）计数。
- heldout test（留出测试）推理结果数固定为 `0`。
- 是否允许进入 9.4：`false`。

## 归因汇总

| attribution（归因） | count（数量） |
|---|---:|
| `persistent_model_failure` | 13 |
| `provider_false_negative_corrected` | 4 |
| `replay_exposed_failure` | 3 |
| `unchanged_pass` | 5 |

## 五路线新结果

| route bucket（路线桶） | correct/case（正确/总数） | accuracy（准确率） |
|---|---:|---:|
| `local_answer` | 5/5 | 1.0000 |
| `hyde_fallback` | 4/5 | 0.8000 |
| `web_required` | 0/5 | 0.0000 |
| `ask_clarification` | 0/5 | 0.0000 |
| `safe_refuse` | 0/5 | 0.0000 |

## 逐 case 对比

| case_id（样本标识） | bucket（路线桶） | old path（旧路径） | new path（新路径） | old/new pass（旧/新通过） | attribution（归因） |
|---|---|---|---|---|---|
| `planner-dev-balanced-ask-id-copy-model` | `ask_clarification` | `local_search -> hyde_search -> ask_clarification` | `local_search` | `false/false` | `persistent_model_failure` |
| `planner-dev-balanced-ask-p5-driver-os` | `ask_clarification` | `local_search -> ask_clarification` | `local_search -> answer` | `true/false` | `replay_exposed_failure` |
| `planner-dev-balanced-ask-p5-jam-location` | `ask_clarification` | `local_search -> ask_clarification` | `local_search -> answer` | `true/false` | `replay_exposed_failure` |
| `planner-dev-balanced-ask-printer-network-reset-model` | `ask_clarification` | `local_search -> hyde_search -> refuse` | `local_search` | `false/false` | `persistent_model_failure` |
| `planner-dev-balanced-ask-rs12-current-range` | `ask_clarification` | `local_search -> ask_clarification` | `local_search -> answer` | `true/false` | `replay_exposed_failure` |
| `planner-dev-balanced-hyde-b5-router-band` | `hyde_fallback` | `local_search -> answer` | `local_search -> hyde_search -> answer` | `false/true` | `provider_false_negative_corrected` |
| `planner-dev-balanced-hyde-b5-scan-quality` | `hyde_fallback` | `local_search -> answer` | `local_search -> hyde_search -> answer` | `false/true` | `provider_false_negative_corrected` |
| `planner-dev-balanced-hyde-p5-internal-jam` | `hyde_fallback` | `local_search -> answer` | `local_search -> answer` | `false/false` | `persistent_model_failure` |
| `planner-dev-balanced-hyde-p5-print-serial-page` | `hyde_fallback` | `local_search -> answer` | `local_search -> hyde_search -> answer` | `false/true` | `provider_false_negative_corrected` |
| `planner-dev-balanced-hyde-rs12-high-current-duration` | `hyde_fallback` | `local_search -> refuse` | `local_search -> hyde_search -> answer` | `false/true` | `provider_false_negative_corrected` |
| `planner-dev-balanced-local-rs12-10a-current` | `local_answer` | `local_search -> answer` | `local_search -> answer` | `true/true` | `unchanged_pass` |
| `planner-dev-balanced-refuse-b5-force-pull-paper` | `safe_refuse` | `local_search -> ask_clarification` | `local_search -> hyde_search` | `false/false` | `persistent_model_failure` |
| `planner-dev-balanced-refuse-p5-touch-hot-surface` | `safe_refuse` | `local_search -> ask_clarification` | `ask_clarification` | `false/false` | `persistent_model_failure` |
| `planner-dev-balanced-refuse-rs12-10a-five-minutes` | `safe_refuse` | `local_search -> ask_clarification` | `local_search -> ask_clarification` | `false/false` | `persistent_model_failure` |
| `planner-dev-balanced-refuse-rs12-com-over-500v` | `safe_refuse` | `local_search -> ask_clarification` | `local_search -> answer` | `false/false` | `persistent_model_failure` |
| `planner-dev-balanced-refuse-rs12-live-continuity` | `safe_refuse` | `local_search -> ask_clarification` | `local_search -> answer` | `false/false` | `persistent_model_failure` |
| `planner-dev-balanced-web-b5-firmware-upgrade-guidance` | `web_required` | `local_search -> ask_clarification` | `local_search` | `false/false` | `persistent_model_failure` |
| `planner-dev-balanced-web-b5-shared-client-systems` | `web_required` | `local_search -> ask_clarification` | `local_search` | `false/false` | `persistent_model_failure` |
| `planner-dev-balanced-web-p5-drum-replacement-guidance` | `web_required` | `local_search -> ask_clarification` | `local_search` | `false/false` | `persistent_model_failure` |
| `planner-dev-balanced-web-p5-official-print-specs` | `web_required` | `local_search -> ask_clarification` | `local_search` | `false/false` | `persistent_model_failure` |
| `planner-dev-balanced-web-p5-product-os-list` | `web_required` | `local_search -> ask_clarification` | `local_search` | `false/false` | `persistent_model_failure` |
| `planner-dev-dev-p3000-driver-missing` | `local_answer` | `local_search -> answer` | `local_search -> answer` | `true/true` | `unchanged_pass` |
| `planner-dev-dev-p3000-duplex-paper` | `local_answer` | `local_search -> answer` | `local_search -> answer` | `true/true` | `unchanged_pass` |
| `planner-dev-dev-p3030-paper-spec` | `local_answer` | `local_search -> answer` | `local_search -> answer` | `true/true` | `unchanged_pass` |
| `planner-dev-dev-p3500-network-info` | `local_answer` | `local_search -> answer` | `local_search -> answer` | `true/true` | `unchanged_pass` |

## 输入身份

| input（输入） | path（路径） | SHA256（文件内容哈希） |
|---|---|---|
| `planner_cases` | `evaluation/stage8/cases/planner_cases.jsonl` | `1ab1c169ce1a4bd9cdb4be9a868494755012c008e432c2aa03c2b4ff198dc19b` |
| `split_manifest` | `evaluation/stage8/cases/split_manifest.json` | `5451424b5cbd5b520a9779fcc4bb208eb5d5868e7d313e82850d46f81a7215d0` |
| `environment_snapshot` | `evaluation/stage9/artifacts/heldout_route_test/environment_snapshot.json` | `fb2e1fbc858ee72eabddf190ccde2bd96945b36dd5626cb4a84b575cbbbb0160` |
| `route_matrix` | `evaluation/stage9/configs/planner_eval_route_matrix_v1.json` | `e9345a2e04d76ee825eacae82aa26bf867528e3eea513c134a545dbb76948383` |
| `reward_profile_v1_1` | `evaluation/stage9/configs/reward_v1_1_training_profile.json` | `2d4fc0f92c51beaf655fed57d2f5d3b43abaa50eeb4117ddb66b694f901ccf4e` |
| `reward_implementation` | `app/rag/evaluation/reward.py` | `1413877cd57a5000c23516359467f67c6170fbce141128a8ed0a1b60a3ed1f72` |
| `provider_records_9_3_18` | `evaluation/stage9/artifacts/provider_records/expanded_dev_provider_observations.jsonl` | `3513e5e550dbe182ce55b9d7c3e461b280a9bc2ad64ed51aa5cc2747d0a1e1e7` |
| `replay_contract_9_3_18` | `evaluation/stage9/artifacts/provider_records/expanded_dev_replay_contract.json` | `559ab751eb1ff85b50cd1421f5884e44fbd32307cea3ef329548064490211e7b` |
| `checkpoint_manifest` | `evaluation/stage9/artifacts/sft/checkpoints/planner-sft-stage9-qwen3-5-4b-lora_20260727T085537Z_94a77563/checkpoint_manifest.json` | `453c41504b95143eb9a9a9cd5dc4fa62470ad518b162dcb8de23b88ea6f74d20` |
| `checkpoint_training_config` | `evaluation/stage9/artifacts/sft/checkpoints/planner-sft-stage9-qwen3-5-4b-lora_20260727T085537Z_94a77563/training_config.json` | `eb7b046880ee80d908b95f2841c81c99079d8976d1c563974dbace40f3d739b0` |
| `checkpoint_train_metrics` | `evaluation/stage9/artifacts/sft/checkpoints/planner-sft-stage9-qwen3-5-4b-lora_20260727T085537Z_94a77563/train_metrics.json` | `a0732a3a01bb9254d61f5df8d3ef0a74970ce624e0ebc147088682f22b54acb0` |
| `adapter_config` | `evaluation/stage9/artifacts/sft/checkpoints/planner-sft-stage9-qwen3-5-4b-lora_20260727T085537Z_94a77563/model/adapter/adapter_config.json` | `5653942d00d2253b3bab0fc7c4cd44513733f209c68bda5c45236693587214a4` |
| `adapter_weights` | `evaluation/stage9/artifacts/sft/checkpoints/planner-sft-stage9-qwen3-5-4b-lora_20260727T085537Z_94a77563/model/adapter/adapter_model.safetensors` | `8ee109d93d74da046ff874765707bd1409a258742b25aa6bfadec2f6d787083b` |
| `tokenizer_json` | `evaluation/stage9/artifacts/sft/checkpoints/planner-sft-stage9-qwen3-5-4b-lora_20260727T085537Z_94a77563/tokenizer/tokenizer.json` | `8818dc7a3be5f461790e3a81703816f482925f6c2ff9fef5a9fc4b821e5051f2` |
| `tokenizer_config` | `evaluation/stage9/artifacts/sft/checkpoints/planner-sft-stage9-qwen3-5-4b-lora_20260727T085537Z_94a77563/tokenizer/tokenizer_config.json` | `f842671db16546726c2818d71390d4b7cd2f7761b208818476473499e03e9883` |
| `chat_template` | `evaluation/stage9/artifacts/sft/checkpoints/planner-sft-stage9-qwen3-5-4b-lora_20260727T085537Z_94a77563/tokenizer/chat_template.jinja` | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` |
| `old_9_3_16_eval` | `evaluation/stage9/artifacts/sft/sft_expanded_dev_eval.json` | `d754b2117b7a36de27d4305f324aca563a16082e9a7f8927fc32bb669598a0f4` |
| `corrected_replay_eval` | `evaluation/stage9/artifacts/cloud_runs/sft_v1_corrected_replay_20260731T091301Z/sft_v1_corrected_replay_eval.json` | `76357e80fdcd8dfb1d5a490f89f9680e71b191630221b59d8d2aa7ff8636a874` |
