"""阶段 9 Planner（规划器）SFT（监督微调）训练入口。"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.stage9.model_planner.checkpoint_io import (  # noqa: E402
    CheckpointManifest,
    Stage9SftTrainingConfig,
    TrainingBackend,
    TuningMethod,
    collect_framework_versions,
    create_checkpoint_dir,
    current_code_version,
    load_training_config,
    write_json,
)
from app.rag.query.model_planner.decision_codec import DECISION_CODEC_VERSION  # noqa: E402
from app.rag.query.model_planner.prompt_builder import PROMPT_BUILDER_VERSION, PlannerPromptConfig  # noqa: E402
from evaluation.stage9.model_planner.sft_dataset import (  # noqa: E402
    SftDatasetStats,
    SftTrainExample,
    load_sft_manifest,
    load_sft_train_examples,
    write_examples_preview,
)
from evaluation.stage9.model_planner.model_profile import load_model_profile  # noqa: E402


TRAINER_VERSION = "stage9-planner-sft-trainer-v1"


def run_sft_training(config: Stage9SftTrainingConfig) -> CheckpointManifest:
    """
    执行一次 SFT（监督微调）训练或本地 smoke（冒烟）训练。

    debug_memorized（调试记忆）后端会保存 prompt_hash -> target_json 映射，用来验证
    checkpoint（检查点）、inference（推理）和 OfflineRagEnvironment（离线 RAG 环境）链路；
    transformers_causal_lm（因果语言模型训练）后端才会真实更新大模型权重。
    """

    _set_seed(config.seed)
    prompt_config = PlannerPromptConfig(max_input_chars=config.max_input_chars)
    examples, dataset_stats = load_sft_train_examples(
        PROJECT_ROOT / config.train_data,
        prompt_config=prompt_config,
        max_samples=config.max_train_samples,
    )
    manifest_payload = load_sft_manifest(PROJECT_ROOT / config.train_manifest)
    reward_profile = json.loads((PROJECT_ROOT / config.reward_profile).read_text(encoding="utf-8"))
    _validate_config_against_inputs(config, manifest_payload, reward_profile)
    model_profile = _load_config_model_profile(config)

    run_id, checkpoint_dir = create_checkpoint_dir(config)
    model_dir = checkpoint_dir / "model"
    tokenizer_dir = checkpoint_dir / "tokenizer"
    metrics_path = checkpoint_dir / "train_metrics.json"
    config_path = checkpoint_dir / "training_config.json"
    preview_path = checkpoint_dir / "training_examples_preview.jsonl"

    write_json(config_path, config.model_dump(mode="json"))
    if config.save_training_preview_count:
        write_examples_preview(examples, preview_path, limit=config.save_training_preview_count)

    if config.training_backend == TrainingBackend.DEBUG_MEMORIZED:
        backend_metrics = _run_debug_memorized_backend(examples, model_dir)
        tokenizer_path = ""
    else:
        backend_metrics = _run_transformers_backend(
            config=config,
            model_profile=model_profile,
            run_id=run_id,
            examples=examples,
            model_dir=model_dir,
            tokenizer_dir=tokenizer_dir,
        )
        tokenizer_path = _portable_path(tokenizer_dir)

    train_metrics = {
        "trainer_version": TRAINER_VERSION,
        "training_backend": config.training_backend.value,
        "dataset": dataset_stats.model_dump(mode="json"),
        "source_manifest_id": manifest_payload.get("manifest_id", ""),
        "reward_profile": reward_profile.get("profile_name", ""),
        "model_profile": model_profile.model_dump(mode="json") if model_profile else None,
        "backend_metrics": backend_metrics,
    }
    write_json(metrics_path, train_metrics)

    manifest = CheckpointManifest(
        run_id=run_id,
        run_name=config.run_name,
        policy_version=f"{config.training_backend.value}:{run_id}",
        training_backend=config.training_backend,
        base_model_id=config.base_model_id,
        model_profile_id=model_profile.profile_id if model_profile else config.model_profile_id,
        model_profile=model_profile,
        tuning_method=backend_metrics.get("tuning_method", config.tuning_method.value),
        adapter_id=backend_metrics.get("adapter_id", ""),
        adapter_path=backend_metrics.get("adapter_path", ""),
        quantization=backend_metrics.get("quantization", ""),
        peft_config=backend_metrics.get("peft_config", {}),
        train_data=config.train_data,
        train_manifest=config.train_manifest,
        reward_profile=config.reward_profile,
        snapshot_id=config.snapshot_id,
        code_version=current_code_version(),
        created_at=datetime_now_utc(),
        seed=config.seed,
        framework_versions=collect_framework_versions(),
        prompt_builder_version=PROMPT_BUILDER_VERSION,
        decision_codec_version=DECISION_CODEC_VERSION,
        model_path=backend_metrics.get("model_path", _portable_path(model_dir)),
        tokenizer_path=tokenizer_path,
        train_metrics_path=_portable_path(metrics_path),
        training_config_path=_portable_path(config_path),
        sample_count=dataset_stats.sample_count,
        source_case_count=dataset_stats.source_case_count,
        action_counts=dataset_stats.action_counts,
        reason_code_counts=dataset_stats.reason_code_counts,
        max_input_tokens=config.max_input_tokens,
        max_target_tokens=config.max_target_tokens,
    )
    write_json(checkpoint_dir / "checkpoint_manifest.json", manifest.model_dump(mode="json"))
    return manifest


def _run_debug_memorized_backend(examples: list[SftTrainExample], model_dir: Path) -> dict[str, Any]:
    """
    本地 smoke（冒烟）后端。

    它不更新大模型权重，只保存上下文到目标 JSON 的确定性映射，专门用于确认后续推理入口、
    codec（编解码器）和离线环境可以加载 checkpoint（检查点）并输出合法决策。
    """

    model_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, dict[str, Any]] = {}
    for example in examples:
        mapping[example.context_key] = {
            "sample_id": example.sample_id,
            "source_case_id": example.source_case_id,
            "turn_index": example.turn_index,
            "target_json": example.target_json,
        }
    payload = {
        "backend": TrainingBackend.DEBUG_MEMORIZED.value,
        "version": "stage9-debug-memorized-policy-v1",
        "prompt_builder_version": PROMPT_BUILDER_VERSION,
        "decision_codec_version": DECISION_CODEC_VERSION,
        "context_key_to_target": mapping,
    }
    write_json(model_dir / "debug_memorized_policy.json", payload)
    return {
        "updated_model_weights": False,
        "memorized_context_count": len(mapping),
        "note": "本地 smoke 后端只验证训练/推理/评测链路，不代表真实 SFT 模型效果。",
    }


def _run_transformers_backend(
        *,
        config: Stage9SftTrainingConfig,
        model_profile: Any | None,
        run_id: str,
        examples: list[SftTrainExample],
        model_dir: Path,
        tokenizer_dir: Path,
) -> dict[str, Any]:
    """
    transformers（大模型训练框架）后端。

    这个函数是云端 GPU（显卡算力）正式放大的入口；本地单元测试不会执行它。训练数据和
    debug_memorized（调试记忆）后端共用同一套 prompt（提示词）与 target_json（目标 JSON）。
    """

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    # model_load_id（模型加载身份）优先来自 profile（配置档案），允许云端用本地模型目录加载，
    # 同时让 base_model_id（基础模型身份）继续保持稳定审计语义。
    model_load_id = model_profile.training_model_id if model_profile else config.base_model_id
    # tokenizer（分词器）必须和训练加载的基础模型一致，否则 prompt（提示词）和 target_json（目标 JSON）
    # 的 token（分词单元）切分会漂，后续 checkpoint（检查点）推理也无法复现训练输入。
    tokenizer = AutoTokenizer.from_pretrained(
        model_load_id,
        trust_remote_code=config.trust_remote_code,
        use_fast=config.use_fast_tokenizer,
    )
    # 部分 causal LM（因果语言模型）没有显式 pad_token（填充符），这里优先复用 eos/unk，
    # 只影响 batch padding（批量填充），不会改变监督目标。
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    # 如果仍然没有 pad/eos/unk token（填充/结束/未知符），批量训练无法构造等长 tensor（张量）。
    if tokenizer.pad_token is None:
        raise ValueError("tokenizer 必须有 pad/eos/unk token 才能进行批量训练")

    # model_kwargs（模型加载参数）只放真实加载时需要的框架参数，避免把项目审计字段传给 transformers。
    model_kwargs: dict[str, Any] = {"trust_remote_code": config.trust_remote_code}
    # torch_dtype（张量精度）用于 LoRA（低秩适配）和 full（全量微调）加载；QLoRA 的计算精度另由 bnb 配置控制。
    dtype = _torch_dtype(config.torch_dtype, torch)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    # QLoRA（4 位量化低秩适配）需要 bitsandbytes 的 4bit（4 位）量化配置，减少基础模型显存占用。
    if config.tuning_method == TuningMethod.QLORA:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=config.load_in_4bit,
            bnb_4bit_compute_dtype=_torch_dtype(config.bnb_compute_dtype, torch),
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        )
        # 4bit（4 位）模型不能再随意 model.to(device)，因此加载阶段直接交给 device_map（设备映射）放置。
        model_kwargs["device_map"] = "auto" if config.device == "auto" else {"": config.device}

    # 这里真正加载基础模型；LoRA/QLoRA 都是在这个基础模型上挂 adapter（适配器）。
    model = AutoModelForCausalLM.from_pretrained(model_load_id, **model_kwargs)
    # 非 QLoRA 场景仍允许显式把模型移动到 cuda/mps/cpu；QLoRA 的设备放置已由 device_map 管。
    if config.tuning_method != TuningMethod.QLORA and config.device != "auto":
        model.to(config.device)
    # gradient checkpointing（梯度检查点）用计算换显存，云端大模型微调默认打开更稳。
    if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    # LoRA/QLoRA 只训练 adapter（适配器）；full（全量微调）保持原模型可训练。
    if config.tuning_method in {TuningMethod.LORA, TuningMethod.QLORA}:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

        # QLoRA（4 位量化低秩适配）在挂 LoRA adapter 前要先准备量化模型的梯度与输入层。
        if config.tuning_method == TuningMethod.QLORA:
            try:
                model = prepare_model_for_kbit_training(
                    model,
                    use_gradient_checkpointing=config.gradient_checkpointing,
                )
            except TypeError:
                model = prepare_model_for_kbit_training(model)
        # LoraConfig（低秩适配配置）决定哪些模块挂 adapter，以及 adapter 的秩、缩放和 dropout。
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        # get_peft_model（创建参数高效微调模型）会冻结基础模型，只让 adapter 参数 requires_grad=True。
        model = get_peft_model(model, lora_config)

    # Dataset（数据集）只保存 prompt 和 target_json，保持和本地 smoke（冒烟）训练完全同源。
    dataset = Dataset.from_list([
        {"prompt": example.prompt, "target_json": example.target_json}
        for example in examples
    ])

    def tokenize(record: dict[str, str]) -> dict[str, list[int]]:
        # prompt_ids（提示词 token）参与前向计算，但不参与 loss（损失）监督。
        prompt_ids = tokenizer(
            record["prompt"],
            add_special_tokens=True,
            truncation=True,
            max_length=config.max_input_tokens,
        )["input_ids"]
        # target_text（目标文本）只包含 PlannerDecision（规划器决策）JSON，末尾补 eos 让模型学会停止。
        target_text = record["target_json"] + (tokenizer.eos_token or "")
        # target_ids（目标 token）才是 SFT（监督微调）真正要预测的部分。
        target_ids = tokenizer(
            target_text,
            add_special_tokens=False,
            truncation=True,
            max_length=config.max_target_tokens,
        )["input_ids"]
        # labels（训练标签）里 prompt 部分填 -100，表示 Trainer（训练器）计算 loss 时忽略上下文。
        return {
            "input_ids": prompt_ids + target_ids,
            "attention_mask": [1] * (len(prompt_ids) + len(target_ids)),
            "labels": [-100] * len(prompt_ids) + target_ids,
        }

    # tokenized（已分词数据）在训练开始前一次性生成，避免每个 step 重复构造 prompt/target。
    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

    def collate(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        # pad_id（填充 token）用于把同一个 batch（批量）里的不同长度样本补齐成矩阵。
        pad_id = int(tokenizer.pad_token_id)
        # max_length（本批最大长度）只按当前 batch 动态补齐，减少无效 padding（填充）计算。
        max_length = max(len(item["input_ids"]) for item in batch)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for item in batch:
            # pad_count（填充数量）表示当前样本距离本批最大长度还差多少 token。
            pad_count = max_length - len(item["input_ids"])
            # input_ids（输入 token）用 tokenizer 的 pad_id 补齐。
            input_ids.append(item["input_ids"] + [pad_id] * pad_count)
            # attention_mask（注意力掩码）中 padding 部分填 0，避免模型关注填充 token。
            attention_mask.append(item["attention_mask"] + [0] * pad_count)
            # labels（训练标签）中 padding 部分也填 -100，避免 padding 进入 loss。
            labels.append(item["labels"] + [-100] * pad_count)
        # Trainer（训练器）期望 collate 返回 torch.Tensor（张量）字典。
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    # TrainingArguments（训练参数）只负责训练循环、batch（批量）、保存策略和日志，不保存业务审计字段。
    args = TrainingArguments(
        output_dir=str(model_dir),
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps if config.max_steps is not None else -1,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        logging_steps=1,
        save_strategy="no",
        report_to=config.report_to,
        seed=config.seed,
        remove_unused_columns=False,
        bf16=config.torch_dtype == "bfloat16",
        fp16=config.torch_dtype == "float16",
    )
    # Trainer（训练器）接管反向传播；模型是否为 full/lora/qlora 已在上方准备完成。
    trainer = Trainer(model=model, args=args, train_dataset=tokenized, data_collator=collate)
    # train_output（训练输出）只保存关键数值，完整审计信息由 checkpoint manifest（检查点清单）负责。
    train_output = trainer.train()
    # tokenizer（分词器）独立保存，推理时必须复用同一个 tokenizer 才能稳定还原训练分布。
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)
    # total/trainable parameter（总参数/可训练参数）用于确认 LoRA/QLoRA 没有误训全量权重。
    total_parameters, trainable_parameters = _count_parameters(model)
    # adapter_id（适配器身份）进入 manifest（清单），后续 SFT/GRPO 对比不能只靠目录名猜。
    adapter_id = _build_adapter_id(config, run_id)
    # LoRA/QLoRA 只保存 adapter（适配器）；full 才保存完整模型权重。
    if config.tuning_method in {TuningMethod.LORA, TuningMethod.QLORA}:
        adapter_dir = model_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir)
        model_path = _portable_path(adapter_dir)
        adapter_path = model_path
    else:
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_dir)
        model_path = _portable_path(model_dir)
        adapter_path = ""
        adapter_id = ""
    return {
        "updated_model_weights": True,
        "model_load_id": str(model_load_id),
        "model_path": model_path,
        "adapter_id": adapter_id,
        "adapter_path": adapter_path,
        "tuning_method": config.tuning_method.value,
        "quantization": _quantization_label(config),
        "peft_config": _peft_config_snapshot(config),
        "trainable_parameter_count": trainable_parameters,
        "total_parameter_count": total_parameters,
        "train_loss": float(train_output.training_loss),
        "global_step": int(train_output.global_step),
    }


def _build_adapter_id(config: Stage9SftTrainingConfig, run_id: str) -> str:
    """生成 adapter_id（适配器身份），把模型 profile、训练方法和 run_id 串起来便于审计。"""

    profile_id = config.model_profile_id or "unknown_profile"
    return f"{profile_id}:{config.tuning_method.value}:{run_id}"


def _quantization_label(config: Stage9SftTrainingConfig) -> str:
    """返回 quantization（量化方式）标签，写入 checkpoint manifest（检查点清单）。"""

    if config.tuning_method == TuningMethod.QLORA:
        return f"4bit:{config.bnb_4bit_quant_type}:{config.bnb_compute_dtype}"
    return "none"


def _peft_config_snapshot(config: Stage9SftTrainingConfig) -> dict[str, Any]:
    """
    构造 PEFT（参数高效微调）配置快照。

    full（全量微调）没有 adapter（适配器），返回空对象；LoRA/QLoRA 需要记录完整参数，
    否则后续报告只能知道“用了适配器”，但不知道训练容量和目标模块。
    """

    if config.tuning_method == TuningMethod.FULL:
        return {}
    return {
        "tuning_method": config.tuning_method.value,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "target_modules": list(config.target_modules),
        "load_in_4bit": config.load_in_4bit,
        "bnb_compute_dtype": config.bnb_compute_dtype,
        "bnb_4bit_quant_type": config.bnb_4bit_quant_type,
        "bnb_4bit_use_double_quant": config.bnb_4bit_use_double_quant,
        "gradient_checkpointing": config.gradient_checkpointing,
    }


def _count_parameters(model: Any) -> tuple[int, int]:
    """
    统计 total/trainable parameters（总参数量/可训练参数量）。

    LoRA/QLoRA 的核心验收点就是可训练参数远小于总参数；这个统计会进入 train_metrics（训练指标）。
    """

    total = 0
    trainable = 0
    for parameter in model.parameters():
        parameter_count = int(parameter.numel())
        total += parameter_count
        if parameter.requires_grad:
            trainable += parameter_count
    return total, trainable


def _validate_config_against_inputs(
        config: Stage9SftTrainingConfig,
        manifest_payload: dict[str, Any],
        reward_profile: dict[str, Any],
) -> None:
    manifest_snapshot = str(manifest_payload.get("snapshot_id") or "").strip()
    if manifest_snapshot and manifest_snapshot != config.snapshot_id:
        raise ValueError(
            f"训练配置 snapshot_id={config.snapshot_id} 与 SFT manifest snapshot_id={manifest_snapshot} 不一致"
        )
    reward_version = str(reward_profile.get("reward_version") or "").strip()
    manifest_reward = str(manifest_payload.get("reward_version") or "").strip()
    if reward_version and manifest_reward and reward_version != manifest_reward:
        raise ValueError(
            f"Reward profile={reward_version} 与 SFT manifest reward_version={manifest_reward} 不一致"
        )


def _load_config_model_profile(config: Stage9SftTrainingConfig):
    """
    读取并校验训练绑定的 model profile（模型配置档案）。

    debug_memorized（调试记忆）允许为空；真实 transformers（训练框架）后端必须绑定 profile，
    否则 checkpoint manifest（检查点清单）无法追踪模型身份和模板边界。
    """

    if not config.model_profile_id and not config.model_profile_path:
        return None
    profile_ref = config.model_profile_path or config.model_profile_id
    profile = load_model_profile(profile_ref)
    if config.model_profile_id and config.model_profile_id != profile.profile_id:
        raise ValueError(
            f"训练配置 model_profile_id={config.model_profile_id} 与 profile_id={profile.profile_id} 不一致"
        )
    if profile.base_model_id != config.base_model_id:
        raise ValueError(
            f"训练配置 base_model_id={config.base_model_id} 与 model profile base_model_id={profile.base_model_id} 不一致"
        )
    if profile.enable_thinking:
        raise ValueError("Planner SFT model profile 必须设置 enable_thinking=false")
    if config.max_input_tokens > profile.max_context_tokens:
        raise ValueError(
            f"max_input_tokens={config.max_input_tokens} 超过 profile max_context_tokens={profile.max_context_tokens}"
        )
    if config.max_target_tokens > profile.max_target_tokens:
        raise ValueError(
            f"max_target_tokens={config.max_target_tokens} 超过 profile max_target_tokens={profile.max_target_tokens}"
        )
    return profile


def _torch_dtype(raw_dtype: str, torch_module: Any) -> Any:
    if raw_dtype == "auto":
        return None
    mapping = {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }
    if raw_dtype not in mapping:
        raise ValueError(f"不支持的 torch_dtype：{raw_dtype}")
    return mapping[raw_dtype]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def datetime_now_utc() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")


def _portable_path(path: Path) -> str:
    """项目内路径写相对路径；项目外临时路径写绝对路径，便于测试和云端挂载目录共用。"""

    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段 9 Planner SFT 训练入口。")
    parser.add_argument("--config", required=True, type=Path, help="训练配置 JSON。")
    args = parser.parse_args(argv)
    manifest = run_sft_training(load_training_config(args.config))
    checkpoint_dir = (PROJECT_ROOT / manifest.model_path).parent
    print(f"run_id={manifest.run_id}")
    print(f"policy_version={manifest.policy_version}")
    print(f"training_backend={manifest.training_backend.value}")
    print(f"checkpoint={checkpoint_dir}")
    print(f"sample_count={manifest.sample_count}")
    print(json.dumps(manifest.action_counts, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
