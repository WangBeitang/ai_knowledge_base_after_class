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
        model_path=_portable_path(model_dir),
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

    model_load_id = model_profile.training_model_id if model_profile else config.base_model_id
    tokenizer = AutoTokenizer.from_pretrained(
        model_load_id,
        trust_remote_code=config.trust_remote_code,
        use_fast=config.use_fast_tokenizer,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token is None:
        raise ValueError("tokenizer 必须有 pad/eos/unk token 才能进行批量训练")

    model_kwargs: dict[str, Any] = {"trust_remote_code": config.trust_remote_code}
    dtype = _torch_dtype(config.torch_dtype, torch)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_load_id, **model_kwargs)
    if config.device != "auto":
        model.to(config.device)

    dataset = Dataset.from_list([
        {"prompt": example.prompt, "target_json": example.target_json}
        for example in examples
    ])

    def tokenize(record: dict[str, str]) -> dict[str, list[int]]:
        prompt_ids = tokenizer(
            record["prompt"],
            add_special_tokens=True,
            truncation=True,
            max_length=config.max_input_tokens,
        )["input_ids"]
        target_text = record["target_json"] + (tokenizer.eos_token or "")
        target_ids = tokenizer(
            target_text,
            add_special_tokens=False,
            truncation=True,
            max_length=config.max_target_tokens,
        )["input_ids"]
        return {
            "input_ids": prompt_ids + target_ids,
            "attention_mask": [1] * (len(prompt_ids) + len(target_ids)),
            "labels": [-100] * len(prompt_ids) + target_ids,
        }

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

    def collate(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        pad_id = int(tokenizer.pad_token_id)
        max_length = max(len(item["input_ids"]) for item in batch)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for item in batch:
            pad_count = max_length - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * pad_count)
            attention_mask.append(item["attention_mask"] + [0] * pad_count)
            labels.append(item["labels"] + [-100] * pad_count)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

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
    )
    trainer = Trainer(model=model, args=args, train_dataset=tokenized, data_collator=collate)
    train_output = trainer.train()
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(tokenizer_dir)
    return {
        "updated_model_weights": True,
        "model_load_id": str(model_load_id),
        "train_loss": float(train_output.training_loss),
        "global_step": int(train_output.global_step),
    }


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
