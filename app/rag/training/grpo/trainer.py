"""真实 Qwen Planner GRPO（群组相对策略优化）训练器。"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import uuid
from collections import Counter
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from app.rag.evaluation.action_providers import RecordingActionProvider, RemoteRealActionProvider
from app.rag.evaluation.case_schema import EnvironmentSnapshot
from app.rag.evaluation.grpo_case_exporter import GrpoTrainingCase, load_grpo_training_cases
from app.rag.evaluation.offline_environment import OfflineRagEnvironment, OfflineTrajectoryResult
from app.rag.evaluation.reward import RewardConfig, RewardWeights, TrajectoryReward, score_trajectory
from app.rag.query.contracts import PlannerContext, PlannerDecision
from app.rag.query.model_planner.checkpoint_runtime import CheckpointManifest, TuningMethod
from app.rag.query.model_planner.decision_codec import DecisionDecodeResult, decode_decision
from app.rag.query.model_planner.prompt_builder import PlannerPromptConfig, build_planner_prompt
from app.rag.training.grpo.config import FormalGrpoConfig
from app.rag.training.grpo.objective import (
    completion_token_log_probs,
    compute_group_advantages,
    grpo_token_objective,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
TRAINER_VERSION = "formal-qwen-planner-grpo-v2-memory-efficient"
FATAL_PROVIDER_ERRORS = {
    "action_provider_failed",
    "provider_recording_failed",
    "candidate_disabled_by_snapshot",
    "candidate_not_in_snapshot",
}


class PolicySamplingPlanner:
    """
    使用当前 policy model（策略模型）逐步采样 PlannerDecision（规划器决策）。

    每次 plan 调用保存模型原始 JSON、token ID 和 old log probability（旧策略对数概率）；
    Environment（环境）仍负责 Action（动作）合法转移和真实 Provider（动作执行器）调用。
    """

    def __init__(
            self,
            *,
            model: Any,
            tokenizer: Any,
            device: Any,
            policy_version: str,
            config: FormalGrpoConfig,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.policy_version = policy_version
        self.config = config
        self.prompt_config = PlannerPromptConfig(max_input_chars=config.max_input_chars)
        self.decision_records: list[dict[str, Any]] = []

    def plan(self, context: PlannerContext) -> PlannerDecision:
        """采样一段真实模型输出，并经正式 codec（编解码器）校验。"""

        import torch

        prompt = build_planner_prompt(context, config=self.prompt_config)
        encoded = self.tokenizer(
            prompt.prompt,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=self.config.max_input_tokens,
        )
        prompt_ids = encoded["input_ids"][0].tolist()
        inputs = {name: value.to(self.device) for name, value in encoded.items()}
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=1.0,
                top_k=0,
                max_new_tokens=self.config.max_new_tokens,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        completion_ids = generated.sequences[0, len(prompt_ids):].tolist()
        if not completion_ids:
            raise RuntimeError("模型没有生成 PlannerDecision token")
        with torch.no_grad():
            old_log_probs = completion_token_log_probs(
                self.model,
                prompt_token_ids=prompt_ids,
                completion_token_ids=completion_ids,
                device=self.device,
            ).detach().float().cpu().tolist()
        raw_output = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        decode_result = decode_decision(
            raw_output,
            allowed_actions=context.allowed_actions,
        )
        self.decision_records.append({
            "step": len(self.decision_records) + 1,
            "prompt_hash": prompt.payload_hash,
            "prompt_truncated": prompt.truncation_applied,
            "allowed_actions": [action.value for action in context.allowed_actions],
            "raw_model_output": raw_output,
            "decode_result": decode_result.model_dump(mode="json"),
            "prompt_token_count": len(prompt_ids),
            "completion_token_count": len(completion_ids),
            "prompt_token_ids": prompt_ids,
            "completion_token_ids": completion_ids,
            "old_log_probs": old_log_probs,
        })
        if not decode_result.success or decode_result.decision is None:
            raise ValueError(
                "模型输出不是合法 PlannerDecision："
                f"{decode_result.error_code} {decode_result.error_message}"
            )
        return decode_result.decision


def _enable_policy_gradient_checkpointing(policy_model: Any) -> None:
    """为可训练 policy model（策略模型）启用非重入式 gradient checkpointing（梯度检查点）。"""

    enable = getattr(policy_model, "gradient_checkpointing_enable", None)
    if not callable(enable):
        raise RuntimeError("policy model 不支持 gradient checkpointing，无法安全执行正式 GRPO")
    enable(gradient_checkpointing_kwargs={"use_reentrant": False})


def _set_policy_optimization_mode(policy_model: Any, torch_module: Any) -> None:
    """开启训练模式以激活梯度检查点，同时关闭 dropout，保持 old/new probability 可比。"""

    policy_model.train()
    for module in policy_model.modules():
        if isinstance(module, torch_module.nn.Dropout):
            module.eval()


def run_formal_grpo_training(
        config: FormalGrpoConfig,
        *,
        command_text: str = "",
        resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """完成 75 组、300 条 rollout、1 个 epoch 的正式 LoRA GRPO 训练。"""

    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("正式 GRPO 要求 CUDA GPU，但当前训练进程未检测到 CUDA")
    _set_seed(config.seed, torch)
    inputs = _preflight_inputs(config)
    cases: list[GrpoTrainingCase] = inputs["cases"]
    snapshot: EnvironmentSnapshot = inputs["snapshot"]
    reward_config: RewardConfig = inputs["reward_config"]
    sft_manifest: CheckpointManifest = inputs["sft_manifest"]

    resume_checkpoint = _resolve_path(resume_from) if resume_from is not None else None
    if resume_checkpoint is None:
        run_id, run_dir = _create_run_dir(config)
        _write_json(run_dir / "training_config.json", config.model_dump(mode="json"))
        _write_json(run_dir / "input_manifest.json", inputs["input_manifest"])
        (run_dir / "command.txt").write_text(command_text.strip() + "\n", encoding="utf-8")
        _write_environment_freeze(run_dir / "environment_freeze.json", torch)
        resume_payload = None
    else:
        run_id, run_dir, resume_payload = _prepare_resume(
            resume_checkpoint,
            config=config,
            current_input_manifest=inputs["input_manifest"],
            torch_module=torch,
        )
        resume_command_dir = run_dir / "resume_commands"
        resume_command_dir.mkdir(parents=True, exist_ok=True)
        resume_step = int(resume_payload["trainer_state"]["optimizer_step"])
        (resume_command_dir / f"resume_step_{resume_step:06d}.txt").write_text(
            command_text.strip() + "\n",
            encoding="utf-8",
        )
        _write_environment_freeze(
            run_dir / f"environment_freeze_resume_step_{resume_step:06d}.json",
            torch,
        )

    provider = RemoteRealActionProvider(
        config.provider_endpoint,
        timeout_seconds=config.provider_timeout_seconds,
    )
    provider_health = provider.health(
        expected_snapshot_id=snapshot.snapshot_id,
        expected_snapshot_sha256=config.expected_environment_snapshot_sha256,
    )
    provider_health_path = (
        run_dir / "provider_health.json"
        if resume_payload is None
        else run_dir / f"provider_health_resume_step_{resume_step:06d}.json"
    )
    _write_json(provider_health_path, provider_health)
    recording_provider = RecordingActionProvider(
        provider,
        output_path=run_dir / "provider_observations.jsonl",
        max_candidate_content_chars=500,
    )
    environment = OfflineRagEnvironment(
        snapshot=snapshot,
        action_provider=recording_provider,
        planner_mode="grpo",
        run_id_prefix=run_id,
        max_steps=config.max_environment_steps,
    )

    dtype = _torch_dtype(config.torch_dtype, torch)
    device = torch.device(config.device)
    tokenizer_path = _resolve_path(sft_manifest.tokenizer_path)
    sft_adapter_path = _resolve_path(sft_manifest.adapter_path or sft_manifest.model_path)
    policy_adapter_path = (
        resume_checkpoint / "adapter"
        if resume_checkpoint is not None
        else sft_adapter_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token_id is None:
        raise RuntimeError("正式 GRPO tokenizer 缺少 pad/eos/unk token")

    model_kwargs = {
        "dtype": dtype,
        "local_files_only": True,
        "device_map": {"": config.device},
    }
    policy_base = AutoModelForCausalLM.from_pretrained(config.base_model_id, **model_kwargs)
    policy_model = PeftModel.from_pretrained(
        policy_base,
        policy_adapter_path,
        is_trainable=True,
    )
    _enable_policy_gradient_checkpointing(policy_model)
    reference_base = AutoModelForCausalLM.from_pretrained(config.base_model_id, **model_kwargs)
    reference_model = PeftModel.from_pretrained(
        reference_base,
        sft_adapter_path,
        is_trainable=False,
    )
    policy_model.eval()
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)

    trainable_parameters = [parameter for parameter in policy_model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("policy model 没有可训练 LoRA 参数")
    reference_versions_before = _parameter_versions(reference_model)
    current_policy_trainable_hash = _parameter_sha256(policy_model, trainable_only=True)
    if resume_payload is None:
        initial_policy_trainable_hash = current_policy_trainable_hash
        _write_json(run_dir / "initial_policy_state.json", {
            "sft_run_id": config.sft_run_id,
            "policy_trainable_sha256": initial_policy_trainable_hash,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        })
    else:
        initial_state = json.loads((run_dir / "initial_policy_state.json").read_text(encoding="utf-8"))
        initial_policy_trainable_hash = initial_state["policy_trainable_sha256"]
        expected_resume_hash = resume_payload["trainer_state"]["policy_trainable_sha256"]
        if current_policy_trainable_hash != expected_resume_hash:
            raise RuntimeError("恢复 adapter 的 LoRA 参数 SHA256 与 checkpoint trainer state 不一致")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_optimizer_steps = config.expected_case_count * config.num_epochs
    warmup_steps = int(total_optimizer_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    if resume_payload is None:
        case_order = list(range(len(cases)))
        random.Random(config.seed).shuffle(case_order)
        optimizer_step = 0
        next_case_index = 0
        processed_case_ids: list[str] = []
        elapsed_before_seconds = 0.0
        last_checkpoint: Path | None = None
    else:
        trainer_state = resume_payload["trainer_state"]
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        scheduler.load_state_dict(resume_payload["scheduler_state"])
        _restore_rng_state(resume_payload["rng_state"], torch)
        case_order = list(trainer_state["case_order"])
        optimizer_step = int(trainer_state["optimizer_step"])
        next_case_index = int(trainer_state["next_case_index"])
        processed_case_ids = list(trainer_state["processed_case_ids"])
        elapsed_before_seconds = float(trainer_state.get("elapsed_seconds", 0.0))
        last_checkpoint = resume_checkpoint
    pending_rollouts: list[dict[str, Any]] = []
    pending_step_metrics: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)

    try:
        for group_index in range(next_case_index, len(case_order)):
            case_index = case_order[group_index]
            case_record = cases[case_index]
            case = case_record.case_contract
            sampled: list[tuple[PolicySamplingPlanner, OfflineTrajectoryResult, TrajectoryReward]] = []
            for sample_index in range(config.group_size):
                planner = PolicySamplingPlanner(
                    model=policy_model,
                    tokenizer=tokenizer,
                    device=device,
                    policy_version=f"{run_id}:old_step_{optimizer_step:06d}",
                    config=config,
                )
                trajectory = environment.run_planner(
                    case,
                    planner,
                    run_id=f"{run_id}_{case.case_id}_g{group_index:03d}_r{sample_index}",
                    planner_mode="grpo",
                )
                _raise_on_provider_contract_failure(trajectory)
                reward = score_trajectory(case, trajectory, reward_config)
                if not math.isfinite(reward.total_reward):
                    raise FloatingPointError(f"case={case.case_id} Reward 出现 NaN/Inf")
                sampled.append((planner, trajectory, reward))

            rewards = torch.tensor(
                [item[2].total_reward for item in sampled],
                dtype=torch.float32,
                device=device,
            )
            advantages = compute_group_advantages(rewards)
            _set_policy_optimization_mode(policy_model, torch)
            group_policy_loss = 0.0
            group_kl = 0.0
            group_total_loss = 0.0
            group_clip_fraction = 0.0

            for sample_index, (planner, trajectory, reward) in enumerate(sampled):
                rollout_token_count = sum(
                    len(record["completion_token_ids"])
                    for record in planner.decision_records
                )
                if rollout_token_count <= 0:
                    raise RuntimeError("rollout 没有模型生成 token")
                decision_audits: list[dict[str, Any]] = []
                rollout_metrics = {"policy_loss": 0.0, "kl": 0.0, "total_loss": 0.0, "clip_fraction": 0.0}
                for decision_record in planner.decision_records:
                    prompt_ids = decision_record["prompt_token_ids"]
                    completion_ids = decision_record["completion_token_ids"]
                    old_log_probs = torch.tensor(
                        decision_record["old_log_probs"],
                        dtype=torch.float32,
                        device=device,
                    )
                    with torch.no_grad():
                        reference_log_probs = completion_token_log_probs(
                            reference_model,
                            prompt_token_ids=prompt_ids,
                            completion_token_ids=completion_ids,
                            device=device,
                        )
                    new_log_probs = completion_token_log_probs(
                        policy_model,
                        prompt_token_ids=prompt_ids,
                        completion_token_ids=completion_ids,
                        device=device,
                    )
                    objective = grpo_token_objective(
                        new_log_probs=new_log_probs,
                        old_log_probs=old_log_probs,
                        reference_log_probs=reference_log_probs,
                        advantage=advantages[sample_index],
                        clip_epsilon=config.clip_epsilon,
                        kl_beta=config.kl_beta,
                    )
                    token_weight = len(completion_ids) / rollout_token_count
                    scaled_loss = objective["total_loss"] * (token_weight / config.group_size)
                    scaled_loss.backward()
                    for name in rollout_metrics:
                        rollout_metrics[name] += float(objective[name].detach().cpu()) * token_weight

                    audit = {
                        name: value
                        for name, value in decision_record.items()
                        if name != "prompt_token_ids"
                    }
                    audit["new_log_probs"] = new_log_probs.detach().float().cpu().tolist()
                    audit["reference_log_probs"] = reference_log_probs.detach().float().cpu().tolist()
                    audit["token_probability_alignment_ok"] = (
                        len(audit["old_log_probs"])
                        == len(audit["new_log_probs"])
                        == len(audit["reference_log_probs"])
                        == len(completion_ids)
                    )
                    decision_audits.append(audit)

                group_policy_loss += rollout_metrics["policy_loss"] / config.group_size
                group_kl += rollout_metrics["kl"] / config.group_size
                group_total_loss += rollout_metrics["total_loss"] / config.group_size
                group_clip_fraction += rollout_metrics["clip_fraction"] / config.group_size
                pending_rollouts.append({
                    "run_id": run_id,
                    "epoch": 1,
                    "group_index": group_index,
                    "sample_index": sample_index,
                    "case_id": case.case_id,
                    "case_record_fingerprint": case_record.record_fingerprint,
                    "reference_target": case_record.reference_trajectory.model_dump(mode="json"),
                    "snapshot_id": snapshot.snapshot_id,
                    "policy_version": planner.policy_version,
                    "reference_policy_version": sft_manifest.policy_version,
                    "model_decisions": decision_audits,
                    "trajectory": trajectory.to_json_trace(),
                    "reward": reward.model_dump(mode="json"),
                    "advantage": float(advantages[sample_index].detach().cpu()),
                    "policy_loss": rollout_metrics["policy_loss"],
                    "kl": rollout_metrics["kl"],
                    "total_loss": rollout_metrics["total_loss"],
                })

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, config.max_grad_norm)
            if not torch.isfinite(torch.as_tensor(grad_norm)):
                raise FloatingPointError("LoRA gradient norm 出现 NaN/Inf")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            processed_case_ids.append(case.case_id)
            step_metric = {
                "optimizer_step": optimizer_step,
                "group_index": group_index,
                "case_id": case.case_id,
                "rollout_count": config.group_size,
                "reward_mean": float(rewards.mean().detach().cpu()),
                "reward_min": float(rewards.min().detach().cpu()),
                "reward_max": float(rewards.max().detach().cpu()),
                "advantage_mean": float(advantages.mean().detach().cpu()),
                "advantage_std": float(advantages.std(unbiased=False).detach().cpu()),
                "policy_loss": group_policy_loss,
                "kl": group_kl,
                "total_loss": group_total_loss,
                "clip_fraction": group_clip_fraction,
                "gradient_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
                "learning_rate": float(scheduler.get_last_lr()[0]),
            }
            _assert_finite_metrics(step_metric)
            pending_step_metrics.append(step_metric)
            print(
                f"formal_grpo case={group_index + 1}/75 id={case.case_id} "
                f"rollouts=4 reward={step_metric['reward_mean']:.4f} "
                f"loss={group_total_loss:.6f} kl={group_kl:.6f}",
                flush=True,
            )

            should_save = (
                optimizer_step % config.save_every_optimizer_steps == 0
                or optimizer_step == total_optimizer_steps
            )
            if should_save:
                last_checkpoint = _save_recovery_checkpoint(
                    run_dir=run_dir,
                    policy_model=policy_model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    optimizer_step=optimizer_step,
                    next_case_index=group_index + 1,
                    processed_case_ids=processed_case_ids,
                    case_order=case_order,
                    rollout_segment=pending_rollouts,
                    metric_segment=pending_step_metrics,
                    previous_checkpoint=last_checkpoint,
                    elapsed_seconds=elapsed_before_seconds + (time.monotonic() - started),
                    policy_trainable_sha256=_parameter_sha256(policy_model, trainable_only=True),
                    torch_module=torch,
                )
                pending_rollouts = []
                pending_step_metrics = []

        if pending_rollouts or pending_step_metrics:
            raise RuntimeError("正式训练结束后仍存在未进入恢复 checkpoint 的审计段")

        reference_unchanged = reference_versions_before == _parameter_versions(reference_model)
        policy_trainable_hash_after = _parameter_sha256(policy_model, trainable_only=True)
        lora_updated = initial_policy_trainable_hash != policy_trainable_hash_after
        if not reference_unchanged:
            raise RuntimeError("reference model 参数在训练期间发生变化")
        if not lora_updated:
            raise RuntimeError("LoRA 可训练参数没有发生更新")
        if last_checkpoint is None:
            raise RuntimeError("正式训练没有生成 checkpoint")

        recovery_state_reload_ok = _verify_recovery_state(last_checkpoint, torch)

        # 最后一组的 planner/sample/objective 局部变量仍会引用 policy model（策略模型）
        # 或 autograd graph（自动求导图）；先显式释放，确保 adapter 重载验证不会同时占用三份基础模型。
        sampled.clear()
        del (
            planner,
            sampled,
            trainable_parameters,
            rewards,
            advantages,
            old_log_probs,
            reference_log_probs,
            new_log_probs,
            objective,
            scaled_loss,
            grad_norm,
        )
        del optimizer, scheduler, reference_model, reference_base, policy_model, policy_base
        gc.collect()
        torch.cuda.empty_cache()
        reload_ok = _verify_adapter_reload(
            base_model_id=config.base_model_id,
            adapter_dir=last_checkpoint / "adapter",
            dtype=dtype,
            device=config.device,
            enabled=config.verify_checkpoint_reload,
        )
        rollout_rows = _consolidate_segments(run_dir, "rollouts_segment.jsonl", run_dir / "rollouts.jsonl")
        metric_rows = _consolidate_segments(run_dir, "metrics_segment.jsonl", run_dir / "step_metrics.jsonl")
        summary = _build_final_summary(
            run_id=run_id,
            config=config,
            rollouts=rollout_rows,
            metrics=metric_rows,
            processed_case_ids=processed_case_ids,
            optimizer_step=optimizer_step,
            elapsed_seconds=elapsed_before_seconds + (time.monotonic() - started),
            reference_unchanged=reference_unchanged,
            lora_updated=lora_updated,
            policy_hash_before=initial_policy_trainable_hash,
            policy_hash_after=policy_trainable_hash_after,
            checkpoint_reload_ok=reload_ok,
            recovery_state_reload_ok=recovery_state_reload_ok,
            final_checkpoint=last_checkpoint,
            sft_manifest=sft_manifest,
        )
        _write_json(run_dir / "final_metrics.json", summary)
        _write_output_manifest(run_dir)
        return {"run_id": run_id, "run_dir": str(run_dir), **summary}
    except Exception as exc:
        failure_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        _write_json(run_dir / f"FAILED_{failure_stamp}.json", {
            "run_id": run_id,
            "failed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "error_type": exc.__class__.__name__,
            "message": str(exc),
            "optimizer_step": optimizer_step,
            "processed_case_count": len(processed_case_ids),
            "last_recovery_checkpoint": str(last_checkpoint or ""),
        })
        raise


def _preflight_inputs(config: FormalGrpoConfig) -> dict[str, Any]:
    """只核对本轮直接输入身份，不重复 SFT V2 数据质量门禁。"""

    paths = {
        "train_cases": _resolve_path(config.train_cases),
        "train_case_manifest": _resolve_path(config.train_case_manifest),
        "source_sft_train_data": _resolve_path(config.source_sft_train_data),
        "reward_profile": _resolve_path(config.reward_profile),
        "environment_snapshot": _resolve_path(config.environment_snapshot),
        "sft_checkpoint_dir": _resolve_path(config.sft_checkpoint_dir),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"正式 GRPO 输入不存在：{name}={path}")
    expected_hashes = {
        "train_cases": config.expected_train_cases_sha256,
        "train_case_manifest": config.expected_case_manifest_sha256,
        "source_sft_train_data": config.expected_source_sft_train_sha256,
        "reward_profile": config.expected_reward_profile_sha256,
        "environment_snapshot": config.expected_environment_snapshot_sha256,
    }
    actual_hashes = {name: _sha256(paths[name]) for name in expected_hashes}
    mismatches = {
        name: {"expected": expected_hashes[name], "actual": actual_hashes[name]}
        for name in expected_hashes
        if expected_hashes[name] != actual_hashes[name]
    }
    if mismatches:
        raise ValueError(f"正式 GRPO 输入 SHA256 漂移：{mismatches}")

    case_manifest = json.loads(paths["train_case_manifest"].read_text(encoding="utf-8"))
    if (
        case_manifest.get("case_count") != 75
        or case_manifest.get("all_cases_train_only") is not True
        or case_manifest.get("all_cases_reviewed") is not True
        or case_manifest.get("rollout_generated") is not False
        or case_manifest.get("training_performed") is not False
    ):
        raise ValueError("GRPO case manifest 身份或 train-only 门禁不符合正式训练要求")
    if (
        case_manifest.get("output_file_sha256", {}).get("grpo_train_cases.jsonl")
        != config.expected_train_cases_sha256
        or case_manifest.get("reward_profile_sha256") != config.expected_reward_profile_sha256
        or case_manifest.get("environment_snapshot_sha256")
        != config.expected_environment_snapshot_sha256
    ):
        raise ValueError("GRPO case manifest 绑定的 case/Reward/snapshot SHA256 不一致")
    cases = load_grpo_training_cases(paths["train_cases"])
    if len(cases) != config.expected_case_count:
        raise ValueError(f"GRPO train case 数不是 75：{len(cases)}")
    case_ids = [row.case_contract.case_id for row in cases]
    if len(set(case_ids)) != config.expected_case_count:
        raise ValueError("GRPO train case_id 不是 75 个唯一值")

    snapshot = EnvironmentSnapshot.model_validate_json(
        paths["environment_snapshot"].read_text(encoding="utf-8")
    )
    reward_payload = json.loads(paths["reward_profile"].read_text(encoding="utf-8"))
    if reward_payload.get("decision") != "frozen":
        raise ValueError("Reward profile 未冻结")
    reward_config = RewardConfig(
        reward_version=reward_payload["reward_version"],
        weights=RewardWeights.model_validate(reward_payload["weights"]),
    )
    sft_manifest_path = paths["sft_checkpoint_dir"] / "checkpoint_manifest.json"
    sft_manifest = CheckpointManifest.model_validate_json(
        sft_manifest_path.read_text(encoding="utf-8")
    )
    if sft_manifest.run_id != config.sft_run_id:
        raise ValueError("SFT V2 run_id 与正式 GRPO 配置不一致")
    if sft_manifest.base_model_id != config.base_model_id:
        raise ValueError("SFT V2 base_model_id 与正式 GRPO 配置不一致")
    if sft_manifest.tuning_method != TuningMethod.LORA:
        raise ValueError("正式 GRPO 第一轮只接受 SFT V2 LoRA adapter 初始化")
    adapter_path = _resolve_path(sft_manifest.adapter_path or sft_manifest.model_path)
    adapter_weight = _adapter_weight_path(adapter_path)
    if adapter_weight.stat().st_size != config.sft_adapter_bytes:
        raise ValueError("SFT V2 adapter 字节数不一致")

    declared_inputs = [
        str(paths[name]) for name in (
            "train_cases",
            "train_case_manifest",
            "source_sft_train_data",
            "reward_profile",
            "environment_snapshot",
        )
    ] + [str(sft_manifest_path), str(adapter_weight), str(_resolve_path(sft_manifest.tokenizer_path))]
    return {
        "cases": cases,
        "snapshot": snapshot,
        "reward_config": reward_config,
        "sft_manifest": sft_manifest,
        "input_manifest": {
            "trainer_version": TRAINER_VERSION,
            "code_identity": _code_identity(),
            "declared_input_files": declared_inputs,
            "input_sha256": {str(paths[name]): actual_hashes[name] for name in actual_hashes},
            "case_count": len(cases),
            "unique_case_count": len({row.case_contract.case_id for row in cases}),
            "group_size": config.group_size,
            "expected_rollout_count": len(cases) * config.group_size,
            "excluded_inputs_read": {
                "heldout": False,
                "dev": False,
                "test": False,
                "reject": False,
                "historical_candidate_pool": False,
            },
            "sft_checkpoint_manifest_sha256": _sha256(sft_manifest_path),
            "sft_adapter_sha256": _sha256(adapter_weight),
            "sft_adapter_bytes": adapter_weight.stat().st_size,
        },
    }


def _save_recovery_checkpoint(
        *,
        run_dir: Path,
        policy_model: Any,
        tokenizer: Any,
        optimizer: Any,
        scheduler: Any,
        optimizer_step: int,
        next_case_index: int,
        processed_case_ids: list[str],
        case_order: list[int],
        rollout_segment: list[dict[str, Any]],
        metric_segment: list[dict[str, Any]],
        previous_checkpoint: Path | None,
        elapsed_seconds: float,
        policy_trainable_sha256: str,
        torch_module: Any,
) -> Path:
    """保存 adapter、优化器、调度器、随机状态和本段审计记录，供失败后恢复。"""

    checkpoint_dir = run_dir / "checkpoints" / f"step_{optimizer_step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    adapter_dir = checkpoint_dir / "adapter"
    policy_model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint_dir / "tokenizer")
    torch_module.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    torch_module.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
    torch_module.save({
        "python_random_state": random.getstate(),
        "torch_rng_state": torch_module.get_rng_state(),
        "cuda_rng_state_all": torch_module.cuda.get_rng_state_all(),
    }, checkpoint_dir / "rng_state.pt")
    _write_jsonl(checkpoint_dir / "rollouts_segment.jsonl", rollout_segment)
    _write_jsonl(checkpoint_dir / "metrics_segment.jsonl", metric_segment)
    trainer_state = {
        "trainer_version": TRAINER_VERSION,
        "optimizer_step": optimizer_step,
        "next_case_index": next_case_index,
        "processed_case_ids": list(processed_case_ids),
        "processed_case_count": len(processed_case_ids),
        "processed_rollout_count": len(processed_case_ids) * 4,
        "case_order": list(case_order),
        "previous_checkpoint": str(previous_checkpoint or ""),
        "elapsed_seconds": elapsed_seconds,
        "policy_trainable_sha256": policy_trainable_sha256,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _write_json(checkpoint_dir / "trainer_state.json", trainer_state)
    files = {
        str(path.relative_to(checkpoint_dir)): {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(checkpoint_dir.rglob("*"))
        if path.is_file() and path.name != "checkpoint_manifest.json"
    }
    _write_json(checkpoint_dir / "checkpoint_manifest.json", {
        "checkpoint_version": "formal-grpo-recovery-v1",
        "optimizer_step": optimizer_step,
        "adapter_path": "adapter",
        "optimizer_state": "optimizer.pt",
        "scheduler_state": "scheduler.pt",
        "rng_state": "rng_state.pt",
        "trainer_state": "trainer_state.json",
        "files": files,
    })
    return checkpoint_dir


def _build_final_summary(
        *,
        run_id: str,
        config: FormalGrpoConfig,
        rollouts: list[dict[str, Any]],
        metrics: list[dict[str, Any]],
        processed_case_ids: list[str],
        optimizer_step: int,
        elapsed_seconds: float,
        reference_unchanged: bool,
        lora_updated: bool,
        policy_hash_before: str,
        policy_hash_after: str,
        checkpoint_reload_ok: bool,
        recovery_state_reload_ok: bool,
        final_checkpoint: Path,
        sft_manifest: CheckpointManifest,
) -> dict[str, Any]:
    case_counts = Counter(row["case_id"] for row in rollouts)
    finite_fields = ("advantage", "policy_loss", "kl", "total_loss")
    all_finite = all(
        math.isfinite(float(row[field]))
        for row in rollouts
        for field in finite_fields
    ) and all(math.isfinite(float(row["reward"]["total_reward"])) for row in rollouts)
    exact_closed_loop = (
        len(processed_case_ids) == len(set(processed_case_ids)) == 75
        and len(rollouts) == 300
        and set(case_counts.values()) == {4}
        and optimizer_step == 75
    )
    if not exact_closed_loop:
        raise RuntimeError(
            "正式 GRPO 数量未闭环："
            f"cases={len(set(processed_case_ids))}, rollouts={len(rollouts)}, "
            f"per_case={sorted(set(case_counts.values()))}, optimizer_steps={optimizer_step}"
        )
    if not all_finite:
        raise FloatingPointError("正式 GRPO 汇总发现 NaN/Inf")
    return {
        "run_id": run_id,
        "trainer_version": TRAINER_VERSION,
        "status": "completed",
        "policy_model": config.base_model_id,
        "policy_initial_adapter": sft_manifest.adapter_id,
        "reference_model": config.base_model_id,
        "reference_adapter": sft_manifest.adapter_id,
        "tokenizer_path": sft_manifest.tokenizer_path,
        "reward_profile": config.reward_profile,
        "epoch_count": 1,
        "unique_case_count": 75,
        "rollouts_per_case": 4,
        "rollout_count": 300,
        "optimizer_step": optimizer_step,
        "optimizer": "torch.optim.AdamW",
        "scheduler": "transformers.get_linear_schedule_with_warmup",
        "learning_rate_configured": config.learning_rate,
        "learning_rate_first_optimizer_step": metrics[0]["learning_rate"],
        "learning_rate_final": metrics[-1]["learning_rate"],
        "warmup_steps": int(75 * config.warmup_ratio),
        "runtime_seconds": elapsed_seconds,
        "reward_mean": sum(row["reward"]["total_reward"] for row in rollouts) / len(rollouts),
        "advantage_mean": sum(row["advantage"] for row in rollouts) / len(rollouts),
        "policy_loss_mean": sum(row["policy_loss"] for row in rollouts) / len(rollouts),
        "kl_mean": sum(row["kl"] for row in rollouts) / len(rollouts),
        "total_loss_mean": sum(row["total_loss"] for row in rollouts) / len(rollouts),
        "all_required_metrics_finite": all_finite,
        "reference_parameters_unchanged": reference_unchanged,
        "lora_parameters_updated": lora_updated,
        "policy_trainable_sha256_before": policy_hash_before,
        "policy_trainable_sha256_after": policy_hash_after,
        "checkpoint_reload_ok": checkpoint_reload_ok,
        "recovery_state_reload_ok": recovery_state_reload_ok,
        "final_checkpoint": str(final_checkpoint),
        "excluded_inputs_read": {
            "heldout": False,
            "dev": False,
            "test": False,
            "reject": False,
            "historical_candidate_pool": False,
        },
    }


def _raise_on_provider_contract_failure(trajectory: OfflineTrajectoryResult) -> None:
    codes = {error.code for error in trajectory.errors}
    fatal = sorted(codes.intersection(FATAL_PROVIDER_ERRORS))
    if fatal or trajectory.config_match_status != "match" or trajectory.corpus_match_status != "match":
        raise RuntimeError(
            "真实 Provider/Observation 契约失败："
            f"case={trajectory.case_id}, errors={fatal}, "
            f"config={trajectory.config_match_status}, corpus={trajectory.corpus_match_status}"
        )


def _verify_adapter_reload(
        *,
        base_model_id: str,
        adapter_dir: Path,
        dtype: Any,
        device: str,
        enabled: bool,
) -> bool:
    if not enabled:
        return False
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=dtype,
        local_files_only=True,
        device_map={"": device},
    )
    reloaded = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    reloaded.eval()
    parameter_count = sum(parameter.numel() for parameter in reloaded.parameters())
    ok = parameter_count > 0
    del reloaded, base
    gc.collect()
    torch.cuda.empty_cache()
    if not ok:
        raise RuntimeError("最终 GRPO adapter 重载后参数为空")
    return True


def _verify_recovery_state(checkpoint_dir: Path, torch_module: Any) -> bool:
    """确认 optimizer/scheduler/RNG/trainer state（恢复状态）均可重新读取。"""

    payload = _load_recovery_payload(checkpoint_dir, torch_module)
    optimizer_state = payload["optimizer_state"]
    scheduler_state = payload["scheduler_state"]
    rng_state = payload["rng_state"]
    trainer_state = payload["trainer_state"]
    checks = (
        isinstance(optimizer_state, dict) and "param_groups" in optimizer_state,
        isinstance(scheduler_state, dict) and "last_epoch" in scheduler_state,
        isinstance(rng_state, dict) and "torch_rng_state" in rng_state,
        trainer_state.get("optimizer_step") == 75,
        trainer_state.get("processed_case_count") == 75,
        trainer_state.get("processed_rollout_count") == 300,
    )
    if not all(checks):
        raise RuntimeError("最终 recovery checkpoint（恢复检查点）状态无法完整重载")
    return True


def _load_recovery_payload(checkpoint_dir: Path, torch_module: Any) -> dict[str, Any]:
    """校验 checkpoint manifest（检查点清单）后读取所有恢复状态。"""

    manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("checkpoint_version") != "formal-grpo-recovery-v1":
        raise ValueError("不是正式 GRPO recovery checkpoint")
    for relative_path, identity in manifest.get("files", {}).items():
        path = checkpoint_dir / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"恢复文件不存在：{path}")
        if path.stat().st_size != int(identity["bytes"]) or _sha256(path) != identity["sha256"]:
            raise ValueError(f"恢复文件身份漂移：{path}")
    optimizer_state = torch_module.load(
        checkpoint_dir / "optimizer.pt",
        map_location="cpu",
        weights_only=False,
    )
    scheduler_state = torch_module.load(
        checkpoint_dir / "scheduler.pt",
        map_location="cpu",
        weights_only=False,
    )
    rng_state = torch_module.load(
        checkpoint_dir / "rng_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    trainer_state = json.loads((checkpoint_dir / "trainer_state.json").read_text(encoding="utf-8"))
    return {
        "manifest": manifest,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "rng_state": rng_state,
        "trainer_state": trainer_state,
    }


def _prepare_resume(
        checkpoint_dir: Path,
        *,
        config: FormalGrpoConfig,
        current_input_manifest: dict[str, Any],
        torch_module: Any,
) -> tuple[str, Path, dict[str, Any]]:
    """只允许从同一正式 run 的最新、身份未漂移 checkpoint（检查点）恢复。"""

    checkpoint_dir = checkpoint_dir.resolve()
    if checkpoint_dir.parent.name != "checkpoints" or not checkpoint_dir.name.startswith("step_"):
        raise ValueError("--resume-from 必须指向 run/checkpoints/step_NNNNNN")
    run_dir = checkpoint_dir.parent.parent
    if (run_dir / "final_metrics.json").exists() or (run_dir / "output_manifest.json").exists():
        raise FileExistsError("该正式 GRPO run 已完成，禁止覆盖或再次恢复")
    latest = max(
        checkpoint_dir.parent.glob("step_*"),
        key=lambda path: int(path.name.removeprefix("step_")),
    )
    if latest.resolve() != checkpoint_dir:
        raise ValueError("只能从当前 run 的最新 recovery checkpoint 恢复")
    saved_config = json.loads((run_dir / "training_config.json").read_text(encoding="utf-8"))
    if saved_config != config.model_dump(mode="json"):
        raise ValueError("恢复时 formal GRPO config 与原 run 配置不一致")
    saved_inputs = json.loads((run_dir / "input_manifest.json").read_text(encoding="utf-8"))
    saved_code = saved_inputs.pop("code_identity", {})
    current_inputs = dict(current_input_manifest)
    current_code = current_inputs.pop("code_identity", {})
    if saved_inputs != current_inputs:
        raise ValueError("恢复时正式输入 manifest 或 SHA256 已漂移")
    if (
        saved_code.get("git_head") != current_code.get("git_head")
        or saved_code.get("source_sha256") != current_code.get("source_sha256")
    ):
        raise ValueError("恢复时正式 GRPO 关键源码身份已漂移")
    payload = _load_recovery_payload(checkpoint_dir, torch_module)
    trainer_state = payload["trainer_state"]
    step = int(checkpoint_dir.name.removeprefix("step_"))
    if not (
        trainer_state.get("optimizer_step") == step
        and trainer_state.get("next_case_index") == trainer_state.get("processed_case_count")
        and trainer_state.get("processed_rollout_count") == trainer_state.get("processed_case_count") * 4
        and 0 <= trainer_state.get("next_case_index", -1) <= 75
        and sorted(trainer_state.get("case_order", [])) == list(range(75))
    ):
        raise ValueError("恢复 checkpoint 的 trainer state 数量或 case 顺序不合法")
    return run_dir.name, run_dir, payload


def _restore_rng_state(rng_state: dict[str, Any], torch_module: Any) -> None:
    """恢复 Python、PyTorch 和 CUDA 随机状态，使后续 rollout 顺序可继续复现。"""

    random.setstate(rng_state["python_random_state"])
    torch_module.set_rng_state(rng_state["torch_rng_state"])
    torch_module.cuda.set_rng_state_all(rng_state["cuda_rng_state_all"])


def _consolidate_segments(run_dir: Path, segment_name: str, output_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint_dir in sorted((run_dir / "checkpoints").glob("step_*")):
        segment_path = checkpoint_dir / segment_name
        if not segment_path.exists():
            raise FileNotFoundError(f"恢复 checkpoint 缺少审计段：{segment_path}")
        rows.extend(_read_jsonl(segment_path))
    _write_jsonl(output_path, rows)
    return rows


def _parameter_versions(model: Any) -> dict[str, int]:
    return {name: int(parameter._version) for name, parameter in model.named_parameters()}


def _parameter_sha256(model: Any, *, trainable_only: bool) -> str:
    import torch

    digest = hashlib.sha256()
    matched = 0
    for name, parameter in model.named_parameters():
        if trainable_only and not parameter.requires_grad:
            continue
        matched += 1
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        raw = parameter.detach().contiguous().view(torch.uint8).cpu()
        digest.update(raw.numpy().tobytes())
    if matched == 0:
        raise RuntimeError("没有可用于参数哈希的张量")
    return digest.hexdigest()


def _adapter_weight_path(adapter_dir: Path) -> Path:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        path = adapter_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"adapter 权重不存在：{adapter_dir}")


def _create_run_dir(config: FormalGrpoConfig) -> tuple[str, Path]:
    output_root = _resolve_path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_name = "".join(char if char.isalnum() or char in "-_" else "-" for char in config.run_name)
    run_id = f"{safe_name}_{stamp}_{uuid.uuid4().hex[:8]}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_id, run_dir


def _write_output_manifest(run_dir: Path) -> None:
    files = {
        str(path.relative_to(run_dir)): {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != "output_manifest.json"
    }
    _write_json(run_dir / "output_manifest.json", {
        "manifest_version": "formal-grpo-output-v1",
        "file_count": len(files),
        "files": files,
    })


def _write_environment_freeze(path: Path, torch_module: Any) -> None:
    versions = {"python": platform.python_version(), "torch": torch_module.__version__}
    for package in ("transformers", "peft", "accelerate", "pydantic", "requests"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    payload = {
        "versions": versions,
        "python_executable": sys.executable,
        "cuda_available": torch_module.cuda.is_available(),
        "cuda_version": torch_module.version.cuda,
        "gpu": (
            torch_module.cuda.get_device_name(0)
            if torch_module.cuda.is_available()
            else "unavailable"
        ),
        "gpu_memory_bytes": (
            torch_module.cuda.get_device_properties(0).total_memory
            if torch_module.cuda.is_available()
            else 0
        ),
    }
    _write_json(path, payload)


def _code_identity() -> dict[str, Any]:
    """记录 Git commit 和正式训练关键源码 SHA256，覆盖未提交代码场景。"""

    source_paths = (
        "app/rag/training/grpo/config.py",
        "app/rag/training/grpo/objective.py",
        "app/rag/training/grpo/trainer.py",
        "app/rag/training/grpo/cli.py",
        "app/rag/evaluation/provider_worker.py",
        "app/rag/evaluation/action_providers.py",
        "app/rag/evaluation/offline_environment.py",
        "app/rag/evaluation/reward.py",
        "app/rag/query/model_planner/prompt_builder.py",
        "app/rag/query/model_planner/decision_codec.py",
        "app/rag/query/contracts.py",
    )
    return {
        "git_head": _git_output("rev-parse", "HEAD"),
        "source_sha256": {
            relative_path: _sha256(PROJECT_ROOT / relative_path)
            for relative_path in source_paths
        },
    }


def _git_output(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _assert_finite_metrics(metric: dict[str, Any]) -> None:
    for field in (
        "reward_mean",
        "reward_min",
        "reward_max",
        "advantage_mean",
        "advantage_std",
        "policy_loss",
        "kl",
        "total_loss",
        "clip_fraction",
        "gradient_norm",
        "learning_rate",
    ):
        if not math.isfinite(float(metric[field])):
            raise FloatingPointError(f"{field} 出现 NaN/Inf")


def _set_seed(seed: int, torch_module: Any) -> None:
    random.seed(seed)
    torch_module.manual_seed(seed)
    torch_module.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def _torch_dtype(name: str, torch_module: Any) -> Any:
    mapping = {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }
    if name not in mapping:
        raise ValueError(f"不支持的 torch_dtype={name}")
    return mapping[name]


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file_obj:
        return [json.loads(line) for line in file_obj if line.strip()]
