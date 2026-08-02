import json
import hashlib
import random
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from app.rag.training.grpo.config import FormalGrpoConfig, load_grpo_config
from app.rag.training.grpo.objective import (
    completion_token_log_probs,
    compute_group_advantages,
    grpo_token_objective,
)
from app.rag.training.grpo.trainer import (
    _enable_policy_gradient_checkpointing,
    _parameter_sha256,
    _prepare_resume,
    _set_policy_optimization_mode,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIG = PROJECT_ROOT / "evaluation/stage9/configs/planner_grpo_qwen3_5_4b_lora_formal.json"


def test_formal_config_is_fixed_to_75_cases_four_rollouts_and_one_epoch():
    config = load_grpo_config(FORMAL_CONFIG)

    assert config.expected_case_count == 75
    assert config.group_size == 4
    assert config.num_epochs == 1
    assert config.expected_case_count * config.group_size == 300
    assert config.verify_checkpoint_reload is True
    assert not ({"max_cases", "max_train_samples", "max_steps"} & set(config.model_fields))


@pytest.mark.parametrize(
    ("field", "value"),
    (("expected_case_count", 2), ("group_size", 2), ("num_epochs", 0)),
)
def test_formal_config_rejects_smoke_scope(field: str, value: int):
    payload = json.loads(FORMAL_CONFIG.read_text(encoding="utf-8"))
    payload[field] = value

    with pytest.raises(ValueError, match="正式 GRPO|第一轮正式 GRPO"):
        FormalGrpoConfig.model_validate(payload)


def test_formal_config_forbids_unknown_truncation_fields():
    payload = json.loads(FORMAL_CONFIG.read_text(encoding="utf-8"))
    payload["max_cases"] = 2

    with pytest.raises(ValueError, match="extra_forbidden"):
        FormalGrpoConfig.model_validate(payload)


def test_group_advantage_is_relative_and_finite():
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])

    advantages = compute_group_advantages(rewards)

    assert torch.isfinite(advantages).all()
    assert float(advantages.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(advantages.std(unbiased=False)) == pytest.approx(1.0, abs=1e-5)


def test_equal_group_rewards_produce_zero_advantage():
    advantages = compute_group_advantages(torch.full((4,), 0.5))

    assert torch.equal(advantages, torch.zeros(4))


def test_grpo_objective_backpropagates_through_new_policy_only():
    new_log_probs = torch.tensor([-0.4, -0.6], requires_grad=True)
    old_log_probs = torch.tensor([-0.5, -0.5], requires_grad=True)
    reference_log_probs = torch.tensor([-0.45, -0.55], requires_grad=True)

    objective = grpo_token_objective(
        new_log_probs=new_log_probs,
        old_log_probs=old_log_probs,
        reference_log_probs=reference_log_probs,
        advantage=torch.tensor(0.75),
        clip_epsilon=0.2,
        kl_beta=0.02,
    )
    objective["total_loss"].backward()

    assert all(torch.isfinite(value) for value in objective.values())
    assert new_log_probs.grad is not None
    assert torch.isfinite(new_log_probs.grad).all()
    assert old_log_probs.grad is None
    assert reference_log_probs.grad is None


def test_grpo_objective_rejects_non_finite_probabilities():
    with pytest.raises(FloatingPointError, match="NaN/Inf"):
        grpo_token_objective(
            new_log_probs=torch.tensor([float("nan")]),
            old_log_probs=torch.tensor([-0.5]),
            reference_log_probs=torch.tensor([-0.5]),
            advantage=1.0,
            clip_epsilon=0.2,
            kl_beta=0.02,
        )


def test_completion_log_probs_use_last_prompt_position_for_first_generated_token():
    calls = []

    class ToyModel(torch.nn.Module):
        def forward(self, *, input_ids, attention_mask, use_cache, logits_to_keep):
            del attention_mask, use_cache
            batch, sequence = input_ids.shape
            calls.append({"sequence": sequence, "logits_to_keep": logits_to_keep})
            first_position = sequence - logits_to_keep
            logits = torch.zeros((batch, logits_to_keep, 8), dtype=torch.float32)
            for local_position, absolute_position in enumerate(range(first_position, sequence)):
                logits[:, local_position, absolute_position + 1] = 3.0
            return SimpleNamespace(logits=logits)

    result = completion_token_log_probs(
        ToyModel(),
        prompt_token_ids=[5, 6],
        completion_token_ids=[2, 3],
        device=torch.device("cpu"),
    )
    expected_first = torch.log_softmax(torch.tensor([0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dim=-1)[2]
    expected_second = torch.log_softmax(torch.tensor([0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0]), dim=-1)[3]

    assert torch.allclose(result, torch.stack((expected_first, expected_second)))
    assert calls == [{"sequence": 4, "logits_to_keep": 3}]


def test_policy_memory_mode_enables_non_reentrant_checkpointing_and_disables_dropout():
    class ToyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = torch.nn.Dropout(p=0.5)
            self.gradient_checkpointing_kwargs = None

        def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs):
            self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs

    policy = ToyPolicy()
    policy.eval()

    _enable_policy_gradient_checkpointing(policy)
    _set_policy_optimization_mode(policy, torch)

    assert policy.training is True
    assert policy.dropout.training is False
    assert policy.gradient_checkpointing_kwargs == {"use_reentrant": False}


def test_trainable_parameter_hash_changes_after_real_parameter_update():
    model = torch.nn.Linear(2, 1, bias=False)
    before = _parameter_sha256(model, trainable_only=True)

    with torch.no_grad():
        model.weight.add_(0.1)

    assert _parameter_sha256(model, trainable_only=True) != before


def test_recovery_checkpoint_can_reload_all_formal_training_state(tmp_path: Path):
    config = load_grpo_config(FORMAL_CONFIG)
    run_dir = tmp_path / "formal_grpo_run"
    checkpoint = run_dir / "checkpoints" / "step_000025"
    checkpoint.mkdir(parents=True)
    input_manifest = {"input_sha256": {"train_cases": "a" * 64}, "case_count": 75}
    (run_dir / "training_config.json").write_text(
        json.dumps(config.model_dump(mode="json")),
        encoding="utf-8",
    )
    (run_dir / "input_manifest.json").write_text(json.dumps(input_manifest), encoding="utf-8")
    torch.save({"state": {}, "param_groups": []}, checkpoint / "optimizer.pt")
    torch.save({"last_epoch": 25}, checkpoint / "scheduler.pt")
    torch.save({
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": [],
    }, checkpoint / "rng_state.pt")
    (checkpoint / "trainer_state.json").write_text(json.dumps({
        "optimizer_step": 25,
        "next_case_index": 25,
        "processed_case_ids": [f"case-{index}" for index in range(25)],
        "processed_case_count": 25,
        "processed_rollout_count": 100,
        "case_order": list(range(75)),
        "policy_trainable_sha256": "b" * 64,
    }), encoding="utf-8")
    (checkpoint / "rollouts_segment.jsonl").write_text("", encoding="utf-8")
    (checkpoint / "metrics_segment.jsonl").write_text("", encoding="utf-8")
    files = {}
    for path in checkpoint.iterdir():
        if path.is_file():
            files[path.name] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
    (checkpoint / "checkpoint_manifest.json").write_text(json.dumps({
        "checkpoint_version": "formal-grpo-recovery-v1",
        "optimizer_step": 25,
        "files": files,
    }), encoding="utf-8")

    run_id, restored_run_dir, payload = _prepare_resume(
        checkpoint,
        config=config,
        current_input_manifest=input_manifest,
        torch_module=torch,
    )

    assert run_id == "formal_grpo_run"
    assert restored_run_dir == run_dir
    assert payload["trainer_state"]["processed_rollout_count"] == 100
    assert "param_groups" in payload["optimizer_state"]
