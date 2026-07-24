import json
from pathlib import Path

import pytest

from app.rag.query.model_planner import load_checkpoint_manifest
from evaluation.stage9.model_planner.checkpoint_io import Stage9SftTrainingConfig, load_training_config
from app.rag.query.model_planner.model_profile import load_model_profile
from evaluation.stage9.model_planner.sft_train import run_sft_training


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = PROJECT_ROOT / "configs/planner_model_profiles"
CLOUD_TEMPLATE = PROJECT_ROOT / "evaluation/stage9/configs/planner_sft_cloud_template.json"


def test_qwen3_5_4b_profile_is_default_base_model():
    profile = load_model_profile("qwen3_5_4b")

    assert profile.profile_id == "qwen3_5_4b"
    assert profile.display_name == "Qwen3.5-4B"
    assert profile.role == "default_base_model"
    assert profile.auto_train_enabled is True
    assert profile.base_model_id == "Qwen/Qwen3.5-4B"
    assert profile.training_model_id == "Qwen/Qwen3.5-4B"
    assert profile.serving_model_id == "qwen3.5:4b"
    assert profile.chat_template == "qwen3.5-chat"
    assert profile.enable_thinking is False
    assert profile.max_context_tokens == 4096
    assert profile.max_target_tokens == 128
    assert profile.recommended_backend == "transformers_causal_lm"


def test_qwen3_5_9b_profile_is_upgrade_only():
    profile = load_model_profile(PROFILE_DIR / "qwen3_5_9b.json")

    assert profile.profile_id == "qwen3_5_9b"
    assert profile.display_name == "Qwen3.5-9B"
    assert profile.role == "upgrade_candidate"
    assert profile.auto_train_enabled is False
    assert profile.enable_thinking is False


def test_cloud_template_binds_default_4b_profile_and_excludes_14b():
    template_text = CLOUD_TEMPLATE.read_text(encoding="utf-8")
    template = json.loads(template_text)
    profile = load_model_profile(template["model_profile_path"])

    assert template["model_profile_id"] == "qwen3_5_4b"
    assert profile.profile_id == template["model_profile_id"]
    assert template["base_model_id"] == profile.base_model_id
    assert template["max_input_tokens"] <= profile.max_context_tokens
    assert template["max_target_tokens"] <= profile.max_target_tokens
    assert "Qwen3-14B" not in template_text
    assert "qwen3-14b" not in template_text.lower()

    loaded = load_training_config(CLOUD_TEMPLATE)
    assert loaded.base_model_id == "Qwen/Qwen3.5-4B"
    assert loaded.model_profile_id == "qwen3_5_4b"


def test_training_checkpoint_manifest_embeds_profile_snapshot(tmp_path):
    profile = load_model_profile("qwen3_5_4b")
    config = Stage9SftTrainingConfig(
        run_name="planner-sft-stage9-profile-test",
        training_backend="debug_memorized",
        base_model_id=profile.base_model_id,
        model_profile_id=profile.profile_id,
        model_profile_path="configs/planner_model_profiles/qwen3_5_4b.json",
        train_data="evaluation/stage9/artifacts/sft/sft_planner_stage9_train.jsonl",
        train_manifest="evaluation/stage9/artifacts/sft/sft_planner_stage9_manifest.json",
        reward_profile="evaluation/stage9/configs/reward_v1_1_training_profile.json",
        snapshot_id="stage85-env-20260721-v2",
        output_root=str(tmp_path),
        max_input_tokens=profile.max_context_tokens,
        max_target_tokens=profile.max_target_tokens,
        num_epochs=0,
        max_train_samples=2,
        save_training_preview_count=0,
    )

    manifest = run_sft_training(config)
    checkpoint_dir = tmp_path / manifest.run_id
    loaded_manifest = load_checkpoint_manifest(checkpoint_dir)
    metrics = json.loads((checkpoint_dir / "train_metrics.json").read_text(encoding="utf-8"))

    assert loaded_manifest.model_profile_id == "qwen3_5_4b"
    assert loaded_manifest.model_profile is not None
    assert loaded_manifest.model_profile.base_model_id == "Qwen/Qwen3.5-4B"
    assert loaded_manifest.model_profile.enable_thinking is False
    assert metrics["model_profile"]["profile_id"] == "qwen3_5_4b"


def test_training_rejects_model_profile_mismatch(tmp_path):
    config = Stage9SftTrainingConfig(
        run_name="planner-sft-stage9-profile-mismatch",
        training_backend="debug_memorized",
        base_model_id="Qwen/Qwen3.5-9B",
        model_profile_id="qwen3_5_4b",
        model_profile_path="configs/planner_model_profiles/qwen3_5_4b.json",
        train_data="evaluation/stage9/artifacts/sft/sft_planner_stage9_train.jsonl",
        train_manifest="evaluation/stage9/artifacts/sft/sft_planner_stage9_manifest.json",
        reward_profile="evaluation/stage9/configs/reward_v1_1_training_profile.json",
        snapshot_id="stage85-env-20260721-v2",
        output_root=str(tmp_path),
        num_epochs=0,
        max_train_samples=1,
        save_training_preview_count=0,
    )

    with pytest.raises(ValueError, match="base_model_id"):
        run_sft_training(config)
