import pytest

from app.rag.query.model_planner import CheckpointManifest, TuningMethod
from app.rag.query.model_planner.checkpoint_runtime import (
    _inference_model_kwargs,
    _select_inference_device,
)
from evaluation.stage9.model_planner.checkpoint_io import (
    Stage9SftTrainingConfig,
    TrainingBackend,
    collect_framework_versions,
    load_training_config,
)
from app.rag.query.model_planner.model_profile import load_model_profile


LORA_CONFIG = "evaluation/stage9/configs/planner_sft_qwen3_5_4b_lora.json"
QLORA_CONFIG = "evaluation/stage9/configs/planner_sft_qwen3_5_4b_qlora.json"
CLOUD_TEMPLATE = "evaluation/stage9/configs/planner_sft_cloud_template.json"


def test_lora_cloud_config_is_first_stage_default():
    config = load_training_config(LORA_CONFIG)

    assert config.training_backend == TrainingBackend.TRANSFORMERS_CAUSAL_LM
    assert config.base_model_id == "Qwen/Qwen3.5-4B"
    assert config.model_profile_id == "qwen3_5_4b"
    assert config.tuning_method == TuningMethod.LORA
    assert config.allow_full_finetune is False
    assert config.load_in_4bit is False
    assert config.lora_r == 16
    assert config.lora_alpha == 32
    assert config.lora_dropout == 0.05
    assert "q_proj" in config.target_modules
    assert "down_proj" in config.target_modules


def test_qlora_cloud_config_requires_4bit_loading():
    config = load_training_config(QLORA_CONFIG)

    assert config.training_backend == TrainingBackend.TRANSFORMERS_CAUSAL_LM
    assert config.tuning_method == TuningMethod.QLORA
    assert config.load_in_4bit is True
    assert config.bnb_compute_dtype == "bfloat16"
    assert config.bnb_4bit_quant_type == "nf4"
    assert config.bnb_4bit_use_double_quant is True


def test_cloud_template_uses_lora_not_full_finetune():
    config = load_training_config(CLOUD_TEMPLATE)

    assert config.tuning_method == TuningMethod.LORA
    assert config.allow_full_finetune is False
    assert config.load_in_4bit is False


def test_transformers_backend_rejects_full_finetune_by_default():
    with pytest.raises(ValueError, match="默认禁止 full"):
        _base_transformers_config(tuning_method="full")


def test_transformers_backend_rejects_invalid_lora_4bit_mix():
    with pytest.raises(ValueError, match="4bit 训练请使用 qlora"):
        _base_transformers_config(tuning_method="lora", load_in_4bit=True)

    with pytest.raises(ValueError, match="qlora 训练必须设置 load_in_4bit=true"):
        _base_transformers_config(tuning_method="qlora", load_in_4bit=False)


def test_checkpoint_manifest_records_adapter_metadata():
    profile = load_model_profile("qwen3_5_4b")
    manifest = CheckpointManifest(
        run_id="stage9-peft-test",
        run_name="stage9-peft-test",
        policy_version="transformers_causal_lm:stage9-peft-test",
        training_backend=TrainingBackend.TRANSFORMERS_CAUSAL_LM,
        base_model_id=profile.base_model_id,
        model_profile_id=profile.profile_id,
        model_profile=profile,
        tuning_method=TuningMethod.LORA,
        adapter_id="qwen3_5_4b:lora:stage9-peft-test",
        adapter_path="evaluation/stage9/artifacts/sft/checkpoints/stage9-peft-test/model/adapter",
        quantization="none",
        peft_config={
            "tuning_method": "lora",
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
        },
        train_data="evaluation/stage9/artifacts/sft/sft_planner_stage9_train.jsonl",
        train_manifest="evaluation/stage9/artifacts/sft/sft_planner_stage9_manifest.json",
        reward_profile="evaluation/stage9/configs/reward_v1_1_training_profile.json",
        snapshot_id="stage85-env-20260721-v2",
        code_version="test",
        created_at="2026-07-24T00:00:00+00:00",
        seed=20260721,
        framework_versions={"python": "test", "peft": "test", "bitsandbytes": "unavailable"},
        prompt_builder_version="test",
        decision_codec_version="test",
        model_path="evaluation/stage9/artifacts/sft/checkpoints/stage9-peft-test/model/adapter",
        tokenizer_path="evaluation/stage9/artifacts/sft/checkpoints/stage9-peft-test/tokenizer",
        train_metrics_path="evaluation/stage9/artifacts/sft/checkpoints/stage9-peft-test/train_metrics.json",
        training_config_path="evaluation/stage9/artifacts/sft/checkpoints/stage9-peft-test/training_config.json",
        sample_count=155,
        source_case_count=70,
        max_input_tokens=4096,
        max_target_tokens=128,
    )

    assert manifest.tuning_method == TuningMethod.LORA
    assert manifest.adapter_id == "qwen3_5_4b:lora:stage9-peft-test"
    assert manifest.adapter_path.endswith("/adapter")
    assert manifest.peft_config["lora_r"] == 16


def test_framework_versions_include_peft_dependencies():
    versions = collect_framework_versions()

    assert "peft" in versions
    assert "bitsandbytes" in versions


def test_checkpoint_runtime_selects_cuda_and_bfloat16_for_supported_gpu():
    torch_module = _FakeTorch(cuda_available=True, bf16_supported=True)

    device = _select_inference_device(torch_module)

    assert device.type == "cuda"
    assert _inference_model_kwargs(torch_module, device) == {"dtype": "bfloat16"}


def test_checkpoint_runtime_keeps_cpu_default_dtype_without_gpu():
    torch_module = _FakeTorch(cuda_available=False, bf16_supported=False)

    device = _select_inference_device(torch_module)

    assert device.type == "cpu"
    assert _inference_model_kwargs(torch_module, device) == {}


class _FakeCuda:
    def __init__(self, *, available: bool, bf16_supported: bool) -> None:
        self._available = available
        self._bf16_supported = bf16_supported

    def is_available(self) -> bool:
        return self._available

    def is_bf16_supported(self) -> bool:
        return self._bf16_supported


class _FakeTorch:
    bfloat16 = "bfloat16"
    float16 = "float16"

    def __init__(self, *, cuda_available: bool, bf16_supported: bool) -> None:
        self.cuda = _FakeCuda(available=cuda_available, bf16_supported=bf16_supported)

    @staticmethod
    def device(device_type: str):
        return type("FakeDevice", (), {"type": device_type})()


def _base_transformers_config(**overrides):
    payload = {
        "run_name": "stage9-peft-validator-test",
        "training_backend": "transformers_causal_lm",
        "base_model_id": "Qwen/Qwen3.5-4B",
        "model_profile_id": "qwen3_5_4b",
        "model_profile_path": "configs/planner_model_profiles/qwen3_5_4b.json",
        "train_data": "evaluation/stage9/artifacts/sft/sft_planner_stage9_train.jsonl",
        "train_manifest": "evaluation/stage9/artifacts/sft/sft_planner_stage9_manifest.json",
        "reward_profile": "evaluation/stage9/configs/reward_v1_1_training_profile.json",
        "snapshot_id": "stage85-env-20260721-v2",
        "output_root": "evaluation/stage9/artifacts/sft/checkpoints",
        "max_input_tokens": 4096,
        "max_target_tokens": 128,
        "num_epochs": 1,
        "tuning_method": "lora",
        "target_modules": ["q_proj", "v_proj"],
        "load_in_4bit": False,
    }
    payload.update(overrides)
    return Stage9SftTrainingConfig(**payload)
