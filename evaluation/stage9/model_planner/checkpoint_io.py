"""阶段 9 Planner（规划器）SFT（监督微调）配置与 checkpoint（检查点）读写。"""

from __future__ import annotations

import json
import subprocess
import uuid
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.rag.query.model_planner.checkpoint_runtime import (
    CheckpointManifest,
    TrainingBackend,
    TuningMethod,
    load_checkpoint_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT_ROOT = "evaluation/stage9/artifacts/sft/checkpoints"


class CheckpointModel(BaseModel):
    """checkpoint（检查点）相关 schema（结构）基类。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


class Stage9SftTrainingConfig(CheckpointModel):
    """
    阶段 9 SFT（监督微调）训练配置。

    同一份 schema（结构）同时服务本地 smoke（冒烟）和云端 GPU（显卡算力）训练。区别只在
    training_backend（训练后端）、base_model_id（基础模型身份）和样本/训练步数配置。
    """

    run_name: str = Field(min_length=1, description="训练运行名称，会进入 run_id 和 checkpoint manifest。")
    training_backend: TrainingBackend = Field(
        default=TrainingBackend.DEBUG_MEMORIZED,
        description="训练后端：本地 smoke 用 debug_memorized，云端真实 SFT 用 transformers_causal_lm。",
    )
    base_model_id: str = Field(
        min_length=1,
        description="基础模型身份，必须写入 checkpoint；真实加载路径由 model profile 的 training_model_id 描述。",
    )
    model_profile_id: str = Field(
        default="",
        description="model profile（模型配置档案）身份；真实模型训练必须填写，debug smoke 可为空。",
    )
    model_profile_path: str = Field(
        default="",
        description="model profile（模型配置档案）JSON 路径；真实模型训练必须填写并写入 checkpoint。",
    )
    train_data: str = Field(min_length=1, description="SFT 训练 JSONL 路径。")
    train_manifest: str = Field(min_length=1, description="SFT 数据 manifest（清单）路径。")
    reward_profile: str = Field(min_length=1, description="冻结 Reward profile（奖励配置）路径。")
    snapshot_id: str = Field(min_length=1, description="训练数据绑定的环境 snapshot（快照）ID。")
    output_root: str = Field(default=DEFAULT_CHECKPOINT_ROOT, min_length=1, description="checkpoint 输出根目录。")
    max_input_tokens: int = Field(default=2048, ge=1, description="模型输入最大 token（分词单元）数。")
    max_target_tokens: int = Field(default=128, ge=1, description="目标 JSON 最大 token（分词单元）数。")
    max_input_chars: int = Field(default=12_000, ge=1, description="无 tokenizer 场景的 prompt 最大字符数。")
    learning_rate: float = Field(default=2e-5, gt=0, description="学习率。")
    num_epochs: float = Field(default=1.0, ge=0, description="训练轮数；debug_memorized 可为 0。")
    batch_size: int = Field(default=1, ge=1, description="每设备 batch（批量）大小。")
    gradient_accumulation_steps: int = Field(default=8, ge=1, description="梯度累积步数。")
    seed: int = Field(default=20260721, ge=0, description="随机种子，用于可复现 smoke 和训练。")
    max_train_samples: int | None = Field(default=None, ge=1, description="最多读取多少训练样本；空表示全量。")
    max_steps: int | None = Field(default=None, ge=1, description="transformers Trainer 最大训练步数；空按 epoch。")
    device: str = Field(default="auto", min_length=1, description="训练设备，auto/cpu/cuda/mps。")
    torch_dtype: str = Field(default="auto", min_length=1, description="torch dtype（张量精度），如 auto/float16/bfloat16。")
    tuning_method: TuningMethod = Field(
        default=TuningMethod.FULL,
        description="训练方法：full 全量微调、lora 低秩适配、qlora 4 位量化低秩适配。",
    )
    allow_full_finetune: bool = Field(
        default=False,
        description="是否允许 full（全量微调）；阶段 9 第一版默认 false，避免云端误跑完整权重微调。",
    )
    lora_r: int = Field(default=16, ge=1, description="LoRA rank（低秩秩数），越大可训练容量越高但显存开销越大。")
    lora_alpha: int = Field(default=32, ge=1, description="LoRA alpha（低秩缩放），通常与 lora_r 配合控制更新幅度。")
    lora_dropout: float = Field(default=0.05, ge=0, lt=1, description="LoRA dropout（低秩随机丢弃），降低小数据过拟合。")
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        description="LoRA target_modules（目标模块），限定哪些线性层挂 adapter。",
    )
    load_in_4bit: bool = Field(default=False, description="是否 4bit（4 位）加载基础模型；仅 qlora 允许为 true。")
    bnb_compute_dtype: str = Field(
        default="bfloat16",
        min_length=1,
        description="bitsandbytes 4bit 计算 dtype（计算精度），如 bfloat16/float16/float32。",
    )
    bnb_4bit_quant_type: str = Field(default="nf4", min_length=1, description="bitsandbytes 4bit 量化类型，默认 nf4。")
    bnb_4bit_use_double_quant: bool = Field(default=True, description="是否启用 nested quantization（双重量化）。")
    gradient_checkpointing: bool = Field(default=True, description="是否启用 gradient checkpointing（梯度检查点）节省显存。")
    trust_remote_code: bool = Field(default=False, description="是否信任远端模型自定义代码。")
    use_fast_tokenizer: bool = Field(default=True, description="是否优先使用 fast tokenizer（快速分词器）。")
    report_to: list[str] = Field(default_factory=list, description="Trainer 上报目标，默认空避免本地误连外部服务。")
    save_training_preview_count: int = Field(default=20, ge=0, description="保存多少条 prompt/target 预览。")

    @model_validator(mode="after")
    def validate_training_backend(self) -> "Stage9SftTrainingConfig":
        if self.training_backend == TrainingBackend.TRANSFORMERS_CAUSAL_LM and self.num_epochs == 0 and self.max_steps is None:
            raise ValueError("transformers_causal_lm 训练必须设置 num_epochs>0 或 max_steps")
        if self.training_backend == TrainingBackend.TRANSFORMERS_CAUSAL_LM and (
                not self.model_profile_id or not self.model_profile_path
        ):
            raise ValueError("transformers_causal_lm 训练必须设置 model_profile_id 和 model_profile_path")
        if self.training_backend == TrainingBackend.TRANSFORMERS_CAUSAL_LM:
            if self.tuning_method == TuningMethod.FULL and not self.allow_full_finetune:
                raise ValueError("阶段 9 第一版默认禁止 full 全量微调；确需使用时必须 allow_full_finetune=true")
            if self.tuning_method in {TuningMethod.LORA, TuningMethod.QLORA} and not self.target_modules:
                raise ValueError("lora/qlora 训练必须配置 target_modules")
            if self.tuning_method == TuningMethod.QLORA and not self.load_in_4bit:
                raise ValueError("qlora 训练必须设置 load_in_4bit=true")
            if self.tuning_method == TuningMethod.LORA and self.load_in_4bit:
                raise ValueError("lora 训练不能设置 load_in_4bit=true；4bit 训练请使用 qlora")
        return self


def load_training_config(path: str | Path) -> Stage9SftTrainingConfig:
    """读取训练 config（配置）JSON 并校验。"""

    return Stage9SftTrainingConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """写 UTF-8 JSON，供 checkpoint（检查点）、metrics（指标）和配置快照共用。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def create_checkpoint_dir(config: Stage9SftTrainingConfig) -> tuple[str, Path]:
    """创建本次训练 checkpoint（检查点）目录并返回 run_id。"""

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_name = "".join(char if char.isalnum() or char in "-_" else "-" for char in config.run_name)
    run_id = f"{safe_name}_{timestamp}_{uuid.uuid4().hex[:8]}"
    output_root = Path(config.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    checkpoint_dir = output_root / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    return run_id, checkpoint_dir


def collect_framework_versions() -> dict[str, str]:
    """收集训练框架版本；缺失依赖写 unavailable，避免本地 smoke（冒烟）被重依赖卡住。"""

    versions: dict[str, str] = {}
    for package in ("python", "torch", "transformers", "datasets", "peft", "bitsandbytes"):
        if package == "python":
            versions[package] = _python_version()
            continue
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def current_code_version() -> str:
    """返回当前 git 版本；dirty 表示工作树有未提交改动。"""

    try:
        revision = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"
    return f"{revision}{'-dirty' if status else ''}"


def _python_version() -> str:
    import sys

    return ".".join(str(part) for part in sys.version_info[:3])
