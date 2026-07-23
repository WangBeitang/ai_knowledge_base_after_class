"""阶段 9 Planner（规划器）SFT（监督微调）配置与 checkpoint（检查点）读写。"""

from __future__ import annotations

import json
import subprocess
import uuid
from datetime import UTC, datetime
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT_ROOT = "evaluation/stage9/artifacts/sft/checkpoints"


class TrainingBackend(str, Enum):
    """
    训练后端枚举。

    DEBUG_MEMORIZED 是本地 smoke（冒烟）后端，只验证数据、prompt、codec、checkpoint 和
    推理链路；TRANSFORMERS_CAUSAL_LM 才是云端 GPU（显卡算力）实际训练入口。
    """

    DEBUG_MEMORIZED = "debug_memorized"
    TRANSFORMERS_CAUSAL_LM = "transformers_causal_lm"


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
    base_model_id: str = Field(min_length=1, description="基础模型 ID 或本地模型路径，必须写入 checkpoint。")
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
    trust_remote_code: bool = Field(default=False, description="是否信任远端模型自定义代码。")
    use_fast_tokenizer: bool = Field(default=True, description="是否优先使用 fast tokenizer（快速分词器）。")
    report_to: list[str] = Field(default_factory=list, description="Trainer 上报目标，默认空避免本地误连外部服务。")
    save_training_preview_count: int = Field(default=20, ge=0, description="保存多少条 prompt/target 预览。")

    @model_validator(mode="after")
    def validate_training_backend(self) -> "Stage9SftTrainingConfig":
        if self.training_backend == TrainingBackend.TRANSFORMERS_CAUSAL_LM and self.num_epochs == 0 and self.max_steps is None:
            raise ValueError("transformers_causal_lm 训练必须设置 num_epochs>0 或 max_steps")
        return self


class CheckpointManifest(CheckpointModel):
    """
    checkpoint manifest（检查点清单）。

    它是训练产物的机器可读索引：后续 eval（评测）、SFT（监督微调）对比和 GRPO（组相对
    策略优化强化训练）必须从这里追踪模型、数据、Reward（奖励）和代码版本。
    """

    run_id: str = Field(min_length=1)
    run_name: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    training_backend: TrainingBackend
    base_model_id: str = Field(min_length=1)
    train_data: str = Field(min_length=1)
    train_manifest: str = Field(min_length=1)
    reward_profile: str = Field(min_length=1)
    snapshot_id: str = Field(min_length=1)
    code_version: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    seed: int = Field(ge=0)
    framework_versions: dict[str, str] = Field(default_factory=dict)
    prompt_builder_version: str = Field(min_length=1)
    decision_codec_version: str = Field(min_length=1)
    model_path: str = Field(min_length=1)
    tokenizer_path: str = ""
    train_metrics_path: str = Field(min_length=1)
    eval_metrics_path: str = ""
    training_config_path: str = Field(min_length=1)
    sample_count: int = Field(ge=0)
    source_case_count: int = Field(ge=0)
    action_counts: dict[str, int] = Field(default_factory=dict)
    reason_code_counts: dict[str, int] = Field(default_factory=dict)
    max_input_tokens: int = Field(ge=1)
    max_target_tokens: int = Field(ge=1)


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


def load_checkpoint_manifest(checkpoint_dir: str | Path) -> CheckpointManifest:
    """读取 checkpoint_manifest.json。"""

    manifest_path = Path(checkpoint_dir) / "checkpoint_manifest.json"
    return CheckpointManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))


def collect_framework_versions() -> dict[str, str]:
    """收集训练框架版本；缺失依赖写 unavailable，避免本地 smoke（冒烟）被重依赖卡住。"""

    versions: dict[str, str] = {}
    for package in ("python", "torch", "transformers", "datasets"):
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
