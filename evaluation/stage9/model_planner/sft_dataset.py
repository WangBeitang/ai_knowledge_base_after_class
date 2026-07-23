"""阶段 9 Planner（规划器）SFT（监督微调）数据读取和训练样本构造。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

from app.rag.evaluation.sft_exporter import SftArtifactStatus, SftPlannerSample
from app.rag.query.contracts import PlannerDecision
from evaluation.stage9.model_planner.decision_codec import decode_decision, encode_decision
from evaluation.stage9.model_planner.prompt_builder import (
    PlannerPromptConfig,
    build_planner_prompt,
)


class Stage9SftDatasetModel(BaseModel):
    """SFT（监督微调）数据集内部 schema（结构）基类。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


class SftTrainExample(Stage9SftDatasetModel):
    """
    单条可训练样本。

    prompt（提示词）是模型输入，target_json（目标 JSON）是模型要学习输出的
    PlannerDecision（规划器决策）。context_key（上下文键）用于本地 smoke（冒烟）checkpoint
    的精确回放，不作为正式模型能力指标。
    """

    sample_id: str = Field(min_length=1)
    source_case_id: str = Field(min_length=1)
    split: str = Field(min_length=1)
    turn_index: int = Field(ge=1)
    prompt: str = Field(min_length=1)
    prompt_hash: str = Field(min_length=1)
    context_key: str = Field(min_length=1)
    input_context: dict[str, Any]
    target_json: str = Field(min_length=1)
    target_decision: dict[str, Any]
    target_action: str = Field(min_length=1)
    target_reason_code: str = Field(min_length=1)
    allowed_actions: list[str] = Field(min_length=1)
    label_source: str = Field(min_length=1)
    review_status: str = Field(min_length=1)
    artifact_status: str = Field(min_length=1)
    prompt_char_count: int = Field(ge=1)
    target_char_count: int = Field(ge=1)


class SftDatasetStats(Stage9SftDatasetModel):
    """
    SFT（监督微调）数据集统计。

    format_parse_rate（格式解析率）只检查训练目标 JSON 是否能解析为合法 PlannerDecision
    （规划器决策），不代表模型训练后的推理正确率。
    """

    sample_count: int = Field(ge=0)
    source_case_count: int = Field(ge=0)
    action_counts: dict[str, int] = Field(default_factory=dict)
    reason_code_counts: dict[str, int] = Field(default_factory=dict)
    split_counts: dict[str, int] = Field(default_factory=dict)
    label_source_counts: dict[str, int] = Field(default_factory=dict)
    review_status_counts: dict[str, int] = Field(default_factory=dict)
    artifact_status_counts: dict[str, int] = Field(default_factory=dict)
    format_parse_rate: float = Field(ge=0, le=1)
    prompt_char_count_avg: float = Field(ge=0)
    target_char_count_avg: float = Field(ge=0)
    max_prompt_char_count: int = Field(ge=0)
    max_target_char_count: int = Field(ge=0)


def load_sft_samples(
        path: str | Path,
        *,
        max_samples: int | None = None,
        require_approved_training_seed: bool = True,
) -> list[SftPlannerSample]:
    """读取 SftPlannerSample（SFT 规划器样本）JSONL，并执行训练边界校验。"""

    samples: list[SftPlannerSample] = []
    with Path(path).open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if max_samples is not None and len(samples) >= max_samples:
                break
            raw_line = line.strip()
            if not raw_line:
                continue
            sample = SftPlannerSample.model_validate_json(raw_line)
            if require_approved_training_seed and sample.artifact_status != SftArtifactStatus.APPROVED_TRAINING_SEED:
                raise ValueError(
                    f"{path}:{line_number} 不是 approved_training_seed，不能进入阶段 9 SFT"
                )
            samples.append(sample)
    if not samples:
        raise ValueError(f"SFT 数据为空：{path}")
    return samples


def load_sft_train_examples(
        path: str | Path,
        *,
        prompt_config: PlannerPromptConfig | None = None,
        max_samples: int | None = None,
) -> tuple[list[SftTrainExample], SftDatasetStats]:
    """从 JSONL 直接读取并构造训练样本。"""

    samples = load_sft_samples(path, max_samples=max_samples)
    return build_sft_train_examples(samples, prompt_config=prompt_config)


def build_sft_train_examples(
        samples: Iterable[SftPlannerSample],
        *,
        prompt_config: PlannerPromptConfig | None = None,
) -> tuple[list[SftTrainExample], SftDatasetStats]:
    """把 SftPlannerSample（SFT 规划器样本）转换为 prompt + target_json。"""

    active_prompt_config = prompt_config or PlannerPromptConfig()
    examples: list[SftTrainExample] = []
    parse_success_count = 0
    for sample in samples:
        decision = PlannerDecision.model_validate(sample.target_decision)
        target_json = encode_decision(decision)
        prompt = build_planner_prompt(sample.input_context, config=active_prompt_config)
        allowed_actions = [str(action) for action in prompt.payload["allowed_actions"]]
        decode_result = decode_decision(target_json, allowed_actions=allowed_actions)
        if not decode_result.success:
            raise ValueError(
                f"样本 {sample.sample_id} 的 target_decision 不合法："
                f"{decode_result.error_code} {decode_result.error_message}"
            )
        parse_success_count += 1
        examples.append(SftTrainExample(
            sample_id=sample.sample_id,
            source_case_id=sample.source_case_id,
            split=sample.split.value,
            turn_index=sample.turn_index,
            prompt=prompt.prompt,
            prompt_hash=prompt.payload_hash,
            context_key=prompt.context_key,
            input_context=prompt.payload,
            target_json=target_json,
            target_decision=json.loads(target_json),
            target_action=decision.action.value,
            target_reason_code=decision.reason_code.value,
            allowed_actions=allowed_actions,
            label_source=sample.label_source,
            review_status=sample.review_status,
            artifact_status=sample.artifact_status.value,
            prompt_char_count=len(prompt.prompt),
            target_char_count=len(target_json),
        ))
    stats = _stats(examples, parse_success_count=parse_success_count)
    return examples, stats


def write_examples_preview(
        examples: Iterable[SftTrainExample],
        path: str | Path,
        *,
        limit: int = 20,
) -> None:
    """写出少量训练样本预览，方便本地和 GPU（显卡算力）服务器核对输入输出。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        for index, example in enumerate(examples):
            if index >= limit:
                break
            file_obj.write(json.dumps(example.model_dump(mode="json"), ensure_ascii=False) + "\n")


def load_sft_manifest(path: str | Path) -> dict[str, Any]:
    """读取 SFT manifest（清单）原始 JSON；阶段 9 merge manifest 暂不强制套旧 schema。"""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _stats(examples: list[SftTrainExample], *, parse_success_count: int) -> SftDatasetStats:
    action_counts = Counter(example.target_action for example in examples)
    reason_counts = Counter(example.target_reason_code for example in examples)
    split_counts = Counter(example.split for example in examples)
    label_source_counts = Counter(example.label_source for example in examples)
    review_status_counts = Counter(example.review_status for example in examples)
    artifact_status_counts = Counter(example.artifact_status for example in examples)
    prompt_lengths = [example.prompt_char_count for example in examples]
    target_lengths = [example.target_char_count for example in examples]
    sample_count = len(examples)
    return SftDatasetStats(
        sample_count=sample_count,
        source_case_count=len({example.source_case_id for example in examples}),
        action_counts=dict(sorted(action_counts.items())),
        reason_code_counts=dict(sorted(reason_counts.items())),
        split_counts=dict(sorted(split_counts.items())),
        label_source_counts=dict(sorted(label_source_counts.items())),
        review_status_counts=dict(sorted(review_status_counts.items())),
        artifact_status_counts=dict(sorted(artifact_status_counts.items())),
        format_parse_rate=(parse_success_count / sample_count) if sample_count else 0.0,
        prompt_char_count_avg=float(mean(prompt_lengths)) if prompt_lengths else 0.0,
        target_char_count_avg=float(mean(target_lengths)) if target_lengths else 0.0,
        max_prompt_char_count=max(prompt_lengths) if prompt_lengths else 0,
        max_target_char_count=max(target_lengths) if target_lengths else 0,
    )
