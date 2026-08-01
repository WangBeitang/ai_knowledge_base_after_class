"""把已审核 Planner case（规划器案例）导出为 GRPO 训练输入。

GRPO（组相对策略优化）的 rollout（采样轨迹）在训练阶段产生。本模块只冻结训练前
需要的 case 契约、参考轨迹身份和审核来源，不采样轨迹、不计算 Reward（奖励分数），
也不执行训练。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.rag.evaluation.case_schema import (
    CaseSplit,
    HumanReviewStatus,
    PlannerEvalCase,
    validate_case_collection,
)
from app.rag.query.contracts import QueryAction


GRPO_CASE_EXPORT_VERSION = "planner-grpo-case-export-v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class GrpoCaseExportModel(BaseModel):
    """GRPO case 导出公共 Schema（数据结构约束）。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


class GrpoCaseArtifactStatus(str, Enum):
    """case 是否已获准作为 GRPO train-only（仅训练）输入。"""

    APPROVED_TRAINING_CASE = "approved_grpo_training_case"


class GrpoReferenceTrajectory(GrpoCaseExportModel):
    """SFT 已审核轨迹在 GRPO case 中的参考身份，不作为强制唯一输出。"""

    source_trace_id: str = Field(min_length=1)
    route: list[QueryAction] = Field(min_length=1)
    content_fingerprint: str = Field(min_length=64, max_length=64)
    origin: str = Field(min_length=1)
    review_evidence: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_fingerprint(self) -> "GrpoReferenceTrajectory":
        if not _SHA256_PATTERN.fullmatch(self.content_fingerprint):
            raise ValueError("reference trajectory content_fingerprint 必须是 SHA256")
        return self


class GrpoTrainingCase(GrpoCaseExportModel):
    """GRPO 训练前冻结的单条 case，不包含训练期 rollout。"""

    dataset_version: str = Field(min_length=1)
    source_dataset_fingerprint: str = Field(min_length=64, max_length=64)
    case_contract: PlannerEvalCase
    reference_trajectory: GrpoReferenceTrajectory
    review_status: str = "reviewed"
    artifact_status: GrpoCaseArtifactStatus = GrpoCaseArtifactStatus.APPROVED_TRAINING_CASE
    record_fingerprint: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_training_boundary(self) -> "GrpoTrainingCase":
        case = self.case_contract
        if case.split != CaseSplit.TRAIN:
            raise ValueError("GRPO case 只能来自 train split")
        if case.human_review_status != HumanReviewStatus.REVIEWED:
            raise ValueError("GRPO case 必须已经人工审核")
        if self.review_status != "reviewed":
            raise ValueError("GRPO case review_status 必须是 reviewed")
        if self.reference_trajectory.route not in case.acceptable_action_paths:
            raise ValueError("参考轨迹不在 acceptable_action_paths 中")
        if not _SHA256_PATTERN.fullmatch(self.source_dataset_fingerprint):
            raise ValueError("source_dataset_fingerprint 必须是 SHA256")
        if not _SHA256_PATTERN.fullmatch(self.record_fingerprint):
            raise ValueError("record_fingerprint 必须是 SHA256")
        return self


class GrpoCaseExportManifest(GrpoCaseExportModel):
    """GRPO case 数据集清单；rollout 和训练状态必须保持 false。"""

    export_version: str = GRPO_CASE_EXPORT_VERSION
    created_at: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    source_dataset_fingerprint: str = Field(min_length=64, max_length=64)
    source_sft_freeze_status: str = Field(min_length=1)
    source_sft_legacy_exceptions: dict[str, Any]
    case_count: int = Field(ge=1)
    unique_query_count: int = Field(ge=1)
    route_counts: dict[str, int]
    terminal_counts: dict[str, int]
    origin_counts: dict[str, int]
    leakage_group_count: int = Field(ge=1)
    input_file_sha256: dict[str, str]
    output_file_sha256: dict[str, str]
    reward_profile_path: str = Field(min_length=1)
    reward_profile_sha256: str = Field(min_length=64, max_length=64)
    environment_snapshot_path: str = Field(min_length=1)
    environment_snapshot_sha256: str = Field(min_length=64, max_length=64)
    all_cases_train_only: bool
    all_cases_reviewed: bool
    all_reference_routes_accepted: bool
    rollout_source: str = "training_time_policy_sampling"
    rollout_generated: bool = False
    training_performed: bool = False

    @model_validator(mode="after")
    def reject_training_side_effects(self) -> "GrpoCaseExportManifest":
        if self.rollout_generated or self.training_performed:
            raise ValueError("GRPO case 导出阶段不得生成 rollout 或执行训练")
        if self.rollout_source != "training_time_policy_sampling":
            raise ValueError("GRPO rollout 必须在训练阶段由策略采样")
        if not (
            self.all_cases_train_only
            and self.all_cases_reviewed
            and self.all_reference_routes_accepted
        ):
            raise ValueError("GRPO case 导出门禁未通过")
        return self


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def stable_sha256(value: Any) -> str:
    """对结构化对象计算稳定 SHA256（文件内容哈希）。"""

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _normalized_query(query: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", query.lower())


def build_grpo_training_cases(
    *,
    trajectory_index: Sequence[Mapping[str, Any]],
    case_sources: Mapping[str, PlannerEvalCase],
    externally_reviewed_case_ids: set[str],
    dataset_version: str,
    source_dataset_fingerprint: str,
) -> list[GrpoTrainingCase]:
    """把冻结轨迹索引和原 case 合成 GRPO train-only case。

    externally_reviewed_case_ids 用于把“原 case 仍保留 pending 字段、但外部审核决定已绑定”的
    case 投影成 reviewed。原对象和原 fingerprint（内容指纹）不会被修改。
    """

    if not _SHA256_PATTERN.fullmatch(source_dataset_fingerprint):
        raise ValueError("source_dataset_fingerprint 必须是 SHA256")
    index_ids = [str(row.get("source_case_id") or "") for row in trajectory_index]
    if not index_ids or any(not case_id for case_id in index_ids):
        raise ValueError("轨迹索引缺少 source_case_id")
    if len(index_ids) != len(set(index_ids)):
        raise ValueError("轨迹索引 source_case_id 不唯一")
    missing = sorted(set(index_ids) - set(case_sources))
    if missing:
        raise ValueError(f"GRPO case 来源缺失：{missing}")

    records: list[GrpoTrainingCase] = []
    for index_row in trajectory_index:
        case_id = str(index_row["source_case_id"])
        source_case = case_sources[case_id]
        if source_case.query != str(index_row["query"]):
            raise ValueError(f"case query 与冻结轨迹不一致：{case_id}")
        if source_case.human_review_status == HumanReviewStatus.REVIEWED:
            reviewed_case = source_case
        elif case_id in externally_reviewed_case_ids:
            reviewed_case = PlannerEvalCase.model_validate({
                **source_case.model_dump(mode="json"),
                "human_review_status": HumanReviewStatus.REVIEWED.value,
                "notes": (
                    f"{source_case.notes}; reviewed_by_external_decision=true"
                    if source_case.notes
                    else "reviewed_by_external_decision=true"
                ),
            })
        else:
            raise ValueError(f"case 没有有效审核决定：{case_id}")

        route = [QueryAction(action) for action in index_row["route"]]
        reference = GrpoReferenceTrajectory(
            source_trace_id=str(index_row["source_trace_id"]),
            route=route,
            content_fingerprint=str(index_row["content_fingerprint"]),
            origin=str(index_row["origin"]),
            review_evidence=str(index_row["review_evidence"]),
        )
        fingerprint_payload = {
            "dataset_version": dataset_version,
            "source_dataset_fingerprint": source_dataset_fingerprint,
            "case_contract": reviewed_case.model_dump(mode="json"),
            "reference_trajectory": reference.model_dump(mode="json"),
            "review_status": "reviewed",
            "artifact_status": GrpoCaseArtifactStatus.APPROVED_TRAINING_CASE.value,
        }
        records.append(GrpoTrainingCase(
            **fingerprint_payload,
            record_fingerprint=stable_sha256(fingerprint_payload),
        ))

    validate_grpo_training_cases(records)
    return records


def validate_grpo_training_cases(cases: Sequence[GrpoTrainingCase]) -> dict[str, Any]:
    """执行GRPO case集合的数量、审核、路径和泄漏分组边界校验。"""

    if not cases:
        raise ValueError("GRPO case 集合不能为空")
    case_ids = [row.case_contract.case_id for row in cases]
    fingerprints = [row.record_fingerprint for row in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("GRPO case_id 不唯一")
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("GRPO record_fingerprint 不唯一")
    normalized_queries = [_normalized_query(row.case_contract.query) for row in cases]
    if len(normalized_queries) != len(set(normalized_queries)):
        raise ValueError("GRPO case 存在完全重复 query")

    planner_cases = [row.case_contract for row in cases]
    validate_case_collection(planner_cases)
    if any(case.split != CaseSplit.TRAIN for case in planner_cases):
        raise ValueError("GRPO case 混入非 train split")
    if any(case.human_review_status != HumanReviewStatus.REVIEWED for case in planner_cases):
        raise ValueError("GRPO case 混入未审核数据")
    if any(
        row.reference_trajectory.route not in row.case_contract.acceptable_action_paths
        for row in cases
    ):
        raise ValueError("GRPO case 参考路线不在可接受路线中")

    route_counts = Counter(
        " -> ".join(action.value for action in row.reference_trajectory.route)
        for row in cases
    )
    terminal_counts = Counter(row.reference_trajectory.route[-1].value for row in cases)
    origin_counts = Counter(row.reference_trajectory.origin for row in cases)
    return {
        "case_count": len(cases),
        "unique_query_count": len(set(normalized_queries)),
        "route_counts": dict(sorted(route_counts.items())),
        "terminal_counts": dict(sorted(terminal_counts.items())),
        "origin_counts": dict(sorted(origin_counts.items())),
        "leakage_group_count": len({case.leakage_group_id for case in planner_cases}),
        "all_cases_train_only": True,
        "all_cases_reviewed": True,
        "all_reference_routes_accepted": True,
    }


def load_grpo_training_cases(path: str | Path) -> list[GrpoTrainingCase]:
    """读取GRPO case JSONL并重新执行Schema和集合门禁。"""

    records: list[GrpoTrainingCase] = []
    with Path(path).open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            try:
                records.append(GrpoTrainingCase.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number} GRPO case 无效") from exc
    validate_grpo_training_cases(records)
    return records


def write_grpo_training_cases(path: str | Path, cases: Iterable[GrpoTrainingCase]) -> None:
    """写出GRPO case JSONL；调用方负责目录级原子替换。"""

    with Path(path).open("w", encoding="utf-8") as file_obj:
        for case in cases:
            file_obj.write(
                json.dumps(case.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
                + "\n"
            )
