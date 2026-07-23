"""阶段 9.2 Provider（执行器）观察记录。

RecordingActionProvider（记录型动作执行器）包在真实 Provider（执行器）外层。真实检索
仍由 MilvusActionProvider（Milvus 动作执行器）完成；本类只在 Environment（环境）生成
Observation（观察结果）后，把本次 Action（动作）的摘要写入 JSONL，供训练审计和异常
排查使用。
"""

from __future__ import annotations

import copy
import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.rag.evaluation.offline_environment import (
    OfflineActionProvider,
    OfflineError,
    OfflineState,
)
from app.rag.query.contracts import (
    PlannerDecision,
    QueryAction,
    RetrievalCandidate,
    RetrievalObservation,
)


PROVIDER_OBSERVATION_RECORD_VERSION = "stage9-provider-observation-v1"


class ProviderRecordModel(BaseModel):
    """Provider（执行器）记录文件的 schema（数据结构）公共基类。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


class ProviderObservationRecord(ProviderRecordModel):
    """
    单次 Action（动作）的 Provider（执行器）观察记录。

    这不是训练样本，也不是长期主环境。它只记录一次真实检索调用的关键边界：代码版本、
    snapshot（快照）、config（配置）、Action（动作）、候选身份和 Observation（观察结果）。
    """

    record_id: str = Field(min_length=1, description="记录唯一 ID；写入时生成，用于排查单条 Action。")
    record_version: str = Field(
        default=PROVIDER_OBSERVATION_RECORD_VERSION,
        description="记录 schema（数据结构）版本；字段变化时必须升级。",
    )
    recorded_at: str = Field(min_length=1, description="UTC ISO 记录时间；表示日志落盘时间。")
    provider_name: str = Field(min_length=1, description="外层 Provider（执行器）名称，通常是 recording_action_provider。")
    wrapped_provider_name: str = Field(min_length=1, description="被包装的真实 Provider（执行器）名称。")
    run_id: str = Field(min_length=1, description="轨迹运行 ID；来自 OfflineState（离线运行状态）。")
    case_id: str = Field(min_length=1, description="评测 case（样本）ID。")
    snapshot_id: str = Field(min_length=1, description="环境 snapshot（快照）ID。")
    step: int = Field(ge=1, description="本次 Action（动作）处于轨迹第几步，从 1 开始。")
    action: QueryAction = Field(description="本次执行的 Action（动作）。")
    query: str = Field(min_length=1, description="本次 Action（动作）实际使用的查询文本。")
    retrieval_config_version: str = Field(min_length=1, description="检索配置版本。")
    retrieval_config_hash: str = Field(min_length=64, max_length=64, description="检索配置 JSON 的 SHA-256。")
    candidate_count: int = Field(ge=0, description="本次 Provider（执行器）返回的候选数量。")
    candidates: list[dict[str, Any]] = Field(default_factory=list, description="候选证据快照；正文会按上限截断。")
    observation: dict[str, Any] = Field(default_factory=dict, description="Environment（环境）生成的 Observation（观察结果）。")
    duration_ms: int = Field(ge=0, description="本次 Action（动作）耗时，单位毫秒。")
    error: dict[str, Any] | None = Field(default=None, description="结构化错误；正常执行为空。")


class RecordingActionProvider:
    """
    记录型 ActionProvider（动作执行器）。

    它实现 OfflineActionProvider（离线动作执行器）的三个检索方法，并额外暴露
    record_observation（记录观察结果）钩子。OfflineRagEnvironment（离线 RAG 环境）会在
    生成 Observation（观察结果）后调用该钩子。
    """

    provider_name = "recording_action_provider"

    def __init__(
            self,
            wrapped_provider: OfflineActionProvider,
            *,
            output_path: str | Path,
            max_candidate_content_chars: int = 500,
    ) -> None:
        if max_candidate_content_chars <= 0:
            raise ValueError("max_candidate_content_chars 必须大于 0")
        self.wrapped_provider = wrapped_provider
        self.output_path = Path(output_path)
        self.max_candidate_content_chars = int(max_candidate_content_chars)

    @property
    def wrapped_provider_name(self) -> str:
        return str(getattr(self.wrapped_provider, "provider_name", self.wrapped_provider.__class__.__name__))

    def local_search(self, state: OfflineState, decision: PlannerDecision) -> list[RetrievalCandidate]:
        """委托执行 local_search（本地检索）。"""
        return copy.deepcopy(self.wrapped_provider.local_search(state, decision))

    def hyde_search(self, state: OfflineState, decision: PlannerDecision) -> list[RetrievalCandidate]:
        """委托执行 hyde_search（假设式改写检索）。"""
        return copy.deepcopy(self.wrapped_provider.hyde_search(state, decision))

    def web_search(self, state: OfflineState, decision: PlannerDecision) -> list[RetrievalCandidate]:
        """委托执行 web_search（网页检索）。"""
        return copy.deepcopy(self.wrapped_provider.web_search(state, decision))

    def record_observation(
            self,
            *,
            state: OfflineState,
            decision: PlannerDecision,
            candidates: list[RetrievalCandidate],
            observation: RetrievalObservation,
            error: OfflineError | None,
            duration_ms: int,
    ) -> None:
        """把单次 Action（动作）看到的 Observation（观察结果）追加写入 JSONL。"""
        record = ProviderObservationRecord(
            record_id=f"provider_record_{uuid.uuid4().hex}",
            recorded_at=datetime.now(UTC).isoformat(timespec="seconds"),
            provider_name=self.provider_name,
            wrapped_provider_name=self.wrapped_provider_name,
            run_id=state.run_id,
            case_id=state.case_id,
            snapshot_id=state.snapshot_id,
            step=state.planner_step + 1,
            action=decision.action,
            query=decision.query,
            retrieval_config_version=state.retrieval_config_version,
            retrieval_config_hash=_stable_hash(state.retrieval_config_snapshot),
            candidate_count=len(candidates),
            candidates=[
                _candidate_payload(candidate, max_content_chars=self.max_candidate_content_chars)
                for candidate in candidates
            ],
            observation=observation.model_dump(mode="json"),
            duration_ms=duration_ms,
            error=error.model_dump(mode="json") if error is not None else None,
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n")


def read_provider_observation_records(path: str | Path) -> list[ProviderObservationRecord]:
    """读取 ProviderObservationRecord（执行器观察记录）JSONL，并执行 schema（数据结构）校验。"""
    records: list[ProviderObservationRecord] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(ProviderObservationRecord.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number} Provider 记录非法：{exc}") from exc
    return records


def _candidate_payload(candidate: RetrievalCandidate, *, max_content_chars: int) -> dict[str, Any]:
    payload = candidate.model_dump(mode="json")
    content = str(payload.get("content") or "")
    if len(content) > max_content_chars:
        payload["content"] = content[:max_content_chars]
        payload["content_truncated"] = True
    else:
        payload["content_truncated"] = False
    return payload


def _stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
