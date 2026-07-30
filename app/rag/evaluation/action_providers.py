"""正式离线评测 ActionProvider（动作执行器）。"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.process.query.agent.state import QueryGraphState, create_query_default_state
from app.rag.evaluation.offline_environment import (
    OfflineActionProvider,
    OfflineError,
    OfflineState,
)
from app.rag.query.config import RETRIEVAL_DEFAULT_MODE, normalize_retrieval_mode
from app.rag.query.contracts import (
    PlannerDecision,
    QueryAction,
    RetrievalCandidate,
    RetrievalObservation,
)


QueryNode = Callable[[QueryGraphState], Mapping[str, Any]]
PROVIDER_OBSERVATION_RECORD_VERSION = "provider-observation-v1"


class MilvusActionProvider:
    """
    真实检索 ActionProvider（动作执行器）。

    它把 OfflineState（离线运行状态）投影成业务 QueryGraphState（查询图状态），再复用
    现有 local_search（本地检索）、HyDE（假设式改写检索）和 Web（网页检索）节点。
    该类属于正式评测基础设施，训练、回归和云端 smoke（冒烟）都应复用这一份实现。
    """

    provider_name = "milvus_action_provider"

    def __init__(
            self,
            *,
            local_search_fn: QueryNode | None = None,
            hyde_search_fn: QueryNode | None = None,
            web_search_fn: QueryNode | None = None,
            chunk_status_filter_enabled: bool = True,
    ) -> None:
        self.local_search_fn = local_search_fn or _default_local_search
        self.hyde_search_fn = hyde_search_fn or _default_hyde_search
        self.web_search_fn = web_search_fn or _default_web_search
        # chunk_status_filter_enabled（切片启停过滤开关）默认开启，表示真实环境读取 Mongo
        # 中人工禁用 chunk。测试或纯离线 smoke（冒烟）可关闭，避免无意连接外部数据库。
        self.chunk_status_filter_enabled = bool(chunk_status_filter_enabled)

    def local_search(self, state: OfflineState, decision: PlannerDecision) -> list[RetrievalCandidate]:
        """执行 local_search（本地检索），返回 Milvus（向量数据库）本地候选。"""

        graph_state = self._query_graph_state(state, decision)
        result_state = self.local_search_fn(graph_state)
        return _candidate_list(result_state, "embedding_chunks")

    def hyde_search(self, state: OfflineState, decision: PlannerDecision) -> list[RetrievalCandidate]:
        """执行 hyde_search（假设式改写检索），返回本地候选。"""

        graph_state = self._query_graph_state(state, decision)
        result_state = self.hyde_search_fn(graph_state)
        return _candidate_list(result_state, "hyde_embedding_chunks")

    def web_search(self, state: OfflineState, decision: PlannerDecision) -> list[RetrievalCandidate]:
        """执行 web_search（网页检索），返回 Web（网页）候选。"""

        if not state.web_search_allowed:
            raise ValueError("当前 State（运行状态）不允许 Web（网页检索）")
        graph_state = self._query_graph_state(state, decision)
        result_state = self.web_search_fn(graph_state)
        return _candidate_list(result_state, "web_search_docs")

    def _query_graph_state(self, state: OfflineState, decision: PlannerDecision) -> QueryGraphState:
        """
        把 OfflineState（离线运行状态）投影成 QueryGraphState（查询图状态）。

        该转换只复制当前 Action（动作）需要的身份、权限、快照和 Planner（规划器）事实，
        不读取聊天历史，不写 Mongo Trace（追踪记录），也不把评测 State（运行状态）原样
        暴露给真实检索服务。
        """

        retrieval_mode = normalize_retrieval_mode(
            state.retrieval_config_snapshot.get("retrieval_mode") or RETRIEVAL_DEFAULT_MODE
        )
        return create_query_default_state(
            session_id=state.session_id,
            original_query=state.original_query,
            is_stream=False,
            owner_user_id=state.owner_user_id,
            tenant_id=state.tenant_id,
            dataset_ids=list(state.dataset_ids),
            query_started_at=datetime.now(UTC).isoformat(timespec="seconds"),
            rewritten_query=decision.query or state.current_query,
            subject_ids=list(state.subject_ids),
            standard_subject_names=list(state.standard_subject_names),
            subject_resolution_status=state.subject_resolution_status,
            subject_candidates=[],
            clarification_question=(
                state.latest_observation.clarification_question
                if state.latest_observation
                else None
            ),
            query_identifiers=copy.deepcopy(state.query_identifiers),
            history=[],
            trace_id=state.run_id,
            planner_step=state.planner_step,
            policy_version=state.policy_version,
            current_planner_decision=decision,
            planner_action_history=list(state.action_history),
            planner_type=state.planner_mode,
            planner_runtime_metadata={
                "provider": self.provider_name,
                "snapshot_id": state.snapshot_id,
            },
            web_search_allowed=state.web_search_allowed,
            safe_guard_triggered=bool(state.errors),
            planner_max_steps=state.planner_max_steps,
            retrieval_observation=state.latest_observation,
            retrieval_mode=retrieval_mode.value,
            retrieval_config_version=state.retrieval_config_version,
            retrieval_config_snapshot=copy.deepcopy(state.retrieval_config_snapshot),
            chunk_status_filter_enabled=self.chunk_status_filter_enabled,
            disabled_chunk_ids=list(state.disabled_chunk_ids),
            trace_persistence_enabled=False,
            history_persistence_enabled=False,
            execution_source="retrieval_test",
            config_match_status=state.config_match_status,
            corpus_match_status=state.corpus_match_status,
        )


RealActionProvider = MilvusActionProvider


class ProviderRecordModel(BaseModel):
    """Provider（执行器）记录文件的 schema（数据结构）公共基类。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


class ProviderObservationRecord(ProviderRecordModel):
    """
    单次 Action（动作）的 Provider（执行器）观察记录。

    这不是训练样本。它只记录一次真实检索调用的关键边界：代码版本、snapshot（快照）、
    config（配置）、Action（动作）、候选身份和 Observation（观察结果）。
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

    它包装真实 Provider（执行器），并在 OfflineRagEnvironment（离线 RAG 环境）生成
    Observation（观察结果）后追加写入 JSONL（逐行 JSON）审计记录。普通诊断可以限制
    candidate 正文长度；需要不可变 Replay（回放）的正式记录应传 None，完整保留候选正文。
    """

    provider_name = "recording_action_provider"

    def __init__(
            self,
            wrapped_provider: OfflineActionProvider,
            *,
            output_path: str | Path,
            max_candidate_content_chars: int | None = 500,
    ) -> None:
        if (
            max_candidate_content_chars is not None
            and max_candidate_content_chars <= 0
        ):
            raise ValueError("max_candidate_content_chars 必须大于 0")
        self.wrapped_provider = wrapped_provider
        self.output_path = Path(output_path)
        self.max_candidate_content_chars = (
            int(max_candidate_content_chars)
            if max_candidate_content_chars is not None
            else None
        )

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


class ReplayActionProvider:
    """
    最小回放 ActionProvider（动作执行器）。

    Replay（回放）只用于 smoke test（冒烟测试）、regression test（回归测试）和单条异常轨迹
    复现。缺少 case/action/query（样本/动作/查询）时必须明确报错，不能用空候选静默通过。
    """

    provider_name = "replay_action_provider"

    def __init__(self, records_path: str | Path) -> None:
        self.records_path = Path(records_path)
        self.records = read_provider_observation_records(self.records_path)
        self._record_by_key = _record_index(self.records)

    def local_search(self, state: OfflineState, decision: PlannerDecision) -> list[RetrievalCandidate]:
        """回放 local_search（本地检索）候选。"""

        return self._candidates_for(state, decision, QueryAction.LOCAL_SEARCH)

    def hyde_search(self, state: OfflineState, decision: PlannerDecision) -> list[RetrievalCandidate]:
        """回放 hyde_search（假设式改写检索）候选。"""

        return self._candidates_for(state, decision, QueryAction.HYDE_SEARCH)

    def web_search(self, state: OfflineState, decision: PlannerDecision) -> list[RetrievalCandidate]:
        """回放 web_search（网页检索）候选。"""

        return self._candidates_for(state, decision, QueryAction.WEB_SEARCH)

    def _candidates_for(
            self,
            state: OfflineState,
            decision: PlannerDecision,
            expected_action: QueryAction,
    ) -> list[RetrievalCandidate]:
        if decision.action != expected_action:
            raise ValueError(f"Replay 期望 Action={expected_action.value}，实际为 {decision.action.value}")
        key = _record_key(
            snapshot_id=state.snapshot_id,
            case_id=state.case_id,
            action=decision.action,
            query=decision.query,
        )
        record = self._record_by_key.get(key)
        if record is None:
            raise KeyError(
                "Replay 缺少对应记录："
                f"snapshot_id={state.snapshot_id}, case_id={state.case_id}, "
                f"action={decision.action.value}, query={decision.query}"
            )
        return [
            copy.deepcopy(RetrievalCandidate.model_validate(_candidate_contract_payload(candidate)))
            for candidate in record.candidates
        ]


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


def _candidate_list(result_state: Mapping[str, Any], field_name: str) -> list[RetrievalCandidate]:
    candidates = result_state.get(field_name) or []
    return [RetrievalCandidate.model_validate(candidate) for candidate in candidates]


def _default_local_search(state: QueryGraphState) -> Mapping[str, Any]:
    from app.rag.query.embedding_search_service import search_by_embedding

    return search_by_embedding(state)


def _default_hyde_search(state: QueryGraphState) -> Mapping[str, Any]:
    from app.rag.query.hyde_search_service import search_by_hyde

    return search_by_hyde(state)


def _default_web_search(state: QueryGraphState) -> Mapping[str, Any]:
    from app.rag.query.web_search_service import search_by_web

    return search_by_web(state)


def _candidate_payload(
        candidate: RetrievalCandidate,
        *,
        max_content_chars: int | None,
) -> dict[str, Any]:
    payload = candidate.model_dump(mode="json")
    content = str(payload.get("content") or "")
    if max_content_chars is not None and len(content) > max_content_chars:
        payload["content"] = content[:max_content_chars]
        payload["content_truncated"] = True
    else:
        payload["content_truncated"] = False
    return payload


def _stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _record_index(records: list[ProviderObservationRecord]) -> dict[tuple[str, str, str, str], ProviderObservationRecord]:
    index: dict[tuple[str, str, str, str], ProviderObservationRecord] = {}
    for record in records:
        key = _record_key(
            snapshot_id=record.snapshot_id,
            case_id=record.case_id,
            action=record.action,
            query=record.query,
        )
        if key in index:
            raise ValueError(
                "Replay 记录存在重复键："
                f"snapshot_id={record.snapshot_id}, case_id={record.case_id}, "
                f"action={record.action.value}, query={record.query}"
            )
        index[key] = record
    return index


def _record_key(
        *,
        snapshot_id: str,
        case_id: str,
        action: QueryAction,
        query: str,
) -> tuple[str, str, str, str]:
    return (
        snapshot_id,
        case_id,
        action.value,
        _normalize_query(query),
    )


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", str(query or "").strip())


def _candidate_contract_payload(candidate: dict) -> dict:
    payload = dict(candidate)
    # content_truncated（正文已截断）是记录文件的审计元数据，不属于 RetrievalCandidate
    #（检索候选）契约；回放执行时必须剥离，避免污染真实候选 schema（数据结构）。
    payload.pop("content_truncated", None)
    return payload
