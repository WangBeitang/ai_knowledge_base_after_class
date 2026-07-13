"""把运行时 QueryGraphState 安全投影为可持久化 Retrieval Trace。"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from app.infra.persistence.retrieval_trace_repository import get_retrieval_trace_repository
from app.rag.query.config import (
    POLICY_VERSION,
    RETRIEVAL_CONFIG_VERSION,
    build_retrieval_config_snapshot,
)
from app.rag.query.contracts import (
    Citation,
    IdentifierResolutionStatus,
    PlannerDecision,
    PlannerExecutionStatus,
    QueryAction,
    RetrievalCandidate,
    RetrievalConfigSnapshot,
    RetrievalMode,
    RetrievalObservation,
    RetrievalTrace,
    RetrievalTraceStatus,
    TraceChannelHit,
    TraceEvidenceSummary,
    TraceObservation,
    TracePlannerStep,
    TraceStepStatus,
    UsageMetrics,
)
from app.rag.query.query_identifier_service import extract_identifiers_from_record
from app.shared.runtime.logger import logger


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed_ms(started_at: str) -> int:
    """根据带时区 ISO 时间计算非负耗时；非法入口时间按 0 处理并由日志定位。"""
    try:
        started = datetime.fromisoformat(started_at)
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        return max(0, int((datetime.now(timezone.utc) - started).total_seconds() * 1000))
    except (TypeError, ValueError):
        return 0


def _content_hash(content: object) -> str:
    """只持久化文本 SHA-256；hash 可核验内容版本，但不能还原私密正文。"""
    return hashlib.sha256(str(content or "").encode("utf-8")).hexdigest()


def _usage_from_metadata(metadata: dict[str, object] | None, *, duration_ms: int | None = None) -> UsageMetrics:
    metadata = dict(metadata or {})
    input_tokens = max(0, int(metadata.get("input_tokens") or 0))
    output_tokens = max(0, int(metadata.get("output_tokens") or 0))
    total_tokens = max(0, int(metadata.get("total_tokens") or input_tokens + output_tokens))
    return UsageMetrics(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        duration_ms=max(0, int(duration_ms if duration_ms is not None else metadata.get("duration_ms") or 0)),
        estimated_cost=max(0.0, float(metadata.get("estimated_cost") or 0.0)),
        currency=str(metadata.get("currency") or "CNY"),
    )


def project_observation(observation: RetrievalObservation | dict | None) -> TraceObservation | None:
    """
    删除 Observation 中的 ``content_excerpt``，只保留正文 hash 和结构化事实。

    该函数是 Trace 的隐私边界：调用方不能直接 ``model_dump`` 运行时 Observation 写 Mongo。
    """
    if observation is None:
        return None
    validated = (
        observation
        if isinstance(observation, RetrievalObservation)
        else RetrievalObservation.model_validate(observation)
    )
    evidence_summaries = [
        TraceEvidenceSummary(
            document_id=summary.document_id,
            chunk_id=summary.chunk_id,
            title=summary.title,
            source_type=summary.source_type,
            rerank_score=summary.rerank_score,
            matched_identifiers=summary.matched_identifiers,
            content_excerpt_hash=_content_hash(summary.content_excerpt),
        )
        for summary in validated.evidence_summaries
    ]
    return TraceObservation(
        action=validated.action,
        status=validated.status,
        channel_counts=validated.channel_counts,
        candidate_count=validated.candidate_count,
        reranked_count=validated.reranked_count,
        top_rerank_score=validated.top_rerank_score,
        requested_identifiers=validated.requested_identifiers,
        matched_identifiers=validated.matched_identifiers,
        identifier_resolution_status=validated.identifier_resolution_status,
        suggested_identifiers=validated.suggested_identifiers,
        citation_count=validated.citation_count,
        evidence_summaries=evidence_summaries,
        evidence_ambiguous=validated.evidence_ambiguous,
        clarification_question=validated.clarification_question,
        duration_ms=validated.duration_ms,
        error_code=validated.error_code,
        used_structured_filter=validated.used_structured_filter,
        filter_fallback=validated.filter_fallback,
    )


def _config_snapshot(state: dict[str, Any]) -> RetrievalConfigSnapshot:
    raw_snapshot = state.get("retrieval_config_snapshot") or build_retrieval_config_snapshot(
        retrieval_mode=state.get("retrieval_mode"),
        web_fallback_enabled=state.get("web_search_allowed"),
    )
    return RetrievalConfigSnapshot.model_validate(raw_snapshot)


def build_running_trace(state: dict[str, Any]) -> RetrievalTrace:
    """创建查询入口的 running Trace；此时主体、步骤和最终引用尚为空。"""
    planner_metadata = dict(state.get("planner_runtime_metadata") or {})
    return RetrievalTrace(
        trace_id=str(state["trace_id"]),
        session_id=str(state["session_id"]),
        owner_user_id=str(state["owner_user_id"]),
        tenant_id=str(state["tenant_id"]),
        dataset_ids=list(state["dataset_ids"]),
        original_query=str(state["original_query"]),
        rewritten_query=str(state.get("rewritten_query") or ""),
        subject_ids=list(state.get("subject_ids") or []),
        standard_subject_names=list(state.get("standard_subject_names") or []),
        query_identifiers=dict(state.get("query_identifiers") or {}),
        policy_version=str(state.get("policy_version") or POLICY_VERSION),
        planner_type=str(state.get("planner_type") or "rule"),
        provider=planner_metadata.get("provider"),
        model_id=planner_metadata.get("model_id"),
        model_revision=planner_metadata.get("model_revision"),
        prompt_version=planner_metadata.get("prompt_version"),
        retrieval_config_version=str(
            state.get("retrieval_config_version") or RETRIEVAL_CONFIG_VERSION
        ),
        retrieval_mode=RetrievalMode(state["retrieval_mode"]),
        retrieval_config_snapshot=_config_snapshot(state),
        status=RetrievalTraceStatus.RUNNING,
        started_at=str(state["query_started_at"]),
    )


def _build_step(
        state: dict[str, Any],
        *,
        decision: PlannerDecision,
        planner_runtime_metadata: dict[str, object],
        output_observation: RetrievalObservation | dict | None,
        execution_status: TraceStepStatus,
        duration_ms: int,
) -> TracePlannerStep:
    return TracePlannerStep(
        step=int(state.get("planner_step", 0)) + (0 if output_observation is not None else 1),
        input_observation=project_observation(state.get("retrieval_observation")),
        decision=decision,
        execution_status=execution_status,
        output_observation=project_observation(output_observation),
        duration_ms=max(0, duration_ms),
        planner_usage=_usage_from_metadata(planner_runtime_metadata),
    )


def _persistence_enabled(state: dict[str, Any]) -> bool:
    return bool(state.get("trace_persistence_enabled"))


def safe_create_running_trace(state: dict[str, Any]) -> None:
    """Trace 创建失败只记日志，不改变查询结果和权限边界。"""
    if not _persistence_enabled(state):
        return
    try:
        trace = build_running_trace(state)
        get_retrieval_trace_repository().create_running(trace.model_dump(mode="json"))
    except Exception as error:  # Trace 是旁路可观测能力，不能让主查询失败。
        logger.exception(
            f"创建 Retrieval Trace 失败，trace_id={state.get('trace_id')}, "
            f"error_type={type(error).__name__}"
        )


def safe_record_planner_decision(
        state: dict[str, Any],
        *,
        decision: PlannerDecision,
        planner_runtime_metadata: dict[str, object],
) -> None:
    """Planner 决策完成后立即追加 pending step。"""
    if not _persistence_enabled(state):
        return
    try:
        step = _build_step(
            state,
            decision=decision,
            planner_runtime_metadata=planner_runtime_metadata,
            output_observation=None,
            execution_status=TraceStepStatus.PENDING,
            duration_ms=int(planner_runtime_metadata.get("duration_ms") or 0),
        )
        get_retrieval_trace_repository().append_step(
            str(state["trace_id"]),
            step.model_dump(mode="json"),
        )
    except Exception as error:
        logger.exception(
            f"追加 Planner Trace step 失败，trace_id={state.get('trace_id')}, "
            f"error_type={type(error).__name__}"
        )


def safe_complete_action_step(
        state: dict[str, Any],
        *,
        observation: RetrievalObservation,
        execution_status: PlannerExecutionStatus,
) -> None:
    """检索 Action 完成后，用输出 Observation 更新本步 Trace。"""
    if not _persistence_enabled(state):
        return
    try:
        raw_decision = state.get("current_planner_decision")
        decision = (
            raw_decision
            if isinstance(raw_decision, PlannerDecision)
            else PlannerDecision.model_validate(raw_decision)
        )
        # 此时 state.planner_step 仍是上一轮已完成数，因此完整 step 的编号仍需 +1。
        pending_state = {**state, "planner_step": int(state.get("planner_step", 0)) + 1}
        step = _build_step(
            pending_state,
            decision=decision,
            planner_runtime_metadata=dict(state.get("planner_runtime_metadata") or {}),
            output_observation=observation,
            execution_status=(
                TraceStepStatus.FAILED
                if execution_status == PlannerExecutionStatus.FAILED
                else TraceStepStatus.COMPLETED
            ),
            duration_ms=observation.duration_ms,
        )
        get_retrieval_trace_repository().complete_step(
            str(state["trace_id"]),
            step.model_dump(mode="json"),
        )
    except Exception as error:
        logger.exception(
            f"完成检索 Trace step 失败，trace_id={state.get('trace_id')}, "
            f"error_type={type(error).__name__}"
        )


def _channel_hits(state: dict[str, Any]) -> list[TraceChannelHit]:
    """从真实执行过的 Action 原始列表生成无正文候选投影。"""
    action_lists = (
        (QueryAction.LOCAL_SEARCH, state.get("embedding_chunks") or []),
        (QueryAction.HYDE_SEARCH, state.get("hyde_embedding_chunks") or []),
        (QueryAction.WEB_SEARCH, state.get("web_search_docs") or []),
    )
    rerank_by_identity: dict[tuple[str, str], float | None] = {}
    for raw_candidate in state.get("reranked_docs") or []:
        candidate = RetrievalCandidate.model_validate(raw_candidate)
        identity = (
            candidate.source_type.value,
            str(candidate.chunk_id if candidate.chunk_id is not None else candidate.url),
        )
        rerank_by_identity[identity] = candidate.rerank_score

    hits: list[TraceChannelHit] = []
    for action, raw_candidates in action_lists:
        channel = {
            QueryAction.LOCAL_SEARCH: "original",
            QueryAction.HYDE_SEARCH: "hyde",
            QueryAction.WEB_SEARCH: "web",
        }[action]
        for rank, raw_candidate in enumerate(raw_candidates, start=1):
            candidate = RetrievalCandidate.model_validate(raw_candidate)
            identity = (
                candidate.source_type.value,
                str(candidate.chunk_id if candidate.chunk_id is not None else candidate.url),
            )
            hits.append(TraceChannelHit(
                channel=channel,
                document_id=candidate.document_id,
                chunk_id=candidate.chunk_id,
                index_version=candidate.index_version,
                rank=rank,
                retrieval_score=candidate.retrieval_score,
                rerank_score=rerank_by_identity.get(identity),
                matched_identifiers=extract_identifiers_from_record(candidate.model_dump(mode="json")),
                content_excerpt_hash=_content_hash(candidate.content),
            ))
    return hits


def safe_complete_terminal_step_and_trace(
        state: dict[str, Any],
        *,
        execution_status: PlannerExecutionStatus,
) -> None:
    """写入最后一个终止 Action，并把 running Trace 收口为 completed。"""
    if not _persistence_enabled(state):
        return
    try:
        raw_decision = state.get("current_planner_decision")
        decision = (
            raw_decision
            if isinstance(raw_decision, PlannerDecision)
            else PlannerDecision.model_validate(raw_decision)
        )
        terminal_step = TracePlannerStep(
            step=int(state.get("planner_step", 0)),
            input_observation=project_observation(state.get("retrieval_observation")),
            decision=decision,
            execution_status=(
                TraceStepStatus.FAILED
                if execution_status == PlannerExecutionStatus.FAILED
                else TraceStepStatus.COMPLETED
            ),
            output_observation=None,
            duration_ms=int((state.get("answer_runtime_metadata") or {}).get("duration_ms") or 0),
            planner_usage=_usage_from_metadata(state.get("planner_runtime_metadata") or {}),
        )
        repository = get_retrieval_trace_repository()
        repository.complete_step(str(state["trace_id"]), terminal_step.model_dump(mode="json"))

        observation = project_observation(state.get("retrieval_observation"))
        answer_metadata = dict(state.get("answer_runtime_metadata") or {})
        citations = [
            item if isinstance(item, Citation) else Citation.model_validate(item)
            for item in state.get("citations") or []
        ]
        channel_hits = _channel_hits(state)
        completed_at = _utc_now_iso()
        repository.complete_trace(str(state["trace_id"]), {
            "rewritten_query": str(state.get("rewritten_query") or ""),
            "subject_ids": list(state.get("subject_ids") or []),
            "standard_subject_names": list(state.get("standard_subject_names") or []),
            "identifier_resolution_status": (
                observation.identifier_resolution_status.value
                if observation is not None
                else IdentifierResolutionStatus.NOT_APPLICABLE.value
            ),
            "suggested_identifiers": (
                observation.suggested_identifiers if observation is not None else {}
            ),
            "policy_version": str(state.get("policy_version") or POLICY_VERSION),
            "planner_type": str(state.get("planner_type") or "rule"),
            "provider": (state.get("planner_runtime_metadata") or {}).get("provider"),
            "model_id": (state.get("planner_runtime_metadata") or {}).get("model_id"),
            "model_revision": (state.get("planner_runtime_metadata") or {}).get("model_revision"),
            "prompt_version": (state.get("planner_runtime_metadata") or {}).get("prompt_version"),
            "planner_usage": _usage_from_metadata(
                state.get("planner_runtime_metadata") or {},
                duration_ms=int(state.get("planner_total_duration_ms") or 0),
            ).model_dump(mode="json"),
            "answer_provider": answer_metadata.get("provider"),
            "answer_model_id": answer_metadata.get("model_id"),
            "answer_model_revision": answer_metadata.get("model_revision"),
            "answer_prompt_version": answer_metadata.get("prompt_version"),
            "answer_usage": _usage_from_metadata(answer_metadata).model_dump(mode="json"),
            "retrieval_config_version": str(
                state.get("retrieval_config_version") or RETRIEVAL_CONFIG_VERSION
            ),
            "retrieval_mode": str(state["retrieval_mode"]),
            "retrieval_config_snapshot": _config_snapshot(state).model_dump(mode="json"),
            "index_versions": sorted({
                hit.index_version for hit in channel_hits if hit.index_version is not None
            }),
            "status": RetrievalTraceStatus.COMPLETED.value,
            "terminal_action": decision.action.value,
            "terminal_reason_code": (
                state.get("terminal_reason_code") or decision.reason_code
            ).value,
            "channel_hits": [hit.model_dump(mode="json") for hit in channel_hits],
            "final_citations": [citation.model_dump(mode="json") for citation in citations],
            "completed_at": completed_at,
            "total_duration_ms": _elapsed_ms(str(state.get("query_started_at") or "")),
            "error_code": (
                "ANSWER_ACTION_EXECUTION_ERROR"
                if execution_status == PlannerExecutionStatus.FAILED
                else None
            ),
        })
    except Exception as error:
        logger.exception(
            f"收口 Retrieval Trace 失败，trace_id={state.get('trace_id')}, "
            f"error_type={type(error).__name__}"
        )


def safe_fail_trace(state: dict[str, Any], error: Exception) -> None:
    """查询图抛出未处理异常时尽力标记 failed，且不持久化异常正文或堆栈。"""
    if not _persistence_enabled(state):
        return
    try:
        get_retrieval_trace_repository().fail_trace(str(state["trace_id"]), {
            "status": RetrievalTraceStatus.FAILED.value,
            "completed_at": _utc_now_iso(),
            "total_duration_ms": _elapsed_ms(str(state.get("query_started_at") or "")),
            "error_code": f"UNHANDLED_{type(error).__name__.upper()}",
        })
    except Exception as trace_error:
        logger.exception(
            f"标记 Retrieval Trace failed 失败，trace_id={state.get('trace_id')}, "
            f"error_type={type(trace_error).__name__}"
        )

