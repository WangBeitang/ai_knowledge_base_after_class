"""把检索 Action 的原始结果收口为强校验 Observation。"""

from collections.abc import Mapping

from app.process.query.agent.state import QueryGraphState
from app.rag.query.contracts import (
    EvidenceSourceType,
    EvidenceSummary,
    IdentifierResolutionStatus,
    ObservationStatus,
    PlannerDecision,
    PlannerExecutionStatus,
    PlannerHistoryItem,
    QueryAction,
    RetrievalObservation,
)
from app.rag.query.query_identifier_service import (
    build_not_found_question,
    build_suggestion_question,
    extract_identifiers_from_record,
    filter_records_matching_requested_identifiers,
    normalize_identifier_mapping,
    rank_identifier_suggestions,
    requested_identifiers_are_covered,
)
from app.rag.query.trace_service import safe_complete_action_step
from app.shared.runtime.logger import logger, node_log


RETRIEVAL_ACTIONS = {
    QueryAction.LOCAL_SEARCH,
    QueryAction.HYDE_SEARCH,
    QueryAction.WEB_SEARCH,
}


def current_decision(state: QueryGraphState) -> PlannerDecision:
    """读取并重新校验当前 Decision，禁止路由层使用未验证字典。"""
    raw_decision = state.get("current_planner_decision")
    if raw_decision is None:
        raise ValueError("执行 Action 前必须存在 current_planner_decision")
    return (
        raw_decision
        if isinstance(raw_decision, PlannerDecision)
        else PlannerDecision.model_validate(raw_decision)
    )


def is_expected_external_error(error: Exception) -> bool:
    """
    区分“可预期外部调用失败”和“代码自身错误”。

    网络超时、连接中断、Milvus/HTTP/MCP 客户端异常可以转成 FAILED Observation，让
    Planner 决定 fallback；ValueError、KeyError、TypeError 等普通编程错误继续抛到查询
    入口返回 500，不能伪装成证据不足。JSONDecodeError 例外，它通常表示外部响应损坏。
    """
    if isinstance(error, (TimeoutError, ConnectionError, OSError)):
        return True
    error_type = type(error)
    module_name = error_type.__module__.lower()
    class_name = error_type.__name__
    if class_name == "JSONDecodeError":
        return True
    return module_name.startswith(
        ("httpx", "httpcore", "requests", "aiohttp", "pymilvus", "openai", "agents")
    )


def build_failed_observation(
        state: QueryGraphState,
        action: QueryAction,
        error: Exception,
        duration_ms: int,
) -> RetrievalObservation:
    """把可预期外部错误转换成不包含异常正文的标准 FAILED Observation。"""
    error_code = f"{action.value.upper()}_{type(error).__name__.upper()}"
    logger.warning(
        f"检索 Action 可预期失败，trace_id={state.get('trace_id')}, "
        f"action={action.value}, error_code={error_code}"
    )
    return RetrievalObservation(
        action=action,
        status=ObservationStatus.FAILED,
        duration_ms=max(0, duration_ms),
        error_code=error_code,
    )


def append_action_history(
        state: QueryGraphState,
        *,
        execution_status: PlannerExecutionStatus,
) -> list[PlannerHistoryItem]:
    """在 Action 真正执行完成后追加历史；未执行的 Decision 绝不能提前写入。"""
    decision = current_decision(state)
    history = [
        item if isinstance(item, PlannerHistoryItem) else PlannerHistoryItem.model_validate(item)
        for item in state.get("planner_action_history", [])
    ]
    history.append(
        PlannerHistoryItem(
            step=len(history) + 1,
            decision=decision,
            execution_status=execution_status,
        )
    )
    return history


def _channel_counts(state: QueryGraphState) -> dict[str, int]:
    """记录各个已实际执行检索 Action 当前保留的原始候选数量。"""
    return {
        QueryAction.LOCAL_SEARCH.value: len(state.get("embedding_chunks") or []),
        QueryAction.HYDE_SEARCH.value: len(state.get("hyde_embedding_chunks") or []),
        QueryAction.WEB_SEARCH.value: len(state.get("web_search_docs") or []),
    }


def _evidence_summaries(reranked_docs: list[Mapping[str, object]]) -> list[EvidenceSummary]:
    """只把前 5 条受限摘要交给 Planner，不复制完整候选列表。"""
    summaries = []
    for document in reranked_docs[:5]:
        summaries.append(
            EvidenceSummary(
                document_id=document.get("document_id"),
                chunk_id=document.get("chunk_id"),
                title=str(document.get("title") or document.get("source_title") or "未命名证据"),
                source_type=EvidenceSourceType(document.get("source_type")),
                rerank_score=document.get("rerank_score"),
                matched_identifiers=extract_identifiers_from_record(document),
                content_excerpt=str(document.get("content") or "")[:500],
            )
        )
    return summaries


def _build_success_observation(
        state: QueryGraphState,
        action: QueryAction,
) -> tuple[RetrievalObservation, list[dict]]:
    """从累计 RRF/rerank 结果生成本轮 Observation，并再次执行编号同码保护。"""
    rrf_candidates = list(state.get("rrf_chunks") or [])
    reranked_docs = list(state.get("reranked_docs") or [])
    requested_identifiers = normalize_identifier_mapping(state.get("query_identifiers"))
    preliminary = state.get("retrieval_observation")
    if preliminary is not None and not isinstance(preliminary, RetrievalObservation):
        preliminary = RetrievalObservation.model_validate(preliminary)

    # 普通本地检索已经确定为“需要确认/未找到”时，不能让后续累计分数覆盖编号结论。
    if (
        isinstance(preliminary, RetrievalObservation)
        and preliminary.action == action
        and preliminary.identifier_resolution_status in {
            IdentifierResolutionStatus.SUGGESTION_REQUIRED,
            IdentifierResolutionStatus.NOT_FOUND,
        }
    ):
        return preliminary, []

    matched_identifiers: dict[str, list[str]] = {}
    suggested_identifiers: dict[str, list[str]] = {}
    clarification_question = None
    identifier_status = IdentifierResolutionStatus.NOT_APPLICABLE
    safe_reranked_docs = reranked_docs

    if requested_identifiers:
        exact_docs, matched_identifiers = filter_records_matching_requested_identifiers(
            reranked_docs,
            requested_identifiers,
        )
        if requested_identifiers_are_covered(requested_identifiers, matched_identifiers):
            safe_reranked_docs = [dict(document) for document in exact_docs]
            identifier_status = IdentifierResolutionStatus.EXACT_MATCH
            if (
                isinstance(preliminary, RetrievalObservation)
                and preliminary.identifier_resolution_status
                == IdentifierResolutionStatus.FALLBACK_EXACT_MATCH
            ):
                identifier_status = IdentifierResolutionStatus.FALLBACK_EXACT_MATCH
        else:
            # 不同编号候选只用于追问，不能进入答案证据。候选仍限定在已经通过当前权限、
            # dataset、subject 过滤并实际召回的累计列表中。
            suggested_identifiers = rank_identifier_suggestions(
                requested_identifiers,
                rrf_candidates,
            )
            if suggested_identifiers:
                identifier_status = IdentifierResolutionStatus.SUGGESTION_REQUIRED
                clarification_question = build_suggestion_question(
                    requested_identifiers,
                    suggested_identifiers,
                )
            else:
                identifier_status = IdentifierResolutionStatus.NOT_FOUND
                clarification_question = build_not_found_question(requested_identifiers)
            safe_reranked_docs = []

    candidate_count = len(rrf_candidates)
    reranked_count = len(safe_reranked_docs)
    top_rerank_score = (
        float(safe_reranked_docs[0]["rerank_score"])
        if reranked_count and safe_reranked_docs[0].get("rerank_score") is not None
        else None
    )
    observation = RetrievalObservation(
        action=action,
        status=ObservationStatus.SUCCESS if candidate_count else ObservationStatus.EMPTY,
        channel_counts=_channel_counts(state),
        candidate_count=candidate_count,
        reranked_count=reranked_count,
        top_rerank_score=top_rerank_score,
        requested_identifiers=requested_identifiers,
        matched_identifiers=matched_identifiers,
        identifier_resolution_status=identifier_status,
        suggested_identifiers=suggested_identifiers,
        citation_count=0,
        evidence_summaries=_evidence_summaries(safe_reranked_docs),
        evidence_ambiguous=False,
        clarification_question=clarification_question,
        duration_ms=max(0, int(state.get("current_action_duration_ms", 0))),
        used_structured_filter=bool(
            isinstance(preliminary, RetrievalObservation) and preliminary.used_structured_filter
        ),
        filter_fallback=bool(
            isinstance(preliminary, RetrievalObservation) and preliminary.filter_fallback
        ),
    )
    return observation, safe_reranked_docs


@node_log("node_retrieval_observation")
def node_retrieval_observation(state: QueryGraphState) -> dict:
    """完成本轮检索 Action，写 Observation、执行历史和安全后的 rerank 证据。"""
    decision = current_decision(state)
    if decision.action not in RETRIEVAL_ACTIONS:
        raise ValueError("node_retrieval_observation 只能收口检索 Action")

    preliminary = state.get("retrieval_observation")
    if preliminary is not None and not isinstance(preliminary, RetrievalObservation):
        preliminary = RetrievalObservation.model_validate(preliminary)

    if (
        isinstance(preliminary, RetrievalObservation)
        and preliminary.action == decision.action
        and preliminary.status == ObservationStatus.FAILED
    ):
        observation = preliminary
        safe_reranked_docs: list[dict] = []
        execution_status = PlannerExecutionStatus.FAILED
    else:
        observation, safe_reranked_docs = _build_success_observation(state, decision.action)
        execution_status = PlannerExecutionStatus.COMPLETED

    history = append_action_history(state, execution_status=execution_status)
    # Observation 已经过强契约校验后再投影到 Mongo；trace_service 会删除正文片段，只保留
    # hash、身份和分数。Trace 写入失败不会改变这里返回给 Planner 的业务结果。
    safe_complete_action_step(
        state,
        observation=observation,
        execution_status=execution_status,
    )
    return {
        "retrieval_observation": observation,
        "reranked_docs": safe_reranked_docs,
        "clarification_question": observation.clarification_question,
        "planner_action_history": history,
        "planner_step": len(history),
    }
