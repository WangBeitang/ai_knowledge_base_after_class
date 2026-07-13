import pytest
from pydantic import ValidationError

from app.rag.query.contracts import (
    Citation,
    EvidenceSourceType,
    EvidenceSummary,
    IdentifierResolutionStatus,
    MAX_EVIDENCE_EXCERPT_CHARS,
    ObservationStatus,
    PlannerContext,
    PlannerDecision,
    PlannerExecutionStatus,
    PlannerHistoryItem,
    PlannerReasonCode,
    QueryAction,
    RetrievalObservation,
    SubjectResolutionStatus,
)
from app.rag.query.planner import QueryPlanner, RULE_BASED_POLICY_VERSION


def _local_evidence(index: int = 1, *, excerpt: str = "开机前检查急停按钮。") -> EvidenceSummary:
    return EvidenceSummary(
        document_id=f"doc_{index}",
        chunk_id=index,
        title="开机检查",
        source_type=EvidenceSourceType.LOCAL,
        rerank_score=0.91,
        matched_identifiers={"equipment_model": ["HAK 180"]},
        content_excerpt=excerpt,
    )


def test_query_action_and_reason_code_are_stable_string_enums():
    assert {action.value for action in QueryAction} == {
        "local_search",
        "hyde_search",
        "web_search",
        "answer",
        "ask_clarification",
        "refuse",
    }
    assert {reason.value for reason in PlannerReasonCode} == {
        "subject_ambiguous",
        "subject_required",
        "subject_not_found",
        "identifier_confirmation_required",
        "identifier_not_found",
        "realtime_query",
        "initial_local_search",
        "local_evidence_sufficient",
        "local_empty",
        "local_low_score",
        "hyde_evidence_sufficient",
        "hyde_still_insufficient",
        "web_evidence_available",
        "web_empty_or_failed",
        "evidence_ambiguous",
        "action_execution_error",
        "safe_guard_triggered",
    }
    assert {status.value for status in IdentifierResolutionStatus} == {
        "not_applicable",
        "exact_match",
        "fallback_exact_match",
        "suggestion_required",
        "not_found",
    }
    assert RULE_BASED_POLICY_VERSION == "rule-v1"


def test_planner_decision_accepts_protocol_strings_and_serializes_as_json_values():
    decision = PlannerDecision(
        action="local_search",
        query="  HAK 180 如何开机？  ",
        reason_code="initial_local_search",
    )

    assert decision.action is QueryAction.LOCAL_SEARCH
    assert decision.reason_code is PlannerReasonCode.INITIAL_LOCAL_SEARCH
    assert decision.query == "HAK 180 如何开机？"
    assert decision.model_dump(mode="json") == {
        "action": "local_search",
        "query": "HAK 180 如何开机？",
        "reason_code": "initial_local_search",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "unknown", "query": "问题", "reason_code": "initial_local_search"},
        {"action": "local_search", "query": "问题", "reason_code": "free_text_reason"},
        {"action": "local_search", "query": "   ", "reason_code": "initial_local_search"},
        {
            "action": "local_search",
            "query": "问题",
            "reason_code": "initial_local_search",
            "private_thought": "不允许的自由文本推理",
        },
    ],
)
def test_planner_decision_rejects_invalid_or_extra_protocol_values(payload):
    with pytest.raises(ValidationError):
        PlannerDecision(**payload)


def test_evidence_summary_enforces_local_and_web_identity_boundaries():
    local = _local_evidence()
    web = EvidenceSummary(
        title="厂商公告",
        source_type="web",
        rerank_score=0.78,
        content_excerpt="厂商发布了最新维护公告。",
    )

    assert local.document_id == "doc_1"
    assert web.document_id is None
    assert web.chunk_id is None

    with pytest.raises(ValidationError, match="本地证据必须同时包含"):
        EvidenceSummary(title="无来源", source_type="local")
    with pytest.raises(ValidationError, match="Web 证据的 document_id"):
        EvidenceSummary(
            document_id="doc_fake",
            chunk_id=1,
            title="伪造来源",
            source_type="web",
        )


def test_evidence_summary_limits_single_excerpt_length():
    with pytest.raises(ValidationError):
        _local_evidence(excerpt="证" * (MAX_EVIDENCE_EXCERPT_CHARS + 1))


def test_retrieval_observation_accepts_structured_success_result():
    observation = RetrievalObservation(
        action="local_search",
        status="success",
        channel_counts={"dense": 5, "learned_sparse": 5},
        candidate_count=6,
        reranked_count=2,
        top_rerank_score=0.91,
        matched_identifiers={"equipment_model": ["HAK 180"]},
        citation_count=1,
        evidence_summaries=[_local_evidence()],
        duration_ms=35,
        used_structured_filter=True,
    )

    assert observation.status is ObservationStatus.SUCCESS
    assert observation.channel_counts["dense"] == 5
    assert observation.evidence_summaries[0].chunk_id == 1


def test_exact_identifier_match_requires_actual_matched_identifier():
    observation = RetrievalObservation(
        action="local_search",
        status="success",
        candidate_count=1,
        reranked_count=1,
        top_rerank_score=0.91,
        requested_identifiers={"alarm_code": ["E020"]},
        matched_identifiers={"alarm_code": ["E020"]},
        identifier_resolution_status="exact_match",
        citation_count=1,
        evidence_summaries=[_local_evidence()],
        used_structured_filter=True,
    )

    assert observation.identifier_resolution_status is IdentifierResolutionStatus.EXACT_MATCH

    with pytest.raises(ValidationError, match="必须包含 matched_identifiers"):
        RetrievalObservation(
            action="local_search",
            status="success",
            candidate_count=1,
            reranked_count=1,
            top_rerank_score=0.91,
            requested_identifiers={"alarm_code": ["E020"]},
            identifier_resolution_status="exact_match",
        )

    with pytest.raises(ValidationError, match="必须覆盖 requested_identifiers"):
        RetrievalObservation(
            action="local_search",
            status="success",
            candidate_count=1,
            reranked_count=1,
            top_rerank_score=0.96,
            requested_identifiers={"alarm_code": ["E020"]},
            matched_identifiers={"alarm_code": ["E021"]},
            identifier_resolution_status="exact_match",
        )


def test_fallback_same_identifier_can_be_evidence_only_when_fallback_is_recorded():
    observation = RetrievalObservation(
        action="local_search",
        status="success",
        candidate_count=1,
        reranked_count=1,
        top_rerank_score=0.88,
        requested_identifiers={"alarm_code": ["E020"]},
        matched_identifiers={"alarm_code": ["E020"]},
        identifier_resolution_status="fallback_exact_match",
        citation_count=1,
        evidence_summaries=[_local_evidence()],
        used_structured_filter=True,
        filter_fallback=True,
    )

    assert observation.identifier_resolution_status is IdentifierResolutionStatus.FALLBACK_EXACT_MATCH

    with pytest.raises(ValidationError, match="必须标记 filter_fallback"):
        RetrievalObservation(
            action="local_search",
            status="success",
            candidate_count=1,
            reranked_count=1,
            top_rerank_score=0.88,
            requested_identifiers={"alarm_code": ["E020"]},
            matched_identifiers={"alarm_code": ["E020"]},
            identifier_resolution_status="fallback_exact_match",
        )


def test_different_identifier_is_suggestion_only_and_cannot_create_citation():
    observation = RetrievalObservation(
        action="local_search",
        status="success",
        candidate_count=1,
        reranked_count=1,
        top_rerank_score=0.96,
        requested_identifiers={"alarm_code": ["E020"]},
        matched_identifiers={"alarm_code": ["E021"]},
        identifier_resolution_status="suggestion_required",
        suggested_identifiers={"alarm_code": ["E021"]},
        clarification_question="当前知识库未找到 E020，是否想查询 E021？",
        citation_count=0,
        evidence_summaries=[_local_evidence()],
        used_structured_filter=True,
        filter_fallback=True,
    )

    assert observation.identifier_resolution_status is IdentifierResolutionStatus.SUGGESTION_REQUIRED
    assert observation.suggested_identifiers == {"alarm_code": ["E021"]}
    serialized_observation = observation.model_dump(mode="json")
    assert serialized_observation["requested_identifiers"] == {"alarm_code": ["E020"]}
    assert serialized_observation["identifier_resolution_status"] == "suggestion_required"
    assert serialized_observation["suggested_identifiers"] == {"alarm_code": ["E021"]}

    with pytest.raises(ValidationError, match="citation_count 必须为 0"):
        RetrievalObservation(
            action="local_search",
            status="success",
            candidate_count=1,
            reranked_count=1,
            top_rerank_score=0.96,
            requested_identifiers={"alarm_code": ["E020"]},
            matched_identifiers={"alarm_code": ["E021"]},
            identifier_resolution_status="suggestion_required",
            suggested_identifiers={"alarm_code": ["E021"]},
            clarification_question="当前知识库未找到 E020，是否想查询 E021？",
            citation_count=1,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"suggested_identifiers": {}},
        {"clarification_question": None},
        {"suggested_identifiers": {"alarm_code": []}},
        {"suggested_identifiers": {"alarm_code": ["E020"]}},
    ],
)
def test_identifier_suggestion_requires_non_empty_candidate_and_question(overrides):
    payload = {
        "action": "local_search",
        "status": "success",
        "requested_identifiers": {"alarm_code": ["E020"]},
        "identifier_resolution_status": "suggestion_required",
        "suggested_identifiers": {"alarm_code": ["E021"]},
        "clarification_question": "当前知识库未找到 E020，是否想查询 E021？",
    }
    payload.update(overrides)

    with pytest.raises(ValidationError):
        RetrievalObservation(**payload)


def test_identifier_not_found_requires_clarification_and_has_no_matched_identifier():
    observation = RetrievalObservation(
        action="local_search",
        status="empty",
        requested_identifiers={"alarm_code": ["E020"]},
        identifier_resolution_status="not_found",
        clarification_question="当前知识库未找到 E020，请核对设备屏幕上的报警码。",
    )

    assert observation.identifier_resolution_status is IdentifierResolutionStatus.NOT_FOUND

    with pytest.raises(ValidationError, match="不能包含 matched_identifiers"):
        RetrievalObservation(
            action="local_search",
            status="empty",
            requested_identifiers={"alarm_code": ["E020"]},
            identifier_resolution_status="not_found",
            matched_identifiers={"alarm_code": ["E021"]},
            clarification_question="请核对报警码。",
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"candidate_count": -1},
        {"candidate_count": 1, "reranked_count": 2, "top_rerank_score": 0.5},
        {"candidate_count": 1, "reranked_count": 1, "citation_count": 2, "top_rerank_score": 0.5},
        {"candidate_count": 1, "reranked_count": 1},
        {"top_rerank_score": 0.5},
        {"status": "failed"},
        {"status": "success", "error_code": "milvus_timeout"},
        {"filter_fallback": True},
    ],
)
def test_retrieval_observation_rejects_inconsistent_fields(overrides):
    payload = {
        "action": "local_search",
        "status": "empty",
    }
    payload.update(overrides)

    with pytest.raises(ValidationError):
        RetrievalObservation(**payload)


def test_failed_observation_requires_machine_readable_error_code():
    observation = RetrievalObservation(
        action="web_search",
        status="failed",
        error_code="web_timeout",
        duration_ms=1_500,
    )

    assert observation.error_code == "web_timeout"


def test_ambiguous_observation_requires_deterministic_clarification_question():
    observation = RetrievalObservation(
        action="local_search",
        status="success",
        candidate_count=2,
        reranked_count=2,
        top_rerank_score=0.82,
        evidence_summaries=[_local_evidence(1), _local_evidence(2)],
        evidence_ambiguous=True,
        clarification_question="请确认设备型号是 HAK 180 还是 HAK 180 Pro？",
    )

    assert observation.evidence_ambiguous is True

    with pytest.raises(ValidationError, match="必须提供 clarification_question"):
        RetrievalObservation(
            action="local_search",
            status="success",
            candidate_count=1,
            reranked_count=1,
            top_rerank_score=0.8,
            evidence_ambiguous=True,
        )


def test_retrieval_observation_limits_total_excerpt_length():
    with pytest.raises(ValidationError, match="证据摘要总长度"):
        RetrievalObservation(
            action="local_search",
            status="success",
            candidate_count=5,
            reranked_count=5,
            top_rerank_score=0.9,
            evidence_summaries=[
                _local_evidence(index, excerpt="证" * 450)
                for index in range(5)
            ],
        )


def test_retrieval_observation_limits_evidence_top_n():
    with pytest.raises(ValidationError):
        RetrievalObservation(
            action="local_search",
            status="success",
            candidate_count=6,
            reranked_count=6,
            top_rerank_score=0.9,
            evidence_summaries=[_local_evidence(index) for index in range(6)],
        )


def test_planner_context_uses_enums_and_isolates_mutable_defaults():
    context = PlannerContext(
        original_query="HAK180 如何开机？",
        current_query="HAK 180 如何开机？",
        subject_resolution_status="confirmed",
        max_steps=4,
        allowed_actions=["local_search", "refuse"],
    )
    another_context = PlannerContext(
        original_query="E021 是什么？",
        current_query="E021 是什么？",
        subject_resolution_status="confirmed",
        max_steps=4,
        allowed_actions=["local_search", "refuse"],
    )

    context.subject_ids.append("subject_hak_180")

    assert context.subject_resolution_status is SubjectResolutionStatus.CONFIRMED
    assert context.allowed_actions == [QueryAction.LOCAL_SEARCH, QueryAction.REFUSE]
    assert another_context.subject_ids == []


def test_planner_context_rejects_duplicate_allowed_actions():
    with pytest.raises(ValidationError, match="allowed_actions 不能包含重复"):
        PlannerContext(
            original_query="问题",
            current_query="问题",
            subject_resolution_status="confirmed",
            max_steps=4,
            allowed_actions=["local_search", "local_search"],
        )


def test_planner_context_requires_refuse_as_safe_exit():
    with pytest.raises(ValidationError, match="必须包含 refuse"):
        PlannerContext(
            original_query="问题",
            current_query="问题",
            subject_resolution_status="confirmed",
            max_steps=4,
            allowed_actions=["local_search"],
        )


def test_planner_history_requires_positive_step_and_closed_execution_status():
    decision = PlannerDecision(
        action="local_search",
        query="问题",
        reason_code="initial_local_search",
    )
    item = PlannerHistoryItem(
        step=1,
        decision=decision,
        execution_status="completed",
    )

    assert item.execution_status is PlannerExecutionStatus.COMPLETED

    with pytest.raises(ValidationError):
        PlannerHistoryItem(step=0, decision=decision, execution_status="completed")
    with pytest.raises(ValidationError):
        PlannerHistoryItem(step=1, decision=decision, execution_status="running")


def test_citation_enforces_local_and_web_source_contracts():
    local = Citation(
        document_id="doc_1",
        chunk_id=1001,
        title="开机说明",
        source="HAK180说明书",
        score=0.91,
        source_type="local",
    )
    web = Citation(
        title="厂商公告",
        source="https://example.com/notice",
        score=0.82,
        source_type="web",
    )

    assert local.source_type is EvidenceSourceType.LOCAL
    assert web.model_dump(mode="json")["source_type"] == "web"

    with pytest.raises(ValidationError, match="本地引用必须同时包含"):
        Citation(title="缺少 ID", source="手册", score=0.5, source_type="local")
    with pytest.raises(ValidationError, match="Web 引用的 document_id"):
        Citation(
            document_id="doc_fake",
            chunk_id=1,
            title="伪造 Web",
            source="https://example.com",
            score=0.5,
            source_type="web",
        )


def test_query_planner_protocol_accepts_structural_implementation():
    class DummyPlanner:
        policy_version = "test-rule-v1"

        def plan(self, context: PlannerContext) -> PlannerDecision:
            return PlannerDecision(
                action=QueryAction.LOCAL_SEARCH,
                query=context.current_query,
                reason_code=PlannerReasonCode.INITIAL_LOCAL_SEARCH,
            )

    planner = DummyPlanner()
    context = PlannerContext(
        original_query="问题",
        current_query="问题",
        subject_resolution_status="confirmed",
        max_steps=4,
        allowed_actions=["local_search", "refuse"],
    )

    assert isinstance(planner, QueryPlanner)
    assert planner.plan(context).action is QueryAction.LOCAL_SEARCH
