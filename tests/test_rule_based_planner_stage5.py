import copy

import pytest

from app.rag.query.contracts import (
    IdentifierResolutionStatus,
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
from app.rag.query.planner import (
    QueryPlanner,
    REALTIME_RULE_VERSION,
    RULE_BASED_POLICY_VERSION,
    RuleBasedPlanner,
    RuleBasedPlannerConfig,
)


ALL_ACTIONS = list(QueryAction)

_HISTORY_REASON = {
    QueryAction.LOCAL_SEARCH: PlannerReasonCode.INITIAL_LOCAL_SEARCH,
    QueryAction.HYDE_SEARCH: PlannerReasonCode.LOCAL_LOW_SCORE,
    QueryAction.WEB_SEARCH: PlannerReasonCode.HYDE_STILL_INSUFFICIENT,
    QueryAction.ANSWER: PlannerReasonCode.LOCAL_EVIDENCE_SUFFICIENT,
    QueryAction.ASK_CLARIFICATION: PlannerReasonCode.EVIDENCE_AMBIGUOUS,
    QueryAction.REFUSE: PlannerReasonCode.SAFE_GUARD_TRIGGERED,
}


def _planner(threshold: float = 0.75) -> RuleBasedPlanner:
    return RuleBasedPlanner(
        config=RuleBasedPlannerConfig(
            rerank_evidence_threshold=threshold,
            retrieval_config_version="retrieval-dev-eval-v1",
        )
    )


def _history(
        *actions: QueryAction,
        failed_last: bool = False,
) -> list[PlannerHistoryItem]:
    items = []
    for index, action in enumerate(actions, start=1):
        execution_status = (
            PlannerExecutionStatus.FAILED
            if failed_last and index == len(actions)
            else PlannerExecutionStatus.COMPLETED
        )
        items.append(
            PlannerHistoryItem(
                step=index,
                decision=PlannerDecision(
                    action=action,
                    query="HAK 180 查询",
                    reason_code=_HISTORY_REASON[action],
                ),
                execution_status=execution_status,
            )
        )
    return items


def _observation(
        action: QueryAction,
        *,
        status: ObservationStatus = ObservationStatus.SUCCESS,
        candidate_count: int = 1,
        rerank_score: float | None = 0.9,
        identifier_status: IdentifierResolutionStatus = IdentifierResolutionStatus.NOT_APPLICABLE,
        evidence_ambiguous: bool = False,
        error_code: str | None = None,
) -> RetrievalObservation:
    reranked_count = 1 if rerank_score is not None else 0
    payload = {
        "action": action,
        "status": status,
        "candidate_count": candidate_count,
        "reranked_count": reranked_count,
        "top_rerank_score": rerank_score,
        "identifier_resolution_status": identifier_status,
        "evidence_ambiguous": evidence_ambiguous,
        "error_code": error_code,
    }
    if evidence_ambiguous:
        payload["clarification_question"] = "请确认要查询哪个设备或故障场景？"
    return RetrievalObservation(**payload)


def _context(
        *,
        subject_status: SubjectResolutionStatus = SubjectResolutionStatus.CONFIRMED,
        query: str = "HAK 180 如何开机？",
        history: list[PlannerHistoryItem] | None = None,
        observation: RetrievalObservation | None = None,
        subject_ids: list[str] | None = None,
        web_search_allowed: bool = True,
        safe_guard_triggered: bool = False,
        max_steps: int = 4,
        allowed_actions: list[QueryAction] | None = None,
) -> PlannerContext:
    history = history or []
    if subject_ids is None:
        subject_ids = ["subject_hak_180"] if subject_status == SubjectResolutionStatus.CONFIRMED else []
    return PlannerContext(
        original_query=query,
        current_query=query,
        subject_resolution_status=subject_status,
        subject_ids=subject_ids,
        latest_observation=observation,
        action_history=history,
        web_search_allowed=web_search_allowed,
        safe_guard_triggered=safe_guard_triggered,
        planner_step=len(history),
        max_steps=max_steps,
        allowed_actions=allowed_actions or ALL_ACTIONS,
    )


def test_rule_based_planner_implements_protocol_and_exposes_versions():
    planner = _planner()

    assert isinstance(planner, QueryPlanner)
    assert planner.policy_version == RULE_BASED_POLICY_VERSION
    assert planner.realtime_rule_version == REALTIME_RULE_VERSION
    assert planner.config.retrieval_config_version == "retrieval-dev-eval-v1"


@pytest.mark.parametrize(
    ("subject_status", "reason_code"),
    [
        (SubjectResolutionStatus.AMBIGUOUS, PlannerReasonCode.SUBJECT_AMBIGUOUS),
        (SubjectResolutionStatus.NO_MENTION, PlannerReasonCode.SUBJECT_REQUIRED),
    ],
)
def test_subject_ambiguity_or_missing_subject_asks_for_clarification(subject_status, reason_code):
    decision = _planner().plan(_context(subject_status=subject_status))

    assert decision.action == QueryAction.ASK_CLARIFICATION
    assert decision.reason_code == reason_code


def test_subject_not_found_refuses_non_realtime_query():
    decision = _planner().plan(_context(subject_status=SubjectResolutionStatus.NOT_FOUND))

    assert decision.action == QueryAction.REFUSE
    assert decision.reason_code == PlannerReasonCode.SUBJECT_NOT_FOUND


def test_subject_not_found_can_use_web_for_obvious_realtime_query():
    context = _context(
        subject_status=SubjectResolutionStatus.NOT_FOUND,
        query="ZX900 厂家今天发布了什么最新公告？",
    )

    decision = _planner().plan(context)

    assert decision.action == QueryAction.WEB_SEARCH
    assert decision.reason_code == PlannerReasonCode.REALTIME_QUERY


def test_obvious_realtime_query_goes_directly_to_web():
    context = _context(query="HAK180 厂家今天有没有发布最新召回公告？")

    decision = _planner().plan(context)

    assert decision.action == QueryAction.WEB_SEARCH
    assert decision.reason_code == PlannerReasonCode.REALTIME_QUERY


def test_realtime_rule_is_conservative_for_local_knowledge_question():
    context = _context(query="HAK180 最新维修方法是什么？")

    decision = _planner().plan(context)

    assert decision.action == QueryAction.LOCAL_SEARCH
    assert decision.reason_code == PlannerReasonCode.INITIAL_LOCAL_SEARCH


def test_ambiguous_subject_still_clarifies_before_realtime_web():
    context = _context(
        subject_status=SubjectResolutionStatus.AMBIGUOUS,
        query="这台机器今天有没有最新召回公告？",
    )

    assert _planner().plan(context).action == QueryAction.ASK_CLARIFICATION


def test_confirmed_subject_without_subject_id_safely_refuses_instead_of_full_search():
    context = _context(subject_ids=[])

    decision = _planner().plan(context)

    assert decision.action == QueryAction.REFUSE
    assert decision.reason_code == PlannerReasonCode.SAFE_GUARD_TRIGGERED


def test_confirmed_subject_without_id_cannot_bypass_guard_via_realtime_rule():
    context = _context(
        query="HAK180 厂家今天有没有发布最新召回公告？",
        subject_ids=[],
    )

    decision = _planner().plan(context)

    assert decision.action == QueryAction.REFUSE
    assert decision.reason_code == PlannerReasonCode.SAFE_GUARD_TRIGGERED


def test_confirmed_local_query_starts_with_local_search():
    decision = _planner().plan(_context())

    assert decision.action == QueryAction.LOCAL_SEARCH
    assert decision.reason_code == PlannerReasonCode.INITIAL_LOCAL_SEARCH


def test_different_identifier_candidate_always_clarifies_even_with_high_score():
    history = _history(QueryAction.LOCAL_SEARCH)
    observation = RetrievalObservation(
        action=QueryAction.LOCAL_SEARCH,
        status=ObservationStatus.SUCCESS,
        candidate_count=1,
        reranked_count=1,
        top_rerank_score=0.99,
        requested_identifiers={"alarm_code": ["E020"]},
        matched_identifiers={"alarm_code": ["E021"]},
        identifier_resolution_status=IdentifierResolutionStatus.SUGGESTION_REQUIRED,
        suggested_identifiers={"alarm_code": ["E021"]},
        clarification_question="当前只找到 E021，是否要查询 E021？",
        used_structured_filter=True,
        filter_fallback=True,
    )

    decision = _planner().plan(_context(history=history, observation=observation))

    assert decision.action == QueryAction.ASK_CLARIFICATION
    assert decision.reason_code == PlannerReasonCode.IDENTIFIER_CONFIRMATION_REQUIRED


def test_identifier_not_found_asks_user_to_check_code():
    history = _history(QueryAction.LOCAL_SEARCH)
    observation = RetrievalObservation(
        action=QueryAction.LOCAL_SEARCH,
        status=ObservationStatus.EMPTY,
        requested_identifiers={"alarm_code": ["E020"]},
        identifier_resolution_status=IdentifierResolutionStatus.NOT_FOUND,
        clarification_question="没有找到 E020，请核对设备屏幕上的报警码。",
    )

    decision = _planner().plan(_context(history=history, observation=observation))

    assert decision.action == QueryAction.ASK_CLARIFICATION
    assert decision.reason_code == PlannerReasonCode.IDENTIFIER_NOT_FOUND


def test_other_evidence_ambiguity_asks_for_clarification():
    history = _history(QueryAction.LOCAL_SEARCH)
    observation = _observation(QueryAction.LOCAL_SEARCH, evidence_ambiguous=True)

    decision = _planner().plan(_context(history=history, observation=observation))

    assert decision.action == QueryAction.ASK_CLARIFICATION
    assert decision.reason_code == PlannerReasonCode.EVIDENCE_AMBIGUOUS


@pytest.mark.parametrize(
    "identifier_status",
    [
        IdentifierResolutionStatus.NOT_APPLICABLE,
        IdentifierResolutionStatus.EXACT_MATCH,
        IdentifierResolutionStatus.FALLBACK_EXACT_MATCH,
    ],
)
def test_sufficient_local_evidence_answers_only_for_safe_identifier_status(identifier_status):
    history = _history(QueryAction.LOCAL_SEARCH)
    if identifier_status == IdentifierResolutionStatus.NOT_APPLICABLE:
        observation = _observation(QueryAction.LOCAL_SEARCH, rerank_score=0.8)
    else:
        observation = RetrievalObservation(
            action=QueryAction.LOCAL_SEARCH,
            status=ObservationStatus.SUCCESS,
            candidate_count=1,
            reranked_count=1,
            top_rerank_score=0.8,
            requested_identifiers={"alarm_code": ["E020"]},
            matched_identifiers={"alarm_code": ["E020"]},
            identifier_resolution_status=identifier_status,
            used_structured_filter=True,
            filter_fallback=(identifier_status == IdentifierResolutionStatus.FALLBACK_EXACT_MATCH),
        )

    decision = _planner().plan(_context(history=history, observation=observation))

    assert decision.action == QueryAction.ANSWER
    assert decision.reason_code == PlannerReasonCode.LOCAL_EVIDENCE_SUFFICIENT


def test_threshold_boundary_is_inclusive_and_bound_to_config():
    history = _history(QueryAction.LOCAL_SEARCH)
    observation = _observation(QueryAction.LOCAL_SEARCH, rerank_score=0.75)

    decision = _planner(threshold=0.75).plan(
        _context(history=history, observation=observation)
    )

    assert decision.action == QueryAction.ANSWER


def test_low_score_local_candidates_trigger_hyde_once():
    history = _history(QueryAction.LOCAL_SEARCH)
    observation = _observation(QueryAction.LOCAL_SEARCH, rerank_score=0.74)

    decision = _planner(threshold=0.75).plan(
        _context(history=history, observation=observation)
    )

    assert decision.action == QueryAction.HYDE_SEARCH
    assert decision.reason_code == PlannerReasonCode.LOCAL_LOW_SCORE


def test_empty_local_result_skips_hyde_and_falls_back_to_web():
    history = _history(QueryAction.LOCAL_SEARCH)
    observation = _observation(
        QueryAction.LOCAL_SEARCH,
        status=ObservationStatus.EMPTY,
        candidate_count=0,
        rerank_score=None,
    )

    decision = _planner().plan(_context(history=history, observation=observation))

    assert decision.action == QueryAction.WEB_SEARCH
    assert decision.reason_code == PlannerReasonCode.LOCAL_EMPTY


def test_sufficient_hyde_evidence_answers():
    history = _history(QueryAction.LOCAL_SEARCH, QueryAction.HYDE_SEARCH)
    observation = _observation(QueryAction.HYDE_SEARCH, rerank_score=0.82)

    decision = _planner().plan(_context(history=history, observation=observation))

    assert decision.action == QueryAction.ANSWER
    assert decision.reason_code == PlannerReasonCode.HYDE_EVIDENCE_SUFFICIENT


def test_insufficient_hyde_evidence_falls_back_to_web():
    history = _history(QueryAction.LOCAL_SEARCH, QueryAction.HYDE_SEARCH)
    observation = _observation(QueryAction.HYDE_SEARCH, rerank_score=0.4)

    decision = _planner().plan(_context(history=history, observation=observation))

    assert decision.action == QueryAction.WEB_SEARCH
    assert decision.reason_code == PlannerReasonCode.HYDE_STILL_INSUFFICIENT


def test_direct_realtime_web_observation_can_answer_without_local_search():
    history = _history(QueryAction.WEB_SEARCH)
    observation = _observation(QueryAction.WEB_SEARCH, rerank_score=0.88)
    context = _context(
        query="HAK180 厂家今天有没有发布最新召回公告？",
        history=history,
        observation=observation,
    )

    decision = _planner().plan(context)

    assert decision.action == QueryAction.ANSWER
    assert decision.reason_code == PlannerReasonCode.WEB_EVIDENCE_AVAILABLE


@pytest.mark.parametrize(
    "observation",
    [
        _observation(
            QueryAction.WEB_SEARCH,
            status=ObservationStatus.EMPTY,
            candidate_count=0,
            rerank_score=None,
        ),
        _observation(QueryAction.WEB_SEARCH, rerank_score=0.2),
        _observation(
            QueryAction.WEB_SEARCH,
            status=ObservationStatus.FAILED,
            candidate_count=0,
            rerank_score=None,
            error_code="web_timeout",
        ),
    ],
)
def test_web_empty_low_score_or_failed_safely_refuses(observation):
    failed_last = observation.status == ObservationStatus.FAILED
    history = _history(QueryAction.WEB_SEARCH, failed_last=failed_last)
    context = _context(
        query="HAK180 厂家今天有没有发布最新召回公告？",
        history=history,
        observation=observation,
    )

    decision = _planner().plan(context)

    assert decision.action == QueryAction.REFUSE
    assert decision.reason_code == PlannerReasonCode.WEB_EMPTY_OR_FAILED


def test_local_execution_failure_can_fall_back_to_web():
    history = _history(QueryAction.LOCAL_SEARCH, failed_last=True)
    observation = _observation(
        QueryAction.LOCAL_SEARCH,
        status=ObservationStatus.FAILED,
        candidate_count=0,
        rerank_score=None,
        error_code="milvus_timeout",
    )

    decision = _planner().plan(_context(history=history, observation=observation))

    assert decision.action == QueryAction.WEB_SEARCH
    assert decision.reason_code == PlannerReasonCode.ACTION_EXECUTION_ERROR


def test_web_disabled_prevents_realtime_or_fallback_web():
    realtime_decision = _planner().plan(
        _context(
            query="HAK180 厂家今天有没有发布最新公告？",
            web_search_allowed=False,
        )
    )
    local_history = _history(QueryAction.LOCAL_SEARCH)
    local_empty = _observation(
        QueryAction.LOCAL_SEARCH,
        status=ObservationStatus.EMPTY,
        candidate_count=0,
        rerank_score=None,
    )
    fallback_decision = _planner().plan(
        _context(
            history=local_history,
            observation=local_empty,
            web_search_allowed=False,
        )
    )

    assert realtime_decision.action == QueryAction.REFUSE
    assert realtime_decision.reason_code == PlannerReasonCode.SAFE_GUARD_TRIGGERED
    assert fallback_decision.action == QueryAction.REFUSE
    assert fallback_decision.reason_code == PlannerReasonCode.LOCAL_EMPTY


def test_safety_guard_and_max_steps_always_refuse():
    safety_decision = _planner().plan(_context(safe_guard_triggered=True))
    history = _history(
        QueryAction.LOCAL_SEARCH,
        QueryAction.HYDE_SEARCH,
        QueryAction.WEB_SEARCH,
    )
    observation = _observation(QueryAction.WEB_SEARCH, rerank_score=0.99)
    max_step_decision = _planner().plan(
        _context(
            history=history,
            observation=observation,
            max_steps=3,
        )
    )

    assert safety_decision.action == QueryAction.REFUSE
    assert max_step_decision.action == QueryAction.REFUSE
    assert max_step_decision.reason_code == PlannerReasonCode.SAFE_GUARD_TRIGGERED


@pytest.mark.parametrize(
    "history",
    [
        _history(QueryAction.LOCAL_SEARCH, QueryAction.LOCAL_SEARCH),
        _history(QueryAction.LOCAL_SEARCH, QueryAction.WEB_SEARCH, QueryAction.HYDE_SEARCH),
    ],
)
def test_repeated_or_backward_retrieval_transition_safely_refuses(history):
    observation = _observation(history[-1].decision.action, rerank_score=0.9)

    decision = _planner().plan(_context(history=history, observation=observation))

    assert decision.action == QueryAction.REFUSE
    assert decision.reason_code == PlannerReasonCode.SAFE_GUARD_TRIGGERED


def test_missing_target_action_in_allowlist_uses_safe_refuse():
    context = _context(allowed_actions=[QueryAction.WEB_SEARCH, QueryAction.REFUSE])

    decision = _planner().plan(context)

    assert decision.action == QueryAction.REFUSE
    assert decision.reason_code == PlannerReasonCode.SAFE_GUARD_TRIGGERED


def test_same_context_is_deterministic_and_not_mutated():
    planner = _planner()
    context = _context()
    before = copy.deepcopy(context.model_dump(mode="json"))

    first = planner.plan(context)
    second = planner.plan(context)

    assert first == second
    assert context.model_dump(mode="json") == before


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rerank_evidence_threshold": -0.1, "retrieval_config_version": "v1"},
        {"rerank_evidence_threshold": 1.1, "retrieval_config_version": "v1"},
        {"rerank_evidence_threshold": True, "retrieval_config_version": "v1"},
        {"rerank_evidence_threshold": 0.7, "retrieval_config_version": "   "},
    ],
)
def test_rule_config_rejects_unversioned_or_invalid_threshold(kwargs):
    with pytest.raises(ValueError):
        RuleBasedPlannerConfig(**kwargs)
