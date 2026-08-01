from collections import Counter
from types import SimpleNamespace

from app.rag.query.contracts import (
    EvidenceSourceType,
    EvidenceSummary,
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

from evaluation.stage9.sft_v2.build_sft_v2_candidates import (
    CandidateSeed,
    CandidateQuestionProfile,
    ClaimEvidenceBinding,
    FORMAL_GAP_COUNT,
    NEW_TRAJECTORY_COUNT,
    RESERVE_COUNT,
    ROUTE_SPECS,
    SourceEvidence,
    SourceConditionedPlanner,
    _hyde_target_improvement,
    _post_retrieval_clarification_supported,
    _post_retrieval_refusal_supported,
    _web_fact_complete,
    _web_answer_coverage,
    bind_observed_source_evidence,
    assess_route_admission,
    build_allocations,
    route_quota,
    validate_allocations,
    validate_pre_provider_profiles,
)
from evaluation.stage9.sft_v2.repair_sft_v2_candidates import (
    audit_round3_lock,
    first_repair_partition,
)


def test_sft_v2_route_quota_matches_section_5() -> None:
    assert route_quota() == {
        "ask_clarification": 7,
        "refuse": 3,
        "local_search -> answer": 12,
        "local_search -> ask_clarification": 9,
        "local_search -> refuse": 5,
        "web_search -> answer": 16,
        "web_search -> ask_clarification": 6,
        "web_search -> refuse": 4,
        "local_search -> hyde_search -> answer": 9,
        "local_search -> hyde_search -> ask_clarification": 6,
        "local_search -> hyde_search -> refuse": 4,
        "local_search -> web_search -> answer": 15,
        "local_search -> web_search -> ask_clarification": 5,
        "local_search -> web_search -> refuse": 3,
        "local_search -> hyde_search -> web_search -> answer": 12,
        "local_search -> hyde_search -> web_search -> ask_clarification": 5,
        "local_search -> hyde_search -> web_search -> refuse": 4,
    }
    assert sum(count for _, count, _ in ROUTE_SPECS) == NEW_TRAJECTORY_COUNT


def test_round3_repair_only_resamples_first_run_failures() -> None:
    lock = audit_round3_lock()
    reuse_ids, resample_ids = first_repair_partition(set(lock["rejected_ids"]))

    assert len(lock["approved_ids"]) == 38
    assert len(reuse_ids) == 44
    assert len(resample_ids) == 43
    assert reuse_ids.isdisjoint(resample_ids)


def test_sft_v2_allocates_before_question_generation_and_passes_diversity_gate() -> None:
    allocations = build_allocations()
    report = validate_allocations(allocations)

    assert len(allocations) == NEW_TRAJECTORY_COUNT
    assert sum(item["reserve"] for item in allocations) == RESERVE_COUNT
    assert len(allocations) - sum(item["reserve"] for item in allocations) == FORMAL_GAP_COUNT
    assert Counter(item["route_name"] for item in allocations) == Counter(route_quota())
    assert report["reserve_count"] == RESERVE_COUNT
    assert len({item["candidate_id"] for item in allocations}) == NEW_TRAJECTORY_COUNT


def test_sft_v2_small_routes_use_distinct_device_families() -> None:
    allocations = build_allocations()
    for route_name, count in route_quota().items():
        if count > 4:
            continue
        selected = [item for item in allocations if item["route_name"] == route_name]
        families = []
        for item in selected:
            source = item["local_source"] or item["web_source"]
            families.append(source.device_family)
        assert len(families) == len(set(families))


def test_sft_v2_question_families_are_frozen_before_drafting() -> None:
    allocations = build_allocations()
    terminal_labels = {"answer", "ask_clarification", "refuse"}
    for route_name, count in route_quota().items():
        selected = [item for item in allocations if item["route_name"] == route_name]
        family_counts = Counter(item["question_family"] for item in selected)
        assert all(family not in terminal_labels for family in family_counts)
        if count >= 5:
            assert max(family_counts.values()) <= int(count * 0.4)
        if count >= 8:
            assert len(family_counts) >= 4
        elif count >= 5:
            assert len(family_counts) >= 3
        else:
            assert len(family_counts) == count


def _local_evidence(*, fact_text: str) -> SourceEvidence:
    return SourceEvidence(
        source_type="local",
        source_id="doc-test:101:1",
        publisher="test",
        source_title="测试手册",
        document_id="doc-test",
        chunk_id="101",
        index_version=1,
        evidence_content_sha256="a" * 64,
        fact_text=fact_text,
    )


def _seed(
        *,
        route: list[QueryAction],
        query: str,
        fact_text: str,
        answer_points: list[str] | None = None,
        profile: CandidateQuestionProfile | None = None,
        candidate_id: str = "sft-v2-new-test",
) -> CandidateSeed:
    return CandidateSeed(
        candidate_id=candidate_id,
        sampling_target_route=route,
        reserve=False,
        device_family="test",
        question_family="test",
        missing_or_safety_trigger="检索事实触发",
        source_evidence=[_local_evidence(fact_text=fact_text)],
        retrieval_subject_id="subject-test",
        retrieval_subject_name="测试设备",
        query=query,
        answer_points=answer_points or [],
        question_profile=profile or CandidateQuestionProfile(),
    )


def _record(
        action: QueryAction,
        *,
        candidates: list[dict],
        identifier_status: str = "not_applicable",
) -> SimpleNamespace:
    return SimpleNamespace(
        action=action,
        candidates=candidates,
        observation={"identifier_resolution_status": identifier_status},
    )


def _candidate(*, rank: int, score: float) -> dict:
    return {
        "source_type": "local",
        "document_id": "doc-test",
        "chunk_id": "101",
        "index_version": 1,
        "retrieval_rank": rank,
        "retrieval_score": score,
        "rerank_score": None,
    }


def test_hyde_gate_requires_bound_target_to_improve() -> None:
    evidence = _local_evidence(fact_text="不同模式分别适用不同的安全条件。")
    passed, detail = _hyde_target_improvement(
        [_candidate(rank=1, score=0.05)],
        [_candidate(rank=3, score=0.04)],
        [evidence],
    )
    assert passed is False
    assert detail["targets"][0]["improved"] is False

    passed, detail = _hyde_target_improvement(
        [_candidate(rank=4, score=0.03)],
        [_candidate(rank=2, score=0.04)],
        [evidence],
    )
    assert passed is True
    assert detail["targets"][0]["improved"] is True


def test_post_retrieval_clarification_rejects_predeclared_missing_field() -> None:
    seed = _seed(
        route=[QueryAction.LOCAL_SEARCH, QueryAction.ASK_CLARIFICATION],
        query="我不知道当前模式，请告诉我该用哪个参数。",
        fact_text="自动模式与手动模式分别使用不同参数，具体选择取决于当前配置。",
    )
    passed, detail = _post_retrieval_clarification_supported(
        seed,
        [_record(QueryAction.LOCAL_SEARCH, candidates=[_candidate(rank=1, score=0.04)])],
    )
    assert passed is False
    assert detail["reason"] == "missing_field_predeclared_in_query"


def test_post_retrieval_clarification_accepts_new_branching_evidence() -> None:
    seed = _seed(
        route=[QueryAction.LOCAL_SEARCH, QueryAction.HYDE_SEARCH, QueryAction.ASK_CLARIFICATION],
        query="这项调整应当使用哪个参数？",
        fact_text="自动模式与手动模式分别使用不同参数，具体选择取决于当前配置。",
    )
    passed, detail = _post_retrieval_clarification_supported(
        seed,
        [
            _record(QueryAction.LOCAL_SEARCH, candidates=[]),
            _record(QueryAction.HYDE_SEARCH, candidates=[_candidate(rank=1, score=0.04)]),
        ],
    )
    assert passed is True
    assert detail["evidence_newly_ambiguous"] is True


def test_post_retrieval_refusal_requires_new_safety_fact_and_hidden_intent() -> None:
    seed = _seed(
        route=[QueryAction.LOCAL_SEARCH, QueryAction.REFUSE],
        query="设备运行中需要进入该区域处理物料，应采用什么作业安排？",
        fact_text="维护期间必须隔离全部能量，并使用防护装置避免人员遭受重伤或死亡。",
    )
    passed, detail = _post_retrieval_refusal_supported(
        seed,
        [_record(QueryAction.LOCAL_SEARCH, candidates=[_candidate(rank=1, score=0.04)])],
    )
    assert passed is True
    assert detail["safety_grounded"] is True

    unsafe_seed = seed.model_copy(update={"query": "怎么绕过防护进入该区域？"})
    passed, detail = _post_retrieval_refusal_supported(
        unsafe_seed,
        [_record(QueryAction.LOCAL_SEARCH, candidates=[_candidate(rank=1, score=0.04)])],
    )
    assert passed is False
    assert detail["reason"] == "unsafe_intent_explicit_before_retrieval"


def test_web_fact_gate_rejects_boilerplate_and_truncation() -> None:
    good = SourceEvidence(
        source_type="web",
        source_id="official-test",
        publisher="test",
        source_title="Official advisory",
        url="https://example.com/advisory",
        captured_at="2026-08-01T00:00:00+00:00",
        response_sha256="b" * 64,
        evidence_content_sha256="c" * 64,
        fact_text=(
            "The current advisory identifies the affected firmware versions and states that "
            "operators must apply the vendor update before enabling remote access."
        ),
    )
    assert _web_fact_complete(good)[0] is True

    bad = good.model_copy(update={
        "fact_text": "Official website of the United States government privacy policy and"
    })
    passed, detail = _web_fact_complete(bad)
    assert passed is False
    assert {"fact_too_short", "page_boilerplate", "truncated_ending"}.issubset(detail["reasons"])


def test_web_answer_coverage_uses_atomic_cross_language_span_binding() -> None:
    fact = (
        "The NIST Cybersecurity Framework 2.0 Community Profile is voluntary and provides "
        "a risk-based approach for managing cybersecurity activities."
    )
    evidence = SourceEvidence(
        source_type="web",
        source_id="nist-test",
        publisher="NIST",
        source_title="NIST profile",
        url="https://example.com/nist",
        captured_at="2026-08-01T00:00:00+00:00",
        response_sha256="b" * 64,
        evidence_content_sha256="c" * 64,
        fact_text=fact,
    )
    claim = "NIST 网络安全框架 2.0 社区配置文件是自愿性的。"
    seed = _seed(
        route=[QueryAction.WEB_SEARCH, QueryAction.ANSWER],
        query="该配置文件是否强制实施？",
        fact_text="本地作者背景。",
        answer_points=[claim],
        profile=CandidateQuestionProfile(claim_evidence_bindings=[ClaimEvidenceBinding(
            claim=claim,
            evidence_span=fact,
            relation="entailed",
        )]),
    )
    assert _web_answer_coverage(seed, evidence)[0] is True

    unsupported = seed.model_copy(update={
        "question_profile": CandidateQuestionProfile(claim_evidence_bindings=[
            ClaimEvidenceBinding(
                claim=claim,
                evidence_span=fact,
                relation="partially_entailed",
            )
        ])
    })
    assert _web_answer_coverage(unsupported, evidence)[0] is False


def _observation(
        action: QueryAction,
        *,
        score: float,
        chunk_id: str = "101",
) -> RetrievalObservation:
    return RetrievalObservation(
        action=action,
        status=ObservationStatus.SUCCESS,
        candidate_count=1,
        reranked_count=1,
        top_rerank_score=score,
        identifier_resolution_status=IdentifierResolutionStatus.NOT_APPLICABLE,
        evidence_summaries=[EvidenceSummary(
            document_id="doc-test",
            chunk_id=chunk_id,
            title="测试手册",
            source_type=EvidenceSourceType.LOCAL,
            rerank_score=score,
            content_excerpt="不同模式对应不同参数；未经授权的安全参数修改可能导致危险。",
        )],
    )


def _context(
        *,
        observation: RetrievalObservation | None = None,
        previous_action: QueryAction | None = None,
) -> PlannerContext:
    history = []
    if previous_action is not None:
        history.append(PlannerHistoryItem(
            step=1,
            decision=PlannerDecision(
                action=previous_action,
                query="现场问题",
                reason_code=PlannerReasonCode.INITIAL_LOCAL_SEARCH,
            ),
            execution_status=PlannerExecutionStatus.COMPLETED,
        ))
    return PlannerContext(
        original_query="现场问题",
        current_query="现场问题",
        subject_resolution_status=SubjectResolutionStatus.CONFIRMED,
        subject_ids=["subject-test"],
        latest_observation=observation,
        action_history=history,
        web_search_allowed=True,
        planner_step=len(history),
        max_steps=4,
        allowed_actions=list(QueryAction),
    )


def test_dynamic_planner_does_not_play_hyde_sampling_target_when_local_is_strong() -> None:
    seed = _seed(
        route=[QueryAction.LOCAL_SEARCH, QueryAction.HYDE_SEARCH, QueryAction.ANSWER],
        query="设备出现轻微摆动时应检查什么？",
        fact_text="振动监测模块用于分析机械摆动。",
        answer_points=["检查振动监测模块。"],
        profile=CandidateQuestionProfile(
            terminology_gap=True,
            user_expression="轻微摆动",
            document_terms=["振动监测模块"],
        ),
    )
    planner = SourceConditionedPlanner(seed)

    assert planner.plan(_context()).action == QueryAction.LOCAL_SEARCH
    decision = planner.plan(_context(
        observation=_observation(QueryAction.LOCAL_SEARCH, score=0.80),
        previous_action=QueryAction.LOCAL_SEARCH,
    ))
    assert decision.action == QueryAction.ANSWER


def test_dynamic_planner_uses_hyde_for_real_weak_search_and_terminology_gap() -> None:
    seed = _seed(
        route=[QueryAction.LOCAL_SEARCH, QueryAction.ANSWER],
        query="设备像喘气一样一抖一抖，该查什么？",
        fact_text="速度环周期振荡需要检查增益参数。",
        answer_points=["检查速度环增益参数。"],
        profile=CandidateQuestionProfile(
            terminology_gap=True,
            user_expression="像喘气一样一抖一抖",
            document_terms=["速度环周期振荡"],
        ),
    )
    planner = SourceConditionedPlanner(seed)
    decision = planner.plan(_context(
        observation=_observation(QueryAction.LOCAL_SEARCH, score=0.10, chunk_id="999"),
        previous_action=QueryAction.LOCAL_SEARCH,
    ))
    assert decision.action == QueryAction.HYDE_SEARCH


def test_dynamic_planner_terminal_is_decided_by_observed_branch_or_boundary() -> None:
    branch_seed = _seed(
        route=[QueryAction.LOCAL_SEARCH, QueryAction.ANSWER],
        query="这项设置应使用哪个值？",
        fact_text="自动与手动模式使用不同参数。",
        profile=CandidateQuestionProfile(
            branch_selector="运行模式",
            branch_values=["自动", "手动"],
            branch_evidence_spans=["自动", "手动"],
            answer_changes_by_branch=True,
        ),
    )
    branch_decision = SourceConditionedPlanner(branch_seed).plan(_context(
        observation=_observation(QueryAction.LOCAL_SEARCH, score=0.80),
        previous_action=QueryAction.LOCAL_SEARCH,
    ))
    assert branch_decision.action == QueryAction.ASK_CLARIFICATION

    boundary_seed = _seed(
        route=[QueryAction.LOCAL_SEARCH, QueryAction.ANSWER],
        query="这个部件应怎样更换？",
        fact_text="该部件只能由获得授权的维修人员更换。",
        profile=CandidateQuestionProfile(
            post_search_boundary="该部件只能由获得授权的维修人员更换。",
            post_search_boundary_span="该部件只能由获得授权的维修人员更换。",
        ),
    )
    boundary_decision = SourceConditionedPlanner(boundary_seed).plan(_context(
        observation=_observation(QueryAction.LOCAL_SEARCH, score=0.80),
        previous_action=QueryAction.LOCAL_SEARCH,
    ))
    assert boundary_decision.action == QueryAction.REFUSE


def test_dynamic_planner_pre_search_terminal_overrides_sampling_target() -> None:
    seed = _seed(
        route=[QueryAction.LOCAL_SEARCH, QueryAction.HYDE_SEARCH, QueryAction.ANSWER],
        query="能源未隔离时怎样进入危险区？",
        fact_text="进入危险区前必须隔离全部能源。",
        profile=CandidateQuestionProfile(pre_search_terminal=QueryAction.REFUSE),
    )
    assert SourceConditionedPlanner(seed).plan(_context()).action == QueryAction.REFUSE


def test_pre_provider_profile_gate_does_not_treat_sampling_route_as_label() -> None:
    seed = _seed(
        route=[QueryAction.LOCAL_SEARCH, QueryAction.HYDE_SEARCH, QueryAction.ANSWER],
        query="设备抖动时应该检查什么？",
        fact_text="速度环周期振荡需要检查增益参数。",
        answer_points=["检查增益参数。"],
        profile=CandidateQuestionProfile(claim_evidence_bindings=[
            ClaimEvidenceBinding(
                claim="检查增益参数。",
                evidence_span="需要检查增益参数",
            ),
        ]),
    )

    assert validate_pre_provider_profiles([seed]) == {}


def test_formal_evidence_is_rebuilt_from_provider_candidate() -> None:
    seed = _seed(
        route=[QueryAction.LOCAL_SEARCH, QueryAction.ANSWER],
        query="这项设置是什么？",
        fact_text="作者预选事实。",
        answer_points=["答案。"],
    )
    record = SimpleNamespace(
        case_id=seed.candidate_id,
        record_id="provider-record-1",
        action=QueryAction.LOCAL_SEARCH,
        candidates=[{
            "source_type": "local",
            "document_id": "doc-test",
            "chunk_id": "101",
            "index_version": 1,
            "retrieval_rank": 2,
            "retrieval_score": 0.31,
            "rerank_score": 0.42,
            "content": "本次 Provider 实际返回的完整原文。",
        }],
    )

    bound = bind_observed_source_evidence([seed], [record])[0].source_evidence[0]

    assert bound.fact_text == "本次 Provider 实际返回的完整原文。"
    assert bound.retrieved_content == bound.fact_text
    assert bound.provider_record_id == "provider-record-1"
    assert bound.observed_action == QueryAction.LOCAL_SEARCH
    assert bound.retrieval_rank == 2
    assert bound.retrieval_score == 0.31
    assert bound.rerank_score == 0.42


def test_route_admission_uses_actual_routes_and_allows_target_swap() -> None:
    first = _seed(
        candidate_id="attempt-1",
        route=[QueryAction.LOCAL_SEARCH, QueryAction.ANSWER],
        query="问题一",
        fact_text="事实一。",
        answer_points=["答案一。"],
    )
    second = _seed(
        candidate_id="attempt-2",
        route=[QueryAction.LOCAL_SEARCH, QueryAction.REFUSE],
        query="问题二",
        fact_text="事实二。",
    )
    trajectories = [
        SimpleNamespace(
            case_id="attempt-1",
            action_path=[QueryAction.LOCAL_SEARCH, QueryAction.REFUSE],
        ),
        SimpleNamespace(
            case_id="attempt-2",
            action_path=[QueryAction.LOCAL_SEARCH, QueryAction.ANSWER],
        ),
    ]
    report = assess_route_admission(
        [first, second], trajectories,
        {"attempt-1": {"passed": True}, "attempt-2": {"passed": True}},
    )
    assert report["complete"] is True
    assert report["sampling_target_hit_count"] == 0
    assert report["sampling_target_miss_count"] == 2
