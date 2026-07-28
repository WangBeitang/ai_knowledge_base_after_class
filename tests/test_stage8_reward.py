import json

from app.rag.evaluation import REWARD_VERSION, RewardConfig, RewardWeights, score_trajectory
from app.rag.evaluation.case_schema import EnvironmentSnapshot, PlannerEvalCase
from app.rag.evaluation.offline_environment import OfflineRagEnvironment, OfflineState
from app.rag.query.config import RETRIEVAL_CONFIG_VERSION
from app.rag.query.contracts import (
    Citation,
    EvidenceSourceType,
    PlannerDecision,
    QueryAction,
    RetrievalCandidate,
    RetrievalChannel,
)


class FakeRewardActionProvider:
    """Reward 单元测试用 ActionProvider，不连接真实 Milvus/Web。"""

    def local_search(self, state: OfflineState, decision: PlannerDecision) -> list[RetrievalCandidate]:
        return [_candidate(12345, retrieval_channel=RetrievalChannel.ORIGINAL)]

    def hyde_search(self, state: OfflineState, decision: PlannerDecision) -> list[RetrievalCandidate]:
        # 故意返回和 local_search 相同的 chunk，用来验证重复命中不会刷高检索分，
        # 但 action_path 里的 hyde_search 仍会被 behavior/cost 评分捕获。
        return [_candidate(12345, retrieval_channel=RetrievalChannel.HYDE)]

    def web_search(self, state: OfflineState, decision: PlannerDecision) -> list[RetrievalCandidate]:
        return [
            RetrievalCandidate(
                title="HAK180 外部网页候选",
                content="网页摘要提到了 HAK180，但不是本地手册证据。",
                source_type=EvidenceSourceType.WEB,
                retrieval_channels=[RetrievalChannel.WEB],
                retrieval_rank=1,
                retrieval_score=0.80,
                rerank_score=0.80,
                url="https://example.com/hak180",
            )
        ]


def _snapshot() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        snapshot_id="stage8-reward-test-v1",
        created_at="2026-07-19T00:00:00+00:00",
        created_by="pytest",
        dataset_ids=["dataset_default_equipment_ops"],
        test_user_ids=["eval_demo_user"],
        documents=[
            {
                "document_id": "doc_hak180_manual",
                "dataset_id": "dataset_default_equipment_ops",
                "index_version": 3,
                "visibility": "public",
                "chunk_count": 1,
            }
        ],
        enabled_chunks={"doc_hak180_manual": [12345]},
        disabled_chunks=[],
        retrieval_config_version=RETRIEVAL_CONFIG_VERSION,
        retrieval_config_snapshot={
            "retrieval_mode": "dense_learned_sparse_bm25",
            "per_channel_topk": 5,
            "fusion_topk": 5,
            "rerank_min_topk": 2,
            "rerank_max_topk": 5,
            "rrf_k": 60,
            "evidence_threshold": 0.75,
            "web_fallback_enabled": True,
        },
        policy_version="rule-v1",
    )


def _env() -> OfflineRagEnvironment:
    return OfflineRagEnvironment(
        snapshot=_snapshot(),
        action_provider=FakeRewardActionProvider(),
        run_id_prefix="pytest_reward",
    )


def _answer_case() -> PlannerEvalCase:
    return PlannerEvalCase(
        case_id="reward-dev-alarm-e020",
        case_group="core",
        split="dev",
        leakage_group_id="reward-hak180-e020",
        query="HAK180 的 E020 是什么故障？",
        dataset_ids=["dataset_default_equipment_ops"],
        owner_user_id="eval_demo_user",
        tenant_id="tenant_default",
        privacy_scope="public_demo",
        source_document_ids=["doc_hak180_manual"],
        source_index_versions={"doc_hak180_manual": 3},
        expected_subject_ids=["subject_hak180"],
        expected_subject_names=["HAK180"],
        expected_chunks=[
            {
                "document_id": "doc_hak180_manual",
                "chunk_id": 12345,
                "index_version": 3,
                "relevance": "required",
                "answer_point_ids": ["offline_answer"],
            }
        ],
        expected_answer_points=["离线评测基于 HAK180 E020 证据 12345 形成 answer 终态"],
        expected_behavior={
            "should_answer": True,
            "should_refuse": False,
            "should_ask_clarification": False,
            "should_call_web": False,
            "forbidden_actions": ["web_search"],
        },
        acceptable_action_paths=[["local_search", "answer"]],
        expected_identifiers={"alarm_code": ["E020"]},
        label_source="manual",
        human_review_status="reviewed",
    )


def _refusal_case() -> PlannerEvalCase:
    return PlannerEvalCase(
        case_id="reward-dev-refuse-no-evidence",
        case_group="refusal",
        split="dev",
        leakage_group_id="reward-refuse-no-evidence",
        query="编一个 HAK180 手册里没有的维修参数。",
        dataset_ids=["dataset_default_equipment_ops"],
        owner_user_id="eval_demo_user",
        tenant_id="tenant_default",
        privacy_scope="public_demo",
        expected_chunks=[],
        expected_answer_points=[],
        expected_behavior={
            "should_answer": False,
            "should_refuse": True,
            "should_ask_clarification": False,
            "should_call_web": False,
            "forbidden_actions": ["web_search", "hyde_search"],
        },
        acceptable_action_paths=[["refuse"]],
        expected_identifiers={},
        label_source="manual",
        human_review_status="reviewed",
    )


def _clarification_case() -> PlannerEvalCase:
    return PlannerEvalCase(
        case_id="reward-dev-ask-missing-subject",
        case_group="clarification",
        split="dev",
        leakage_group_id="reward-ask-missing-subject",
        query="这个报警应该怎么处理？",
        dataset_ids=["dataset_default_equipment_ops"],
        owner_user_id="eval_demo_user",
        tenant_id="tenant_default",
        privacy_scope="public_demo",
        expected_chunks=[],
        expected_answer_points=[],
        expected_behavior={
            "should_answer": False,
            "should_refuse": False,
            "should_ask_clarification": True,
            "should_call_web": False,
            "forbidden_actions": ["web_search", "hyde_search"],
        },
        acceptable_action_paths=[["ask_clarification"]],
        expected_identifiers={},
        label_source="manual",
        human_review_status="reviewed",
    )


def _web_answer_case() -> PlannerEvalCase:
    return PlannerEvalCase(
        case_id="reward-dev-web-hak180",
        case_group="realtime",
        split="dev",
        leakage_group_id="reward-web-hak180",
        query="请根据外部网页说明 HAK180 的最新信息。",
        dataset_ids=["dataset_default_equipment_ops"],
        owner_user_id="eval_demo_user",
        tenant_id="tenant_default",
        privacy_scope="public_demo",
        expected_chunks=[],
        expected_web_evidence=[
            {
                "source_id": "example-hak180",
                "publisher": "Example",
                "source_title": "HAK180 外部网页候选",
                "url": "https://example.com/hak180",
                "captured_at": "2026-07-28T07:33:24+00:00",
                "response_sha256": "a" * 64,
                "evidence_content_sha256": "b" * 64,
                "fact_ids": ["hak180_web_fact"],
                "answer_point_ids": ["hak180_web_point"],
            }
        ],
        expected_answer_points=["HAK180 外部网页候选"],
        expected_behavior={
            "should_answer": True,
            "should_refuse": False,
            "should_ask_clarification": False,
            "should_call_web": True,
            "web_required_reason": "需要外部网页证据",
            "forbidden_actions": [],
        },
        acceptable_action_paths=[["web_search", "answer"]],
        expected_identifiers={},
        label_source="manual",
        gold_origin="heldout_gold",
        human_review_status="reviewed",
    )


def _candidate(chunk_id: int, *, retrieval_channel: RetrievalChannel) -> RetrievalCandidate:
    return RetrievalCandidate(
        document_id="doc_hak180_manual",
        chunk_id=chunk_id,
        dataset_id="dataset_default_equipment_ops",
        index_version=3,
        chunk_index=1,
        enabled=True,
        title=f"HAK180 E020 证据 {chunk_id}",
        source_title="HAK180 操作手册",
        content="HAK180 报警码 E020 表示温控异常，需要停机检查温度传感器。",
        equipment_model="HAK180",
        alarm_code="E020",
        source_type=EvidenceSourceType.LOCAL,
        retrieval_channels=[retrieval_channel],
        retrieval_rank=1,
        retrieval_score=0.92,
        rerank_score=0.92,
    )


def test_stage85_reward_v1_1_default_weights_prioritize_planner_route():
    weights = RewardWeights()
    config = RewardConfig()

    assert REWARD_VERSION == "reward-v1.1"
    assert config.reward_version == "reward-v1.1"
    assert weights.as_dict() == {
        "format": 0.15,
        "retrieval": 0.12,
        "citation": 0.08,
        "answer": 0.15,
        "behavior": 0.35,
        "cost": 0.15,
    }
    assert weights.behavior + weights.cost == 0.50
    assert weights.retrieval + weights.citation == 0.20
    assert weights.behavior + weights.cost > weights.retrieval + weights.citation


def test_stage8_reward_scores_valid_answer_path_and_serializes_json():
    case = _answer_case()
    trajectory = _env().run_action_path(
        case,
        [QueryAction.LOCAL_SEARCH, QueryAction.ANSWER],
        run_id="reward_valid_answer",
    )

    reward = score_trajectory(case, trajectory)

    assert reward.reward_version == "reward-v1.1"
    assert reward.capped_by is None
    assert reward.format_valid is True
    assert set(reward.components) == {
        "format",
        "retrieval",
        "citation",
        "answer",
        "behavior",
        "cost",
    }
    assert reward.components["retrieval"].details["recall_at_k"] == 1.0
    assert reward.components["citation"].details["citation_hit_rate"] == 1.0
    assert reward.components["answer"].details["answer_point_coverage"] == 1.0
    assert reward.components["behavior"].weight > reward.components["retrieval"].weight
    assert reward.components["cost"].weight > reward.components["citation"].weight
    for component in reward.components.values():
        assert component.details

    json.dumps(reward.to_json_dict(), ensure_ascii=False)


def test_stage9_reward_scores_frozen_web_url_as_retrieval_and_citation_evidence():
    case = _web_answer_case()
    trajectory = _env().run_action_path(
        case,
        [QueryAction.WEB_SEARCH, QueryAction.ANSWER],
        run_id="reward_valid_web_answer",
    )

    reward = score_trajectory(case, trajectory)

    assert trajectory.status.value == "completed"
    assert reward.capped_by is None
    assert reward.components["retrieval"].details["recall_at_k"] == 1.0
    assert reward.components["retrieval"].details["expected_web_evidence_count"] == 1
    assert reward.components["citation"].details["citation_hit_rate"] == 1.0
    assert reward.components["citation"].details["invalid_citation_count"] == 0
    assert reward.components["answer"].details["answer_point_coverage"] == 1.0


def test_stage9_reward_rejects_non_http_web_citation_identity():
    case = _web_answer_case()
    trajectory = _env().run_action_path(
        case,
        [QueryAction.WEB_SEARCH, QueryAction.ANSWER],
        run_id="reward_invalid_web_citation",
    ).model_copy(
        update={
            "citations": [
                Citation(
                    title="无效 Web 引用",
                    source="not-a-web-url",
                    score=0.9,
                    source_type=EvidenceSourceType.WEB,
                )
            ]
        }
    )

    reward = score_trajectory(case, trajectory)

    assert reward.components["citation"].details["citation_hit_rate"] == 0.0
    assert reward.components["citation"].details["invalid_citation_count"] == 1
    assert reward.components["citation"].score == 0.0


def test_stage8_reward_caps_invalid_action_path_total_score():
    case = _answer_case()
    trajectory = _env().run_action_path(
        case,
        [QueryAction.HYDE_SEARCH],
        run_id="reward_invalid_first_action",
    )

    reward = score_trajectory(case, trajectory, RewardConfig(invalid_format_cap=0.20))

    assert reward.capped_by == "invalid_format"
    assert reward.total_reward <= 0.20
    assert reward.components["format"].score < 1.0


def test_stage8_reward_does_not_penalize_refusal_for_missing_answer_points():
    case = _refusal_case()
    trajectory = _env().run_action_path(
        case,
        [QueryAction.REFUSE],
        run_id="reward_refusal",
    )

    reward = score_trajectory(case, trajectory)

    assert reward.components["answer"].score == 1.0
    assert reward.components["answer"].details["not_applicable_answer_points"] is True
    assert reward.components["retrieval"].score == 1.0
    assert reward.components["retrieval"].details["not_applicable"] is True


def test_stage85_reward_does_not_penalize_clarification_for_missing_expected_chunks():
    case = _clarification_case()
    trajectory = _env().run_action_path(
        case,
        [QueryAction.ASK_CLARIFICATION],
        run_id="reward_clarification",
    )

    reward = score_trajectory(case, trajectory)

    assert reward.components["answer"].score == 1.0
    assert reward.components["answer"].details["not_applicable_answer_points"] is True
    assert reward.components["retrieval"].score == 1.0
    assert reward.components["retrieval"].details["not_applicable"] is True
    assert reward.components["citation"].score == 1.0
    assert reward.components["citation"].details["not_applicable"] is True


def test_stage8_reward_penalizes_unnecessary_hyde_and_web_actions():
    case = _answer_case()
    env = _env()
    direct = env.run_action_path(
        case,
        [QueryAction.LOCAL_SEARCH, QueryAction.ANSWER],
        run_id="reward_direct_path",
    )
    hyde = env.run_action_path(
        case,
        [QueryAction.LOCAL_SEARCH, QueryAction.HYDE_SEARCH, QueryAction.ANSWER],
        run_id="reward_hyde_path",
    )

    direct_reward = score_trajectory(case, direct)
    hyde_reward = score_trajectory(case, hyde)

    assert hyde_reward.components["behavior"].score < direct_reward.components["behavior"].score
    assert hyde_reward.components["cost"].score < direct_reward.components["cost"].score
    assert any("HyDE" in reason for reason in hyde_reward.components["cost"].reasons)

    web_like = direct.model_copy(update={
        "action_path": [
            QueryAction.LOCAL_SEARCH,
            QueryAction.WEB_SEARCH,
            QueryAction.ANSWER,
        ]
    })
    web_reward = score_trajectory(case, web_like)

    assert web_reward.components["behavior"].score < direct_reward.components["behavior"].score
    assert web_reward.components["cost"].score < direct_reward.components["cost"].score
    assert any("Web" in reason for reason in web_reward.components["cost"].reasons)


def test_stage85_reward_v1_1_route_components_move_total_more_than_retrieval_components():
    case = _answer_case()
    env = _env()
    direct = env.run_action_path(
        case,
        [QueryAction.LOCAL_SEARCH, QueryAction.ANSWER],
        run_id="reward_route_direct_path",
    )
    unnecessary_route = direct.model_copy(update={
        "action_path": [
            QueryAction.LOCAL_SEARCH,
            QueryAction.WEB_SEARCH,
            QueryAction.ANSWER,
        ]
    })
    direct_reward = score_trajectory(case, direct)
    route_reward = score_trajectory(case, unnecessary_route)

    route_weight_delta = (
        direct_reward.components["behavior"].weighted_score
        + direct_reward.components["cost"].weighted_score
        - route_reward.components["behavior"].weighted_score
        - route_reward.components["cost"].weighted_score
    )
    retrieval_weight_delta = (
        direct_reward.components["retrieval"].weighted_score
        + direct_reward.components["citation"].weighted_score
        - route_reward.components["retrieval"].weighted_score
        - route_reward.components["citation"].weighted_score
    )

    assert route_weight_delta > 0
    assert abs(retrieval_weight_delta) < 1e-12
    assert route_weight_delta > retrieval_weight_delta
