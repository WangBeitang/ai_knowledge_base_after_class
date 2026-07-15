from app.rag.evaluation.case_schema import EnvironmentSnapshot, PlannerEvalCase
from app.rag.evaluation.offline_environment import (
    OfflineRagEnvironment,
    OfflineState,
    OfflineTrajectoryStatus,
)
from app.rag.query.config import RETRIEVAL_CONFIG_VERSION
from app.rag.query.contracts import (
    EvidenceSourceType,
    PlannerDecision,
    PlannerReasonCode,
    QueryAction,
    RetrievalCandidate,
    RetrievalChannel,
)
from app.rag.query.planner import RuleBasedPlanner, RuleBasedPlannerConfig


class FakeOfflineActionProvider:
    def __init__(self, *, disabled_candidate: bool = False) -> None:
        self.disabled_candidate = disabled_candidate

    def local_search(self, state: OfflineState, decision: PlannerDecision) -> list[RetrievalCandidate]:
        return [_candidate(12345, retrieval_channel=RetrievalChannel.ORIGINAL)]

    def hyde_search(self, state: OfflineState, decision: PlannerDecision) -> list[RetrievalCandidate]:
        return [_candidate(67890, retrieval_channel=RetrievalChannel.HYDE)]

    def web_search(self, state: OfflineState, decision: PlannerDecision) -> list[RetrievalCandidate]:
        return [
            RetrievalCandidate(
                title="HAK180 最新召回公告",
                content="公开网页摘要显示 HAK180 有最新召回公告。",
                source_type=EvidenceSourceType.WEB,
                retrieval_channels=[RetrievalChannel.WEB],
                retrieval_rank=1,
                retrieval_score=0.91,
                rerank_score=0.91,
                url="https://example.com/hak180-recall",
            )
        ]


def _snapshot(*, disabled_chunk_id: int | None = None) -> EnvironmentSnapshot:
    disabled_chunks = []
    if disabled_chunk_id is not None:
        disabled_chunks.append({
            "document_id": "doc_hak180_manual",
            "chunk_id": disabled_chunk_id,
            "index_version": 3,
        })
    return EnvironmentSnapshot(
        snapshot_id="stage8-env-test-v1",
        created_at="2026-07-15T00:00:00+00:00",
        created_by="pytest",
        dataset_ids=["dataset_default_equipment_ops"],
        test_user_ids=["eval_demo_user"],
        documents=[
            {
                "document_id": "doc_hak180_manual",
                "dataset_id": "dataset_default_equipment_ops",
                "index_version": 3,
                "visibility": "public",
                "chunk_count": 2,
            }
        ],
        enabled_chunks={"doc_hak180_manual": [12345, 67890]},
        disabled_chunks=disabled_chunks,
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


def _case() -> PlannerEvalCase:
    return PlannerEvalCase(
        case_id="planner-dev-alarm-e020",
        case_group="core",
        split="dev",
        leakage_group_id="hak180-e020",
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
                "answer_point_ids": ["alarm_meaning"],
            }
        ],
        expected_answer_points=["说明 E020 的故障含义"],
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


def _candidate(chunk_id: int, *, retrieval_channel: RetrievalChannel) -> RetrievalCandidate:
    return RetrievalCandidate(
        document_id="doc_hak180_manual",
        chunk_id=chunk_id,
        dataset_id="dataset_default_equipment_ops",
        index_version=3,
        chunk_index=1 if chunk_id == 12345 else 2,
        enabled=True,
        title=f"HAK180 E020 证据 {chunk_id}",
        source_title="HAK180 使用手册",
        content="HAK180 报警码 E020 表示温控异常，需要先停机检查温度传感器。",
        equipment_model="HAK180",
        alarm_code="E020",
        source_type=EvidenceSourceType.LOCAL,
        retrieval_channels=[retrieval_channel],
        retrieval_rank=1,
        retrieval_score=0.92,
        rerank_score=0.92,
    )


def test_offline_environment_executes_fixed_action_path_and_json_trace():
    env = OfflineRagEnvironment(
        snapshot=_snapshot(),
        action_provider=FakeOfflineActionProvider(),
        run_id_prefix="pytest",
    )

    result = env.run_action_path(
        _case(),
        [QueryAction.LOCAL_SEARCH, QueryAction.ANSWER],
        run_id="run_fixed_path",
    )

    assert result.status is OfflineTrajectoryStatus.COMPLETED
    assert result.action_path == [QueryAction.LOCAL_SEARCH, QueryAction.ANSWER]
    assert result.terminal_action is QueryAction.ANSWER
    assert result.config_match_status == "match"
    assert result.corpus_match_status == "match"
    assert result.final_state.history_persistence_enabled is False
    assert result.citations[0].chunk_id == 12345
    assert result.to_json_trace()["steps"][0]["decision"]["action"] == "local_search"


def test_offline_environment_runs_multiple_trajectories_without_state_pollution():
    env = OfflineRagEnvironment(
        snapshot=_snapshot(),
        action_provider=FakeOfflineActionProvider(),
    )

    first = env.run_action_path(_case(), [QueryAction.LOCAL_SEARCH, QueryAction.ANSWER], run_id="run_a")
    second = env.run_action_path(
        _case(),
        [QueryAction.LOCAL_SEARCH, QueryAction.HYDE_SEARCH, QueryAction.ANSWER],
        run_id="run_b",
    )

    assert first.status is OfflineTrajectoryStatus.COMPLETED
    assert second.status is OfflineTrajectoryStatus.COMPLETED
    assert first.action_path == [QueryAction.LOCAL_SEARCH, QueryAction.ANSWER]
    assert second.action_path == [QueryAction.LOCAL_SEARCH, QueryAction.HYDE_SEARCH, QueryAction.ANSWER]
    assert [candidate.chunk_id for candidate in first.retrieved_candidates] == [12345]
    assert [candidate.chunk_id for candidate in second.retrieved_candidates] == [12345, 67890]


def test_offline_environment_marks_illegal_action_path():
    env = OfflineRagEnvironment(
        snapshot=_snapshot(),
        action_provider=FakeOfflineActionProvider(),
    )

    result = env.run_action_path(_case(), [QueryAction.HYDE_SEARCH], run_id="run_illegal")

    assert result.status is OfflineTrajectoryStatus.FAILED
    assert result.errors[0].code == "illegal_action_transition"
    assert result.action_path == [QueryAction.HYDE_SEARCH]
    assert result.terminal_action is None


def test_offline_environment_rejects_snapshot_disabled_candidate():
    env = OfflineRagEnvironment(
        snapshot=_snapshot(disabled_chunk_id=12345),
        action_provider=FakeOfflineActionProvider(),
    )

    result = env.run_action_path(
        _case(),
        [QueryAction.LOCAL_SEARCH, QueryAction.ANSWER],
        run_id="run_disabled",
    )

    assert result.status is OfflineTrajectoryStatus.FAILED
    assert result.corpus_match_status == "mismatch"
    assert result.errors[0].code == "candidate_disabled_by_snapshot"


def test_offline_environment_run_planner_reaches_answer_from_observation():
    env = OfflineRagEnvironment(
        snapshot=_snapshot(),
        action_provider=FakeOfflineActionProvider(),
    )
    planner = RuleBasedPlanner(
        config=RuleBasedPlannerConfig(
            rerank_evidence_threshold=0.75,
            retrieval_config_version=RETRIEVAL_CONFIG_VERSION,
        )
    )

    result = env.run_planner(_case(), planner, run_id="run_rule")

    assert result.status is OfflineTrajectoryStatus.COMPLETED
    assert result.action_path == [QueryAction.LOCAL_SEARCH, QueryAction.ANSWER]
    assert result.terminal_reason_code is PlannerReasonCode.LOCAL_EVIDENCE_SUFFICIENT
