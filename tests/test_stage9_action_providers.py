import json
from pathlib import Path

import pytest

from app.rag.evaluation.case_schema import EnvironmentSnapshot, PlannerEvalCase
from app.rag.evaluation.offline_environment import OfflineRagEnvironment
from app.rag.query.config import RETRIEVAL_CONFIG_VERSION
from app.rag.query.contracts import (
    EvidenceSourceType,
    PlannerDecision,
    PlannerReasonCode,
    QueryAction,
    RetrievalCandidate,
    RetrievalChannel,
)
from app.rag.evaluation.action_providers import (
    MilvusActionProvider,
    RecordingActionProvider,
    ReplayActionProvider,
    read_provider_observation_records,
)


def test_milvus_action_provider_uses_query_graph_state_adapter():
    captured = {}

    def local_search_fn(graph_state):
        captured.update(graph_state)
        return {"embedding_chunks": [_local_candidate().model_dump(mode="json")]}

    provider = MilvusActionProvider(
        local_search_fn=local_search_fn,
        chunk_status_filter_enabled=False,
    )
    trajectory = OfflineRagEnvironment(
        snapshot=_snapshot(),
        action_provider=provider,
        planner_mode="real_provider",
        run_id_prefix="pytest_stage9_real_provider",
    ).run_action_path(
        _answer_case(),
        [QueryAction.LOCAL_SEARCH, QueryAction.ANSWER],
        run_id="pytest_stage9_real_provider_local",
        planner_mode="real_provider",
    )

    assert not trajectory.errors
    assert captured["trace_id"] == "pytest_stage9_real_provider_local"
    assert captured["current_planner_decision"].action is QueryAction.LOCAL_SEARCH
    assert captured["retrieval_mode"] == "dense_learned_sparse_bm25"
    assert captured["chunk_status_filter_enabled"] is False
    assert captured["history_persistence_enabled"] is False
    assert trajectory.retrieved_candidates[0].chunk_id == 12345


def test_milvus_action_provider_rejects_web_when_state_disallows_it():
    provider = MilvusActionProvider(
        web_search_fn=lambda graph_state: {"web_search_docs": [_web_candidate().model_dump(mode="json")]},
        chunk_status_filter_enabled=False,
    )
    state = OfflineRagEnvironment(snapshot=_snapshot(), action_provider=provider).reset(_answer_case())

    with pytest.raises(ValueError, match="不允许 Web"):
        provider.web_search(
            state,
            PlannerDecision(
                action=QueryAction.WEB_SEARCH,
                query=state.current_query,
                reason_code=PlannerReasonCode.REALTIME_QUERY,
            ),
        )


def test_recording_provider_writes_observation_record_and_replay_reads_it(tmp_path: Path):
    records_path = tmp_path / "provider_records.jsonl"
    recording_provider = RecordingActionProvider(
        _FakeProvider(),
        output_path=records_path,
        max_candidate_content_chars=40,
    )
    env = OfflineRagEnvironment(
        snapshot=_snapshot(),
        action_provider=recording_provider,
        planner_mode="recording",
    )
    trajectory = env.run_action_path(
        _answer_case(),
        [QueryAction.LOCAL_SEARCH, QueryAction.ANSWER],
        run_id="pytest_stage9_recording_local",
        planner_mode="recording",
    )

    assert not trajectory.errors
    records = read_provider_observation_records(records_path)
    assert len(records) == 1
    record = records[0]
    assert record.action is QueryAction.LOCAL_SEARCH
    assert record.candidate_count == 1
    assert record.observation["status"] == "success"
    assert record.candidates[0]["content_truncated"] is True

    replay_provider = ReplayActionProvider(records_path)
    replay_trajectory = OfflineRagEnvironment(
        snapshot=_snapshot(),
        action_provider=replay_provider,
        planner_mode="replay",
    ).run_action_path(
        _answer_case(),
        [QueryAction.LOCAL_SEARCH, QueryAction.ANSWER],
        run_id="pytest_stage9_replay_local",
        planner_mode="replay",
    )
    assert not replay_trajectory.errors
    assert replay_trajectory.retrieved_candidates[0].chunk_id == 12345


def test_replay_provider_missing_record_fails_loudly(tmp_path: Path):
    records_path = tmp_path / "provider_records.jsonl"
    records_path.write_text("", encoding="utf-8")
    provider = ReplayActionProvider(records_path)
    state = OfflineRagEnvironment(snapshot=_snapshot(), action_provider=provider).reset(_answer_case())

    with pytest.raises(KeyError, match="Replay 缺少对应记录"):
        provider.local_search(
            state,
            PlannerDecision(
                action=QueryAction.LOCAL_SEARCH,
                query=state.current_query,
                reason_code=PlannerReasonCode.INITIAL_LOCAL_SEARCH,
            ),
        )


def test_replay_provider_returns_independent_candidate_objects(tmp_path: Path):
    records_path = tmp_path / "provider_records.jsonl"
    recording_provider = RecordingActionProvider(_FakeProvider(), output_path=records_path)
    env = OfflineRagEnvironment(snapshot=_snapshot(), action_provider=recording_provider)
    env.run_action_path(
        _answer_case(),
        [QueryAction.LOCAL_SEARCH, QueryAction.ANSWER],
        run_id="pytest_stage9_replay_copy",
    )
    replay_provider = ReplayActionProvider(records_path)
    state = OfflineRagEnvironment(snapshot=_snapshot(), action_provider=replay_provider).reset(_answer_case())
    decision = PlannerDecision(
        action=QueryAction.LOCAL_SEARCH,
        query=state.current_query,
        reason_code=PlannerReasonCode.INITIAL_LOCAL_SEARCH,
    )

    first = replay_provider.local_search(state, decision)
    first[0].content = "mutated"
    second = replay_provider.local_search(state, decision)

    assert second[0].content != "mutated"


class _FakeProvider:
    provider_name = "fake_real_provider"

    def local_search(self, state, decision):
        return [_local_candidate()]

    def hyde_search(self, state, decision):
        return []

    def web_search(self, state, decision):
        return [_web_candidate()]


def _snapshot() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        snapshot_id="stage9-provider-test-v1",
        created_at="2026-07-23T00:00:00+00:00",
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


def _answer_case() -> PlannerEvalCase:
    return PlannerEvalCase(
        case_id="stage9-provider-answer",
        case_group="core",
        split="dev",
        leakage_group_id="stage9-provider-answer",
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


def _local_candidate() -> RetrievalCandidate:
    return RetrievalCandidate(
        document_id="doc_hak180_manual",
        chunk_id=12345,
        dataset_id="dataset_default_equipment_ops",
        index_version=3,
        chunk_index=0,
        enabled=True,
        title="HAK180 E020 证据",
        source_title="HAK180 使用手册",
        content="HAK180 报警码 E020 表示温控异常，需要先停机检查温度传感器。" * 3,
        equipment_model="HAK180",
        alarm_code="E020",
        source_type=EvidenceSourceType.LOCAL,
        retrieval_channels=[RetrievalChannel.ORIGINAL],
        retrieval_rank=1,
        retrieval_score=0.92,
        rerank_score=0.92,
    )


def _web_candidate() -> RetrievalCandidate:
    return RetrievalCandidate(
        title="HAK180 最新公开公告",
        source_title="HAK180 最新公开公告",
        content="公开网页摘要。",
        source_type=EvidenceSourceType.WEB,
        retrieval_channels=[RetrievalChannel.WEB],
        retrieval_rank=1,
        retrieval_score=0.70,
        rerank_score=0.70,
        url="https://example.com/hak180",
    )
