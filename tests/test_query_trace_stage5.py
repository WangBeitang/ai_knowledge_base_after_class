import copy

import pytest

from app.process.query.agent.state import create_query_default_state
from app.rag.query import trace_service
from app.rag.query.config import build_retrieval_config_snapshot
from app.rag.query.contracts import (
    Citation,
    EvidenceSourceType,
    EvidenceSummary,
    ObservationStatus,
    PlannerDecision,
    PlannerExecutionStatus,
    PlannerReasonCode,
    QueryAction,
    RetrievalCandidate,
    RetrievalChannel,
    RetrievalObservation,
)


class FakeTraceRepository:
    def __init__(self):
        self.running = []
        self.appended_steps = []
        self.completed_steps = []
        self.completed_traces = []
        self.failed_traces = []

    def create_running(self, trace):
        self.running.append(copy.deepcopy(trace))

    def append_step(self, trace_id, step):
        self.appended_steps.append((trace_id, copy.deepcopy(step)))

    def complete_step(self, trace_id, step):
        self.completed_steps.append((trace_id, copy.deepcopy(step)))

    def complete_trace(self, trace_id, fields):
        self.completed_traces.append((trace_id, copy.deepcopy(fields)))

    def fail_trace(self, trace_id, fields):
        self.failed_traces.append((trace_id, copy.deepcopy(fields)))


def _state(**overrides):
    state = create_query_default_state(
        session_id="session-trace",
        original_query="HAK 180 出现 E020 怎么处理？",
        owner_user_id="user-a",
        tenant_id="tenant_default",
        dataset_ids=["dataset-default"],
        trace_id="trace-1",
        query_started_at="2026-07-13T00:00:00+00:00",
        query_identifiers={"alarm_code": ["E020"]},
        policy_version="rule-v1",
        planner_type="rule",
        retrieval_config_version="retrieval-stage5-dev-v1",
        retrieval_mode="dense_learned_sparse",
        retrieval_config_snapshot=build_retrieval_config_snapshot(),
        trace_persistence_enabled=True,
    )
    state.update(overrides)
    return state


def _observation() -> RetrievalObservation:
    return RetrievalObservation(
        action=QueryAction.LOCAL_SEARCH,
        status=ObservationStatus.SUCCESS,
        channel_counts={"local_search": 1},
        candidate_count=1,
        reranked_count=1,
        top_rerank_score=0.9,
        requested_identifiers={"alarm_code": ["E020"]},
        matched_identifiers={"alarm_code": ["E020"]},
        identifier_resolution_status="exact_match",
        evidence_summaries=[EvidenceSummary(
            document_id="doc-1",
            chunk_id="chunk-1",
            title="E020 报警处理",
            source_type=EvidenceSourceType.LOCAL,
            rerank_score=0.9,
            matched_identifiers={"alarm_code": ["E020"]},
            content_excerpt="按下复位按钮前先确认急停回路。",
        )],
        duration_ms=12,
        used_structured_filter=True,
    )


def _candidate() -> dict:
    return RetrievalCandidate(
        document_id="doc-1",
        chunk_id="chunk-1",
        dataset_id="dataset-default",
        index_version=3,
        chunk_index=0,
        enabled=True,
        title="E020 报警处理",
        source_title="HAK 180 维修手册",
        content="按下复位按钮前先确认急停回路。",
        source_type=EvidenceSourceType.LOCAL,
        retrieval_channels=[RetrievalChannel.ORIGINAL],
        retrieval_rank=1,
        retrieval_score=0.5,
        rerank_score=0.9,
        alarm_code="E020",
    ).model_dump(mode="json")


def test_observation_projection_removes_excerpt_and_keeps_sha256_hash():
    projected = trace_service.project_observation(_observation())
    payload = projected.model_dump(mode="json")

    assert "content_excerpt" not in payload["evidence_summaries"][0]
    assert len(payload["evidence_summaries"][0]["content_excerpt_hash"]) == 64
    assert "按下复位" not in str(payload)


def test_trace_lifecycle_records_pending_completed_and_terminal_steps(monkeypatch):
    repository = FakeTraceRepository()
    monkeypatch.setattr(trace_service, "get_retrieval_trace_repository", lambda: repository)
    state = _state()

    trace_service.safe_create_running_trace(state)
    decision = PlannerDecision(
        action=QueryAction.LOCAL_SEARCH,
        query=state["original_query"],
        reason_code=PlannerReasonCode.INITIAL_LOCAL_SEARCH,
    )
    state["current_planner_decision"] = decision
    planner_metadata = {"duration_ms": 2, "input_tokens": 0, "output_tokens": 0}
    state["planner_runtime_metadata"] = planner_metadata
    trace_service.safe_record_planner_decision(
        state,
        decision=decision,
        planner_runtime_metadata=planner_metadata,
    )

    observation = _observation()
    state["retrieval_observation"] = observation
    trace_service.safe_complete_action_step(
        state,
        observation=observation,
        execution_status=PlannerExecutionStatus.COMPLETED,
    )

    terminal_decision = PlannerDecision(
        action=QueryAction.ANSWER,
        query=state["original_query"],
        reason_code=PlannerReasonCode.LOCAL_EVIDENCE_SUFFICIENT,
    )
    state.update({
        "planner_step": 2,
        "current_planner_decision": terminal_decision,
        "terminal_reason_code": PlannerReasonCode.LOCAL_EVIDENCE_SUFFICIENT,
        "embedding_chunks": [_candidate()],
        "reranked_docs": [_candidate()],
        "citations": [Citation(
            document_id="doc-1",
            chunk_id="chunk-1",
            title="E020 报警处理",
            source="HAK 180 维修手册",
            score=0.9,
            source_type=EvidenceSourceType.LOCAL,
        )],
        "planner_total_duration_ms": 4,
        "answer_runtime_metadata": {
            "provider": "openai-compatible",
            "model_id": "test-model",
            "prompt_version": "answer-out-v1",
            "duration_ms": 20,
        },
    })
    trace_service.safe_complete_terminal_step_and_trace(
        state,
        execution_status=PlannerExecutionStatus.COMPLETED,
    )

    assert repository.running[0]["status"] == "running"
    assert repository.running[0]["retrieval_config_snapshot"]["rrf_k"] == 60
    assert repository.appended_steps[0][1]["step"] == 1
    assert repository.appended_steps[0][1]["execution_status"] == "pending"
    assert repository.completed_steps[0][1]["step"] == 1
    assert repository.completed_steps[0][1]["output_observation"]["status"] == "success"
    assert repository.completed_steps[1][1]["step"] == 2
    final_trace = repository.completed_traces[0][1]
    assert final_trace["status"] == "completed"
    assert final_trace["terminal_action"] == "answer"
    assert final_trace["index_versions"] == [3]
    assert final_trace["final_citations"][0]["chunk_id"] == "chunk-1"
    assert final_trace["channel_hits"][0]["enabled"] is True
    assert final_trace["channel_hits"][0]["retrieval_channels"] == ["original"]
    assert final_trace["channel_hits"][0]["entered_rerank"] is True
    assert final_trace["channel_hits"][0]["became_citation"] is True
    assert "按下复位" not in str(final_trace["channel_hits"])


def test_trace_failure_uses_structured_error_code_without_exception_message(monkeypatch):
    repository = FakeTraceRepository()
    monkeypatch.setattr(trace_service, "get_retrieval_trace_repository", lambda: repository)
    state = _state()

    trace_service.safe_fail_trace(state, ValueError("secret internal detail"))

    failed = repository.failed_traces[0][1]
    assert failed["status"] == "failed"
    assert failed["error_code"] == "UNHANDLED_VALUEERROR"
    assert "secret internal detail" not in str(failed)


def test_disabled_trace_persistence_does_not_initialize_repository(monkeypatch):
    monkeypatch.setattr(
        trace_service,
        "get_retrieval_trace_repository",
        lambda: pytest.fail("离线图重放不应初始化 Mongo repository"),
    )
    trace_service.safe_create_running_trace(_state(trace_persistence_enabled=False))
