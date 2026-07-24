from app.process.query.agent.nodes import node_query_planner as planner_node
from app.process.query.agent.state import create_query_default_state
from app.rag.query.contracts import (
    PlannerContext,
    PlannerDecision,
    PlannerReasonCode,
    QueryAction,
    SubjectResolutionStatus,
)
from app.rag.query.model_planner import PlannerClientError
from app.rag.query.planner_registry import PlannerMode, PlannerRuntime


class FakeModelPlanner:
    policy_version = "http:qwen3.5:4b"

    def __init__(self, decision: PlannerDecision | None = None, error: Exception | None = None) -> None:
        self.decision = decision
        self.error = error
        self.seen_context: PlannerContext | None = None

    def plan(self, context: PlannerContext) -> PlannerDecision:
        self.seen_context = context
        if self.error is not None:
            raise self.error
        assert self.decision is not None
        return self.decision


def _state(**overrides):
    state = create_query_default_state(
        session_id="session-stage9-model-planner",
        original_query="HAK 180 E020 如何处理？",
        owner_user_id="user-a",
        tenant_id="tenant_default",
        dataset_ids=["dataset_default_equipment_ops"],
        trace_id="trace-stage9-model-planner",
    )
    state.update({
        "rewritten_query": "HAK 180 E020 如何处理？",
        "subject_ids": ["subject_hak_180"],
        "subject_resolution_status": SubjectResolutionStatus.CONFIRMED,
        "query_identifiers": {"alarm_code": ["E020"]},
    })
    state.update(overrides)
    return state


def _runtime(planner: FakeModelPlanner, *, mode: PlannerMode = PlannerMode.SFT) -> PlannerRuntime:
    return PlannerRuntime(
        mode=mode,
        planner=planner,
        planner_type="model",
        policy_version=f"{mode.value}:http:qwen3.5:4b",
        retrieval_config_version="retrieval-stage5-final-v1",
        provider="http",
        model_id="qwen3.5:4b",
        prompt_version="stage9-planner-prompt-v1",
        endpoint="http://localhost:11434/v1/chat/completions",
    )


def test_node_query_planner_uses_registry_model_runtime(monkeypatch):
    decision = PlannerDecision(
        action=QueryAction.LOCAL_SEARCH,
        query="HAK 180 E020 如何处理？",
        reason_code=PlannerReasonCode.INITIAL_LOCAL_SEARCH,
    )
    fake_planner = FakeModelPlanner(decision=decision)
    runtime = _runtime(fake_planner, mode=PlannerMode.LOCAL_BASE)
    monkeypatch.setattr(planner_node, "get_current_planner_runtime", lambda: runtime)

    output = planner_node.node_query_planner(_state())

    assert output["current_planner_decision"] == decision
    assert output["planner_mode"] == "local_base"
    assert output["planner_type"] == "model"
    assert output["policy_version"] == "local_base:http:qwen3.5:4b"
    assert output["planner_runtime_metadata"]["planner_mode"] == "local_base"
    assert output["planner_runtime_metadata"]["provider"] == "http"
    assert output["planner_runtime_metadata"]["model_id"] == "qwen3.5:4b"
    assert output["planner_runtime_metadata"]["prompt_version"] == "stage9-planner-prompt-v1"
    assert output["planner_runtime_metadata"]["error_code"] == ""
    assert fake_planner.seen_context is not None


def test_node_query_planner_records_model_error_as_safe_refuse(monkeypatch):
    fake_planner = FakeModelPlanner(error=PlannerClientError(
        "http_timeout",
        "PlannerModelServer 请求超时",
    ))
    runtime = _runtime(fake_planner, mode=PlannerMode.SFT)
    monkeypatch.setattr(planner_node, "get_current_planner_runtime", lambda: runtime)

    output = planner_node.node_query_planner(_state())

    decision = output["current_planner_decision"]
    assert decision.action is QueryAction.REFUSE
    assert decision.reason_code is PlannerReasonCode.ACTION_EXECUTION_ERROR
    assert output["planner_mode"] == "sft"
    assert output["planner_runtime_metadata"]["error_code"] == "http_timeout"
    assert output["planner_runtime_metadata"]["error_message"] == "PlannerModelServer 请求超时"
