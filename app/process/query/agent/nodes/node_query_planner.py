"""把可切换 Planner（规划器）接入 LangGraph State（图状态）。"""

from time import perf_counter

from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import (
    PLANNER_MAX_STEPS,
    RETRIEVAL_CONFIG_VERSION,
)
from app.rag.query.contracts import (
    PlannerContext,
    PlannerDecision,
    PlannerHistoryItem,
    PlannerReasonCode,
    QueryAction,
    RetrievalObservation,
    SubjectResolutionStatus,
)
from app.rag.query.model_planner import ModelPlannerOutputError, PlannerClientError
from app.rag.query.planner_registry import (
    PlannerRegistryError,
    get_current_planner_runtime,
)
from app.rag.query.trace_service import safe_record_planner_decision
from app.shared.runtime.logger import node_log


def build_planner_context(state: QueryGraphState) -> PlannerContext:
    """
    把 LangGraph 的宽 State 投影成 Planner 唯一允许读取的强校验 Context。

    投影的价值是隔离职责：Planner 看不到完整 chunk、HTTP Request、Mongo 或 Milvus
    客户端，只能根据主体状态、最近 Observation 和已执行 Action 历史做确定性决策。
    """
    subject_status = state.get("subject_resolution_status")
    if subject_status is None:
        raise ValueError("主体确认节点必须先写入 subject_resolution_status")

    history = [
        item if isinstance(item, PlannerHistoryItem) else PlannerHistoryItem.model_validate(item)
        for item in state.get("planner_action_history", [])
    ]
    observation = state.get("retrieval_observation")
    if observation is not None and not isinstance(observation, RetrievalObservation):
        observation = RetrievalObservation.model_validate(observation)

    return PlannerContext(
        original_query=state.get("original_query", ""),
        current_query=state.get("rewritten_query") or state.get("original_query", ""),
        subject_resolution_status=SubjectResolutionStatus(subject_status),
        subject_ids=list(state.get("subject_ids", [])),
        subject_candidates=list(state.get("subject_candidates", [])),
        clarification_question=state.get("clarification_question"),
        query_identifiers=dict(state.get("query_identifiers", {})),
        latest_observation=observation,
        action_history=history,
        web_search_allowed=bool(state.get("web_search_allowed", True)),
        safe_guard_triggered=bool(state.get("safe_guard_triggered", False)),
        planner_step=int(state.get("planner_step", 0)),
        max_steps=int(state.get("planner_max_steps", PLANNER_MAX_STEPS)),
        # allowed actions 的中文含义是“当前图实际支持的动作白名单”。路由层必须与这里
        # 使用同一关闭枚举；未来禁用 Web 时优先使用 web_search_allowed，而不是删安全出口。
        allowed_actions=list(QueryAction),
    )


@node_log("node_query_planner")
def node_query_planner(state: QueryGraphState) -> dict:
    """调用当前 registry（注册表）选中的 Planner，并写入本轮契约校验后的 Decision。"""
    started_at = perf_counter()
    context = build_planner_context(state)
    try:
        runtime = get_current_planner_runtime()
        decision = runtime.plan(context)
        duration_ms = int((perf_counter() - started_at) * 1000)
        planner_runtime_metadata = runtime.runtime_metadata(duration_ms=duration_ms)
        policy_version = runtime.policy_version
        planner_type = runtime.planner_type
        planner_mode = runtime.mode.value
        retrieval_config_version = runtime.retrieval_config_version
    except PlannerRegistryError as error:
        duration_ms = int((perf_counter() - started_at) * 1000)
        decision = PlannerDecision(
            action=QueryAction.REFUSE,
            query=context.current_query,
            reason_code=PlannerReasonCode.ACTION_EXECUTION_ERROR,
        )
        planner_mode = error.planner_mode or "unknown"
        policy_version = "planner-registry-error"
        planner_type = "unavailable"
        retrieval_config_version = RETRIEVAL_CONFIG_VERSION
        planner_runtime_metadata = {
            "planner_mode": planner_mode,
            "provider": None,
            "model_id": None,
            "model_revision": None,
            "prompt_version": None,
            "endpoint": None,
            "realtime_rule_version": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost": 0,
            "duration_ms": duration_ms,
            "error_code": error.error_code,
            "error_message": error.message,
        }
    except (PlannerClientError, ModelPlannerOutputError) as error:
        duration_ms = int((perf_counter() - started_at) * 1000)
        decision = PlannerDecision(
            action=QueryAction.REFUSE,
            query=context.current_query,
            reason_code=PlannerReasonCode.ACTION_EXECUTION_ERROR,
        )
        error_code = getattr(error, "error_code", type(error).__name__)
        error_message = getattr(error, "message", str(error))
        planner_runtime_metadata = runtime.runtime_metadata(
            duration_ms=duration_ms,
            error_code=str(error_code),
            error_message=str(error_message),
        )
        policy_version = runtime.policy_version
        planner_type = runtime.planner_type
        planner_mode = runtime.mode.value
        retrieval_config_version = runtime.retrieval_config_version
    # Planner Decision 一产生就先写 pending step。即使后续 Milvus/LLM 发生未处理异常，
    # Trace 仍能说明最后选择了哪个 Action，而不是只剩一条模糊 failed 终态。
    safe_record_planner_decision(
        state,
        decision=decision,
        planner_runtime_metadata=planner_runtime_metadata,
    )

    return {
        "current_planner_decision": decision,
        "policy_version": policy_version,
        "planner_mode": planner_mode,
        "planner_type": planner_type,
        "retrieval_config_version": retrieval_config_version,
        # runtime metadata 的中文含义是“本次决策运行信息”。9.3.3 尚未接入模型 token
        # usage（token 用量）回传，所以 token 和费用暂为 0；不保存自由文本推理或隐藏思维链。
        "planner_runtime_metadata": planner_runtime_metadata,
        "planner_total_duration_ms": int(state.get("planner_total_duration_ms", 0)) + duration_ms,
        # 每个新检索 Action 从 0 开始累计本轮外部调用和 rerank 耗时。
        "current_action_duration_ms": 0,
    }
