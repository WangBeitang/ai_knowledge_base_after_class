"""把无 I/O 的 RuleBasedPlanner 接入 LangGraph State。"""

from time import perf_counter

from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import (
    PLANNER_MAX_STEPS,
    RERANK_EVIDENCE_THRESHOLD,
    RETRIEVAL_CONFIG_VERSION,
)
from app.rag.query.contracts import (
    PlannerContext,
    PlannerHistoryItem,
    QueryAction,
    RetrievalObservation,
    SubjectResolutionStatus,
)
from app.rag.query.planner import RuleBasedPlanner, RuleBasedPlannerConfig
from app.rag.query.trace_service import safe_record_planner_decision
from app.shared.runtime.logger import node_log


# 规则 Planner 是无状态纯函数，配置在进程启动后不可变，因此可以安全复用一个实例。
# 这里显式注入阈值和版本，不在节点内部散落魔法数字，后续评测换配置时只修改 config。
rule_based_planner = RuleBasedPlanner(
    config=RuleBasedPlannerConfig(
        rerank_evidence_threshold=RERANK_EVIDENCE_THRESHOLD,
        retrieval_config_version=RETRIEVAL_CONFIG_VERSION,
    )
)


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
    """调用规则 Planner，并只写入本轮经过契约校验的 PlannerDecision。"""
    started_at = perf_counter()
    context = build_planner_context(state)
    decision = rule_based_planner.plan(context)
    duration_ms = int((perf_counter() - started_at) * 1000)

    planner_runtime_metadata = {
        "provider": None,
        "model_id": None,
        "model_revision": None,
        "prompt_version": None,
        "realtime_rule_version": rule_based_planner.realtime_rule_version,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "estimated_cost": 0,
        "duration_ms": duration_ms,
    }
    # Planner Decision 一产生就先写 pending step。即使后续 Milvus/LLM 发生未处理异常，
    # Trace 仍能说明最后选择了哪个 Action，而不是只剩一条模糊 failed 终态。
    safe_record_planner_decision(
        state,
        decision=decision,
        planner_runtime_metadata=planner_runtime_metadata,
    )

    return {
        "current_planner_decision": decision,
        "policy_version": rule_based_planner.policy_version,
        "planner_type": "rule",
        "retrieval_config_version": rule_based_planner.config.retrieval_config_version,
        # runtime metadata 的中文含义是“本次决策运行信息”。规则 Planner 不调用模型，
        # 所以 token 和费用固定为 0；不保存自由文本推理或隐藏思维链。
        "planner_runtime_metadata": planner_runtime_metadata,
        "planner_total_duration_ms": int(state.get("planner_total_duration_ms", 0)) + duration_ms,
        # 每个新检索 Action 从 0 开始累计本轮外部调用和 rerank 耗时。
        "current_action_duration_ms": 0,
    }
