"""收口最后一个终止 Action，并持久化阶段 5 Retrieval Trace。"""

from app.process.query.agent.state import QueryGraphState
from app.rag.query.contracts import PlannerExecutionStatus, PlannerReasonCode, QueryAction
from app.shared.runtime.logger import node_log
from app.rag.query.trace_service import safe_complete_terminal_step_and_trace

from app.process.query.agent.nodes.node_retrieval_observation import (
    append_action_history,
    current_decision,
)


@node_log("node_trace_finalize")
def node_trace_finalize(state: QueryGraphState) -> dict:
    """
    把终止 Decision 追加到内存 Action Trace，并固定最终原因码。

    finalize 的中文含义是“收口/完成”。节点先形成最终内存 Action history，再把去除
    证据正文后的结构化投影交给 Trace service；Mongo 写入失败只记录日志，不影响答案。
    """
    decision = current_decision(state)
    if decision.action not in {
        QueryAction.ANSWER,
        QueryAction.ASK_CLARIFICATION,
        QueryAction.REFUSE,
    }:
        raise ValueError("Trace 收口时 current_planner_decision 必须是终止 Action")

    terminal_reason = state.get("terminal_reason_code") or decision.reason_code
    execution_status = (
        PlannerExecutionStatus.FAILED
        if terminal_reason == PlannerReasonCode.ACTION_EXECUTION_ERROR
        else PlannerExecutionStatus.COMPLETED
    )
    history = append_action_history(state, execution_status=execution_status)
    finalized_state = {
        **state,
        "planner_action_history": history,
        "planner_step": len(history),
        "terminal_reason_code": terminal_reason,
    }
    safe_complete_terminal_step_and_trace(
        finalized_state,
        execution_status=execution_status,
    )
    return {
        "planner_action_history": history,
        "planner_step": len(history),
        "terminal_reason_code": terminal_reason,
    }
