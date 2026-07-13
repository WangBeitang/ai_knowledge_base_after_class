import sys
from time import perf_counter

from app.rag.query.contracts import ObservationStatus, RetrievalObservation
from app.shared.runtime.logger import node_log
from app.rag.query.rerank_service import rerank_documents
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.query.agent.nodes.node_retrieval_observation import (
    build_failed_observation,
    current_decision,
    is_expected_external_error,
)

@node_log("node_rerank")
def node_rerank(state):
    """
    节点功能：使用 Cross-Encoder 模型对累计 local/Web Candidate 统一打分重排。

    rerank 只写 ``rerank_score`` 并调整顺序，document/chunk/dataset/index、URL、召回
    通道和 RRF 分数必须保留。节点只返回 ``reranked_docs`` partial state，避免用旧
    State 覆盖其他并发节点字段。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    decision = current_decision(state)
    existing_duration_ms = max(0, int(state.get("current_action_duration_ms", 0)))
    preliminary = state.get("retrieval_observation")
    if preliminary is not None and not isinstance(preliminary, RetrievalObservation):
        preliminary = RetrievalObservation.model_validate(preliminary)
    # 外部检索已经失败时不再调用 reranker；失败事实必须原样交给 Observation 节点。
    if (
        isinstance(preliminary, RetrievalObservation)
        and preliminary.action == decision.action
        and preliminary.status == ObservationStatus.FAILED
    ):
        add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
        return {"reranked_docs": [], "current_action_duration_ms": existing_duration_ms}

    started_at = perf_counter()
    try:
        result_state = rerank_documents(state)
    except Exception as error:
        if not is_expected_external_error(error):
            raise
        duration_ms = existing_duration_ms + int((perf_counter() - started_at) * 1000)
        observation = build_failed_observation(state, decision.action, error, duration_ms)
        add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
        return {
            "reranked_docs": [],
            "retrieval_observation": observation,
            "current_action_duration_ms": duration_ms,
        }
    duration_ms = existing_duration_ms + int((perf_counter() - started_at) * 1000)
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {
        "reranked_docs": result_state.get("reranked_docs", []),
        "current_action_duration_ms": duration_ms,
    }
