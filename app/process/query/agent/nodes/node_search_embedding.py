import sys
from time import perf_counter

from app.rag.query.contracts import QueryAction
from app.shared.runtime.logger import node_log
from app.rag.query.embedding_search_service import search_by_embedding
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.query.agent.nodes.node_retrieval_observation import (
    build_failed_observation,
    current_decision,
    is_expected_external_error,
)

@node_log("node_search_embedding")
def node_search_embedding(state):
    """
    节点功能：使用原问题/改写问题执行普通本地向量检索。

    service 层为了兼容现有实现会返回一份包含完整字段的字典，但 LangGraph 节点边界
    只提交本节点负责的 ``query_identifiers``、``embedding_chunks``、禁用 chunk 快照、
    检索 Observation 和可能的 ``clarification_question``。这种返回方式称为 partial state
    （局部状态）：LangGraph 会把它合并回主 State，其他节点已经写入的字段不会被本节点
    用旧副本覆盖。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    decision = current_decision(state)
    if decision.action != QueryAction.LOCAL_SEARCH:
        raise ValueError("普通本地检索节点只能执行 local_search Decision")
    started_at = perf_counter()
    try:
        result_state = search_by_embedding(state)
    except Exception as error:
        if not is_expected_external_error(error):
            raise
        duration_ms = int((perf_counter() - started_at) * 1000)
        observation = build_failed_observation(state, decision.action, error, duration_ms)
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
        return {
            "embedding_chunks": [],
            "retrieval_observation": observation,
            "disabled_chunk_ids": state.get("disabled_chunk_ids", []),
            "current_action_duration_ms": duration_ms,
        }
    duration_ms = int((perf_counter() - started_at) * 1000)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {
        "query_identifiers": result_state.get("query_identifiers", {}),
        "embedding_chunks": result_state.get("embedding_chunks"),
        "disabled_chunk_ids": result_state.get("disabled_chunk_ids", []),
        "retrieval_observation": result_state.get("retrieval_observation"),
        "clarification_question": result_state.get("clarification_question"),
        "current_action_duration_ms": duration_ms,
    }

if __name__ == "__main__":
    test_state = {
        "session_id": "test_search_embedding_001",
        "rewritten_query": "HAK 180 烫金机使用说明",
        "subject_ids": ["subject_hak_180"],
        "standard_subject_names": ["HAK 180 烫金机"],
        "is_stream": False,
    }
    result = node_search_embedding(test_state)
    print(result)
