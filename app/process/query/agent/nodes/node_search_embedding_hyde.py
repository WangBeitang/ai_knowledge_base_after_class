import sys
from time import perf_counter

from app.rag.query.contracts import QueryAction
from app.shared.runtime.logger import node_log
from app.rag.query.hyde_search_service import search_by_hyde
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.query.agent.nodes.node_retrieval_observation import (
    build_failed_observation,
    current_decision,
    is_expected_external_error,
)

@node_log("node_search_embedding_hyde")
def node_search_embedding_hyde(state):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。

    这里只返回 ``hyde_embedding_chunks`` partial state（局部状态），不把 service 返回的
    整个 State 原样交回 LangGraph，避免并发检索节点用各自收到的旧 State 覆盖彼此结果。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    decision = current_decision(state)
    if decision.action != QueryAction.HYDE_SEARCH:
        raise ValueError("HyDE 检索节点只能执行 hyde_search Decision")
    started_at = perf_counter()
    try:
        result_state = search_by_hyde(state)
    except Exception as error:
        if not is_expected_external_error(error):
            raise
        duration_ms = int((perf_counter() - started_at) * 1000)
        observation = build_failed_observation(state, decision.action, error, duration_ms)
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
        return {
            "hyde_embedding_chunks": [],
            "retrieval_observation": observation,
            "current_action_duration_ms": duration_ms,
        }
    duration_ms = int((perf_counter() - started_at) * 1000)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {
        "hyde_embedding_chunks": result_state.get("hyde_embedding_chunks"),
        "current_action_duration_ms": duration_ms,
    }


if __name__ == "__main__":
    mock_state = {
        "session_id": "test_hyde_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤是什么？",
        "subject_ids": ["subject_hak_180"],
        "standard_subject_names": ["HAK 180 烫金机"],
        "is_stream": False,
    }
    result = node_search_embedding_hyde(mock_state)
    print(result)
