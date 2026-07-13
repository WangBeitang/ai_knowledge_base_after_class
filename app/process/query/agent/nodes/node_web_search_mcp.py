import sys
from time import perf_counter

from app.rag.query.contracts import QueryAction
from app.shared.runtime.logger import node_log
from app.rag.query.web_search_service import search_by_web
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.query.agent.nodes.node_retrieval_observation import (
    build_failed_observation,
    current_decision,
    is_expected_external_error,
)

@node_log("node_web_search_mcp")
def node_web_search_mcp(state):
    """
    节点功能：调用外部搜索引擎补充信息，并输出统一 Web RetrievalCandidate。

    每个候选使用真实 URL 作为身份，本地 document/chunk/index 字段必须为空。这里只
    返回 ``web_search_docs`` partial state，不能把 service 的完整 State 原样返回，否则
    并发分支可能互相覆盖检索结果。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    decision = current_decision(state)
    if decision.action != QueryAction.WEB_SEARCH:
        raise ValueError("联网检索节点只能执行 web_search Decision")
    started_at = perf_counter()
    try:
        result_state = search_by_web(state)
    except Exception as error:
        if not is_expected_external_error(error):
            raise
        duration_ms = int((perf_counter() - started_at) * 1000)
        observation = build_failed_observation(state, decision.action, error, duration_ms)
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
        return {
            "web_search_docs": [],
            "retrieval_observation": observation,
            "current_action_duration_ms": duration_ms,
        }
    duration_ms = int((perf_counter() - started_at) * 1000)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    return {
        "web_search_docs": result_state.get("web_search_docs"),
        "current_action_duration_ms": duration_ms,
    }
