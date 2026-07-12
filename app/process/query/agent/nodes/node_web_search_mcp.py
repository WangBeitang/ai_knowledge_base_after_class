import sys

from app.shared.runtime.logger import node_log
from app.rag.query.web_search_service import search_by_web
from app.shared.utils.task_utils import add_done_task, add_running_task

@node_log("node_web_search_mcp")
def node_web_search_mcp(state):
    """
    节点功能：调用外部搜索引擎补充信息，并输出统一 Web RetrievalCandidate。

    每个候选使用真实 URL 作为身份，本地 document/chunk/index 字段必须为空。这里只
    返回 ``web_search_docs`` partial state，不能把 service 的完整 State 原样返回，否则
    并发分支可能互相覆盖检索结果。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    result_state = search_by_web(state)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    return {
        "web_search_docs": result_state.get("web_search_docs")
    }
