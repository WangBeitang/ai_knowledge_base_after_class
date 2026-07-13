import sys

from app.shared.runtime.logger import node_log
from app.rag.query.subject_name_confirm_service import confirm_subject_name
from app.shared.utils.task_utils import add_done_task, add_running_task

@node_log("node_subject_name_confirm")
def node_subject_name_confirm(state):
    """
    节点功能：确认用户问题中的核心主体名称。
    输入：state['original_query']
    输出：主体状态、候选、稳定 ID、标准名称、改写问题和历史快照。

    返回值只列出主体确认节点真正负责写入的字段，也就是 partial state（局部状态）。
    LangGraph 会负责与主 State 合并，节点不能返回收到的完整 State 副本。
    """
    # 先登记节点开始，前端进度区可以立即感知"主体确认"已启动。
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    # 调用 rag/query service 层
    result_state = confirm_subject_name(state)
    # 识别完成后写入完成列表，方便前端展示当前节点已结束。
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    return {
        "subject_ids": result_state.get("subject_ids", []),
        "standard_subject_names": result_state.get("standard_subject_names", []),
        "rewritten_query": result_state.get("rewritten_query", ""),
        "history": result_state.get("history", []),
        "subject_resolution_status": result_state.get("subject_resolution_status"),
        "subject_candidates": result_state.get("subject_candidates", []),
        "clarification_question": result_state.get("clarification_question"),
    }


if __name__ == "__main__":
    mock_state = {
        "session_id": "test_session_001",
        "original_query": "HAK 180 烫金机？",
        "is_stream": False,
    }
    result_state = node_subject_name_confirm(mock_state)
    print(result_state)
