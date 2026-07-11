import sys

from app.shared.runtime.logger import node_log
from app.rag.query.answer_service import generate_answer
from app.shared.utils.task_utils import add_done_task, add_running_task

@node_log("node_answer_output")
def node_answer_output(state):
    """
    节点功能：生成最终回答并交付给用户（支持流式/非流式）。

    答案节点只返回 ``answer`` 和 ``image_urls`` 的 partial state（局部状态）。检索候选、
    Planner 字段和身份字段继续保留在 LangGraph 主 State 中，不从 service 的完整结果回写。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    result_state = generate_answer(state)
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state["is_stream"])
    return {
        "answer": result_state.get("answer", ""),
        "image_urls": result_state.get("image_urls", []),
    }
