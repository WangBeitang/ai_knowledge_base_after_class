import sys

from app.shared.runtime.logger import node_log
from app.rag.query.answer_service import generate_answer
from app.shared.utils.task_utils import add_done_task, add_running_task

@node_log("node_answer_output")
def node_answer_output(state):
    """
    节点功能：生成最终回答并交付给用户（支持流式/非流式）。

    答案节点只返回自己负责的交付字段。编号需要确认时，``citations`` 会显式为空，并
    返回机器可读 ``terminal_reason_code``；检索候选、身份字段不会从 service 完整回写。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    result_state = generate_answer(state)
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state["is_stream"])
    return {
        "answer": result_state.get("answer", ""),
        "image_urls": result_state.get("image_urls", []),
        "citations": result_state.get("citations", []),
        "clarification_question": result_state.get("clarification_question"),
        "terminal_reason_code": result_state.get("terminal_reason_code"),
    }
