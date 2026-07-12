import sys

from app.shared.runtime.logger import node_log
from app.rag.query.rerank_service import rerank_documents
from app.shared.utils.task_utils import add_done_task, add_running_task

@node_log("node_rerank")
def node_rerank(state):
    """
    节点功能：使用 Cross-Encoder 模型对累计 local/Web Candidate 统一打分重排。

    rerank 只写 ``rerank_score`` 并调整顺序，document/chunk/dataset/index、URL、召回
    通道和 RRF 分数必须保留。节点只返回 ``reranked_docs`` partial state，避免用旧
    State 覆盖其他并发节点字段。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    result_state = rerank_documents(state)
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {
        "reranked_docs": result_state.get("reranked_docs", []),
    }
