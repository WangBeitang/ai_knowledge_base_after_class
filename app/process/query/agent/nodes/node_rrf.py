import sys

from app.shared.runtime.logger import node_log
from app.rag.query.rrf_service import fuse_by_rrf
from app.shared.utils.task_utils import add_done_task, add_running_task

@node_log("node_rrf")
def node_rrf(state):
    """
    节点功能：跨 Action Reciprocal Rank Fusion（倒数排名融合）。

    它读取当前实际有结果的 original、HyDE、Web 原始排名列表，一次性按名次融合；不
    直接相加原始分数，也不把上一轮 RRF 结果继续套一层 RRF。相同本地 chunk/Web URL
    会去重并合并 retrieval_channels。

    只返回融合产物 ``rrf_chunks`` 的 partial state（局部状态）；该字段名是过渡名称，
    其中现在可以同时包含 local 和 Web ``RetrievalCandidate``。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    result_state = fuse_by_rrf(state)
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {
        "rrf_chunks": result_state.get("rrf_chunks", []),
    }

if __name__ == "__main__":
    mock_state = {
        "session_id": "test_rrf_session",
        "is_stream": False,
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤是什么？",
        "subject_ids": ["subject_hak_180"],
        "standard_subject_names": ["HAK 180 烫金机"],
    }

    from app.process.query.agent.nodes.node_search_embedding import node_search_embedding
    from app.process.query.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde

    emb_res = node_search_embedding(mock_state)
    hyde_res = node_search_embedding_hyde(mock_state)
    mock_state["embedding_chunks"] = emb_res.get("embedding_chunks") or []
    mock_state["hyde_embedding_chunks"] = hyde_res.get("hyde_embedding_chunks") or []

    result = node_rrf(mock_state)
    print(result)
