import sys

from app.shared.runtime.logger import node_log
from app.rag.query.embedding_search_service import search_by_embedding
from app.shared.utils.task_utils import add_done_task, add_running_task

@node_log("node_search_embedding")
def node_search_embedding(state):
    """
    节点功能：使用原问题/改写问题执行普通本地向量检索。

    service 层为了兼容现有实现会返回一份包含完整字段的字典，但 LangGraph 节点边界
    只提交本节点负责的 ``embedding_chunks``。这种返回方式称为 partial state
    （局部状态）：LangGraph 会把它合并回主 State，其他节点已经写入的字段不会被本节点
    用旧副本覆盖。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    result_state = search_by_embedding(state)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {
        "embedding_chunks": result_state.get("embedding_chunks")
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
