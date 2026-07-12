from app.process.query.agent.state import QueryGraphState
from app.rag.query.query_identifier_service import identifier_requires_clarification
from app.shared.runtime.logger import logger,step_log

def check_params(state):
    embedding_chunks = state.get("embedding_chunks", [])
    hyde_embedding_chunks = state.get("hyde_embedding_chunks", [])
    if len(embedding_chunks) == 0 and len(hyde_embedding_chunks) == 0:
        # 当前 Planner 尚未接入固定图。若普通检索已经明确要求确认编号，允许空候选安全
        # 通过 RRF，后续答案出口只负责交付追问，不调用答案模型。任务 9 接入 Planner 后
        # 会由 ask_clarification 路由在更早位置终止。
        if identifier_requires_clarification(state.get("retrieval_observation")):
            return embedding_chunks, hyde_embedding_chunks
        logger.error("向量检索结果为空，业务无法继续进行")
        raise ValueError("向量检索结果为空，业务无法继续进行")
    return embedding_chunks, hyde_embedding_chunks


def rrf_fuse(chunks_list, limit, k):
    # 1. chunk_id : score   chunk_id : chunk
    score_dict: dict[str, float] = {}
    chunk_dict: dict[str, dict] = {}

    # 2.遍历chunks_list
    # chunks_list = [(0.5, embedding_chunks),(0.5, hyde_embedding_chunks)]
    for weight, chunks in chunks_list:
        for rank, chunk in enumerate(chunks):
            chunk_id = chunk["chunk_id"]
            # 3. 公式：1 / (rank + k)
            score_dict[chunk_id] = score_dict.get(chunk_id, 0) + (weight * (1 / (rank + k)))
            chunk_dict[chunk_id] = chunk

    # 4.处理chunk列表并排序
    chunk_list = [
        {**chunk_dict[chunk_id], "score": score}
        for chunk_id, score in score_dict.items()
    ]
    chunk_list.sort(key=lambda x: x["score"], reverse=True)

    return chunk_list[:limit]

@step_log()
def fuse_by_rrf(state: QueryGraphState) -> QueryGraphState:
    """
    RRF 融合服务：
    1. 合并来自不同检索源的文档列表
    2. 应用 RRF 算法消除分数差异
    3. 给出综合排名最高的文档列表（Top 10）
    4. 回写 rrf_chunks
    """
    # 1.参数校验
    embedding_chunks, hyde_embedding_chunks = check_params(state)

    # 2.封装带有权重的结构；阶段 1 允许单路召回，避免 HyDE 或普通召回为空时中断流程。
    active_chunk_lists = [
        chunks
        for chunks in (embedding_chunks, hyde_embedding_chunks)
        if chunks
    ]
    if not active_chunk_lists:
        state["rrf_chunks"] = []
        return state
    weight = 1 / len(active_chunk_lists)
    chunks_list = [(weight, chunks) for chunks in active_chunk_lists]

    # 3.使用RRF算法重构文档列表
    rrf_chunks = rrf_fuse(chunks_list, limit=5, k=60)

    # 4.回写
    state["rrf_chunks"] = rrf_chunks
    return state
