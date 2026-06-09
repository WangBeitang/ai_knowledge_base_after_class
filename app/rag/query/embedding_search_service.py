from app.infra.llm.providers import llm_provider
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import RETRIEVAL_RANKER_WEIGHTS, RETRIEVAL_DEFAULT_LIMIT
from app.shared.runtime.logger import logger,step_log


def check_params(state):
    item_names = state.get("item_names", [])
    rewritten_query = state.get("rewritten_query")
    if len(item_names) == 0:
        logger.error("主体名称为空，业务无法继续进行")
        raise ValueError("请输入商品名称")
    if not rewritten_query:
        logger.error("请输入问题")
        raise ValueError("请输入问题")
    logger.warning(f"{rewritten_query},类型{type(rewritten_query)}")
    return item_names, rewritten_query


def query_chunk_by_milvus(item_names, rewritten_query):
    # 1.向量化问题
    embedding_result = llm_provider.embed_documents([rewritten_query])
    dense_vector = embedding_result["dense"][0]
    sparse_vector = embedding_result["sparse"][0]

    # 2.reqs
    reqs = milvus_gateway.create_requests(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        expr=f"item_name in {item_names}",
    )

    # 3.执行混合搜索
    hybrid_result = milvus_gateway.hybrid_search(
        collection_name=milvus_gateway.chunk_collection_name,
        reqs=reqs,
        ranker_weights=RETRIEVAL_RANKER_WEIGHTS,
        limit=RETRIEVAL_DEFAULT_LIMIT,
        output_fields=["chunk_id", "item_name", "content", "title", "parent_title", "part", "file_title"],
    )

    # 4.格式化结果
    if hybrid_result[0] and len(hybrid_result[0]) > 0:
        return [
            {
                "chunk_id": item.get("id") or item.get("entity", {}).get("chunk_id"),
                "item_name": item.get("entity", {}).get("item_name"),
                "content": item.get("entity", {}).get("content"),
                "title": item.get("entity", {}).get("title"),
                "parent_title": item.get("entity", {}).get("parent_title"),
                "part": item.get("entity", {}).get("part"),
                "file_title": item.get("entity", {}).get("file_title"),
                "score": item.get("distance", 0.0),
                "type": "milvus",  # 来源类型（向量库）
                "url": None,
            }
            for item in hybrid_result[0]
        ]
    return []




@step_log()
def search_by_embedding(state: QueryGraphState) -> QueryGraphState:
    """
    向量检索服务：
    1. 根据改写后的问题和限定的商品范围
    2. 利用 BGEM3 混合检索（稠密+稀疏）技术
    3. 从 Milvus 向量数据库中召回 Top-K 最相关的知识切片
    4. 回写 embedding_chunks
    """
    # 1.参数校验
    item_names,rewritten_query = check_params(state)

    # 2.混合检索
    embedding_chunks = query_chunk_by_milvus(item_names, rewritten_query)

    # 3.结果回写
    state["embedding_chunks"] = embedding_chunks
    return state