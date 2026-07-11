from app.infra.llm.providers import llm_provider
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.query.agent.state import QueryGraphState
from app.rag.query.chunk_retrieval_utils import (
    CHUNK_OUTPUT_FIELDS,
    build_chunk_retrieval_filter_from_state,
    format_chunk_search_item,
)
from app.rag.query.config import RETRIEVAL_RANKER_WEIGHTS, RETRIEVAL_DEFAULT_LIMIT
from app.shared.runtime.logger import logger,step_log


def check_params(state):
    rewritten_query = state.get("rewritten_query")
    if not rewritten_query:
        logger.error("请输入问题")
        raise ValueError("请输入问题")
    # 普通检索和 HyDE 都通过共享函数读取 dataset、subject、用户和 tenant。任何必填
    # 范围为空都会在这里明确失败，不允许通过删除 expr 条件回退成全库搜索。
    filter_expr = build_chunk_retrieval_filter_from_state(state)
    logger.warning(f"{rewritten_query},类型{type(rewritten_query)}")
    return rewritten_query, filter_expr


def query_chunk_by_milvus(rewritten_query, filter_expr):
    # 1.向量化问题
    embedding_result = llm_provider.embed_documents([rewritten_query])
    dense_vector = embedding_result["dense"][0]
    sparse_vector = embedding_result["sparse"][0]

    # 2.使用共享构建器生成的完整 expr。这里不再自行拼 subject 或权限条件，避免普通检索
    # 与 HyDE/BM25 在后续修改时出现不同的访问范围。
    reqs = milvus_gateway.create_requests(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        expr=filter_expr,
    )

    # 3.执行混合搜索。
    hybrid_result = milvus_gateway.hybrid_search(
        collection_name=milvus_gateway.chunk_collection_name,
        reqs=reqs,
        ranker_weights=RETRIEVAL_RANKER_WEIGHTS,
        limit=RETRIEVAL_DEFAULT_LIMIT,
        output_fields=CHUNK_OUTPUT_FIELDS,
    )

    if hybrid_result and hybrid_result[0] and len(hybrid_result[0]) > 0:
        return [
            format_chunk_search_item(item, source_type="milvus")
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
    rewritten_query, filter_expr = check_params(state)

    # 2.混合检索
    embedding_chunks = query_chunk_by_milvus(rewritten_query, filter_expr)

    # 3.结果回写
    state["embedding_chunks"] = embedding_chunks
    return state
