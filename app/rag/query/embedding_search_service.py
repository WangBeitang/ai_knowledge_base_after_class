from app.infra.llm.providers import llm_provider
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.query.agent.state import QueryGraphState
from app.rag.query.chunk_retrieval_utils import (
    CHUNK_OUTPUT_FIELDS,
    build_subject_filter_expr,
    format_chunk_search_item,
    resolve_subject_filter_values,
)
from app.rag.query.config import RETRIEVAL_RANKER_WEIGHTS, RETRIEVAL_DEFAULT_LIMIT
from app.shared.runtime.logger import logger,step_log


def check_params(state):
    subject_ids = resolve_subject_filter_values(state)
    rewritten_query = state.get("rewritten_query")
    if len(subject_ids) == 0:
        logger.error("标准主题ID为空，业务无法继续进行")
        raise ValueError("请输入标准主题ID")
    if not rewritten_query:
        logger.error("请输入问题")
        raise ValueError("请输入问题")
    logger.warning(f"{rewritten_query},类型{type(rewritten_query)}")
    return subject_ids, rewritten_query


def query_chunk_by_milvus(subject_ids, rewritten_query):
    # 1.向量化问题
    embedding_result = llm_provider.embed_documents([rewritten_query])
    dense_vector = embedding_result["dense"][0]
    sparse_vector = embedding_result["sparse"][0]

    # 2.阶段 2 后只按 subject_id 过滤 chunk。
    reqs = milvus_gateway.create_requests(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        expr=build_subject_filter_expr(subject_ids),
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
    subject_ids, rewritten_query = check_params(state)

    # 2.混合检索
    embedding_chunks = query_chunk_by_milvus(subject_ids, rewritten_query)

    # 3.结果回写
    state["embedding_chunks"] = embedding_chunks
    return state
