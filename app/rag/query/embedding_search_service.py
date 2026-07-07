from app.infra.llm.providers import llm_provider
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.query.agent.state import QueryGraphState
from app.rag.query.chunk_retrieval_utils import (
    CHUNK_OUTPUT_FIELDS,
    build_subject_filter_expr_candidates,
    format_chunk_search_item,
    resolve_subject_filter_values,
)
from app.rag.query.config import RETRIEVAL_RANKER_WEIGHTS, RETRIEVAL_DEFAULT_LIMIT
from app.shared.runtime.logger import logger,step_log


def check_params(state):
    subject_ids, subject_names = resolve_subject_filter_values(state)
    rewritten_query = state.get("rewritten_query")
    if len(subject_ids) == 0 and len(subject_names) == 0:
        logger.error("主体名称为空，业务无法继续进行")
        raise ValueError("请输入主体名称")
    if not rewritten_query:
        logger.error("请输入问题")
        raise ValueError("请输入问题")
    logger.warning(f"{rewritten_query},类型{type(rewritten_query)}")
    return subject_ids, subject_names, rewritten_query


def query_chunk_by_milvus(subject_ids, subject_names, rewritten_query):
    # 1.向量化问题
    embedding_result = llm_provider.embed_documents([rewritten_query])
    dense_vector = embedding_result["dense"][0]
    sparse_vector = embedding_result["sparse"][0]

    # 2.按优先级构造过滤表达式：
    #    新数据优先 subject_id，旧数据无召回时再 fallback 到 subject_name。
    filter_exprs = build_subject_filter_expr_candidates(subject_ids=subject_ids, subject_names=subject_names)

    # 3.执行混合搜索。有 subject_id 结果时直接返回；只有空召回才尝试旧字段兜底。
    for index, filter_expr in enumerate(filter_exprs):
        reqs = milvus_gateway.create_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            expr=filter_expr,
        )
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

        if index == 0 and len(filter_exprs) > 1:
            logger.info(f"subject_id过滤无召回，fallback到旧subject_name过滤。当前表达式：{filter_expr}")
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
    subject_ids, subject_names, rewritten_query = check_params(state)

    # 2.混合检索
    embedding_chunks = query_chunk_by_milvus(subject_ids, subject_names, rewritten_query)

    # 3.结果回写
    state["embedding_chunks"] = embedding_chunks
    return state
