from app.infra.llm.providers import llm_provider
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.query.agent.state import QueryGraphState
from app.rag.query.chunk_retrieval_utils import (
    CHUNK_OUTPUT_FIELDS,
    build_chunk_retrieval_filter_from_state,
    format_chunk_search_item,
)
from app.rag.query.chunk_status_filter_service import get_disabled_chunk_ids_for_query
from app.rag.query.config import (
    BM25_SPARSE_FIELD,
    LEARNED_SPARSE_FIELD,
    RETRIEVAL_DEFAULT_LIMIT,
    RETRIEVAL_RRF_K,
    channels_for_retrieval_mode,
    normalize_retrieval_mode,
)
from app.rag.query.contracts import RetrievalChannel, RetrievalMode
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger,step_log


def check_params(state):
    rewritten_query = state.get("rewritten_query")
    if not rewritten_query:
        logger.error("请输入问题")
        raise ValueError("请输入问题")
    # HyDE 必须和普通检索使用同一个权限与范围入口。即使 HyDE 的向量文本不同，允许
    # 访问的 dataset、subject 和私有文档范围也绝不能变化。
    disabled_chunk_ids = get_disabled_chunk_ids_for_query(state)
    state["disabled_chunk_ids"] = disabled_chunk_ids
    filter_expr = build_chunk_retrieval_filter_from_state({
        **state,
        "disabled_chunk_ids": disabled_chunk_ids,
    })
    logger.warning(f"{rewritten_query},类型{type(rewritten_query)}")
    return rewritten_query, filter_expr


def query_chunk_by_milvus(
        rewritten_query,
        hyde_answer,
        filter_expr,
        *,
        retrieval_mode: RetrievalMode | str = RetrievalMode.DENSE_LEARNED_SPARSE,
):
    # 1.向量化问题
    hyde_query = rewritten_query + ":" + hyde_answer
    embedding_result = llm_provider.embed_documents([hyde_query])
    dense_vector = embedding_result["dense"][0]
    sparse_vector = embedding_result["sparse"][0]
    normalized_mode = normalize_retrieval_mode(retrieval_mode)

    # 2.直接复用共享构建器生成的完整 expr，不在 HyDE 通道内重复拼接过滤条件。
    reqs = milvus_gateway.create_requests(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        expr=filter_expr,
        retrieval_mode=normalized_mode.value,
        query_text=hyde_query,
        learned_sparse_field=LEARNED_SPARSE_FIELD,
        bm25_sparse_field=BM25_SPARSE_FIELD,
    )

    # 3.执行混合搜索。
    hybrid_result = milvus_gateway.hybrid_search(
        collection_name=milvus_gateway.chunk_collection_name,
        reqs=reqs,
        ranker_type="rrf",
        rrf_k=RETRIEVAL_RRF_K,
        limit=RETRIEVAL_DEFAULT_LIMIT,
        output_fields=CHUNK_OUTPUT_FIELDS,
    )

    if hybrid_result and hybrid_result[0]:
        retrieval_channels = [
            *channels_for_retrieval_mode(normalized_mode),
            RetrievalChannel.HYDE,
        ]
        return [
            format_chunk_search_item(
                item,
                retrieval_channels=retrieval_channels,
                retrieval_rank=rank,
            )
            for rank, item in enumerate(hybrid_result[0], start=1)
        ]
    return []


def generate_hyde_answer(rewritten_query):
    from langchain_core.messages import HumanMessage
    from langchain_core.output_parsers import StrOutputParser

    llm_client = llm_provider.chat()
    prompt = load_prompt("hyde_prompt",rewritten_query=rewritten_query)
    messages = [
        HumanMessage(content=prompt)
    ]
    chain = llm_client | StrOutputParser()
    result = chain.invoke(messages)
    return result


def search_by_hyde(state: QueryGraphState) -> QueryGraphState:
    """
    HyDE 检索服务：
    1. 让 LLM 基于问题虚构一个"理想答案"
    2. 对这个假设性答案进行向量化
    3. 用答案向量在 Milvus 中检索真实文档
    4. 回写 hyde_embedding_chunks
    """
    # 1.参数校验
    rewritten_query, filter_expr = check_params(state)
    retrieval_mode = normalize_retrieval_mode(state.get("retrieval_mode"))
    state["retrieval_mode"] = retrieval_mode.value

    # 2.根据问题调用模型生成假设性答案
    hyde_answer = generate_hyde_answer(rewritten_query)

    # 2.混合检索
    hyde_embedding_chunks = query_chunk_by_milvus(
        rewritten_query,
        hyde_answer,
        filter_expr,
        retrieval_mode=retrieval_mode,
    )

    # 3.结果回写
    state["hyde_embedding_chunks"] = hyde_embedding_chunks
    return state
