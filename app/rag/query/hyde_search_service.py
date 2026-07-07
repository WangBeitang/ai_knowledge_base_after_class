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
from app.shared.runtime.load_prompt import load_prompt
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


def query_chunk_by_milvus(subject_ids, subject_names, rewritten_query, hyde_answer):
    # 1.向量化问题
    embedding_result = llm_provider.embed_documents([rewritten_query + ":" + hyde_answer])
    dense_vector = embedding_result["dense"][0]
    sparse_vector = embedding_result["sparse"][0]

    # 2.reqs
    reqs = milvus_gateway.create_requests(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        expr=build_subject_filter_expr(subject_ids=subject_ids, subject_names=subject_names),
    )

    # 3.执行混合搜索
    hybrid_result = milvus_gateway.hybrid_search(
        collection_name=milvus_gateway.chunk_collection_name,
        reqs=reqs,
        ranker_weights=RETRIEVAL_RANKER_WEIGHTS,
        limit=RETRIEVAL_DEFAULT_LIMIT,
        output_fields=CHUNK_OUTPUT_FIELDS,
    )

    # 4.格式化结果
    if hybrid_result[0] and len(hybrid_result[0]) > 0:
        return [
            format_chunk_search_item(item, source_type="hyde")
            for item in hybrid_result[0]
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
    subject_ids, subject_names, rewritten_query = check_params(state)

    # 2.根据问题调用模型生成假设性答案
    hyde_answer = generate_hyde_answer(rewritten_query)

    # 2.混合检索
    hyde_embedding_chunks = query_chunk_by_milvus(subject_ids, subject_names, rewritten_query, hyde_answer)

    # 3.结果回写
    state["hyde_embedding_chunks"] = hyde_embedding_chunks
    return state
