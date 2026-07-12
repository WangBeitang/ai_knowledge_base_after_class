from app.infra.llm.providers import llm_provider
from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import RERANK_MAX_INPUT_TOKENS, RERANK_MIN_SUMMARY_CHARS, RERANK_SUMMARY_CHAR_RATIO, \
    RERANK_MIN_TOPK, RERANK_MAX_TOPK, RERANK_GAP_ABS, RERANK_GAP_RATIO
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger
from app.rag.query.query_identifier_service import identifier_requires_clarification


def check_params(state):
    rrf_chunks = state.get("rrf_chunks") or []
    web_search_docs = state.get("web_search_docs") or []
    rewritten_query = state.get("rewritten_query")
    if not rewritten_query:
        logger.error("请输入有效的查询")
        raise ValueError("请输入有效的查询")
    if len(rrf_chunks) == 0 and len(web_search_docs) == 0:
        logger.error("请输入有效的查询")
        raise ValueError("请输入有效的查询")
    return rrf_chunks, web_search_docs, rewritten_query


def fuse_documents(rrf_chunks, web_search_docs):
    fused_data_list = []
    # 1.处理rrf_chunks
    for chunk in rrf_chunks:
        current_data = {
            "title": chunk.get("title", ""),
            "text": chunk.get("content", ""),
            "url": None,
            "type": "milvus",
            "score": 0.0,
        }
        fused_data_list.append(current_data)

    # 2.处理web_search_docs
    for doc in web_search_docs:
        current_doc = {
            "title": doc.get("title", ""),
            "text": doc.get("snippet", ""),
            "url": doc.get("url", ""),
            "type": "web",
            "score": 0.0,
        }
        fused_data_list.append(current_doc)

    return fused_data_list


def refine_long_answer(rewritten_query, long_answer, limit):
    from langchain_core.messages import HumanMessage
    from langchain_core.output_parsers import StrOutputParser

    # 利用llm压缩超长的回答
    llm_client = llm_provider.chat()
    prompt = load_prompt("rerank_text_refine",question=rewritten_query,answer=long_answer,limit=limit)
    messages = [
        HumanMessage(content=prompt)
    ]
    chains = llm_client | StrOutputParser()
    result = chains.invoke(messages)
    return result


def build_qa_pairs(rewritten_query, fused_data_list):
    qa_pairs = []

    reranker = llm_provider.reranker_model()
    tokenizer = reranker.tokenizer
    # 对问题进行token编码（不添加特殊符号）[2123,321321,43545,6565,77675,8787878,98989,1]
    query_tokens = tokenizer.encode(rewritten_query, add_special_tokens=False)
    query_tokens_number = len(query_tokens)

    for data in fused_data_list:
        text = data["text"]
        text_tokens = tokenizer.encode(text, add_special_tokens=False)
        text_tokens_number = len(text_tokens)
        # 判断是否超长
        if (query_tokens_number + text_tokens_number + 4) > RERANK_MAX_INPUT_TOKENS:
            available_tokens = RERANK_MAX_INPUT_TOKENS - query_tokens_number - 4
            if available_tokens <= 0:
                logger.error("rewritten_query 过长，无法进入 reranker")
                raise ValueError("rewritten_query 过长，无法进入 reranker")
            limit = max(
                RERANK_MIN_SUMMARY_CHARS,
                int(available_tokens / RERANK_SUMMARY_CHAR_RATIO)
            )
            text = refine_long_answer(rewritten_query, text, limit)
        qa_pairs.append([rewritten_query, text])

    return qa_pairs


def reranker_score(qa_pairs):
    reranker = llm_provider.reranker_model()
    scores = reranker.compute_score(qa_pairs,normalize=True)
    return  scores


def fused_and_sort(fused_data_list, scores):
    if len(fused_data_list) != len(scores):
        logger.error("reranker 分数数量和文档数量不一致")
        raise ValueError("reranker 分数数量和文档数量不一致")
    for score, data in zip(scores, fused_data_list):
        data["score"] = score

    fused_data_list.sort(key=lambda x: x["score"], reverse=True)
    return fused_data_list


def dynamic_topk(sorted_data_list):
    """
    动态截断 TopK：
    1. 至少保留 RERANK_MIN_TOPK 条
    2. 最多保留 RERANK_MAX_TOPK 条
    3. 从 min_topk 后开始检查相邻分数，如果出现明显断崖，则提前截断
    """
    if not sorted_data_list:
        return []

    total = len(sorted_data_list)
    min_topk = min(RERANK_MIN_TOPK, total)
    max_topk = min(RERANK_MAX_TOPK, total)

    selected_topk = max_topk

    if max_topk < min_topk:
        return sorted_data_list

    for index in range(min_topk-1,max_topk-1):
        score_1 = sorted_data_list[index].get("score", 0.0)
        score_2 = sorted_data_list[index+1].get("score", 0.0)
        abs_score = score_1 - score_2
        ratio_score = abs_score / (score_1 + 1.0e-6)

        if ratio_score > RERANK_GAP_RATIO or abs_score > RERANK_GAP_ABS:
            selected_topk = index + 1
            break

    return sorted_data_list[:selected_topk]



def rerank_documents(state: QueryGraphState) -> QueryGraphState:
    """
    重排序服务：
    1. 合并 RRF 和 Web Search 的文档
    2. 使用 BGE Reranker 模型计算相关性得分
    3. 根据得分动态截断，智能截取 TopK
    4. 回写 reranked_docs
    """
    # 当前固定图仍会经过 rerank 节点；编号只得到相近候选或完全未找到时，候选不得进入
    # reranker 和答案证据。这里返回空 partial result，最终由答案出口交付确定性追问。
    if identifier_requires_clarification(state.get("retrieval_observation")):
        state["reranked_docs"] = []
        return state

    # 1.参数校验
    rrf_chunks, web_search_docs, rewritten_query = check_params(state)

    # 2.两路数据融合
    fused_data_list = fuse_documents(rrf_chunks, web_search_docs)

    # 3.组装问题和答案列表
    qa_pairs = build_qa_pairs(rewritten_query, fused_data_list)

    # 4.调用reranker打分
    scores = reranker_score(qa_pairs)

    # 5.融合和排序
    sorted_data_list = fused_and_sort(fused_data_list, scores)

    # 6.动态topK
    reranked_docs = dynamic_topk(sorted_data_list)

    # 7.回写state
    state["reranked_docs"] = reranked_docs

    return state
