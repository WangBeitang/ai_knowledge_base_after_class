"""对跨 Action RRF 后的统一候选执行最终相关性重排序。"""

from app.infra.llm.providers import llm_provider
from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import (
    RERANK_GAP_ABS,
    RERANK_GAP_RATIO,
    RERANK_MAX_INPUT_TOKENS,
    RERANK_MAX_TOPK,
    RERANK_MIN_SUMMARY_CHARS,
    RERANK_MIN_TOPK,
    RERANK_SUMMARY_CHAR_RATIO,
)
from app.rag.query.contracts import RetrievalCandidate
from app.rag.query.query_identifier_service import identifier_requires_clarification
from app.rag.query.rrf_service import candidate_identity
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger


def check_params(state):
    """校验累计候选和用户问题；Web 已在外层 RRF 中，不再在这里二次拼接。"""
    rrf_chunks = state.get("rrf_chunks") or []
    rewritten_query = state.get("rewritten_query")
    if not rewritten_query:
        logger.error("请输入有效的查询")
        raise ValueError("请输入有效的查询")
    if not rrf_chunks:
        logger.error("跨 Action RRF 候选为空，无法执行 rerank")
        raise ValueError("跨 Action RRF 候选为空，无法执行 rerank")
    candidates = [RetrievalCandidate.model_validate(item) for item in rrf_chunks]
    return candidates, rewritten_query


def refine_long_answer(rewritten_query, long_answer, limit):
    from langchain_core.messages import HumanMessage
    from langchain_core.output_parsers import StrOutputParser

    llm_client = llm_provider.chat()
    prompt = load_prompt("rerank_text_refine", question=rewritten_query, answer=long_answer, limit=limit)
    messages = [HumanMessage(content=prompt)]
    chains = llm_client | StrOutputParser()
    return chains.invoke(messages)


def build_qa_pairs(rewritten_query, candidates: list[RetrievalCandidate]):
    """
    使用原始用户问题和统一候选正文构造 reranker 输入。

    local、HyDE、Web 在此处使用完全相同的相关性判断，不把 RRF 分数或来源类型写入问题
    文本影响模型。超长正文只压缩本次模型输入，Candidate 中的原始 content 保持不变。
    """
    qa_pairs = []
    reranker = llm_provider.reranker_model()
    tokenizer = reranker.tokenizer
    query_tokens_number = len(tokenizer.encode(rewritten_query, add_special_tokens=False))

    for candidate in candidates:
        text = candidate.content
        text_tokens_number = len(tokenizer.encode(text, add_special_tokens=False))
        if query_tokens_number + text_tokens_number + 4 > RERANK_MAX_INPUT_TOKENS:
            available_tokens = RERANK_MAX_INPUT_TOKENS - query_tokens_number - 4
            if available_tokens <= 0:
                logger.error("rewritten_query 过长，无法进入 reranker")
                raise ValueError("rewritten_query 过长，无法进入 reranker")
            limit = max(
                RERANK_MIN_SUMMARY_CHARS,
                int(available_tokens / RERANK_SUMMARY_CHAR_RATIO),
            )
            text = refine_long_answer(rewritten_query, text, limit)
        qa_pairs.append([rewritten_query, text])
    return qa_pairs


def reranker_score(qa_pairs):
    """调用 BGE reranker 并要求归一化分数；该分数才用于证据阈值。"""
    reranker = llm_provider.reranker_model()
    scores = reranker.compute_score(qa_pairs, normalize=True)
    return [float(scores)] if isinstance(scores, (int, float)) else [float(score) for score in scores]


def attach_rerank_scores_and_sort(
        candidates: list[RetrievalCandidate],
        scores: list[float],
) -> list[dict]:
    """写入 rerank_score，同时保留候选的全部本地/Web 身份和召回元数据。"""
    if len(candidates) != len(scores):
        logger.error("reranker 分数数量和候选数量不一致")
        raise ValueError("reranker 分数数量和候选数量不一致")

    scored_candidates = []
    for candidate, score in zip(candidates, scores):
        scored_candidates.append(RetrievalCandidate.model_validate({
            **candidate.model_dump(mode="json"),
            "rerank_score": score,
        }))
    scored_candidates.sort(
        key=lambda candidate: (
            -(candidate.rerank_score or 0.0),
            candidate_identity(candidate),
        )
    )
    return [candidate.model_dump(mode="json") for candidate in scored_candidates]


def dynamic_topk(sorted_candidates: list[dict]) -> list[dict]:
    """根据归一化 rerank 分数的相邻断崖，在固定最小/最大 TopK 内截断。"""
    if not sorted_candidates:
        return []

    total = len(sorted_candidates)
    min_topk = min(RERANK_MIN_TOPK, total)
    max_topk = min(RERANK_MAX_TOPK, total)
    selected_topk = max_topk

    for index in range(min_topk - 1, max_topk - 1):
        score_1 = sorted_candidates[index].get("rerank_score") or 0.0
        score_2 = sorted_candidates[index + 1].get("rerank_score") or 0.0
        abs_score = score_1 - score_2
        ratio_score = abs_score / (score_1 + 1.0e-6)
        if ratio_score > RERANK_GAP_RATIO or abs_score > RERANK_GAP_ABS:
            selected_topk = index + 1
            break
    return sorted_candidates[:selected_topk]


def rerank_documents(state: QueryGraphState) -> QueryGraphState:
    """
    对 original/HyDE/Web 累计候选统一 rerank，并保留完整 ``RetrievalCandidate``。

    编号只得到相近候选或完全未找到时仍执行阶段 5 安全保护：直接返回空证据，不让候选
    进入 reranker 和答案 Prompt。任务 9 接入 Planner 后会由追问 Action 更早终止。
    """
    if identifier_requires_clarification(state.get("retrieval_observation")):
        state["reranked_docs"] = []
        return state

    candidates, rewritten_query = check_params(state)
    qa_pairs = build_qa_pairs(rewritten_query, candidates)
    scores = reranker_score(qa_pairs)
    sorted_candidates = attach_rerank_scores_and_sort(candidates, scores)
    state["reranked_docs"] = dynamic_topk(sorted_candidates)
    return state
