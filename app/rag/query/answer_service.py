import re
from time import perf_counter

from app.infra.llm.providers import llm_provider
from app.infra.config.providers import infra_config
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.utils.task_utils import push_to_session, set_task_result
from app.shared.utils.sse_utils import SSEEvent
from app.shared.runtime.logger import logger
from app.infra.persistence.history_repository import history_repository
from app.rag.query.citation_service import build_citations
from app.rag.query.contracts import Citation, IdentifierResolutionStatus, PlannerReasonCode, RetrievalObservation
from app.rag.query.query_identifier_service import identifier_requires_clarification


def apply_identifier_clarification_guard(state) -> bool:
    """
    在答案 LLM 前拦截“相近编号待确认”和“编号未找到”两种状态。

    guard 的中文含义是“安全保护”。当前 Planner 尚未接入固定 LangGraph，因此检索节点
    不能直接路由到 ask_clarification；本函数作为阶段 5 分步实施期间的兼容保护，只把
    Observation 中确定性生成的追问交付给用户，并清空图片/引用。它不会把候选 chunk
    送入答案 Prompt。任务 9 接入 Planner 后，这条规则仍可保留为答案出口的最后防线。
    """
    observation = state.get("retrieval_observation")
    if not identifier_requires_clarification(observation):
        return False

    if isinstance(observation, RetrievalObservation):
        resolution_status = observation.identifier_resolution_status
        clarification_question = observation.clarification_question
    else:
        resolution_status = observation.get("identifier_resolution_status")
        clarification_question = observation.get("clarification_question")

    if not clarification_question:
        raise ValueError("编号需要确认时 RetrievalObservation.clarification_question 不能为空")

    state["answer"] = clarification_question
    state["clarification_question"] = clarification_question
    state["image_urls"] = []
    state["citations"] = []
    state["terminal_reason_code"] = (
        PlannerReasonCode.IDENTIFIER_CONFIRMATION_REQUIRED
        if resolution_status in {
            IdentifierResolutionStatus.SUGGESTION_REQUIRED,
            IdentifierResolutionStatus.SUGGESTION_REQUIRED.value,
        }
        else PlannerReasonCode.IDENTIFIER_NOT_FOUND
    )
    return True


def try_return_existing_answer(state):
    answer = state.get("answer")
    is_stream = state.get("is_stream", False)
    session_id = state.get("session_id")

    if not answer:
        return False

    if is_stream:
        for ch in answer:
            push_to_session(session_id, SSEEvent.DELTA, {"delta": ch})
    set_task_result(session_id, "answer", answer)
    return True


def check_params(state):
    reranked_docs = state.get("reranked_docs") or []
    standard_subject_names = state.get("standard_subject_names", [])
    rewritten_query = state.get("rewritten_query", "")

    if len(reranked_docs) == 0 or not rewritten_query:
        logger.error("reranked_docs或rewritten_query为空")
        raise ValueError("reranked_docs或rewritten_query为空")
    # 明显实时问题在未来 Planner 中可以直接 Web，因此全 Web 证据不强制要求本地主题名。
    # 只要最终候选含本地 chunk，就仍要求 standard_subject_names，避免本地答案失去主体。
    has_local_candidate = any(doc.get("source_type") == "local" for doc in reranked_docs)
    if has_local_candidate and len(standard_subject_names) == 0:
        logger.error("本地 reranked_docs 存在时 standard_subject_names 不能为空")
        raise ValueError("本地 reranked_docs 存在时 standard_subject_names 不能为空")

    history = state.get("history", [])
    return reranked_docs, standard_subject_names, rewritten_query, history


def build_answer_prompt(reranked_docs, rewritten_query, standard_subject_names, history):
    # 构建Prompt context history standard_subject_names question
    # 1.拼接context
    context = ""
    for doc in reranked_docs:
        # reranked_docs 从阶段 5 第六部分起保持统一 RetrievalCandidate 结构。本地/Web
        # 只通过 source_type 区分，正文和 rerank 分数不再使用容易混淆的 text/type/score。
        context += (
            f"标题: {doc['title']},来源："
            f"{'联网查询' if doc['source_type'] == 'web' else '本地知识库'},"
            f"reranker模型评分：{doc['rerank_score']}，\n内容：{doc['content']}\n\n"
        )

    # 2.拼接history
    history_text = ""
    if len(history) > 0:
        for index, item in enumerate(history, start=1):
            standard_names = ",".join(item.get("standard_subject_names") or [])
            history_text += (f"序号:{index},类型:{'提问' if item['role'] == 'user' else '回答'},"
                             f"内容:{item['rewritten_query'] if item['role'] == 'user' else item['text']},"
                             f"关联标准主题:{standard_names or '未记录'}\n")
    else:
        history_text = "无历史对话记录"

    # 3.拼接标准主题名
    standard_subject_names_text = ",".join(standard_subject_names) or "未限定本地标准主题"

    # 4.加载提示词
    prompt_text = load_prompt("answer_out", context=context, history=history_text,
                              standard_subject_names=standard_subject_names_text, question=rewritten_query)

    return prompt_text


ANSWER_PROMPT_VERSION = "answer-out-v1"


def _extract_usage_metadata(message) -> dict[str, int]:
    """兼容不同 LangChain provider 的 token 用量字段；缺失时诚实返回 0。"""
    usage = getattr(message, "usage_metadata", None) or {}
    response_metadata = getattr(message, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
    input_tokens = int(usage.get("input_tokens") or token_usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or token_usage.get("completion_tokens") or 0)
    total_tokens = int(
        usage.get("total_tokens")
        or token_usage.get("total_tokens")
        or input_tokens + output_tokens
    )
    return {
        "input_tokens": max(0, input_tokens),
        "output_tokens": max(0, output_tokens),
        "total_tokens": max(0, total_tokens),
    }


def generate_final_answer(state, prompt):
    final_answer = ""
    usage_metadata = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    started_at = perf_counter()
    # 1.获取模型对象
    client = llm_provider.chat()

    # 2.判断是否流式调用
    is_stream = state.get("is_stream", False)
    if is_stream:
        stream = client.stream(prompt)
        for chunk in stream:
            logger.warning(f"大模型流式返回结果为：======={str(chunk)}")
            current_content = chunk.content
            push_to_session(
                state.get("session_id"),
                SSEEvent.DELTA,
                {"delta": current_content}
            )
            final_answer += current_content
            chunk_usage = _extract_usage_metadata(chunk)
            # 流式 provider 通常只在最后一个 chunk 返回累计 usage；取最大值可以兼容累计
            # 和仅末包两种形态，又不会把累计 token 在每个 chunk 上重复相加。
            for key, value in chunk_usage.items():
                usage_metadata[key] = max(usage_metadata[key], value)
    else:
        result = client.invoke(prompt)
        logger.warning(f"大模型invoke返回结果为：======={str(result)}")
        final_answer = result.content
        usage_metadata = _extract_usage_metadata(result)

    state["answer"] = final_answer
    state["answer_runtime_metadata"] = {
        # 当前客户端遵循 OpenAI-compatible 接口，但实际服务商可能是本地或云端代理；不从
        # base_url 猜厂商名称，避免 Trace 把兼容协议误记为真实 provider。
        "provider": "openai-compatible",
        "model_id": infra_config.llm.llm_model,
        "model_revision": None,
        "prompt_version": ANSWER_PROMPT_VERSION,
        **usage_metadata,
        "duration_ms": max(0, int((perf_counter() - started_at) * 1000)),
        "estimated_cost": 0.0,
        "currency": "CNY",
    }


def extract_image_urls(reranked_docs, state):
    # 1.定义一个正则 匹配 markdown 图片正则
    reg = re.compile(r"\!\[.*?\]\((.*?)\)")

    # 2.定义存储数据的列表
    image_urls: list[str] = []

    # 3.循环
    for doc in reranked_docs:
        url = doc.get("url", "")
        text = doc.get("content", "")

        # 提取url
        if url and url.endswith((".jpg", ".png", ".gif", ".jpeg", ".svg")) and url not in image_urls:
            image_urls.append(url)

        # 提取text
        for url in reg.findall(text):
            if url not in image_urls:
                image_urls.append(url)

    # 4.给state赋值
    state["image_urls"] = image_urls
    return state


def save_assistant_message(state):
    if not state.get("history_persistence_enabled", True):
        return
    citations = [
        item if isinstance(item, Citation) else Citation.model_validate(item)
        for item in state.get("citations") or []
    ]
    history_repository.save_message(
        user_id=str(state.get("owner_user_id") or ""),
        session_id=state.get("session_id"),
        role="assistant",
        text=state.get("answer"),
        rewritten_query=state.get("rewritten_query"),
        standard_subject_names=state.get("standard_subject_names", []),
        image_urls=state.get("image_urls", []),
        citations=[item.model_dump(mode="json") for item in citations],
        trace_id=state.get("trace_id", ""),
        terminal_reason_code=(
            state.get("terminal_reason_code").value
            if hasattr(state.get("terminal_reason_code"), "value")
            else str(state.get("terminal_reason_code") or "")
        ),
    )


def generate_answer(state: QueryGraphState) -> QueryGraphState:
    """
    答案生成服务：
    1. 检查前置答案（如有追问或拒绝回答，直接输出）
    2. 构建 Prompt（用户问题 + 历史对话 + TopK 文档）
    3. 调用 LLM 生成最终答案（支持流式推送）
    4. 从引用文档中提取图片 URL
    5. 写入 MongoDB 历史记录
    6. 回写 answer 和 image_urls
    """
    ""
    # 0.编号候选尚未被用户确认时先触发安全保护。保护会写入已有 answer，随后复用统一的
    # 流式/非流式交付逻辑；因此不会调用答案 LLM，也不会生成 Citation。
    apply_identifier_clarification_guard(state)

    # 1.如果已有答案，直接返回
    if not try_return_existing_answer(state):
        # 校验输入
        reranked_docs, standard_subject_names, rewritten_query, history = check_params(state)
        # citations 只从即将进入 Prompt 的最终 reranked_docs 生成，不能把所有召回候选或
        # LLM 在答案中自行提到的来源当成正式引用。
        state["citations"] = build_citations(reranked_docs)
        # 构建提示词
        prompt = build_answer_prompt(reranked_docs, rewritten_query, standard_subject_names, history)
        state["prompt"] = prompt
        # 生成答案
        generate_final_answer(state, prompt)
        # 提取图片 URL
        extract_image_urls(reranked_docs, state)

    # 保存助手消息到历史
    save_assistant_message(state)
    return state
