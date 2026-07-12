import re
from app.infra.llm.providers import llm_provider
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.utils.task_utils import push_to_session, set_task_result
from app.shared.utils.sse_utils import SSEEvent
from app.shared.runtime.logger import logger
from app.infra.persistence.history_repository import history_repository
from app.rag.query.contracts import IdentifierResolutionStatus, PlannerReasonCode, RetrievalObservation
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

    if len(reranked_docs) == 0 or len(standard_subject_names) == 0 or not rewritten_query:
        logger.error("reranked_docs或standard_subject_names或rewritten_query为空")
        raise ValueError("reranked_docs或standard_subject_names或rewritten_query为空")

    history = state.get("history", [])
    return reranked_docs, standard_subject_names, rewritten_query, history


def build_answer_prompt(reranked_docs, rewritten_query, standard_subject_names, history):
    # 构建Prompt context history standard_subject_names question
    # 1.拼接context
    context = ""
    for doc in reranked_docs:
        context += (f"标题: {doc['title']},来源：{'联网查询' if doc['type'] == 'web' else '向量数据库'} ,"
                    f"reranker模型评分：{doc['score']}，\n内容：{doc['text']}\n\n")

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
    standard_subject_names_text = ",".join(standard_subject_names)

    # 4.加载提示词
    prompt_text = load_prompt("answer_out", context=context, history=history_text,
                              standard_subject_names=standard_subject_names_text, question=rewritten_query)

    return prompt_text


def generate_final_answer(state, prompt):
    final_answer = ""
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
    else:
        result = client.invoke(prompt)
        logger.warning(f"大模型invoke返回结果为：======={str(result)}")
        final_answer = result.content

    state["answer"] = final_answer


def extract_image_urls(reranked_docs, state):
    # 1.定义一个正则 匹配 markdown 图片正则
    reg = re.compile(r"\!\[.*?\]\((.*?)\)")

    # 2.定义存储数据的列表
    image_urls: list[str] = []

    # 3.循环
    for doc in reranked_docs:
        url = doc.get("url", "")
        text = doc.get("text", "")

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
    history_repository.save_message(
        session_id=state.get("session_id"),
        role="assistant",
        text=state.get("answer"),
        rewritten_query=state.get("rewritten_query"),
        standard_subject_names=state.get("standard_subject_names", []),
        image_urls=state.get("image_urls", [])
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
        # 构建提示词
        prompt = build_answer_prompt(reranked_docs, rewritten_query, standard_subject_names, history)
        # 生成答案
        generate_final_answer(state, prompt)
        # 提取图片 URL
        extract_image_urls(reranked_docs, state)

    # 保存助手消息到历史
    save_assistant_message(state)
    return state
