import re
from app.infra.llm.providers import llm_provider
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.utils.task_utils import push_to_session, set_task_result
from app.shared.utils.sse_utils import SSEEvent
from app.shared.runtime.logger import logger
from app.infra.persistence.history_repository import history_repository


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
