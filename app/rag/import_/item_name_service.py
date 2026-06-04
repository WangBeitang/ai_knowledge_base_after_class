from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.infra.llm.providers import llm_provider
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import ITEM_NAME_CONTEXT_CHUNK_K, ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger


def validate_chunks_and_title(state):
    # 1.参数获取
    chunks, file_title = state.get("chunks"), state.get("file_title")

    # 2.判空
    if not chunks:
        logger.error("chunks为空，无法进行主体识别！")
        raise ValueError("chunks为空，无法进行主体识别！")
    if not file_title:
        file_title = chunks[0].get("file_title") or chunks[0].get("title") or "default_file_title"
        state["file_title"] = file_title

    return chunks, file_title


def build_document_context(chunks, file_title):
    # 1.获取数据
    chunks = chunks[:ITEM_NAME_CONTEXT_CHUNK_K]

    # 2.拼接上下文
    context = "".join([chunk.get("content") for chunk in chunks])

    # 3.限制最大长度
    context = context[:ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS]

    return context


def recognize_item_name(context, file_title):
    # 1.获取llm
    llm = llm_provider.chat()

    # 2.llm请求参数组装
    # 2.1 加载提示词模板，组装提示词
    system_prompt = load_prompt("product_recognition_system")
    human_prompt = load_prompt("item_name_recognition", file_title=file_title, context=context)

    # 2.2 构造消息列表
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]
    chains = llm | StrOutputParser()

    # 3.调用llm, 获取结果
    result = chains.invoke(messages)

    # 4.判空,file_title作为兜底
    if not result:
        result = file_title

    return result


def recognize_and_index_item_name(state: ImportGraphState) -> ImportGraphState:
    """
    主体识别服务：
    1. 基于 chunks 构造上下文
    2. 调用 LLM 识别 item_name
    3. 将 item_name 回填到 state 和 chunks
    4. 同步写入主体名称索引
    """
    # 1.参数校验
    chunks, file_title = validate_chunks_and_title(state)

    # 2.基于chunks构造上下文
    context = build_document_context(chunks, file_title)

    # 3.调用LLM识别item_name
    item_name = recognize_item_name(context, file_title)
    logger.warning(f"识别结果：{item_name}")

    return state