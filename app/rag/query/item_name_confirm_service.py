from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from pymilvus.milvus_client import milvus_client

from app.infra.llm.providers import llm_provider
from app.infra.persistence.history_repository import history_repository
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import QUERY_HISTORY_LIMIT, ITEM_NAME_CONFIRM_THRESHOLD, ITEM_NAME_CANDIDATE_THRESHOLD, \
    ITEM_NAME_OPTIONS_TOPK
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import step_log,logger

@step_log()
def params_check(state):
    original_query = state.get("original_query")
    session_id = state.get("session_id")
    if not original_query:
        logger.error("请输入问题")
        raise ValueError("请输入问题")
    if not session_id:
        logger.error("请输入会话ID")
        raise ValueError("请输入会话ID")
    return original_query, session_id


def load_history(session_id):
    return history_repository.list_recent(session_id,QUERY_HISTORY_LIMIT)


def build_history_text(history_messages):
    result = []

    # 1.剔除item_names为空的历史消息
    history_messages = [message for message in history_messages if message.get("item_name") and len(message.get("item_name") > 0)]

    # 2.消息拼接
    for index, message in enumerate(history_messages):
        is_user = message.get("role") == "user"
        content = message.get("rewritten_query") if is_user else message.get("text")
        type = "问题" if is_user else "答案"
        item_names = ",".join(message.get("item_names"))

        result.append(f"第{index+1}条消息记录，类型为{type}，主体名为{item_names}，内容如下：{content}")

    return "\n".join(result)


def query_rewrite_and_item_name_recognition(original_query, history_text):
    result_dict = {}

    # 1.获取llm客户端
    llm_client = llm_provider.chat(json_mode=True)

    # 2.加载并渲染提示词模板
    prompt = load_prompt("rewritten_query_and_itemnames",history_text=history_text,query=original_query)

    # 3.模型调用
    chain = llm_client | JsonOutputParser()
    result_dict = chain.invoke([HumanMessage(content=prompt)])

    # 4.结果处理
    if "item_names" not in result_dict:
        result_dict["item_names"] = []
    if "rewritten_query" not in result_dict:
        result_dict["rewritten_query"] = original_query

    return result_dict["rewritten_query"], result_dict["item_names"]


def search_item_name_in_milvus(item_names):
    result_dict = {}

    for item_name in item_names:
        # 1.生成稠密+稀疏向量
        embed_result = llm_provider.embed_documents([item_name])
        dense_vector = embed_result["dense"][0]
        sparse_vector = embed_result["sparse"][0]

        # 2. 创建搜索请求
        reqs = milvus_gateway.create_requests(dense_vector, sparse_vector,limit=10)

        # 3.执行混合搜索
        hybrid_result = milvus_gateway.hybrid_search(
            collection_name=milvus_gateway.item_name_collection,
            reqs=reqs,
            ranker_weights=(0.6, 0.4),
            limit=5,
        )

        # 4.搜索结果解析 [[{id, distance, entity:{item_name:}}]]
        item_name_results = []
        if hybrid_result and hybrid_result[0]:
            for item in hybrid_result[0]:
                score = item.get("distance", 0)
                name = item.get("entity", {}).get("item_name", "")
                item_name_results.append({"item_name": name, "score": score})

        result_dict[item_name] = item_name_results
        logger.info(f"商品[{item_name}]搜索结果：{item_name_results}")

    return result_dict


def classify_item_names(search_result_dict):
    confirmed_list = []
    candidate_list = []

    # 数据格式：{item_name_llm: [{item_name:xxx,score:xx},{item_name:xxx,score:xx},...],item_name_llm: [{item_name:xxx,score:xx},{item_name:xxx,score:xx},...],...}
    for item_name_llm, milvus_results in search_result_dict.items():
        high_score_item_name_list = [
            item_dict.get("item_name")
            for item_dict in milvus_results
            if item_dict.get("score") > ITEM_NAME_CONFIRM_THRESHOLD
        ]

        mid_score_item_name_list = [
            item_dict.get("item_name")
            for item_dict in milvus_results
            if ITEM_NAME_CANDIDATE_THRESHOLD < item_dict.get("score") <= ITEM_NAME_CONFIRM_THRESHOLD
        ]

        if len(high_score_item_name_list) > 0:
            confirmed_list.append(high_score_item_name_list[0])
            continue
        if len(mid_score_item_name_list) > 0:
            candidate_list.extend(mid_score_item_name_list[:ITEM_NAME_OPTIONS_TOPK])

    return confirmed_list, candidate_list


def confirm_item_name(state: QueryGraphState) -> QueryGraphState:
    """
    意图确认服务：
    1. 结合历史对话提取商品名
    2. 将模糊问题改写为完整独立的精准问题
    3. 在 Milvus 向量库中进行混合搜索
    4. 根据评分高低自动对齐标准型号，或生成反问让用户手动确认
    5. 同步历史记录到 MongoDB
    """

    # 1.参数校验
    original_query, session_id = params_check(state)

    # 2. 加载历史对话
    history_messages = load_history(session_id)

    # 3.拼接历史对话文本
    history_text = build_history_text(history_messages)

    # 4.问题重写和item_name识别
    rewritten_query, item_names = query_rewrite_and_item_name_recognition(original_query, history_text)

    # 5. milvus 中查询和确认 item_name（前提：item_names 不为空）
    if item_names and len(item_names) > 0:
        # 5.1 在 milvus 中混合搜索每个 item_name
        search_result_dict = search_item_name_in_milvus(item_names)

        # 6. 根据评分分类：确认列表 / 可选列表
        confirmed_list, candidate_list = classify_item_names(search_result_dict)

        # 7. state 更新
        if confirmed_list and len(confirmed_list) > 0:
            state["item_names"] = confirmed_list
            state["rewritten_query"] = rewritten_query
            return  state
        if candidate_list and len(candidate_list) > 0:
            state["item_names"] = []
            state["rewritten_query"] = rewritten_query
            state["answer"] = f"请问您想咨询的是[{','.join(candidate_list)}]吗？"
            return state
        state["item_names"] = []
        state["rewritten_query"] = rewritten_query
        state["answer"] = "非常抱歉，没有找到匹配的答案，请重新提问。"
        return state

    # 8. 保留此次对话的历史记录
    history_repository.save_message(
        session_id=session_id,
        role="user",
        text=original_query,
        rewritten_query=state.get("rewritten_query", original_query),
        item_names=state.get("item_names", [])
    )
    return  state

