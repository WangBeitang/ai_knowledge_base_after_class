from app.infra.llm.providers import llm_provider
from app.infra.persistence.history_repository import history_repository
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import QUERY_HISTORY_LIMIT, SUBJECT_NAME_CONFIRM_THRESHOLD, SUBJECT_NAME_CANDIDATE_THRESHOLD, \
    SUBJECT_NAME_OPTIONS_TOPK
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

    # 1.剔除subject_names为空的历史消息
    history_messages = [message for message in history_messages if message.get("subject_names")]

    # 2.消息拼接
    for index, message in enumerate(history_messages):
        is_user = message.get("role") == "user"
        content = message.get("rewritten_query") if is_user else message.get("text")
        type = "问题" if is_user else "答案"
        subject_names = ",".join(message.get("subject_names", []))

        result.append(f"第{index+1}条消息记录，类型为{type}，主体名为{subject_names}，内容如下：{content}")

    return "\n".join(result)


def query_rewrite_and_subject_name_recognition(original_query, history_text):
    from langchain_core.messages import HumanMessage
    from langchain_core.output_parsers import JsonOutputParser

    result_dict = {}

    # 1.获取llm客户端
    llm_client = llm_provider.chat(json_mode=True)

    # 2.加载并渲染提示词模板
    prompt = load_prompt("rewritten_query_and_subject_names", history_text=history_text, query=original_query)

    # 3.模型调用
    chain = llm_client | JsonOutputParser()
    result_dict = chain.invoke([HumanMessage(content=prompt)])

    # 4.结果处理
    if "subject_names" not in result_dict:
        result_dict["subject_names"] = []
    if "rewritten_query" not in result_dict:
        result_dict["rewritten_query"] = original_query

    return result_dict["rewritten_query"], result_dict["subject_names"]


def search_subject_name_in_milvus(subject_names):
    result_dict = {}

    for subject_name in subject_names:
        # 1.生成稠密+稀疏向量
        embed_result = llm_provider.embed_documents([subject_name])
        dense_vector = embed_result["dense"][0]
        sparse_vector = embed_result["sparse"][0]

        # 2. 创建搜索请求
        reqs = milvus_gateway.create_requests(dense_vector, sparse_vector,limit=10)

        # 3.执行混合搜索
        hybrid_result = milvus_gateway.hybrid_search(
            collection_name=milvus_gateway.subject_name_collection,
            reqs=reqs,
            ranker_weights=(0.6, 0.4),
            limit=5,
            output_fields=["subject_name"],
        )

        # 4.搜索结果解析 [[{id, distance, entity:{subject_name:}}]]
        subject_name_results = []
        if hybrid_result and hybrid_result[0]:
            for item in hybrid_result[0]:
                score = item.get("distance", 0)
                name = item.get("entity", {}).get("subject_name", "")
                subject_name_results.append({"subject_name": name, "score": score})

        result_dict[subject_name] = subject_name_results
        logger.info(f"主体[{subject_name}]搜索结果：{subject_name_results}")

    return result_dict


def search_subject_alias_in_milvus(subject_names):
    """
    在阶段 2 的别名索引中确认标准主题。

    输入的 subject_names 仍来自问题改写/主体抽取 LLM，它可能是标准名，也可能是用户口语化别名。
    因此这里不再直接查旧 subject_name_collection，而是把用户输入的主体文本向量化后，
    检索 subject_alias_collection。命中 alias 后，记录里会直接返回标准主题关联信息：
    - subject_id：后续 chunk 检索优先使用的稳定过滤键。
    - standard_subject_name：展示、Prompt 和兼容 subject_names 使用的标准名称。

    返回结构保留“按用户输入分组”，便于后续判断每个输入是否确认或需要反问。
    """
    result_dict = {}

    for subject_name in subject_names:
        # 1.用户输入可能是别名，先生成稠密+稀疏向量用于混合召回
        embed_result = llm_provider.embed_documents([subject_name])
        dense_vector = embed_result["dense"][0]
        sparse_vector = embed_result["sparse"][0]

        # 2.别名集合不需要过滤条件，直接在全部 alias 中找最相近项
        reqs = milvus_gateway.create_requests(dense_vector, sparse_vector, limit=10)

        # 3.执行混合搜索，返回 alias 以及其映射到的标准主题信息
        hybrid_result = milvus_gateway.hybrid_search(
            collection_name=milvus_gateway.subject_alias_collection,
            reqs=reqs,
            ranker_weights=(0.6, 0.4),
            limit=5,
            output_fields=["alias", "alias_type", "subject_id", "standard_subject_name"],
        )

        alias_results = []
        if hybrid_result and hybrid_result[0]:
            for item in hybrid_result[0]:
                entity = item.get("entity", {})
                alias_results.append(
                    {
                        "alias": entity.get("alias", ""),
                        "alias_type": entity.get("alias_type", ""),
                        "subject_id": entity.get("subject_id", ""),
                        "standard_subject_name": entity.get("standard_subject_name", ""),
                        "score": item.get("distance", 0),
                    }
                )

        result_dict[subject_name] = alias_results
        logger.info(f"主体别名[{subject_name}]搜索结果：{alias_results}")

    return result_dict


def classify_subject_names(search_result_dict):
    confirmed_list = []
    candidate_list = []

    # 数据格式：{subject_name_llm: [{subject_name:xxx,score:xx},...],...}
    for subject_name_llm, milvus_results in search_result_dict.items():
        high_score_subject_name_list = [
            item_dict.get("subject_name")
            for item_dict in milvus_results
            if item_dict.get("score") > SUBJECT_NAME_CONFIRM_THRESHOLD
        ]

        mid_score_subject_name_list = [
            item_dict.get("subject_name")
            for item_dict in milvus_results
            if SUBJECT_NAME_CANDIDATE_THRESHOLD < item_dict.get("score") <= SUBJECT_NAME_CONFIRM_THRESHOLD
        ]

        if len(high_score_subject_name_list) > 0:
            confirmed_list.append(high_score_subject_name_list[0])
            continue
        if len(mid_score_subject_name_list) > 0:
            candidate_list.extend(mid_score_subject_name_list[:SUBJECT_NAME_OPTIONS_TOPK])

    return confirmed_list, candidate_list


def _dedupe_subject_records(records):
    """
    按 subject_id 去重标准主题记录。

    同一个标准主题可能有多个别名同时命中，例如“HAK180”和“HAK 180 烫金机”。
    state 里只需要保留一份 subject_id/standard_subject_name，避免后续检索重复过滤。
    """
    deduped_records = []
    seen_subject_ids = set()
    for record in records:
        subject_id = record.get("subject_id", "")
        standard_subject_name = record.get("standard_subject_name", "")
        if not subject_id or not standard_subject_name or subject_id in seen_subject_ids:
            continue
        deduped_records.append(
            {
                "subject_id": subject_id,
                "standard_subject_name": standard_subject_name,
            }
        )
        seen_subject_ids.add(subject_id)
    return deduped_records


def classify_subject_aliases(search_result_dict):
    """
    根据 alias 检索分数划分“已确认主题”和“候选主题”。

    confirmed_records 用于直接进入后续 chunk 检索。
    candidate_names 用于分数不够高但有可能命中的场景，返回给用户做二次确认。
    阈值沿用原 subject_name 逻辑，后续可以根据 alias 集合的真实召回分布单独调参。
    """
    confirmed_records = []
    candidate_names = []
    seen_candidate_names = set()

    for subject_name_llm, alias_results in search_result_dict.items():
        high_score_records = [
            item
            for item in alias_results
            if item.get("score", 0) > SUBJECT_NAME_CONFIRM_THRESHOLD
        ]
        mid_score_records = [
            item
            for item in alias_results
            if SUBJECT_NAME_CANDIDATE_THRESHOLD < item.get("score", 0) <= SUBJECT_NAME_CONFIRM_THRESHOLD
        ]

        if high_score_records:
            confirmed_records.append(high_score_records[0])
            continue

        for item in mid_score_records[:SUBJECT_NAME_OPTIONS_TOPK]:
            standard_subject_name = item.get("standard_subject_name", "")
            if not standard_subject_name or standard_subject_name in seen_candidate_names:
                continue
            candidate_names.append(standard_subject_name)
            seen_candidate_names.add(standard_subject_name)

    return _dedupe_subject_records(confirmed_records), candidate_names


def apply_confirmed_subjects_to_state(state, confirmed_records, rewritten_query):
    """
    把 alias 确认结果同时写入新旧字段。

    - subject_ids：后续 chunk 检索优先使用，稳定且不依赖展示名。
    - standard_subject_names：标准主题名，用于展示、日志和 Prompt。
    - subject_names：兼容旧字段，保持与 standard_subject_names 一致。
    """
    standard_subject_names = [record["standard_subject_name"] for record in confirmed_records]
    state["subject_ids"] = [record["subject_id"] for record in confirmed_records]
    state["standard_subject_names"] = standard_subject_names
    state["subject_names"] = standard_subject_names
    state["rewritten_query"] = rewritten_query
    return state


def confirm_subject_name(state: QueryGraphState) -> QueryGraphState:
    """
    意图确认服务：
    1. 结合历史对话提取主体名
    2. 将模糊问题改写为完整独立的精准问题
    3. 在 Milvus 向量库中进行混合搜索
    4. 根据评分高低自动对齐标准型号，或生成反问让用户手动确认
    5. 同步历史记录到 MongoDB
    """

    # 1.参数校验
    original_query, session_id = params_check(state)

    # 2. 加载历史对话
    history_messages = load_history(session_id)
    state["history"] = history_messages

    # 3.拼接历史对话文本
    history_text = build_history_text(history_messages)

    # 4.问题重写和subject_name识别
    rewritten_query, subject_names = query_rewrite_and_subject_name_recognition(original_query, history_text)

    # 5. milvus 中查询和确认 subject_name（前提：subject_names 不为空）
    if subject_names and len(subject_names) > 0:
        # 5.1 先查阶段2别名索引：用户输入别名 -> subject_id / standard_subject_name
        search_result_dict = search_subject_alias_in_milvus(subject_names)

        # 6. 根据评分分类：确认列表 / 可选列表
        confirmed_records, candidate_list = classify_subject_aliases(search_result_dict)

        # 7. state 更新
        if confirmed_records and len(confirmed_records) > 0:
            return apply_confirmed_subjects_to_state(state, confirmed_records, rewritten_query)
        if candidate_list and len(candidate_list) > 0:
            state["subject_ids"] = []
            state["standard_subject_names"] = []
            state["subject_names"] = []
            state["rewritten_query"] = rewritten_query
            state["answer"] = f"请问您想咨询的是[{','.join(candidate_list)}]吗？"
            return state
        state["subject_ids"] = []
        state["standard_subject_names"] = []
        state["subject_names"] = []
        state["rewritten_query"] = rewritten_query
        state["answer"] = "非常抱歉，没有找到匹配的答案，请重新提问。"
        return state

    # 8. 保留此次对话的历史记录
    history_repository.save_message(
        session_id=session_id,
        role="user",
        text=original_query,
        rewritten_query=state.get("rewritten_query", original_query),
        subject_names=state.get("subject_names", [])
    )
    return  state
