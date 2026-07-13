from app.infra.llm.providers import llm_provider
from app.infra.persistence.history_repository import history_repository
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import QUERY_HISTORY_LIMIT, SUBJECT_NAME_CONFIRM_THRESHOLD, SUBJECT_NAME_CANDIDATE_THRESHOLD, \
    SUBJECT_NAME_OPTIONS_TOPK
from app.rag.query.contracts import SubjectResolutionStatus
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

    # 历史记录现在不再依赖旧主体名称字段。
    # 这里只拼接最近对话内容，供 LLM 做必要的指代消解；如果历史已经清空，则自然返回空文本。
    for index, message in enumerate(history_messages):
        is_user = message.get("role") == "user"
        content = message.get("rewritten_query") if is_user else message.get("text")
        message_type = "问题" if is_user else "答案"

        result.append(f"第{index+1}条消息记录，类型为{message_type}，内容如下：{content}")

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
    if "subject_mentions" not in result_dict:
        result_dict["subject_mentions"] = []
    if "rewritten_query" not in result_dict:
        result_dict["rewritten_query"] = original_query

    return result_dict["rewritten_query"], result_dict["subject_mentions"]


def search_subject_alias_in_milvus(subject_mentions):
    """
    在阶段 2 的别名索引中确认标准主题。

    输入的 subject_mentions 来自问题改写/主体抽取 LLM，它可能是标准名，也可能是用户口语化别名。
    因此这里把用户输入的主体文本向量化后，检索 subject_alias_collection。
    命中 alias 后，记录里会直接返回标准主题关联信息：
    - subject_id：后续 chunk 检索优先使用的稳定过滤键。
    - standard_subject_name：展示、Prompt 和历史记录使用的标准名称。

    返回结构保留“按用户输入分组”，便于后续判断每个输入是否确认或需要反问。
    """
    result_dict = {}

    for subject_mention in subject_mentions:
        # 1.用户输入可能是别名，先生成稠密+稀疏向量用于混合召回
        embed_result = llm_provider.embed_documents([subject_mention])
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

        result_dict[subject_mention] = alias_results
        logger.info(f"主体别名[{subject_mention}]搜索结果：{alias_results}")

    return result_dict


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
    阈值沿用原主体确认逻辑，后续可以根据 alias 集合的真实召回分布单独调参。
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
    把 alias 确认结果写入查询 state。

    - subject_ids：后续 chunk 检索优先使用，稳定且不依赖展示名。
    - standard_subject_names：标准主题名，用于展示、日志和 Prompt。
    """
    standard_subject_names = [record["standard_subject_name"] for record in confirmed_records]
    state["subject_ids"] = [record["subject_id"] for record in confirmed_records]
    state["standard_subject_names"] = standard_subject_names
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

    # 4.问题重写和主体提及识别
    rewritten_query, subject_mentions = query_rewrite_and_subject_name_recognition(original_query, history_text)
    state["rewritten_query"] = rewritten_query

    # 5. 先初始化结构化主体输出。阶段 9 起主体节点不再把追问/未找到文本偷偷写进 answer；
    # Planner 只读取下面这些事实，决定 ask_clarification、refuse 或继续检索。
    state["subject_ids"] = []
    state["standard_subject_names"] = []
    state["subject_candidates"] = []
    state["clarification_question"] = None

    # 6. Milvus 中查询和确认主体（前提：LLM 识别到主体提及）。
    if subject_mentions and len(subject_mentions) > 0:
        # 6.1 先查阶段2别名索引：用户输入别名 -> subject_id / standard_subject_name
        search_result_dict = search_subject_alias_in_milvus(subject_mentions)

        # 6.2 根据评分分类：确认列表 / 可选列表。
        confirmed_records, candidate_list = classify_subject_aliases(search_result_dict)

        # 6.3 只写结构化结果，不提前生成最终 answer。
        if confirmed_records and len(confirmed_records) > 0:
            apply_confirmed_subjects_to_state(state, confirmed_records, rewritten_query)
            state["subject_resolution_status"] = SubjectResolutionStatus.CONFIRMED
        elif candidate_list and len(candidate_list) > 0:
            state["subject_resolution_status"] = SubjectResolutionStatus.AMBIGUOUS
            state["subject_candidates"] = list(candidate_list)
            state["clarification_question"] = f"请问您想咨询的是[{','.join(candidate_list)}]吗？"
        else:
            state["subject_resolution_status"] = SubjectResolutionStatus.NOT_FOUND
    else:
        # no_mention 表示问题和历史中都没有可确认主体。它和“提到了但库里没找到”不同，
        # Planner 会选择追问，而不是把未限定主体的问题直接扩成全库搜索。
        state["subject_resolution_status"] = SubjectResolutionStatus.NO_MENTION
        state["clarification_question"] = "请说明您要咨询的设备型号或标准设备名称。"

    # 7. 所有正常主体结果都必须经过同一个用户消息保存点。旧实现的 confirmed/ambiguous/
    # not_found 分支会提前 return，导致用户问题漏存；现在先完成状态分类，再统一落历史。
    history_repository.save_message(
        session_id=session_id,
        role="user",
        text=original_query,
        rewritten_query=state.get("rewritten_query", original_query),
        standard_subject_names=state.get("standard_subject_names", [])
    )
    return state
