import copy

from typing_extensions import TypedDict

from app.rag.query.contracts import (
    Citation,
    PlannerDecision,
    PlannerHistoryItem,
    PlannerReasonCode,
    RetrievalMode,
    RetrievalObservation,
    SubjectResolutionStatus,
)
from app.rag.query.config import (
    PLANNER_MAX_STEPS,
    RETRIEVAL_CONFIG_VERSION,
    RETRIEVAL_DEFAULT_MODE,
    WEB_FALLBACK_ENABLED,
)


class QueryGraphState(TypedDict):
    """
    一次查询在 LangGraph 节点之间传递的运行时状态。

    State（状态）只在当前查询执行期间存在；除非后续由 Trace Repository 显式投影，
    否则不会自动持久化到 Mongo。节点应只返回自己修改的 partial state（局部状态），
    由 LangGraph 合并到主状态，不能把收到的整个 state 原样返回。

    ``TypedDict`` 主要提供静态类型提示，不会像 Pydantic 一样在运行时自动校验；需要
    严格校验的 Planner、Observation 和 Citation 使用阶段 5 第二部分的 Pydantic 模型。
    """

    # ==================== 请求身份与知识库范围 ====================

    # session 的中文含义是“会话”。该 ID 标识一次聊天会话，用于读取历史消息、关联
    # SSE 流式连接和查询任务进度；它不表示用户身份，也不能代替 owner_user_id。
    # 来源：POST /query 请求体。默认空字符串仅用于创建模板，真实查询必须传入或生成。
    session_id: str

    # original query 的中文含义是“用户原始问题”。保存用户未经过改写的输入，供主体
    # 确认、Trace 和最终问题对照使用；后续节点不能用 rewritten_query 覆盖该字段。
    # 来源：POST /query 的 query 字段。默认空字符串，真实查询入口会写入。
    original_query: str

    # stream 的中文含义是“流式输出”。True 表示答案通过 SSE 增量推送，False 表示同步
    # 返回完整结果；它只影响结果交付方式，不应改变检索范围或 Planner 决策。
    # 来源：POST /query 的 is_stream 字段。默认 False。
    is_stream: bool

    # owner 的中文含义是“所有者/归属用户”。保存当前轻量用户 ID，来源是经过 HTTP
    # 边界强制校验的 X-User-Id；后续用于私有 chunk 的 owner 过滤和 Trace 归属。
    # 默认空字符串只用于模板；缺失 header 的真实请求会在进入图前返回 400。
    owner_user_id: str

    # tenant 的中文含义是“租户/组织空间”。当前轻量单租户阶段固定为 tenant_default，
    # 后续用于 shared 内容的租户范围过滤；它不是已经完成多租户鉴权的证明。
    # 来源：查询入口默认配置。默认空字符串，真实查询入口写入 tenant_default。
    tenant_id: str

    # dataset 的中文含义是“知识库数据集”。该列表限定本次查询允许面向哪些知识库，
    # 已在 API 边界完成去空、保序去重和最多 10 个不同 ID 的校验。
    # 默认空列表只用于模板；省略请求字段时，API 会写入默认设备运维知识库 ID。
    dataset_ids: list[str]

    # query started at 的中文含义是“查询开始时间”。使用 UTC ISO 8601 字符串记录查询
    # 入口创建 State 的时间，后续用于计算总耗时、排序 Trace 和重放时核对时间上下文。
    # 默认空字符串；query_graph_invoke 会为每次真实查询写入当前时间，不由节点修改。
    query_started_at: str

    # ==================== 主体确认与设备标识 ====================

    # rewritten query 的中文含义是“改写后的问题”。主体确认服务结合多轮历史，把指代
    # 不完整的问题改写成可独立检索的文本；检索 Action 优先使用该字段。
    # 默认空字符串，当前主体确认节点写入；它不会覆盖 original_query。
    rewritten_query: str

    # subject ID 的中文含义是“标准主题业务 ID”。保存已经确认的稳定主题主键，后续
    # Milvus chunk 检索按该 ID 过滤，不依赖可能重命名的主题展示名。
    # 默认空列表，主体确认成功后写入；列表支持一个问题涉及多个标准主题。
    subject_ids: list[str]

    # standard subject name 的中文含义是“标准主题名称”。它用于界面展示、日志、Prompt
    # 和历史消息，不作为稳定关联主键；真正过滤应使用 subject_ids。
    # 默认空列表，主体确认成功后与 subject_ids 一起写入。
    standard_subject_names: list[str]

    # subject resolution status 的中文含义是“主体确认状态”。使用关闭枚举表达已确认、
    # 有歧义、未找到或问题没有主体；它让 Planner 不再通过 answer 是否为空猜测结果。
    # 默认 None 表示主体节点尚未执行；阶段 9 主体节点完成后必须写入四种枚举之一。
    subject_resolution_status: SubjectResolutionStatus | None

    # subject candidates 的中文含义是“候选标准主题”。当主体匹配分数不足以直接确认时，
    # 保存可供用户选择的主题名称；Planner 可据此选择 ask_clarification（向用户追问）。
    # 默认空列表；只有主体歧义场景才应包含值。
    subject_candidates: list[str]

    # clarification question 的中文含义是“澄清问题/追问文本”。保存由结构化差异规则
    # 生成、可以直接返回给用户的问题；不能保存模型私有思维链。
    # 默认 None；仅主体或证据确实存在可解决歧义时写入。
    clarification_question: str | None

    # query identifiers 的中文含义是“用户问题中实际出现的结构化标识”。按类型保存设备
    # 型号、报警码、SOP 编号、零件编号等规范化结果，例如 {"alarm_code": ["E020"]}。
    # 该字段必须忠实保留用户输入，不能因为向量检索找到相近的 E021 就静默覆盖成 E021；
    # 系统猜测的纠错候选属于 RetrievalObservation.suggested_identifiers，必须经用户确认。
    # 默认空字典；阶段 5 第五部分已在查询入口写入，并用于第一段精确过滤和第二段查询增强。
    query_identifiers: dict[str, list[str]]

    # history 的中文含义是“当前会话历史消息”。主体改写和答案 Prompt 读取该列表；当前
    # 仍按 session_id 获取，完整的 user_id + session_id 隔离属于阶段 7。
    # 默认空列表；它是运行时副本，不代表 QueryGraphState 自己持久化聊天记录。
    history: list[dict]

    # ==================== Planner 决策与轨迹上下文 ====================

    # trace 的中文含义是“追踪记录”。trace_id 唯一标识一次完整查询执行，区别于表示聊天
    # 会话的 session_id；后续 Action、Observation、引用和耗时都通过它关联。
    # 默认空字符串；query_graph_invoke 为每次真实查询生成 UUID，Trace 持久化尚未接入。
    trace_id: str

    # planner step 的中文含义是“Planner 决策步数”。记录当前已经进行到第几次决策，
    # 用于最大步骤保护和轨迹排序；0 表示尚未执行任何 Planner 决策。
    # 初始为 0；阶段 9 在每个 Action 真正执行结束后随 Action history 一起递增。
    planner_step: int

    # policy version 的中文含义是“策略版本”。用于区分 rule-v1、后续模型版本和灰度策略，
    # 使同一 query 的 Action Trace 能关联到明确决策规则。
    # 默认空字符串；阶段 9 Planner 节点首次执行时写入 rule-v1。
    policy_version: str

    # current planner decision 的中文含义是“当前 Planner 决策”。保存最近一次经过 Pydantic
    # 校验的 action/query/reason_code，后续路由只读取 decision.action。
    # 默认 None；阶段 9 Planner 节点每轮决策后覆盖为最新 Decision。
    current_planner_decision: PlannerDecision | None

    # planner action history 的中文含义是“Planner 动作历史”。按步骤保存已经执行过的
    # Decision 和执行终态，用于阻止重复 Action、检测循环和后续轨迹重放。
    # 默认空列表；未执行的 Decision 不能提前写入历史。
    planner_action_history: list[PlannerHistoryItem]

    # planner type 的中文含义是“Planner 类型”。预期值如 rule（规则）或 model（模型），
    # 用于 Trace/评测区分决策来源，不应通过类名或 provider 名称推断。
    # 默认空字符串；阶段 9 规则 Planner 执行后写入 rule。
    planner_type: str

    # planner runtime metadata 的中文含义是“Planner 运行元数据”。后续记录 provider
    #（模型服务方）、model_id、model_revision、prompt_version、token 用量、耗时和成本。
    # 默认空字典；规则 Planner 的模型相关值应为空/0，且这里不保存思维链。
    planner_runtime_metadata: dict[str, object]

    # planner total duration 的中文含义是“本次查询全部 Planner 决策累计耗时”。每次进入
    # node_query_planner 都累加本轮纯决策耗时，Trace 顶层据此统计规则或模型 Planner 成本。
    planner_total_duration_ms: int

    # web search allowed 的中文含义是“本次查询是否允许联网检索”。来源应是 API、租户或
    # 部署策略；Planner 只能读取并收紧，不能自行把 False 改成 True。阶段 9 默认允许，
    # 后续接入更细权限策略时由查询入口显式覆盖。
    web_search_allowed: bool

    # safe guard triggered 的中文含义是“是否已经触发安全保护”。一旦为 True，Planner
    # 必须直接 refuse，不再执行检索或答案生成。默认 False，只在上游安全规则确认风险时写入。
    safe_guard_triggered: bool

    # planner max steps 的中文含义是“本次查询最多允许完成多少个 Action”。它是运行时
    # 防循环边界，不是模型参数；达到上限后即使还有候选也必须安全终止。
    planner_max_steps: int

    # current action duration 的中文含义是“最近一次检索 Action 的累计耗时”。检索节点先
    # 写入外部调用耗时，rerank 节点继续累加模型耗时，Observation 消费后供 Trace 投影。
    # 该值只存在于当前 State，不单独持久化，默认 0 毫秒。
    current_action_duration_ms: int

    # ==================== Observation 与检索配置 ====================

    # retrieval observation 的中文含义是“检索观察结果”。它是最近一个检索 Action 执行后
    # 返回给 Planner 的结构化事实，包含数量、分数、标识命中、耗时和错误。
    # 默认 None；阶段 9 每个 local/HyDE/Web Action 完成后统一生成或更新该字段。
    retrieval_observation: RetrievalObservation | None

    # retrieval mode 的中文含义是“召回组合模式”。阶段 5 第六部分固定三种关闭枚举，
    # 阶段 5B 评测后默认 dense_learned_sparse_bm25；另外两种模式保留用于回归和诊断。
    # 来源：查询 State 或评测覆盖值；普通检索和 HyDE 必须读取同一个值。
    retrieval_mode: str

    # retrieval config version 的中文含义是“检索配置版本”。它关联 top-k、RRF k、rerank
    # 阈值等一整套配置快照，避免只看到 mode 却不知道具体参数。
    # 默认空字符串；阶段 9 Planner 节点写入当前冻结的检索配置版本。
    retrieval_config_version: str

    # retrieval channel results 的中文含义是“各召回通道原始结果”。key 表示 dense、
    # learned_sparse、bm25、hyde 或 web 等通道，value 保存该通道的候选列表。
    # 默认空字典；后续统一召回服务写入。当前旧字段仍暂时维持主链路运行。
    retrieval_channel_results: dict[str, list[dict]]

    # ==================== 当前旧召回字段（阶段 5 分步过渡） ====================

    # embedding chunks 的中文含义是“原问题本地检索 Action 候选”。列表内每项已经通过
    # RetrievalCandidate 校验，包含本地身份、模式通道、Action 来源、召回名次和分数。
    # 模式通道表示本次 Action 启用项，不等于 Milvus 返回了逐通道命中明细。
    # 当前字段名为分步过渡命名，Planner 主链路接入后归入按 Action 保存的结果集合。
    embedding_chunks: list[dict] | None

    # HyDE embedding chunks 的中文含义是“假设答案增强检索 Action 候选”。每项也是统一
    # RetrievalCandidate，并通过 retrieval_channels 标记 hyde 和实际底层召回通道。
    hyde_embedding_chunks: list[dict] | None

    # web search docs 的中文含义是“联网搜索 Action 候选”。每项使用统一 Candidate，真实
    # URL 是去重和 Citation 身份，document/chunk/dataset/index 等本地字段必须为 None。
    web_search_docs: list[dict] | None

    # RRF 是 Reciprocal Rank Fusion，中文为“倒数排名融合”。rrf_chunks 保存所有已执行
    # original/HyDE/Web Action 原始列表一次性重算后的累计 Candidate；本地按 chunk_id、
    # Web 按规范化 URL 去重。字段名仍是过渡命名，内容已经不再只包含本地 chunk。
    rrf_chunks: list[dict]

    # rerank 的中文含义是“重排序”。reranked_docs 保存统一 reranker 对累计 Candidate
    # 写入 rerank_score 后的最终证据列表；所有本地/Web 身份和召回元数据必须原样保留。
    reranked_docs: list[dict]

    # ==================== 终止结果、引用和答案运行信息 ====================

    # citations 的中文含义是“结构化引用列表”。只允许保存最终进入答案上下文的 Citation，
    # 本地引用携带 document_id/chunk_id，Web 引用通过 source 保存 URL。
    # 默认空列表；当前 answer/API/SSE 尚未接入引用输出，后续答案任务写入。
    citations: list[Citation]

    # terminal reason code 的中文含义是“流程终止原因码”。记录为什么最终 answer、追问或
    # refuse，使用与 Planner 决策一致的机器可读枚举，不能写自由文本思维过程。
    # 默认 None；阶段 9 终态节点固定为最后一个 Planner Decision 的 reason_code。
    terminal_reason_code: PlannerReasonCode | None

    # answer runtime metadata 的中文含义是“答案模型运行元数据”。后续记录答案模型的
    # provider、model_id、model_revision、prompt_version、token、耗时和成本。
    # 默认空字典；它和 planner_runtime_metadata 分开，便于区分策略成本与生成成本。
    answer_runtime_metadata: dict[str, object]

    # retrieval config snapshot 的中文含义是“本次查询实际使用的检索配置快照”。与只表示
    # 名称的 retrieval_config_version 不同，这里直接保存 mode、top-k、RRF k、证据阈值
    # 和 Web 开关等真实数值；查询入口创建后不得在同一条 Trace 中途修改。
    retrieval_config_snapshot: dict[str, object]

    # chunk status filter enabled 的中文含义是“是否启用人工禁用 chunk 查询过滤”。True 时
    # 本地检索和 HyDE 会先读取 Mongo ``chunk_status_overrides`` 中 manual_status=disabled
    # 的 chunk_id，再追加到 Milvus expr。默认 False 仅用于无 Mongo 的单元测试和离线重放；
    # 真实 HTTP 查询入口必须显式写 True。
    chunk_status_filter_enabled: bool

    # disabled chunk IDs 的中文含义是“本轮查询范围内被人工禁用的 chunk_id 快照”。它来自
    # Mongo ``chunk_status_overrides.manual_status=disabled``，只用于构建当前 Milvus expr；
    # 默认空列表表示尚无禁用覆盖或当前路径未启用 Mongo 读取，不代表全库没有禁用数据。
    disabled_chunk_ids: list[int | str]

    # trace persistence enabled 的中文含义是“是否由查询入口负责持久化 Trace”。默认 False
    # 让纯图单元测试和离线 Planner 重放不依赖 Mongo；真实 HTTP 查询入口会显式写 True。
    trace_persistence_enabled: bool

    # history persistence enabled 的中文含义是“是否写入聊天历史”。真实聊天为 True；
    # Retrieval Test 为 False，因为它只用于调试/评测，不应污染用户对话记录。
    history_persistence_enabled: bool

    # execution source 的中文含义是“Trace 来源”。chat 表示真实聊天，retrieval_test 表示
    # 检索测试，replay 表示基于历史 Trace 的重放。它只影响可观测归类，不改变权限过滤。
    execution_source: str

    # replay of trace id 的中文含义是“重放来源 Trace ID”。非重放查询为空；重放时用于
    # 说明本次 Trace 是哪条历史记录的复现尝试。
    replay_of_trace_id: str | None

    # config/corpus match status 分别表示“配置是否一致”和“语料快照是否一致”。普通聊天
    # 为 unknown；Trace replay 会根据原 Trace 的配置、index_version 和启停快照更新。
    config_match_status: str
    corpus_match_status: str

    # prompt 的中文含义是“提交给答案模型的完整提示词”。当前 answer_service 构造它，
    # 主要用于运行时调用和排查；后续 Trace 是否保存全文必须单独评估隐私与存储成本。
    # 默认空字符串。
    prompt: str

    # answer 的中文含义是“最终交付文本”。阶段 9 由 terminal response 节点统一写入，
    # 可能是答案、追问或拒答；具体终止类型必须结合 current decision/terminal reason 判断。
    # 默认空字符串。
    answer: str

    # image URLs 的中文含义是“答案引用图片地址列表”。从最终证据正文中的 Markdown 图片
    # 或 Web 图片地址提取，用于前端展示；它和 Citation 来源列表不是同一个概念。
    # 默认空列表。
    image_urls: list[str]


# 默认 State 是“尚未开始处理”的模板。所有 list/dict 都放在这个模板中，再由工厂函数
# deepcopy（深拷贝）生成每次查询的独立对象，避免不同请求共享可变容器。
query_graph_default_state: QueryGraphState = {
    "session_id": "",
    "original_query": "",
    "is_stream": False,
    "owner_user_id": "",
    "tenant_id": "",
    "dataset_ids": [],
    "query_started_at": "",
    "rewritten_query": "",
    "subject_ids": [],
    "standard_subject_names": [],
    "subject_resolution_status": None,
    "subject_candidates": [],
    "clarification_question": None,
    "query_identifiers": {},
    "history": [],
    "trace_id": "",
    "planner_step": 0,
    "policy_version": "",
    "current_planner_decision": None,
    "planner_action_history": [],
    "planner_type": "",
    "planner_runtime_metadata": {},
    "planner_total_duration_ms": 0,
    "web_search_allowed": WEB_FALLBACK_ENABLED,
    "safe_guard_triggered": False,
    "planner_max_steps": PLANNER_MAX_STEPS,
    "current_action_duration_ms": 0,
    "retrieval_observation": None,
    "retrieval_mode": RETRIEVAL_DEFAULT_MODE.value,
    "retrieval_config_version": RETRIEVAL_CONFIG_VERSION,
    "retrieval_channel_results": {},
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "web_search_docs": [],
    "rrf_chunks": [],
    "reranked_docs": [],
    "citations": [],
    "terminal_reason_code": None,
    "answer_runtime_metadata": {},
    "retrieval_config_snapshot": {},
    "chunk_status_filter_enabled": False,
    "disabled_chunk_ids": [],
    "trace_persistence_enabled": False,
    "history_persistence_enabled": True,
    "execution_source": "chat",
    "replay_of_trace_id": None,
    "config_match_status": "unknown",
    "corpus_match_status": "unknown",
    "prompt": "",
    "answer": "",
    "image_urls": [],
}


def create_query_default_state(**overrides) -> QueryGraphState:
    """
    创建一份独立的查询 State，并允许调用方覆盖入口字段。

    这里先深拷贝模板，再合并 API 传入的 session/query/owner 等值。``overrides`` 不会在
    运行时自动执行 Pydantic 校验，因此 HTTP 输入仍必须先经过对应 Schema 校验。
    """
    state = copy.deepcopy(query_graph_default_state)
    state.update(overrides)
    return state


def get_query_default_state() -> QueryGraphState:
    """
    返回一份没有业务覆盖值的干净 State，主要用于测试和非 HTTP 调用。

    每次都深拷贝，确保 history、action history、citations 和各种 dict/list 不会在查询
    之间共享引用。
    """
    return copy.deepcopy(query_graph_default_state)


def copy_query_state(state: QueryGraphState, **overrides) -> QueryGraphState:
    """
    深拷贝一份现有 State，并可覆盖少量字段。

    该函数适合评测重放或分支轨迹：复制后的 Action history、Observation、Citation 和
    通道结果都与原 State 隔离，修改副本不会污染原始轨迹。
    """
    new_state = copy.deepcopy(state)
    new_state.update(overrides)
    return new_state


if __name__ == "__main__":
    # 测试
    state = create_query_default_state(
        session_id="test_001",
        original_query="华为P60怎么样?",
        is_stream=False
    )
    print("初始化状态：", state)

    # 复制状态
    new_state = copy_query_state(
        state,
        original_query="修改后的问题"
    )
    print("复制后的状态：", new_state)
