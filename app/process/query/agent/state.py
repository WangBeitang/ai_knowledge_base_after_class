from typing_extensions import TypedDict
import copy

class QueryGraphState(TypedDict):
    # 请求输入：由 API 或调用方传入，作为一次查询流程的原始上下文
    session_id: str  # 会话唯一标识，用于历史记录、SSE、任务状态
    original_query: str  # 用户原始问题
    is_stream: bool  # 是否使用流式输出
    # 阶段 5 第一部分只把身份和知识库范围稳定传入查询图；真正基于这些字段拼接
    # Milvus visibility/owner 过滤表达式放在后续“查询权限过滤”部分实现。
    owner_user_id: str  # 当前轻量用户 ID，来自强制校验后的 X-User-Id 请求头
    tenant_id: str  # 当前租户上下文；轻量单租户阶段固定为 tenant_default
    dataset_ids: list[str]  # 本次允许检索的知识库 ID，已在 API 边界完成去空、去重和上限校验

    # 主体确认：从问题和历史中识别用户要问的主体
    rewritten_query: str  # 结合历史改写后的独立问题
    subject_ids: list[str]  # 已确认的标准主题 ID 列表，用于查询过滤
    standard_subject_names: list[str]  # 已确认的标准主题名称列表，用于展示、日志和 Prompt
    history: list  # 当前会话历史记录，用于改写问题和构造答案上下文

    # 多路召回：不同检索路径的原始召回结果
    embedding_chunks: list | None  # 标准向量检索召回的知识库切片
    hyde_embedding_chunks: list | None  # HyDE 检索召回的知识库切片
    web_search_docs: list | None  # 联网搜索结果，后续阶段会改为 fallback

    # 召回融合：将多路召回结果合并成统一候选集
    rrf_chunks: list  # RRF 融合后的知识库候选切片

    # 重排序：对候选文档做 rerank 后的最终上下文
    reranked_docs: list  # 重排序后的 Top-K 文档，供答案生成使用

    # 答案生成：最终 prompt、答案和可展示引用资源
    prompt: str  # 组装后的最终 Prompt
    answer: str  # 最终回答；也可用于主体不明确时的追问/拒答
    image_urls: list[str]  # 答案引用到的图片链接


# ========================
# 默认状态（全部为空）
# ========================
query_graph_default_state: QueryGraphState = {
    "session_id": "",
    "original_query": "",
    "is_stream": False,
    "owner_user_id": "",
    "tenant_id": "",
    "dataset_ids": [],
    "rewritten_query": "",
    "subject_ids": [],
    "standard_subject_names": [],
    "history": [],
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "web_search_docs": [],
    "rrf_chunks": [],
    "reranked_docs": [],
    "prompt": "",
    "answer": "",
    "image_urls": [],
}


# ========================
# 创建默认状态（可覆盖）
# ========================
def create_query_default_state(**overrides) -> QueryGraphState:
    """
    创建查询流程的默认状态，支持覆盖字段
    """
    state = copy.deepcopy(query_graph_default_state)
    state.update(overrides)
    return state


# ========================
# 获取干净状态
# ========================
def get_query_default_state() -> QueryGraphState:
    """
    返回一个新的状态实例，避免全局变量污染。
    """
    return copy.deepcopy(query_graph_default_state)


# ========================
# 状态复制函数
# ========================
def copy_query_state(state: QueryGraphState, **overrides) -> QueryGraphState:
    """
    复制现有状态并可覆盖字段，深拷贝，不污染原数据
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
