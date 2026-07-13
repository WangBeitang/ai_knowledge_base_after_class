from app.rag.query.contracts import RetrievalChannel, RetrievalMode


# ====================== 全局配置 ======================
# 拉取历史消息最大条数
QUERY_HISTORY_LIMIT = 10
# 主体名称确认阈值：高于该分数 → 直接确认 [0.75]
SUBJECT_NAME_CONFIRM_THRESHOLD = 0.75
# 主体名称候选阈值：介于两者之间 → 让用户选择
SUBJECT_NAME_CANDIDATE_THRESHOLD = 0.60
# 给用户选择时，最多展示几个候选
SUBJECT_NAME_OPTIONS_TOPK = 2

# ====================== 检索配置 ======================
# 默认返回的最大知识库片段数量
RETRIEVAL_DEFAULT_LIMIT = 5
# 标准主题/别名 collection 仍使用加权融合时的 dense/sparse 权重。chunk 检索从阶段 5
# 第六部分起显式使用 RRF，不再直接相加不同量级的 dense、learned sparse、BM25 分数。
RETRIEVAL_RANKER_WEIGHTS = (0.9, 0.1)
# 阶段 5A 的三种实验模式已具备 schema/请求能力，但默认仍保持 dense + learned sparse，
# 避免在固定评测结论出来前把实验通道直接当成线上最优方案。
RETRIEVAL_DEFAULT_MODE = RetrievalMode.DENSE_LEARNED_SPARSE
# 单次本地 Action 内 Milvus RRF 和跨 Action RRF 使用同一个可版本化 k 基线。
RETRIEVAL_RRF_K = 60
# chunk collection 的 BGE-M3 学习式稀疏字段名。标准主题/别名 collection 仍使用它们
# 自己历史上的 sparse_vector，二者不是同一个 schema，不能做全局机械替换。
LEARNED_SPARSE_FIELD = "learned_sparse_vector"
# Milvus BM25 Function 输出字段。查询侧只向该字段提交原始文本；应用不生成或写入向量。
BM25_SPARSE_FIELD = "bm25_sparse_vector"


def normalize_retrieval_mode(value) -> RetrievalMode:
    """把 State/config 中的字符串统一校验成关闭枚举，未知模式立即失败。"""
    if value in (None, ""):
        return RETRIEVAL_DEFAULT_MODE
    if isinstance(value, RetrievalMode):
        return value
    try:
        return RetrievalMode(str(value))
    except ValueError as exc:
        supported = ", ".join(mode.value for mode in RetrievalMode)
        raise ValueError(f"不支持的 retrieval_mode={value!r}，可选值：{supported}") from exc


def channels_for_retrieval_mode(mode: RetrievalMode | str) -> list[RetrievalChannel]:
    """返回一个本地 Action 启用的模式通道，顺序与 AnnSearchRequest 创建顺序一致。"""
    normalized_mode = normalize_retrieval_mode(mode)
    channels = [RetrievalChannel.DENSE]
    if normalized_mode in {
        RetrievalMode.DENSE_LEARNED_SPARSE,
        RetrievalMode.DENSE_LEARNED_SPARSE_BM25,
    }:
        channels.append(RetrievalChannel.LEARNED_SPARSE)
    if normalized_mode in {
        RetrievalMode.DENSE_BM25,
        RetrievalMode.DENSE_LEARNED_SPARSE_BM25,
    }:
        channels.append(RetrievalChannel.BM25)
    return channels


RERANK_MAX_TOPK: int = 5
RERANK_MIN_TOPK: int = 2   #动态topk
RERANK_GAP_RATIO: float = 0.25
RERANK_GAP_ABS: float = 0.25
RERANK_MAX_INPUT_TOKENS: int = 512
RERANK_SUMMARY_CHAR_RATIO: float = 1.3
RERANK_MIN_SUMMARY_CHARS: int = 50
