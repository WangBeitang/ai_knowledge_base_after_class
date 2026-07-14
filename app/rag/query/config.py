import os

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
# policy version 的中文含义是“Planner 策略版本”。只要规则顺序、规则语义或安全边界
# 发生变化就必须升级；单纯调整检索参数时保持该值不变，改 RETRIEVAL_CONFIG_VERSION。
POLICY_VERSION = "rule-v1"
# realtime patterns version 的中文含义是“实时问题识别规则版本”。实时关键词可以独立于
# Planner 主规则迭代，因此单独记录，便于 Trace 解释为什么某个问题直接进入 Web。
REALTIME_PATTERNS_VERSION = "realtime-keywords-v1"


def _env_bool(name: str, default: bool) -> bool:
    """读取布尔环境变量；未知值立即失败，避免拼写错误静默改变联网边界。"""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized_value = raw_value.strip().lower()
    if normalized_value in {"1", "true", "yes", "on"}:
        return True
    if normalized_value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"环境变量 {name} 必须是 true/false，当前值为 {raw_value!r}")


# per-channel top-k 的中文含义是“每条底层召回通道最多取多少条候选”。Milvus 的 dense、
# learned sparse、BM25 请求公平对比时必须使用相同值；它不等于最终答案证据数量。
RETRIEVAL_PER_CHANNEL_TOPK = 5
# 保留旧名称作为当前调用点的可读别名；二者是同一个配置，不能分别修改。
RETRIEVAL_DEFAULT_LIMIT = RETRIEVAL_PER_CHANNEL_TOPK
# 标准主题/别名 collection 仍使用加权融合时的 dense/sparse 权重。chunk 检索从阶段 5
# 第六部分起显式使用 RRF，不再直接相加不同量级的 dense、learned sparse、BM25 分数。
RETRIEVAL_RANKER_WEIGHTS = (0.9, 0.1)
# 阶段 5 的 60 条固定样本已完成四轮公平对比。三路模式在 50 条核心集上四轮 Recall@K
# 和引用命中率均为 1.0，因此阶段 5B 将它冻结为默认模式；环境变量只用于受控回退和诊断。
RETRIEVAL_DEFAULT_MODE = RetrievalMode(
    os.getenv("RETRIEVAL_MODE", RetrievalMode.DENSE_LEARNED_SPARSE_BM25.value).strip()
)
# 单次本地 Action 内 Milvus RRF 和跨 Action RRF 使用同一个可版本化 k 基线。
RETRIEVAL_RRF_K = 60
# retrieval config version 的中文含义是“检索配置版本”。阶段 9 把 Planner 接入真实链路
# 后，每次决策都必须能说明自己使用的是哪组召回、RRF、rerank 参数。阶段 5B 选择三路
# 模式后升级为 final-v1；以后只要阈值或召回参数变化，仍必须同步升级版本字符串。
RETRIEVAL_CONFIG_VERSION = "retrieval-stage5-final-v1"
# 证据充分阈值：最终累计候选经过统一 reranker 后，最高分达到该值才允许进入 answer。
# 本次检索 schema 评测只比较候选排名，没有独立优化答案门槛，因此继续冻结 0.75 作为
# 当前运行基线；后续若用答案级开发集重新标定，必须升级 retrieval_config_version。
RERANK_EVIDENCE_THRESHOLD = 0.75
# 单次查询最多允许完成的 Planner Action 数。当前最长合法路径为 local -> HyDE -> Web
# -> 终止，6 步保留少量扩展空间，同时能在状态异常时阻止无限循环。
PLANNER_MAX_STEPS = 6
# fusion top-k 的中文含义是“跨 Action RRF 最终保留的累计候选数量”。当前与每路 top-k
# 相同，但仍独立进入配置快照，后续调参时不会把两个不同含义的数值混为一谈。
RETRIEVAL_FUSION_TOPK = 5
# Web fallback 开关决定本地/HyDE 证据不足后是否允许 Planner 联网。它是部署能力边界，
# 不是 Planner 自行放宽的选项；环境变量只能在查询入口创建 State 时读取一次。
WEB_FALLBACK_ENABLED = _env_bool("WEB_FALLBACK_ENABLED", True)
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


def build_retrieval_config_snapshot(
        *,
        retrieval_mode: RetrievalMode | str | None = None,
        web_fallback_enabled: bool | None = None,
) -> dict[str, object]:
    """
    返回可持久化的本次检索配置快照。

    version 只能告诉我们“配置叫什么”，snapshot（快照）才保存真正运行的数值。二者必须
    一起写入 Trace，避免未来同名版本被误改后无法重放历史查询。
    """
    normalized_mode = normalize_retrieval_mode(retrieval_mode)
    return {
        "retrieval_mode": normalized_mode.value,
        "per_channel_topk": RETRIEVAL_PER_CHANNEL_TOPK,
        "fusion_topk": RETRIEVAL_FUSION_TOPK,
        "rerank_min_topk": RERANK_MIN_TOPK,
        "rerank_max_topk": RERANK_MAX_TOPK,
        "rrf_k": RETRIEVAL_RRF_K,
        "evidence_threshold": RERANK_EVIDENCE_THRESHOLD,
        "web_fallback_enabled": (
            WEB_FALLBACK_ENABLED
            if web_fallback_enabled is None
            else bool(web_fallback_enabled)
        ),
    }
