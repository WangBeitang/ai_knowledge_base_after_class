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
# 混合检索权重：dense向量权重 0.9，sparse向量权重 0.1
RETRIEVAL_RANKER_WEIGHTS = (0.9, 0.1)


RERANK_MAX_TOPK: int = 5
RERANK_MIN_TOPK: int = 2   #动态topk
RERANK_GAP_RATIO: float = 0.25
RERANK_GAP_ABS: float = 0.25
RERANK_MAX_INPUT_TOKENS: int = 512
RERANK_SUMMARY_CHAR_RATIO: float = 1.3
RERANK_MIN_SUMMARY_CHARS: int = 50
