"""
查询 Planner 的稳定业务契约。

本模块只定义 Planner、执行环境和 API/Trace 之间交换的数据形状，不执行 Milvus
检索、不访问 Mongo、不调用 LLM，也不参与 LangGraph 路由。把契约和具体策略分开，
可以让后续 ``RuleBasedPlanner`` 与 ``ModelPlanner`` 复用同一组输入输出，而检索节点
不需要知道当前使用的是规则还是模型。

所有对外枚举都继承 ``str, Enum``：
- Enum 成员让 Python 代码避免散落魔法字符串；
- str 值可以稳定写入 JSON、LangGraph State、Mongo Trace 和后续训练数据；
- Pydantic 会在入口拒绝未知值，避免非法 Action 进入路由层。
"""

from enum import Enum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


# Planner 只能看到有限数量、有限长度的证据片段。这里限制的是单次决策上下文，不是
# 最终答案上下文；Trace 持久化层后续只保存证据标识和摘要 hash，不默认保存正文片段。
MAX_EVIDENCE_SUMMARY_COUNT = 5
MAX_EVIDENCE_EXCERPT_CHARS = 500
MAX_EVIDENCE_TOTAL_CHARS = 2_000

NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveStep = Annotated[int, Field(ge=1)]


class QueryContractModel(BaseModel):
    """
    查询契约公共基类。

    ``extra='forbid'`` 很重要：如果 Action 执行节点或未来 Model Planner 拼错字段名，
    应在契约边界立即失败，而不是由 Pydantic 静默忽略后产生难以解释的默认行为。
    字符串统一去除首尾空白，配合字段 ``min_length`` 拒绝只有空白的必填文本。
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class QueryAction(str, Enum):
    """Planner 可以选择的固定动作集合。成员名用于代码，value 用于持久化协议。"""

    LOCAL_SEARCH = "local_search"
    HYDE_SEARCH = "hyde_search"
    WEB_SEARCH = "web_search"
    ANSWER = "answer"
    ASK_CLARIFICATION = "ask_clarification"
    REFUSE = "refuse"


class PlannerReasonCode(str, Enum):
    """
    Planner 决策原因的机器可读枚举。

    reason code 只描述“触发了哪条可验证规则”，不保存自由文本分析，更不能承载模型
    私有思维链。后续 Trace、评测和 GRPO 数据可以按这些稳定值聚合行为。
    """

    # 主体确认阶段的结果。
    SUBJECT_AMBIGUOUS = "subject_ambiguous"  # 主体歧义
    SUBJECT_REQUIRED = "subject_required"  # 需要明确主体
    SUBJECT_NOT_FOUND = "subject_not_found"  # 未找到主体

    # 初始路由和本地知识库检索结果。
    REALTIME_QUERY = "realtime_query"  # 实时查询
    INITIAL_LOCAL_SEARCH = "initial_local_search"  # 初始本地检索
    LOCAL_EVIDENCE_SUFFICIENT = "local_evidence_sufficient"  # 本地证据充足
    LOCAL_EMPTY = "local_empty"  # 本地检索为空
    LOCAL_LOW_SCORE = "local_low_score"  # 本地检索得分低

    # HyDE 与 Web fallback 的结构化结果。
    HYDE_EVIDENCE_SUFFICIENT = "hyde_evidence_sufficient"  # HyDE 证据充足
    HYDE_STILL_INSUFFICIENT = "hyde_still_insufficient"  # HyDE 证据仍不足
    WEB_EVIDENCE_AVAILABLE = "web_evidence_available"  # Web 证据可用
    WEB_EMPTY_OR_FAILED = "web_empty_or_failed"  # Web 检索为空或失败

    # 证据冲突、执行失败和安全兜底。
    EVIDENCE_AMBIGUOUS = "evidence_ambiguous"  # 证据冲突/歧义
    ACTION_EXECUTION_ERROR = "action_execution_error"  # 动作执行失败
    SAFE_GUARD_TRIGGERED = "safe_guard_triggered"  # 安全兜底触发


class SubjectResolutionStatus(str, Enum):
    """主体确认节点输出给 Planner 的有限状态；具体识别逻辑仍由后续任务改造。"""

    CONFIRMED = "confirmed"
    AMBIGUOUS = "ambiguous"
    NOT_FOUND = "not_found"
    NO_MENTION = "no_mention"


class ObservationStatus(str, Enum):
    """一个检索 Action 的可观测执行结果，不等同于整次查询的最终状态。"""

    SUCCESS = "success"
    EMPTY = "empty"
    FAILED = "failed"


class PlannerExecutionStatus(str, Enum):
    """Action history 中某一步的执行终态。未执行的 Decision 不应写入 history。"""

    COMPLETED = "completed"
    FAILED = "failed"


class EvidenceSourceType(str, Enum):
    """
    证据来源类型。

    HyDE 是生成查询向量的检索动作，不是独立数据来源；HyDE 命中的 Milvus chunk 仍是
    ``local``，Web 搜索结果才是 ``web``。
    """

    LOCAL = "local"
    WEB = "web"


class PlannerDecision(QueryContractModel):
    """
    Planner 一次决策的唯一业务输出。

    ``action`` 决定 LangGraph 后续路由，``query`` 是该 Action 实际使用的查询文本，
    ``reason_code`` 说明命中了哪条确定性规则。它不包含最终答案，也不包含自由文本推理。
    """

    action: QueryAction
    query: str = Field(min_length=1)
    reason_code: PlannerReasonCode


class EvidenceSummary(QueryContractModel):
    """
    提供给 Planner 的受限证据摘要。

    Planner 只需要判断证据是否充分或冲突，不需要看到全部 chunk。单条 excerpt 和总量
    都有限制，防止 Observation 逐步膨胀；该字段后续不得默认原样写入持久化 Trace。
    """

    document_id: str | None = None
    chunk_id: str | int | None = None
    title: str = Field(min_length=1)
    source_type: EvidenceSourceType
    rerank_score: float | None = None
    matched_identifiers: dict[str, list[str]] = Field(default_factory=dict)
    content_excerpt: str = Field(default="", max_length=MAX_EVIDENCE_EXCERPT_CHARS)

    @model_validator(mode="after")
    def validate_source_identity(self) -> Self:
        """本地证据必须可追踪到 document/chunk；Web 证据不能伪造本地 ID。"""
        if self.source_type == EvidenceSourceType.LOCAL:
            if not self.document_id or self.chunk_id is None:
                raise ValueError("本地证据必须同时包含 document_id 和 chunk_id")
        elif self.document_id is not None or self.chunk_id is not None:
            raise ValueError("Web 证据的 document_id 和 chunk_id 必须为空")
        return self


class RetrievalObservation(QueryContractModel):
    """
    一个 Action 执行后返回给 Planner 的结构化事实。

    Observation 只记录可验证结果：通道数量、候选数量、rerank 分数、标识命中、耗时和
    错误。它不允许加入“模型认为应该如何”等自由文本思维过程。
    """

    action: QueryAction
    status: ObservationStatus
    channel_counts: dict[str, NonNegativeInt] = Field(default_factory=dict)
    candidate_count: NonNegativeInt = 0
    reranked_count: NonNegativeInt = 0
    top_rerank_score: float | None = None
    matched_identifiers: dict[str, list[str]] = Field(default_factory=dict)
    citation_count: NonNegativeInt = 0
    evidence_summaries: list[EvidenceSummary] = Field(
        default_factory=list,
        max_length=MAX_EVIDENCE_SUMMARY_COUNT,
    )
    evidence_ambiguous: bool = False
    clarification_question: str | None = Field(default=None, min_length=1)
    duration_ms: NonNegativeInt = 0
    error_code: str | None = Field(default=None, min_length=1)
    used_structured_filter: bool = False
    filter_fallback: bool = False

    @model_validator(mode="after")
    def validate_observation_consistency(self) -> Self:
        """校验计数、分数、错误和追问字段之间不会互相矛盾。"""
        if self.reranked_count > self.candidate_count:
            raise ValueError("reranked_count 不能大于 candidate_count")
        if self.citation_count > self.reranked_count:
            raise ValueError("citation_count 不能大于 reranked_count")
        if len(self.evidence_summaries) > self.reranked_count:
            raise ValueError("evidence_summaries 数量不能大于 reranked_count")

        if self.reranked_count == 0 and self.top_rerank_score is not None:
            raise ValueError("没有 rerank 结果时 top_rerank_score 必须为空")
        if self.reranked_count > 0 and self.top_rerank_score is None:
            raise ValueError("存在 rerank 结果时必须提供 top_rerank_score")

        total_excerpt_chars = sum(len(item.content_excerpt) for item in self.evidence_summaries)
        if total_excerpt_chars > MAX_EVIDENCE_TOTAL_CHARS:
            raise ValueError(f"证据摘要总长度不能超过 {MAX_EVIDENCE_TOTAL_CHARS} 个字符")

        if self.evidence_ambiguous and not self.clarification_question:
            raise ValueError("证据存在歧义时必须提供 clarification_question")
        if not self.evidence_ambiguous and self.clarification_question is not None:
            raise ValueError("证据不存在歧义时 clarification_question 必须为空")

        if self.status == ObservationStatus.EMPTY:
            if any((self.candidate_count, self.reranked_count, self.citation_count)):
                raise ValueError("EMPTY Observation 的候选、rerank 和引用数量必须为 0")
            if self.evidence_summaries:
                raise ValueError("EMPTY Observation 不能包含 evidence_summaries")
        if self.status == ObservationStatus.FAILED and not self.error_code:
            raise ValueError("FAILED Observation 必须提供 error_code")
        if self.status != ObservationStatus.FAILED and self.error_code is not None:
            raise ValueError("非 FAILED Observation 不能携带 error_code")

        if self.filter_fallback and not self.used_structured_filter:
            raise ValueError("只有尝试过结构化过滤后才能标记 filter_fallback")
        return self


class PlannerHistoryItem(QueryContractModel):
    """已经执行过的一步 Planner 决策及其执行终态。step 从 1 开始。"""

    step: PositiveStep
    decision: PlannerDecision
    execution_status: PlannerExecutionStatus


class PlannerContext(QueryContractModel):
    """
    QueryPlanner 每次 ``plan`` 调用的完整、可重放输入。

    Context 只组合已经存在的结构化事实，不负责读取外部系统。未来 ModelPlanner 也必须
    接收这份契约，不能绕过它直接读取 Mongo、Milvus 或 HTTP Request。
    """

    original_query: str = Field(min_length=1)
    current_query: str = Field(min_length=1)
    subject_resolution_status: SubjectResolutionStatus
    subject_ids: list[str] = Field(default_factory=list)
    subject_candidates: list[str] = Field(default_factory=list)
    clarification_question: str | None = Field(default=None, min_length=1)
    query_identifiers: dict[str, list[str]] = Field(default_factory=dict)
    latest_observation: RetrievalObservation | None = None
    action_history: list[PlannerHistoryItem] = Field(default_factory=list)
    planner_step: NonNegativeInt = 0
    max_steps: PositiveStep
    allowed_actions: list[QueryAction] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_allowed_actions(self) -> Self:
        """允许动作列表必须保序且无重复，避免路由白名单自身存在歧义。"""
        if len(self.allowed_actions) != len(set(self.allowed_actions)):
            raise ValueError("allowed_actions 不能包含重复 Action")
        return self


class Citation(QueryContractModel):
    """
    最终答案可展示的结构化引用。

    引用由代码根据最终证据生成，不要求 LLM 输出可解析 JSON。本地引用必须能追踪到
    document/chunk；Web 引用用 ``source`` 保存 URL，两个本地 ID 必须为空。
    """

    document_id: str | None = None
    chunk_id: str | int | None = None
    title: str = Field(min_length=1)
    source: str = Field(min_length=1)
    score: float
    source_type: EvidenceSourceType

    @model_validator(mode="after")
    def validate_source_identity(self) -> Self:
        """保持本地和 Web 引用的身份字段语义互斥。"""
        if self.source_type == EvidenceSourceType.LOCAL:
            if not self.document_id or self.chunk_id is None:
                raise ValueError("本地引用必须同时包含 document_id 和 chunk_id")
        elif self.document_id is not None or self.chunk_id is not None:
            raise ValueError("Web 引用的 document_id 和 chunk_id 必须为空")
        return self

