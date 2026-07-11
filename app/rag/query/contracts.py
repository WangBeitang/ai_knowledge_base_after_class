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

    LOCAL_SEARCH = "local_search"  # 本地检索：查询有权限访问的 Milvus 知识库 chunk。
    HYDE_SEARCH = "hyde_search"  # HyDE 检索：先生成假设答案，再用增强文本检索本地知识库。
    WEB_SEARCH = "web_search"  # 联网检索：本地证据不足或问题明显需要实时信息时调用。
    ANSWER = "answer"  # 进入答案生成：证据已充分；该 Action 本身不是最终答案文本。
    ASK_CLARIFICATION = "ask_clarification"  # 向用户追问：主体或证据存在可解决的歧义。
    REFUSE = "refuse"  # 安全拒答：证据不足、超出范围或无法可靠回答时终止。


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

    # 设备型号、报警码等精确标识的确认结果。
    IDENTIFIER_CONFIRMATION_REQUIRED = "identifier_confirmation_required"  # 找到疑似纠错候选，必须让用户确认。
    IDENTIFIER_NOT_FOUND = "identifier_not_found"  # 当前范围内未找到用户输入的标识或可靠候选。

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

    CONFIRMED = "confirmed"  # 已确认：已经得到可用于检索的稳定 subject_id。
    AMBIGUOUS = "ambiguous"  # 有歧义：存在多个候选主题，需要用户进一步确认。
    NOT_FOUND = "not_found"  # 未找到：用户提到了主体，但知识库中没有可靠匹配项。
    NO_MENTION = "no_mention"  # 未提及：问题和历史中没有可用于确认主体的信息。


class ObservationStatus(str, Enum):
    """一个检索 Action 的可观测执行结果，不等同于整次查询的最终状态。"""

    SUCCESS = "success"  # 执行成功：产生了可继续评估的候选或证据。
    EMPTY = "empty"  # 执行成功但结果为空：没有候选，不等同于系统异常。
    FAILED = "failed"  # 执行失败：外部服务或节点异常，必须同时提供 error_code。


class IdentifierResolutionStatus(str, Enum):
    """
    用户输入的设备运维标识与检索结果之间的确认状态。

    resolution 的中文含义是“确认/消歧结果”。该枚举只记录可验证的匹配关系，不允许
    embedding 或 rerank 分数把一个不同的报警码自动改写成用户真正想问的报警码。
    """

    NOT_APPLICABLE = "not_applicable"  # 不适用：用户问题中没有可用于确认的设备型号、报警码等标识。
    EXACT_MATCH = "exact_match"  # 精确命中：结构化字段直接命中用户输入的规范化标识。
    FALLBACK_EXACT_MATCH = "fallback_exact_match"  # 降级同码命中：metadata 未命中，但正文仍找到相同标识。
    SUGGESTION_REQUIRED = "suggestion_required"  # 需要确认：只找到不同但相近的候选标识，不能直接回答。
    NOT_FOUND = "not_found"  # 未找到：精确和宽松检索都没有找到相同标识或可靠纠错候选。


class PlannerExecutionStatus(str, Enum):
    """Action history 中某一步的执行终态。未执行的 Decision 不应写入 history。"""

    COMPLETED = "completed"  # 已完成：该 Decision 对应的 Action 已正常执行并产出结果。
    FAILED = "failed"  # 已失败：Action 执行异常，Planner 可根据 Observation 决定 fallback。


class EvidenceSourceType(str, Enum):
    """
    证据来源类型。

    HyDE 是生成查询向量的检索动作，不是独立数据来源；HyDE 命中的 Milvus chunk 仍是
    ``local``，Web 搜索结果才是 ``web``。
    """

    LOCAL = "local"  # 本地证据：来自 Milvus chunk，必须可追踪到 document_id/chunk_id。
    WEB = "web"  # 联网证据：来自外部 URL，不能伪造本地 document_id/chunk_id。


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
    # requested identifiers 的中文含义是“本次检索实际使用的用户标识快照”。它应从
    # PlannerContext.query_identifiers 复制，供 Observation 自身校验 E020/E021 是否一致，
    # 也让持久化 Trace 不依赖外层 State 才能重放当时的编号判断。
    requested_identifiers: dict[str, list[str]] = Field(default_factory=dict)
    # matched identifiers 的中文含义是“证据实际命中的标识”。它记录候选 chunk 中真正
    # 出现的设备型号、报警码等值，不能用系统猜测值覆盖用户的 query_identifiers。
    matched_identifiers: dict[str, list[str]] = Field(default_factory=dict)
    # identifier resolution status 的中文含义是“编号确认状态”。它明确区分结构化精确
    # 命中、metadata 缺失后的同编号命中、只找到纠错候选，以及完全未找到。
    identifier_resolution_status: IdentifierResolutionStatus = IdentifierResolutionStatus.NOT_APPLICABLE
    # suggested identifiers 的中文含义是“建议用户确认的候选标识”。例如用户输入 E020、
    # 当前设备只找到 E021 时写入 {"alarm_code": ["E021"]}；候选不能进入最终 Citation。
    suggested_identifiers: dict[str, list[str]] = Field(default_factory=dict)
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

        # 标识字典中的 key 和 value 都属于后续 Planner/Trace 的机器协议。空 key、空列表
        # 或空白 value 会让“是否命中编号”的判断失真，因此在 Observation 边界统一拒绝。
        for mapping_name, identifier_mapping in (
            ("requested_identifiers", self.requested_identifiers),
            ("matched_identifiers", self.matched_identifiers),
            ("suggested_identifiers", self.suggested_identifiers),
        ):
            for identifier_type, identifier_values in identifier_mapping.items():
                if not str(identifier_type).strip():
                    raise ValueError(f"{mapping_name} 的标识类型不能为空")
                if not identifier_values or any(not str(value).strip() for value in identifier_values):
                    raise ValueError(f"{mapping_name}.{identifier_type} 必须包含非空标识")

        has_requested_identifiers = bool(self.requested_identifiers)
        has_matched_identifiers = bool(self.matched_identifiers)
        has_suggested_identifiers = bool(self.suggested_identifiers)
        identifier_requires_clarification = self.identifier_resolution_status in {
            IdentifierResolutionStatus.SUGGESTION_REQUIRED,
            IdentifierResolutionStatus.NOT_FOUND,
        }
        clarification_required = self.evidence_ambiguous or identifier_requires_clarification

        if clarification_required and not self.clarification_question:
            raise ValueError("证据存在歧义或标识需要确认时必须提供 clarification_question")
        if not clarification_required and self.clarification_question is not None:
            raise ValueError("证据和标识都不需要确认时 clarification_question 必须为空")

        if self.identifier_resolution_status == IdentifierResolutionStatus.NOT_APPLICABLE:
            if has_requested_identifiers:
                raise ValueError("NOT_APPLICABLE 状态不能包含 requested_identifiers")
        elif not has_requested_identifiers:
            raise ValueError("存在编号确认状态时必须包含 requested_identifiers")

        if self.identifier_resolution_status in {
            IdentifierResolutionStatus.EXACT_MATCH,
            IdentifierResolutionStatus.FALLBACK_EXACT_MATCH,
        } and not has_matched_identifiers:
            raise ValueError("标识命中状态必须包含 matched_identifiers")

        if self.identifier_resolution_status in {
            IdentifierResolutionStatus.EXACT_MATCH,
            IdentifierResolutionStatus.FALLBACK_EXACT_MATCH,
        }:
            for identifier_type, requested_values in self.requested_identifiers.items():
                matched_values = set(self.matched_identifiers.get(identifier_type, []))
                if not set(requested_values).issubset(matched_values):
                    raise ValueError(
                        "EXACT_MATCH/FALLBACK_EXACT_MATCH 的 matched_identifiers "
                        "必须覆盖 requested_identifiers"
                    )

        if self.identifier_resolution_status == IdentifierResolutionStatus.EXACT_MATCH and self.filter_fallback:
            raise ValueError("EXACT_MATCH 不能同时标记 filter_fallback")
        if (
            self.identifier_resolution_status == IdentifierResolutionStatus.FALLBACK_EXACT_MATCH
            and not self.filter_fallback
        ):
            raise ValueError("FALLBACK_EXACT_MATCH 必须标记 filter_fallback")

        if self.identifier_resolution_status == IdentifierResolutionStatus.SUGGESTION_REQUIRED:
            if not has_suggested_identifiers:
                raise ValueError("SUGGESTION_REQUIRED 必须包含 suggested_identifiers")
            if self.citation_count != 0:
                raise ValueError("用户确认候选标识前 citation_count 必须为 0")
            for identifier_type, suggested_values in self.suggested_identifiers.items():
                requested_values = set(self.requested_identifiers.get(identifier_type, []))
                if requested_values.intersection(suggested_values):
                    raise ValueError("suggested_identifiers 只能包含与用户请求不同的候选标识")
        elif has_suggested_identifiers:
            raise ValueError("只有 SUGGESTION_REQUIRED 状态可以包含 suggested_identifiers")

        if self.identifier_resolution_status == IdentifierResolutionStatus.NOT_FOUND and has_matched_identifiers:
            raise ValueError("NOT_FOUND 状态不能包含 matched_identifiers")

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
