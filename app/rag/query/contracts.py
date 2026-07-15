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


class RetrievalMode(str, Enum):
    """
    单次本地检索 Action 可以启用的固定召回组合。

    mode 的中文含义是“模式/组合”。它只决定一次 local/HyDE Action 内部创建哪些
    Milvus AnnSearchRequest，不决定 Planner 是否继续执行 HyDE 或 Web。
    """

    DENSE_LEARNED_SPARSE = "dense_learned_sparse"  # 稠密语义向量 + BGE-M3 学习式稀疏向量。
    DENSE_BM25 = "dense_bm25"  # 稠密语义向量 + Milvus BM25 精确词法通道。
    DENSE_LEARNED_SPARSE_BM25 = "dense_learned_sparse_bm25"  # 三路同时召回并按名次做 RRF。


class RetrievalChannel(str, Enum):
    """
    参与生成候选的底层通道配置和检索 Action 来源。

    前三个成员描述单次本地 Action 内的底层召回通道，后三个成员描述跨 Action 来源。
    当前 Milvus hybrid_search 只返回内部 RRF 后的列表，无法证明单个 chunk 逐条命中了
    哪个底层请求；因此本地候选记录的是该 Action 启用的模式通道集合，不是逐通道命中
    事实。original/HyDE/Web Action 来源则是确定的。
    """

    DENSE = "dense"  # 稠密向量通道：主要匹配自然语言语义。
    LEARNED_SPARSE = "learned_sparse"  # BGE-M3 学习式稀疏通道：兼顾关键词和模型学习权重。
    BM25 = "bm25"  # BM25 词法通道：擅长设备型号、报警码和其他精确字符。
    ORIGINAL = "original"  # 使用用户原问题/改写问题执行的普通本地检索 Action。
    HYDE = "hyde"  # 使用假设答案增强文本执行的本地检索 Action。
    WEB = "web"  # 由 Web Search Action 返回的联网候选。


class RetrievalCandidate(QueryContractModel):
    """
    从召回、跨 Action RRF、rerank 一直传到引用阶段的统一候选结构。

    Candidate 的中文含义是“候选证据”。本地候选必须始终保留 document/chunk/dataset
    身份；Web 候选没有本地身份，只能使用真实 URL。``retrieval_score`` 是 RRF 排名融合
    分，不能当证据充分阈值；``rerank_score`` 才是统一 reranker 对相关性的重新评分。
    """

    # 来源文档 ID。本地 chunk 的稳定文档身份；Web 候选必须为 None。
    document_id: str | None = None
    # 来源 chunk ID。本地去重、Trace 和 Citation 的核心键；Web 候选必须为 None。
    chunk_id: str | int | None = None
    # 所属知识库 ID。本地候选用于证明它仍处于本次 dataset 权限范围；Web 候选为 None。
    dataset_id: str | None = None
    # 文档索引产物版本。本地候选用于重放时确认使用哪一版 chunk；Web 候选为 None。
    index_version: int | None = Field(default=None, ge=0)
    # chunk 在文档中的顺序。用于后续相邻片段扩展；旧数据缺失时允许 None。
    chunk_index: int | None = Field(default=None, ge=0)
    # 查询时该候选是否处于启用状态。本地 chunk 必须为 bool；Web 没有本地启停状态。
    # 阶段 6 路线 B 接入查询过滤后，这里继续使用 enabled 表达“实际可召回状态”。
    enabled: bool | None = None
    # 当前切片或网页标题。供 rerank、答案上下文和引用展示使用。
    title: str = Field(min_length=1)
    # 来源文件标题。本地通常是原文件名；Web 可以为空字符串。
    source_title: str = ""
    # 已确认标准主题 ID。本地候选保留它以便 Trace 核对 subject filter；Web 为 None。
    subject_id: str | None = None
    # 标准主题展示名。用于答案上下文和调试，不代替稳定 subject_id。
    standard_subject_name: str | None = None
    # 当前标题的父级标题。用于保留文档章节层次，缺失时为 None。
    parent_title: str | None = None
    # 候选正文。本地来自 chunk content，Web 来自搜索摘要；不得只保留标题丢失正文。
    content: str = Field(min_length=1)
    # 以下六项是当前 chunk schema 已落地的设备运维 metadata。它们继续穿过 RRF/rerank，
    # 供编号同码核验、Trace 和后续 Citation 使用；Web 候选通常全部为 None。
    equipment_model: str | None = None  # 设备型号，例如 HAK 180。
    alarm_code: str | None = None  # 报警码/故障码，例如 E020。
    part_name: str | None = None  # 中文部件名称，例如温度传感器。
    sop_type: str | None = None  # SOP 类型，例如开机、维修或点检。
    safety_level: str | None = None  # 安全等级或风险提示。
    maintenance_stage: str | None = None  # 维护阶段，例如故障定位或复机确认。
    # 来源类型。local 表示 Milvus chunk，web 表示外部 URL，二者身份规则不同。
    source_type: EvidenceSourceType
    # 参与产生本候选的模式通道和 Action 来源，去重融合时取并集。Milvus 模式通道表示
    # “本 Action 启用了这些通道”，不能伪装成单个候选的逐通道命中明细。
    retrieval_channels: list[RetrievalChannel] = Field(min_length=1)
    # 当前排名，从 1 开始。每次 Action 内或跨 Action 重新排序后覆盖为新名次。
    retrieval_rank: PositiveStep
    # 当前 RRF/召回融合分，只用于候选排序与 Trace，不直接判断证据是否充分。
    retrieval_score: float = Field(ge=0)
    # 统一 reranker 分数。进入 rerank 前为 None，完成后通常是 0～1 的归一化分数。
    rerank_score: float | None = Field(default=None, ge=0, le=1)
    # Web 的真实链接；本地候选通常为空。Web 去重使用规范化 URL，不伪造 chunk_id。
    url: str | None = None

    @model_validator(mode="after")
    def validate_candidate_identity(self) -> Self:
        """保证本地和 Web 身份字段不会在格式转换或 rerank 时互相伪装。"""
        if len(set(self.retrieval_channels)) != len(self.retrieval_channels):
            raise ValueError("retrieval_channels 不能包含重复通道")

        if self.source_type == EvidenceSourceType.LOCAL:
            if (
                not self.document_id
                or self.chunk_id is None
                or not self.dataset_id
                or self.index_version is None
                or self.chunk_index is None
                or self.enabled is None
            ):
                raise ValueError(
                    "本地候选必须包含 document_id、chunk_id、dataset_id、index_version、chunk_index 和 enabled"
                )
            if self.url is not None and not self.url.strip():
                raise ValueError("本地候选的 url 如果存在就不能只有空白")
        else:
            if any(value is not None for value in (
                self.document_id,
                self.chunk_id,
                self.dataset_id,
                self.index_version,
                self.chunk_index,
                self.subject_id,
                self.enabled,
            )):
                raise ValueError("Web 候选不能伪造 document/chunk/dataset/index/enabled 本地身份")
            if not self.url or not self.url.strip():
                raise ValueError("Web 候选必须包含真实 url")
        return self


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
    # web_search_allowed 的中文含义是“本次查询是否允许联网检索”。它由 API、租户策略
    # 或后续图入口写入，不由 Planner 自行放宽。False 时即使本地证据不足，Planner 也
    # 只能安全拒答，不能为了提高回答率绕过调用方的联网边界。
    web_search_allowed: bool = True
    # safe_guard_triggered 的中文含义是“安全约束是否已经触发”。例如敏感操作规则、
    # 非法 Action 转移或上游防护节点发现风险时写 True；Planner 必须优先选择 refuse，
    # 不能继续执行检索或 answer。
    safe_guard_triggered: bool = False
    # planner_step 表示已经完成了多少次 Planner 决策。第八部分只读取并执行最大步数
    # 保护；真正递增和写入 Action history 由后续 LangGraph Planner/Action 节点负责。
    planner_step: NonNegativeInt = 0
    # max_steps 是单次查询允许的 Planner 决策上限。达到上限后必须安全终止，防止
    # local -> HyDE -> Web 等路径因状态错误产生无限循环。
    max_steps: PositiveStep
    # allowed_actions 是运行环境允许路由的 Action 白名单。REFUSE 必须始终存在，确保
    # 目标 Action 不合法或不可用时仍有确定性的安全出口。
    allowed_actions: list[QueryAction] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_allowed_actions(self) -> Self:
        """允许动作列表必须保序且无重复，避免路由白名单自身存在歧义。"""
        if len(self.allowed_actions) != len(set(self.allowed_actions)):
            raise ValueError("allowed_actions 不能包含重复 Action")
        if QueryAction.REFUSE not in self.allowed_actions:
            raise ValueError("allowed_actions 必须包含 refuse，保证 Planner 始终有安全出口")
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


class RetrievalTraceStatus(str, Enum):
    """一次完整查询 Trace 的持久化终态。"""

    RUNNING = "running"  # 运行中：入口已创建 Trace，但查询图尚未终止。
    COMPLETED = "completed"  # 已完成：answer、追问或业务拒答已经正常交付。
    FAILED = "failed"  # 已失败：出现未被 Planner 收口的编程错误或基础设施异常。


class TraceStepStatus(str, Enum):
    """Trace 中单个 Planner Action 的执行状态。"""

    PENDING = "pending"  # 待执行：Planner 已做出 Decision，对应 Action 尚未返回。
    COMPLETED = "completed"  # 已完成：Action 已正常执行，可能得到 SUCCESS 或 EMPTY Observation。
    FAILED = "failed"  # 已失败：Action 得到 FAILED Observation，或答案模型执行失败。


class UsageMetrics(QueryContractModel):
    """
    Planner 或答案模型的一次调用开销。

    规则 Planner 不调用模型，所以 token 和成本均为 0；答案 provider 没有返回 token 时也
    只能诚实记录 0，不能根据文本长度伪造精确 token 数。
    """

    input_tokens: NonNegativeInt = 0  # 输入 token 数；provider 未返回时为 0。
    output_tokens: NonNegativeInt = 0  # 输出 token 数；provider 未返回时为 0。
    total_tokens: NonNegativeInt = 0  # 总 token 数；通常等于输入与输出之和。
    duration_ms: NonNegativeInt = 0  # 本次 Planner 或答案调用的墙钟耗时，单位毫秒。
    estimated_cost: float = Field(default=0.0, ge=0)  # 估算费用；未配置计价规则时为 0。
    currency: str = "CNY"  # 费用币种。当前仅固化契约，不代表已经配置模型计价。


class RetrievalConfigSnapshot(QueryContractModel):
    """本次查询真正生效的检索参数快照；与版本字符串一起保存。"""

    retrieval_mode: RetrievalMode  # 单次 local/HyDE Action 内启用的召回通道组合。
    per_channel_topk: PositiveStep  # 每条 Milvus 底层通道最多返回的候选数。
    fusion_topk: PositiveStep  # 跨 Action RRF 融合后最多保留的累计候选数。
    rerank_min_topk: PositiveStep  # 动态 rerank 至少保留的证据数。
    rerank_max_topk: PositiveStep  # 动态 rerank 最多保留的证据数。
    rrf_k: PositiveStep  # RRF 公式中的平滑参数 k，不是最终候选数量。
    evidence_threshold: float = Field(ge=0, le=1)  # 允许进入 answer 的最低归一化 rerank 分数。
    web_fallback_enabled: bool  # 本地/HyDE 不足时是否允许 Planner 调用 Web。


class TraceEvidenceSummary(QueryContractModel):
    """Observation 的持久化证据摘要；只留身份、分数和正文 hash，不保存正文片段。"""

    document_id: str | None = None  # 本地文档 ID；Web 证据为空。
    chunk_id: str | int | None = None  # 本地 chunk ID；Web 证据为空。
    title: str = Field(min_length=1)  # 可读标题，用于人工排查 Trace。
    source_type: EvidenceSourceType  # local 或 web，决定身份字段语义。
    rerank_score: float | None = Field(default=None, ge=0, le=1)  # 统一 reranker 相关性分数。
    matched_identifiers: dict[str, list[str]] = Field(default_factory=dict)  # 证据实际命中的设备标识。
    content_excerpt_hash: str = Field(min_length=64, max_length=64)  # 摘要正文 SHA-256，用于核验而非还原正文。


class TraceObservation(QueryContractModel):
    """运行时 RetrievalObservation 去除正文后的可持久化投影。"""

    action: QueryAction  # 产生本 Observation 的检索 Action。
    status: ObservationStatus  # success、empty 或 failed。
    channel_counts: dict[str, NonNegativeInt] = Field(default_factory=dict)  # 已执行 Action 的候选数量。
    candidate_count: NonNegativeInt = 0  # 外层 RRF 前后参与判断的候选总数。
    reranked_count: NonNegativeInt = 0  # 通过编号保护并完成 rerank 的证据数。
    top_rerank_score: float | None = Field(default=None, ge=0, le=1)  # 第一名归一化 rerank 分数。
    requested_identifiers: dict[str, list[str]] = Field(default_factory=dict)  # 用户原始问题中的规范化标识。
    matched_identifiers: dict[str, list[str]] = Field(default_factory=dict)  # 证据命中的同编号标识。
    identifier_resolution_status: IdentifierResolutionStatus = IdentifierResolutionStatus.NOT_APPLICABLE
    suggested_identifiers: dict[str, list[str]] = Field(default_factory=dict)  # 只能用于追问的不同编号候选。
    citation_count: NonNegativeInt = 0  # 本轮形成的最终引用数；检索 Observation 通常为 0。
    evidence_summaries: list[TraceEvidenceSummary] = Field(default_factory=list)  # 不含正文的证据摘要。
    evidence_ambiguous: bool = False  # 是否存在必须由用户补充信息解决的证据冲突。
    clarification_question: str | None = None  # 可直接交付用户的确定性追问文本。
    duration_ms: NonNegativeInt = 0  # 当前检索 Action 与 rerank 的累计耗时。
    error_code: str | None = None  # failed 时的机器错误码，不保存异常堆栈和敏感正文。
    used_structured_filter: bool = False  # 是否尝试过设备标识精确过滤。
    filter_fallback: bool = False  # 精确过滤零命中后是否执行过宽松同码降级。


class TracePlannerStep(QueryContractModel):
    """一次 Planner Decision 及对应 Action/Observation 的持久化轨迹单元。"""

    step: PositiveStep  # 从 1 开始的稳定步骤号。
    input_observation: TraceObservation | None = None  # Planner 做决定前能看到的最近 Observation。
    decision: PlannerDecision  # 本步选择的 Action、实际 query 和 reason_code。
    execution_status: TraceStepStatus  # pending/completed/failed。
    output_observation: TraceObservation | None = None  # 检索 Action 完成后的结构化结果；终止 Action 为空。
    duration_ms: NonNegativeInt = 0  # 对应 Action 的耗时；pending 时先记录 Planner 决策耗时。
    planner_usage: UsageMetrics = Field(default_factory=UsageMetrics)  # 本步 Planner 的 token、耗时和成本。


class TraceChannelHit(QueryContractModel):
    """各真实检索 Action 返回的候选摘要，不复制完整 chunk 正文。"""

    channel: RetrievalChannel  # original、HyDE、Web 或本地底层模式通道。
    document_id: str | None = None  # 本地文档 ID；Web 为空。
    chunk_id: str | int | None = None  # 本地 chunk ID；Web 为空。
    index_version: int | None = Field(default=None, ge=0)  # 本地索引版本；Web 为空。
    enabled: bool | None = None  # 本地候选当时是否可召回；Web 没有本地启停状态。
    # 候选携带的召回通道集合。它表示“这个 Action 启用了哪些通道/来源”，不虚构
    # Milvus hybrid_search 内部逐底层请求命中事实。
    retrieval_channels: list[RetrievalChannel] = Field(min_length=1)
    entered_rerank: bool = False  # 该候选身份是否进入统一 rerank 结果集。
    became_citation: bool = False  # 该候选身份是否成为最终答案引用。
    rank: PositiveStep  # 候选在该 Action 原始列表中的排名。
    retrieval_score: float = Field(ge=0)  # 召回/RRF 分，仅用于排序记录。
    rerank_score: float | None = Field(default=None, ge=0, le=1)  # 最终统一 rerank 分，未入选时可为空。
    matched_identifiers: dict[str, list[str]] = Field(default_factory=dict)  # 候选文本/metadata 命中的标识。
    content_excerpt_hash: str = Field(min_length=64, max_length=64)  # 候选正文 SHA-256。


class RetrievalTrace(QueryContractModel):
    """Mongo ``retrieval_traces`` collection 中一条完整查询轨迹。"""

    trace_id: str = Field(min_length=1)  # 一次查询执行 ID；全局唯一，不等于 session_id。
    session_id: str = Field(min_length=1)  # 聊天会话 ID；一个会话可关联多条 Trace。
    owner_user_id: str = Field(min_length=1)  # 发起查询的用户，用于归属和后续权限查询。
    tenant_id: str = Field(min_length=1)  # 当前租户范围；现阶段通常为 tenant_default。
    dataset_ids: list[str] = Field(min_length=1)  # 本次允许查询的知识库范围快照。
    original_query: str = Field(min_length=1)  # 用户未经改写的原始问题。
    rewritten_query: str = ""  # 主体确认后用于检索的改写问题。
    subject_ids: list[str] = Field(default_factory=list)  # 已确认的稳定主题 ID。
    standard_subject_names: list[str] = Field(default_factory=list)  # 主题可读名称。
    query_identifiers: dict[str, list[str]] = Field(default_factory=dict)  # 用户输入的型号/报警码等标识。
    identifier_resolution_status: IdentifierResolutionStatus = IdentifierResolutionStatus.NOT_APPLICABLE
    suggested_identifiers: dict[str, list[str]] = Field(default_factory=dict)  # 需用户确认的不同编号候选。
    policy_version: str = Field(min_length=1)  # Planner 规则/模型策略版本。
    planner_type: str = Field(min_length=1)  # rule 或 model，不能从类名猜测。
    provider: str | None = None  # Planner 模型服务方；规则 Planner 为 null。
    model_id: str | None = None  # Planner 模型 ID；规则 Planner 为 null。
    model_revision: str | None = None  # Planner 模型修订版本；规则 Planner 为 null。
    prompt_version: str | None = None  # Planner 提示词版本；规则 Planner 为 null。
    planner_usage: UsageMetrics = Field(default_factory=UsageMetrics)  # 全部 Planner 步骤累计开销。
    answer_provider: str | None = None  # 最终答案模型服务方；追问/拒答时为 null。
    answer_model_id: str | None = None  # 最终答案模型 ID；追问/拒答时为 null。
    answer_model_revision: str | None = None  # 最终答案模型修订版本。
    answer_prompt_version: str | None = None  # 答案 Prompt 版本，只记版本不保存完整 Prompt。
    answer_usage: UsageMetrics = Field(default_factory=UsageMetrics)  # 答案模型 token、耗时和成本。
    retrieval_config_version: str = Field(min_length=1)  # 检索配置版本名。
    retrieval_mode: RetrievalMode  # 本次 local/HyDE 使用的召回组合。
    retrieval_config_snapshot: RetrievalConfigSnapshot  # 本次真正生效的完整参数快照。
    index_versions: list[int] = Field(default_factory=list)  # 最终候选涉及的本地索引版本集合。
    status: RetrievalTraceStatus  # running/completed/failed。
    terminal_action: QueryAction | None = None  # 最终 answer、追问或拒答动作。
    terminal_reason_code: PlannerReasonCode | None = None  # 最终命中的机器可读规则原因。
    planner_steps: list[TracePlannerStep] = Field(default_factory=list)  # 按执行顺序保存的 Action Trace。
    channel_hits: list[TraceChannelHit] = Field(default_factory=list)  # 各 Action 候选的无正文投影。
    final_citations: list[Citation] = Field(default_factory=list)  # 最终实际进入答案上下文的引用。
    started_at: str = Field(min_length=1)  # 查询入口 UTC ISO 时间。
    completed_at: str | None = None  # completed/failed 的 UTC ISO 时间；running 时为空。
    total_duration_ms: NonNegativeInt = 0  # 从查询入口到终态的总墙钟耗时。
    error_code: str | None = None  # 未处理异常的类型码；不保存敏感异常正文。
