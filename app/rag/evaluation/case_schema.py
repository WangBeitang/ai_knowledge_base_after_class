"""
阶段 8 评测样本、环境快照和评测结果的稳定数据契约。

这个模块只定义 schema（数据形状和校验规则），不执行评测、不调用模型、不访问数据库。
阶段 8 后续脚本会用这些对象读取 JSONL 样本、保存环境快照、记录 baseline 评测结果，
阶段 9 的 SFT/GRPO 数据导出也会复用这些字段边界。

为什么这里注释要比较细：
- 这些字段会长期写入 JSONL、报告和训练导出文件，名字一旦漂移会影响可复现性。
- split、leakage_group、snapshot、Reward、Trace 等术语会直接决定“是否发生测试集泄漏”。
- 用户私有文档和公共 Demo 知识库必须分开标记，避免把私有数据混进公开报告或训练样本。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.rag.query.contracts import QueryAction
from app.rag.query.rrf_service import canonicalize_web_url
from app.shared.config.knowledge_base_config import DEFAULT_DATASET_ID, DEFAULT_TENANT_ID


# Milvus 里 chunk_id 可能是 int，也可能在 JSON/Trace 中以字符串形式流转；这里显式
# 允许两种类型，但后续校验会拒绝空字符串和 bool，避免把 True/False 误当作 1/0。
ChunkId = str | int


class Stage8SchemaModel(BaseModel):
    """
    阶段 8 schema 公共基类。

    extra='forbid' 的中文含义是“拒绝未知字段”：如果 JSONL 拼错字段名，应立即报错，
    不能静默忽略后让评测报告缺少关键标注。validate_assignment 让测试中动态改字段时
    也能触发同样校验，避免构造出的对象和真实文件加载行为不一致。
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


class CaseSplit(str, Enum):
    """评测样本用途拆分。split 是训练/调参/最终测试的硬边界。"""

    TRAIN = "train"  # 训练候选集：允许导出 SFT/阶段 9 训练样本，但仍需 Reward 或人工复核筛选。
    DEV = "dev"  # 开发集：用于调规则、Reward 权重、Prompt 和模型候选，不作为最终结论。
    TEST = "test"  # 隔离测试集：最终考试集，不允许参与调参、模型选择或训练轨迹挑选。
    DEMO_REGRESSION = "demo_regression"  # 面试 Demo/快速回归集：可展示，不进入训练。


class CaseGroup(str, Enum):
    """样本业务分组，用于报告按设备运维场景汇总指标。"""

    CORE = "core"  # 核心设备运维问题，例如报警码、SOP、维护步骤。
    COLLOQUIAL = "colloquial"  # 口语化问法，用于检查同义表达是否影响召回和 Planner 路由。
    PRIVATE_DOC = "private_doc"  # 用户私有文档样本，用于验证权限隔离和私有资料评测。
    REALTIME = "realtime"  # 明确需要实时信息的样本，通常期望触发 web_search。
    REFUSAL = "refusal"  # 应拒答样本，例如本地/外部都无可靠证据或超出范围。
    CLARIFICATION = "clarification"  # 应追问样本，例如主体歧义或报警码相近需要确认。
    DEMO = "demo"  # 面试展示样本，要求稳定、可解释，但不用于训练。


class PrivacyScope(str, Enum):
    """样本数据范围。用于控制报告展示和训练导出的隐私边界。"""

    PUBLIC_DEMO = "public_demo"  # 公共 Demo 知识库样本，可进入面试报告和公开演示。
    PRIVATE_USER = "private_user"  # 用户私有文档样本，报告和导出时必须确认脱敏/授权。


class ChunkRelevance(str, Enum):
    """期望 chunk 的证据强度。Reward 计算时 required 权重大于 supporting/acceptable。"""

    REQUIRED = "required"  # 必须命中的核心证据，未命中会明显影响 R_retrieval/R_citation。
    SUPPORTING = "supporting"  # 支撑性证据，命中有加分，未命中不一定判整条失败。
    ACCEPTABLE = "acceptable"  # 可接受替代证据，用于同一答案要点有多个等价 chunk 的场景。


class LabelSource(str, Enum):
    """标注来源。来源不同会影响可信度和是否能导出训练。"""

    MANUAL = "manual"  # 人工直接标注，通常可信度最高。
    FEEDBACK = "feedback"  # 来自 Trace Feedback 的人工反馈，需要汇总成完整 case 后使用。
    API_ASSISTED = "api_assisted"  # 顶级 API 辅助生成，必须经过 Reward 筛选或人工复核。
    SYNTHETIC = "synthetic"  # 文档约束合成样本，关键样本仍需人工复核。


class GoldOrigin(str, Enum):
    """
    Gold 数据的生产方式和允许用途。

    它与 ``label_source`` 不同：label_source 说明标签由人工、反馈还是 API 辅助产生；
    gold_origin 说明证据是在人工策划边界后入库，还是先经过生产切分再反向生成问题。
    该字段会进入训练样本和 manifest，防止不同证据难度的数据被无标记混合。
    """

    UNSPECIFIED = "unspecified"  # 非 Gold 或历史 case；旧阶段 8 文件默认使用，不自动获得 Gold 身份。
    CURATED_SEED_GOLD = "curated_seed_gold"  # 人工策划原子证据的高置信种子；只允许 train/回归。
    ROUTE_SEED_GOLD = "route_seed_gold"  # 阶段 9 人工路线种子；用于训练 Planner Action 覆盖，只允许 train，不代表 held-out 事实评测。
    PRODUCTION_CHUNK_GOLD = "production_chunk_gold"  # 生产文档先切分入库、再基于真实 chunk 生成的 Gold。
    HELDOUT_GOLD = "heldout_gold"  # 独立来源冻结 Gold；只用于 dev/test，禁止训练导出。


class HumanReviewStatus(str, Enum):
    """人工复核状态。它决定样本能否作为高可信训练/评测证据。"""

    REVIEWED = "reviewed"  # 已复核，可用于正式评测；是否进训练还要看 split 和 Reward。
    PENDING = "pending"  # 待复核，可暂存为候选，不能当作高可信标签。
    REJECTED = "rejected"  # 已拒绝，不应进入正式评测或训练导出。


class PlannerMode(str, Enum):
    """阶段 8 可评测的 Planner 模式。mode 会写入结果文件用于 baseline 对比。"""

    RULE = "rule"  # 当前线上规则 Planner，也是阶段 8 的最低可用 baseline。
    API = "api"  # 顶级 API Planner，离线强参考/teacher 候选，不直接当标准答案。
    LOCAL_BASE = "local_base"  # 本地开源模型零样本 Planner，用于评估训练前起点。
    SFT = "sft"  # 阶段 9 SFT 后模型，阶段 8 只预留结果枚举。
    GRPO = "grpo"  # 阶段 9 GRPO 后模型，阶段 8 不训练也不启用。


def _normalize_text_list(values: list[str], *, field_name: str, allow_empty: bool = True) -> list[str]:
    """
    清洗字符串列表。

    保持输入顺序是为了让报告和导出文件稳定；去重是为了避免同一个 expected_subject 或
    case_id 因重复填写影响统计。allow_empty=false 用在 dataset_ids/test_user_ids 这类
    必须存在的边界字段。
    """
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        value = str(raw_value or "").strip()
        if not value:
            continue
        if value not in seen:
            normalized.append(value)
            seen.add(value)
    if not allow_empty and not normalized:
        raise ValueError(f"{field_name} 至少包含一个非空值")
    return normalized


def _validate_chunk_id(value: ChunkId) -> ChunkId:
    """
    校验 chunk_id 的运行时身份。

    chunk_id 同时被 Trace、Citation、ExpectedChunk 和 Reward 使用，是连接“命中证据”
    与“最终引用”的核心主键。bool 在 Python 中是 int 子类，必须显式拒绝，避免 True 被
    序列化为 1 后污染真实 Milvus chunk_id。
    """
    if isinstance(value, bool):
        raise ValueError("chunk_id 不能是 bool")
    if isinstance(value, int):
        return value
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError("chunk_id 不能为空")
    return normalized


class ExpectedChunk(Stage8SchemaModel):
    """
    一条评测样本期望命中的 chunk 版本身份。

    必须同时保存 document_id、chunk_id、index_version。单独保存 chunk_id 不够，因为
    文档重建索引后可能产生新版本 chunk，旧标注不能套到新 index_version 上。
    """

    # 来源文档 ID。来自 documents.document_id，用于证明期望证据属于哪个文档。
    document_id: str = Field(min_length=1)
    # Milvus chunk 主键。后续和 Trace/Citation 的 chunk_id 对齐，用于算召回和引用命中。
    chunk_id: ChunkId
    # 文档索引版本。用于防止旧 chunk 标注误套到重建后的新索引版本。
    index_version: int = Field(ge=0)
    # 证据强度。Reward 可按 required/supporting/acceptable 给予不同权重。
    relevance: ChunkRelevance = ChunkRelevance.REQUIRED
    # 该 chunk 支撑的答案要点 ID。用于把检索命中和答案覆盖率关联起来。
    answer_point_ids: list[str] = Field(default_factory=list)

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id(cls, chunk_id: ChunkId) -> ChunkId:
        return _validate_chunk_id(chunk_id)

    @field_validator("answer_point_ids")
    @classmethod
    def normalize_answer_point_ids(cls, answer_point_ids: list[str]) -> list[str]:
        return _normalize_text_list(answer_point_ids, field_name="answer_point_ids")

    @model_validator(mode="after")
    def reject_template_placeholders(self) -> Self:
        # 模板文件可以放示例值，但正式 case 不能保留“请替换”占位符，否则报告会看起来
        # 可运行却实际没有真实标注。
        if "请替换" in self.document_id or "请替换" in str(self.chunk_id):
            raise ValueError("expected_chunks 不能保留模板占位符")
        return self


class ExpectedWebEvidence(Stage8SchemaModel):
    """
    一条 Web Gold（网页标准证据）的冻结身份。

    Web 页面没有 ``document_id + chunk_id + index_version``。因此用规范化 URL 作为运行
    时检索/引用身份，并用抓取响应和人工摘取事实的 SHA256 固定审核时看到的页面版本。
    Reward 只按规范化 URL 评价运行时命中；两个 hash 用于数据构建、复核和漂移审计，
    不能伪装成运行时已经重新下载并验证了同一页面内容。
    """

    # 同一网页快照在 manifest、case 和审核报告之间使用的稳定 ID。
    source_id: str = Field(min_length=1)
    # 官方发布者，例如 Huawei；用于报告展示，不参与 URL 身份匹配。
    publisher: str = Field(min_length=1)
    # 冻结页面标题。
    source_title: str = Field(min_length=1)
    # 抓取时的原始 URL；运行时会先规范化再与 Citation.source 比较。
    url: str = Field(min_length=1)
    # UTC ISO 时间，说明网页事实在哪个时点被冻结。
    captured_at: str = Field(min_length=1)
    # 抓取到的原始 HTTP 响应正文 hash；证明审核时看到的页面版本。
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    # 事实列表规范化 JSON 的 hash；页面排版变化时仍可独立核对审核事实。
    evidence_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    # 该页面中被当前 case 使用的事实 ID。
    fact_ids: list[str] = Field(min_length=1)
    # 该网页证据支撑的答案要点 ID。
    answer_point_ids: list[str] = Field(default_factory=list)

    @field_validator("fact_ids", "answer_point_ids")
    @classmethod
    def normalize_fact_ids(cls, values: list[str]) -> list[str]:
        return _normalize_text_list(values, field_name="web_evidence_ids")

    @field_validator("url")
    @classmethod
    def validate_web_url(cls, url: str) -> str:
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            raise ValueError("expected_web_evidence.url 必须是可追溯的 HTTP(S) URL")
        return url

    @field_validator("captured_at")
    @classmethod
    def validate_captured_at(cls, captured_at: str) -> str:
        try:
            parsed = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                "expected_web_evidence.captured_at 必须是 UTC ISO 时间"
            ) from exc
        if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
            raise ValueError(
                "expected_web_evidence.captured_at 必须包含 UTC 时区"
            )
        return captured_at

    @property
    def canonical_url(self) -> str:
        """返回与 Web 候选和 Citation 共用的 URL 身份。"""
        return canonicalize_web_url(self.url)


class ExpectedBehavior(Stage8SchemaModel):
    """
    人工期望的 Planner 终态行为和受控 Action 边界。

    这里描述“应该怎么做”，不是描述模型实际做了什么。实际 Action path 会写入
    PlannerEvalResult，再由 Reward 对比 expected_behavior 评分。
    """

    # 是否应该回答。为 true 时必须至少存在本地 chunk 或冻结 Web 证据，并填写答案要点。
    should_answer: bool
    # 是否应该拒答。例如缺少可靠证据、超出知识库范围或安全边界触发。
    should_refuse: bool
    # 是否应该追问。例如主体不明确、报警码相近但不能直接纠错。
    should_ask_clarification: bool
    # 是否应该调用 Web。只有明显实时信息或本地知识不足且允许联网时才应为 true。
    should_call_web: bool
    # 需要 Web 的业务原因。用于报告解释，不参与当前硬校验。
    web_required_reason: str = ""
    # 明确禁止出现的 Action。例如内部设备手册问题通常禁止 web_search。
    forbidden_actions: list[QueryAction] = Field(default_factory=list)

    @field_validator("forbidden_actions")
    @classmethod
    def dedupe_forbidden_actions(cls, forbidden_actions: list[QueryAction]) -> list[QueryAction]:
        normalized: list[QueryAction] = []
        seen: set[QueryAction] = set()
        for action in forbidden_actions:
            if action not in seen:
                normalized.append(action)
                seen.add(action)
        return normalized

    @model_validator(mode="after")
    def validate_terminal_expectation(self) -> Self:
        # 一条样本的终态只能是回答、拒答、追问三选一；Web 是中间 Action，不是终态。
        terminal_count = sum((
            self.should_answer,
            self.should_refuse,
            self.should_ask_clarification,
        ))
        if terminal_count != 1:
            raise ValueError("expected_behavior 必须且只能选择 answer/refuse/ask_clarification 中的一种")
        if self.should_call_web and QueryAction.WEB_SEARCH in self.forbidden_actions:
            raise ValueError("should_call_web=true 时不能把 web_search 放入 forbidden_actions")
        return self


class PlannerEvalCase(Stage8SchemaModel):
    """
    一条阶段 8 Planner 评测样本。

    它是后续评测、baseline 对比、Reward 计算和 SFT 导出的共同输入。注意它不是一次
    查询 Trace，而是“人工定义的问题与期望”。真实运行得到的轨迹写入 PlannerEvalResult。
    """

    # 样本稳定 ID。文件、报告、结果和训练导出都用它关联，不能复用。
    case_id: str = Field(min_length=1)
    # 业务分组。用于报告按核心/口语/拒答/追问等场景拆指标。
    case_group: CaseGroup
    # 数据用途拆分。test/demo 是硬隔离边界，不能进入训练导出。
    split: CaseSplit
    # 泄漏隔离组。同一问题的改写、口语变体、同一报警码同源样本必须在同一 split。
    leakage_group_id: str = Field(min_length=1)
    # 用户原始问题。Environment reset 时应原样写入 State.original_query。
    query: str = Field(min_length=1)
    # 同义改写候选。用于后续扩充数据，不默认全部参与评测。
    query_variants: list[str] = Field(default_factory=list)
    # 本次样本允许查询的知识库范围。默认设备运维 Demo 知识库。
    dataset_ids: list[str] = Field(default_factory=lambda: [DEFAULT_DATASET_ID])
    # 固定测试用户。用于复现权限过滤，避免不同用户导致召回范围漂移。
    owner_user_id: str = Field(min_length=1)
    # 租户上下文。当前通常为 tenant_default，但仍要写入以复现 shared/public 过滤。
    tenant_id: str = DEFAULT_TENANT_ID
    # 隐私范围。公共 Demo 和用户私有文档在报告、导出、展示时必须分开处理。
    privacy_scope: PrivacyScope = PrivacyScope.PUBLIC_DEMO
    # 来源文档集合。用于文档级泄漏隔离和 expected_chunks 归属校验。
    source_document_ids: list[str] = Field(default_factory=list)
    # 来源文档索引版本快照。key=document_id，value=index_version。
    source_index_versions: dict[str, int] = Field(default_factory=dict)
    # 期望主题 ID。用于固定主体，避免主体识别 LLM 波动污染 Planner 评测。
    expected_subject_ids: list[str] = Field(default_factory=list)
    # 期望主题展示名。用于报告阅读，不作为稳定主键。
    expected_subject_names: list[str] = Field(default_factory=list)
    # 期望本地证据 chunk。本地回答型样本必须填写，用于 R_retrieval 和 R_citation。
    expected_chunks: list[ExpectedChunk] = Field(default_factory=list)
    # 期望 Web 证据。Web 回答型样本必须填写，绑定 URL、抓取 hash 和事实 ID。
    expected_web_evidence: list[ExpectedWebEvidence] = Field(default_factory=list)
    # 期望答案要点。用于 R_answer，回答型样本必须填写。
    expected_answer_points: list[str] = Field(default_factory=list)
    # 期望行为。描述应该回答/拒答/追问/是否 Web。
    expected_behavior: ExpectedBehavior
    # 可接受 Action 路径。允许多条合法路径，例如 local->answer 或 local->hyde->answer。
    acceptable_action_paths: list[list[QueryAction]] = Field(default_factory=list)
    # 期望结构化标识，例如设备型号、报警码、SOP 编号。用于编号安全和检索评价。
    expected_identifiers: dict[str, list[str]] = Field(default_factory=dict)
    # 标注来源。API 辅助、合成、人工反馈的可信度不同。
    label_source: LabelSource
    # Gold 生产方式。默认 unspecified 保持阶段 8 历史 case 兼容；正式训练种子必须显式填写。
    gold_origin: GoldOrigin = GoldOrigin.UNSPECIFIED
    # 人工复核状态。pending/rejected 不能当作高可信训练标签。
    human_review_status: HumanReviewStatus
    # 人工备注。仅用于排查和报告，不参与评分。
    notes: str = ""

    @field_validator("query_variants", "source_document_ids", "expected_subject_ids", "expected_subject_names")
    @classmethod
    def normalize_optional_text_lists(cls, values: list[str]) -> list[str]:
        return _normalize_text_list(values, field_name="text_list")

    @field_validator("dataset_ids")
    @classmethod
    def normalize_dataset_ids(cls, dataset_ids: list[str]) -> list[str]:
        return _normalize_text_list(dataset_ids, field_name="dataset_ids", allow_empty=False)

    @field_validator("expected_answer_points")
    @classmethod
    def normalize_answer_points(cls, expected_answer_points: list[str]) -> list[str]:
        return _normalize_text_list(expected_answer_points, field_name="expected_answer_points")

    @field_validator("source_index_versions")
    @classmethod
    def validate_source_index_versions(cls, source_index_versions: dict[str, int]) -> dict[str, int]:
        normalized: dict[str, int] = {}
        for raw_document_id, raw_index_version in source_index_versions.items():
            document_id = str(raw_document_id or "").strip()
            if not document_id:
                raise ValueError("source_index_versions 的 document_id 不能为空")
            if int(raw_index_version) < 0:
                raise ValueError("source_index_versions 的 index_version 不能小于 0")
            normalized[document_id] = int(raw_index_version)
        return normalized

    @field_validator("expected_identifiers")
    @classmethod
    def normalize_expected_identifiers(cls, expected_identifiers: dict[str, list[str]]) -> dict[str, list[str]]:
        normalized: dict[str, list[str]] = {}
        for raw_key, raw_values in expected_identifiers.items():
            key = str(raw_key or "").strip()
            if not key:
                raise ValueError("expected_identifiers 的 key 不能为空")
            values = _normalize_text_list(raw_values, field_name=f"expected_identifiers.{key}")
            if values:
                normalized[key] = values
        return normalized

    @model_validator(mode="after")
    def validate_case_boundaries(self) -> Self:
        # 回答型样本没有任何证据或答案要点，会让 Reward 无法判断“答对了没有”，
        # 因此在 schema 层直接拒绝。本地和 Web 使用不同身份，不能强迫 Web 伪造 chunk。
        if self.expected_behavior.should_answer:
            if not self.expected_chunks and not self.expected_web_evidence:
                raise ValueError(
                    "should_answer=true 的样本必须标注 expected_chunks 或 expected_web_evidence"
                )
            if not self.expected_answer_points:
                raise ValueError("should_answer=true 的样本必须标注 expected_answer_points")

        if self.expected_web_evidence and not self.expected_behavior.should_call_web:
            raise ValueError(
                "存在 expected_web_evidence 时 expected_behavior.should_call_web 必须为 true"
            )
        if (
            self.expected_behavior.should_call_web
            and self.expected_behavior.should_answer
            and not self.expected_web_evidence
        ):
            raise ValueError(
                "Web 回答型样本必须标注 expected_web_evidence，不能只用本地 chunk 代替网页事实"
            )

        if not self.acceptable_action_paths:
            raise ValueError("acceptable_action_paths 至少包含一条可接受 Action 路径")

        for path in self.acceptable_action_paths:
            if not path:
                raise ValueError("acceptable_action_paths 不能包含空路径")
            forbidden = set(self.expected_behavior.forbidden_actions)
            if forbidden.intersection(path):
                raise ValueError("acceptable_action_paths 不能包含 forbidden_actions 中的 Action")

        expected_terminal = self._expected_terminal_action()
        if not any(path[-1] == expected_terminal for path in self.acceptable_action_paths):
            raise ValueError("acceptable_action_paths 必须包含符合 expected_behavior 的终止 Action")

        if self.expected_behavior.should_call_web and not any(
            QueryAction.WEB_SEARCH in path for path in self.acceptable_action_paths
        ):
            raise ValueError("should_call_web=true 的样本至少需要一条包含 web_search 的可接受路径")

        source_document_ids = set(self.source_document_ids)
        for chunk in self.expected_chunks:
            # 如果填写了 source_document_ids，expected chunk 必须来自其中之一；这能避免
            # 人工标注把其他文档的 chunk 错挂到当前 case。
            if source_document_ids and chunk.document_id not in source_document_ids:
                raise ValueError("expected_chunks 的 document_id 必须属于 source_document_ids")
            expected_version = self.source_index_versions.get(chunk.document_id)
            # index_version 是防泄漏和可复现的关键边界。版本不一致时不能继续评测。
            if expected_version is not None and expected_version != chunk.index_version:
                raise ValueError("expected_chunks 的 index_version 必须与 source_index_versions 一致")

        canonical_web_urls = [
            evidence.canonical_url for evidence in self.expected_web_evidence
        ]
        if len(canonical_web_urls) != len(set(canonical_web_urls)):
            raise ValueError("expected_web_evidence 不能重复标注同一个规范化 URL")
        return self

    def _expected_terminal_action(self) -> QueryAction:
        """把 expected_behavior 映射成终态 Action，用于校验 acceptable_action_paths。"""
        if self.expected_behavior.should_answer:
            return QueryAction.ANSWER
        if self.expected_behavior.should_refuse:
            return QueryAction.REFUSE
        return QueryAction.ASK_CLARIFICATION


class SnapshotDocument(Stage8SchemaModel):
    """环境快照中的文档版本摘要，用于证明评测使用的是哪一版语料。"""

    # 文档业务 ID，关联 documents.document_id。
    document_id: str = Field(min_length=1)
    # 所属知识库 ID，必须在 EnvironmentSnapshot.dataset_ids 内。
    dataset_id: str = Field(min_length=1)
    # 文档索引产物版本。重建索引后必须变化，用于 replay/corpus match 判断。
    index_version: int = Field(ge=0)
    # 快照创建时的可见性，例如 private/shared/public。
    visibility: str = Field(min_length=1)
    # 当前版本 chunk 数量，用于快照摘要和基本完整性检查。
    chunk_count: int = Field(ge=0)


class SnapshotChunkIdentity(Stage8SchemaModel):
    """快照中用于校验启停状态的 chunk 版本身份。"""

    # 所属文档 ID。
    document_id: str = Field(min_length=1)
    # chunk 主键，可能是 int 或字符串。
    chunk_id: ChunkId
    # 所属文档索引版本。
    index_version: int = Field(ge=0)

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id(cls, chunk_id: ChunkId) -> ChunkId:
        return _validate_chunk_id(chunk_id)


class EnvironmentSnapshot(Stage8SchemaModel):
    """
    一次离线评测使用的 dataset、语料、配置和模型边界快照。

    Snapshot 的中文含义是“快照”。它不是当前线上状态的引用，而是评测运行时必须固定的
    事实清单。后续 Offline Environment 应优先读取 snapshot，发现当前 Mongo/Milvus 与
    快照不一致时标记 mismatch，不能静默当作同一次评测。
    """

    # 快照 ID。写入每条 PlannerEvalResult，用于证明结果来自同一环境。
    snapshot_id: str = Field(min_length=1)
    # 快照创建时间。建议使用 UTC ISO 字符串。
    created_at: str = Field(min_length=1)
    # 创建者。可以是操作者 ID 或脚本名，用于审计。
    created_by: str = Field(min_length=1)
    # 固定知识库范围。所有 case.dataset_ids 必须属于这里或由后续脚本显式检查。
    dataset_ids: list[str] = Field(default_factory=lambda: [DEFAULT_DATASET_ID])
    # 固定测试用户。用于复现权限过滤和私有文档范围。
    test_user_ids: list[str] = Field(default_factory=list)
    # 文档版本摘要。用于判断 corpus_match_status。
    documents: list[SnapshotDocument] = Field(default_factory=list)
    # 启用 chunk 快照。key=document_id，value=当时可参与召回的 chunk_id 列表。
    enabled_chunks: dict[str, list[ChunkId]] = Field(default_factory=dict)
    # 人工禁用覆盖快照。来自 chunk_status_overrides，执行时不能读取漂移后的当前值。
    disabled_chunks: list[SnapshotChunkIdentity] = Field(default_factory=list)
    # 检索配置版本名，例如 retrieval-stage5-final-v1。
    retrieval_config_version: str = Field(min_length=1)
    # 检索配置真实参数快照，例如 mode、top-k、RRF k、rerank 阈值、Web 开关。
    retrieval_config_snapshot: dict[str, Any] = Field(default_factory=dict)
    # Planner 策略版本，例如 rule-v1。policy version 变化时不能混入同一 baseline。
    policy_version: str = Field(min_length=1)
    # reranker 模型和阈值信息。用于解释重排分数和引用质量。
    reranker: dict[str, Any] = Field(default_factory=dict)
    # 答案模型信息。用于区分 Planner 质量和答案生成模型变化。
    answer_model: dict[str, Any] = Field(default_factory=dict)
    # 可评测 Planner 注册摘要。记录 rule/api/local_base 是否可运行及原因。
    planner_registry: list[dict[str, Any]] = Field(default_factory=list)
    # 可选 hash。用于快速检测源文件、索引或配置是否漂移。
    source_hashes: dict[str, str] = Field(default_factory=dict)

    @field_validator("dataset_ids", "test_user_ids")
    @classmethod
    def normalize_required_text_lists(cls, values: list[str]) -> list[str]:
        return _normalize_text_list(values, field_name="snapshot_text_list", allow_empty=False)

    @field_validator("enabled_chunks")
    @classmethod
    def normalize_enabled_chunks(cls, enabled_chunks: dict[str, list[ChunkId]]) -> dict[str, list[ChunkId]]:
        normalized: dict[str, list[ChunkId]] = {}
        for raw_document_id, chunk_ids in enabled_chunks.items():
            document_id = str(raw_document_id or "").strip()
            if not document_id:
                raise ValueError("enabled_chunks 的 document_id 不能为空")
            values: list[ChunkId] = []
            seen: set[str] = set()
            for raw_chunk_id in chunk_ids:
                chunk_id = _validate_chunk_id(raw_chunk_id)
                key = str(chunk_id)
                if key not in seen:
                    values.append(chunk_id)
                    seen.add(key)
            normalized[document_id] = values
        return normalized

    @model_validator(mode="after")
    def validate_document_identity(self) -> Self:
        # 同一 document_id + index_version 只能出现一次；否则快照无法判断某个 chunk 属于
        # 哪份文档版本。
        document_keys: set[tuple[str, int]] = set()
        document_ids: set[str] = set()
        for document in self.documents:
            key = (document.document_id, document.index_version)
            if key in document_keys:
                raise ValueError("documents 不能包含重复的 document_id + index_version")
            document_keys.add(key)
            document_ids.add(document.document_id)
            if document.dataset_id not in self.dataset_ids:
                raise ValueError("documents 的 dataset_id 必须属于 snapshot.dataset_ids")
        for document_id in self.enabled_chunks:
            if document_ids and document_id not in document_ids:
                raise ValueError("enabled_chunks 的 document_id 必须属于 documents")
        for chunk in self.disabled_chunks:
            if document_ids and chunk.document_id not in document_ids:
                raise ValueError("disabled_chunks 的 document_id 必须属于 documents")
        return self


class PlannerEvalResult(Stage8SchemaModel):
    """
    单条 case 在一个 planner baseline 下的评测结果。

    Result 是“实际运行结果”，与 PlannerEvalCase 的“人工期望”分开保存。后续报告会按
    case_id + planner_mode + snapshot_id 聚合比较 rule/api/local_base。
    """

    # 本次评测运行 ID。一个 run 可以包含多个 case 和多个 planner。
    run_id: str = Field(min_length=1)
    # 来源 case ID，关联 PlannerEvalCase.case_id。
    case_id: str = Field(min_length=1)
    # 来源 split。复制到结果里是为了报告无需回查 case 文件。
    split: CaseSplit
    # 实际运行的 planner 模式。
    planner_mode: PlannerMode
    # 使用的环境快照 ID。
    snapshot_id: str = Field(min_length=1)
    # 使用的 Reward 版本。Reward 权重变化必须换版本，不能覆盖旧结果。
    reward_version: str = Field(min_length=1)
    # 如果运行写入 retrieval_traces，这里保存 trace_id；纯文件评测可为空。
    trace_id: str = ""
    # 实际 Action 路径，例如 [local_search, answer]。
    action_path: list[QueryAction] = Field(default_factory=list)
    # 最终 Action，通常是 answer/refuse/ask_clarification。
    terminal_action: QueryAction | None = None
    # 终止原因码。保留字符串，便于兼容不同 Planner 的 reason code。
    terminal_reason_code: str = ""
    # rerank 后候选 chunk ID，用于 recall/MRR/nDCG。
    retrieved_chunk_ids: list[ChunkId] = Field(default_factory=list)
    # 最终 Citation 指向的 chunk ID，用于引用命中率。
    citation_chunk_ids: list[ChunkId] = Field(default_factory=list)
    # 指标明细，例如 recall_at_k、mrr、ndcg、citation_hit。
    metrics: dict[str, float | int | bool | None] = Field(default_factory=dict)
    # Reward 分项和总分。这里先作为结构化 dict，Reward v1 任务再细化 schema。
    reward: dict[str, Any] = Field(default_factory=dict)
    # token、耗时、成本等运行用量。规则 Planner 通常为 0 或空。
    usage: dict[str, Any] = Field(default_factory=dict)
    # 格式错误、执行错误或跳过原因。评测脚本不能只丢弃失败 case。
    errors: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("retrieved_chunk_ids", "citation_chunk_ids")
    @classmethod
    def validate_chunk_ids(cls, chunk_ids: list[ChunkId]) -> list[ChunkId]:
        return [_validate_chunk_id(chunk_id) for chunk_id in chunk_ids]


class LogicalTestSet(Stage8SchemaModel):
    """同一 ``test`` split 内的逻辑测试集边界和运行权限。"""

    # 该逻辑测试集冻结的 case 数量；必须和对应 case_id 清单一致。
    case_count: int = Field(ge=0)
    # 是否计入五路线覆盖矩阵。原 35 条核心回答集为 false，新路线留出集为 true。
    counts_toward_route_matrix: bool
    # 是否允许参与 Prompt、Reward 或 checkpoint 选择；heldout 必须为 false。
    allowed_for_model_selection: bool | None = None
    # 数据不可进入训练等静态用途说明。
    policy: str = ""
    # 何时允许执行该逻辑测试集的运行门禁。
    run_policy: str = ""


class SplitManifest(Stage8SchemaModel):
    """
    阶段 8 split 边界记录。

    Manifest 的中文含义是“清单”。它记录 case_id 和 leakage_group 到 split 的映射，
    后续导出训练数据、生成报告或复跑评测时可以证明 train/dev/test/demo 没有混用。
    """

    # manifest 版本 ID。真实样本填充后应更新。
    manifest_id: str = Field(min_length=1)
    # 创建时间。建议 UTC ISO 字符串。
    created_at: str = Field(min_length=1)
    # 对应环境快照 ID。任务 8.1 允许为空，任务 8.3 后应填写。
    snapshot_id: str = ""
    # 训练候选 case_id 列表。
    train_case_ids: list[str] = Field(default_factory=list)
    # 开发验证 case_id 列表。
    dev_case_ids: list[str] = Field(default_factory=list)
    # 隔离测试 case_id 列表，禁止导出训练。
    test_case_ids: list[str] = Field(default_factory=list)
    # 原有核心回答测试集的 case_id 子集；只做历史回答能力回归。
    core_answer_test_case_ids: list[str] = Field(default_factory=list)
    # 新增五路线留出测试集的 case_id 子集；禁止参与模型选择。
    route_heldout_test_case_ids: list[str] = Field(default_factory=list)
    # test split 内逻辑测试集的数量、用途和运行门禁。
    logical_test_sets: dict[str, LogicalTestSet] = Field(default_factory=dict)
    # Demo 回归 case_id 列表，禁止导出训练。
    demo_regression_case_ids: list[str] = Field(default_factory=list)
    # 泄漏组到 split 的映射。用于防止同一问题改写跨 split。
    leakage_group_to_split: dict[str, CaseSplit] = Field(default_factory=dict)
    # 拆分说明和例外原因。
    notes: str = ""

    @field_validator(
        "train_case_ids",
        "dev_case_ids",
        "test_case_ids",
        "core_answer_test_case_ids",
        "route_heldout_test_case_ids",
        "demo_regression_case_ids",
    )
    @classmethod
    def normalize_case_ids(cls, case_ids: list[str]) -> list[str]:
        return _normalize_text_list(case_ids, field_name="case_ids")

    @model_validator(mode="after")
    def reject_case_id_cross_split(self) -> Self:
        split_lists = {
            CaseSplit.TRAIN: self.train_case_ids,
            CaseSplit.DEV: self.dev_case_ids,
            CaseSplit.TEST: self.test_case_ids,
            CaseSplit.DEMO_REGRESSION: self.demo_regression_case_ids,
        }
        seen: dict[str, CaseSplit] = {}
        for split, case_ids in split_lists.items():
            for case_id in case_ids:
                previous_split = seen.get(case_id)
                if previous_split and previous_split != split:
                    raise ValueError(f"case_id={case_id} 不能同时属于 {previous_split.value} 和 {split.value}")
                seen[case_id] = split
        if self.core_answer_test_case_ids or self.route_heldout_test_case_ids:
            core_ids = set(self.core_answer_test_case_ids)
            heldout_ids = set(self.route_heldout_test_case_ids)
            test_ids = set(self.test_case_ids)
            if core_ids & heldout_ids:
                raise ValueError(
                    "core_answer_test 与 route_heldout_test 的 case_id 不能重叠"
                )
            if core_ids | heldout_ids != test_ids:
                raise ValueError(
                    "两个逻辑测试集的 case_id 并集必须恰好等于 test_case_ids"
                )
            expected_keys = {"core_answer_test", "route_heldout_test"}
            if set(self.logical_test_sets) != expected_keys:
                raise ValueError(
                    "logical_test_sets 必须同时定义 core_answer_test 和 "
                    "route_heldout_test"
                )
            if (
                self.logical_test_sets["core_answer_test"].case_count
                != len(core_ids)
                or self.logical_test_sets["route_heldout_test"].case_count
                != len(heldout_ids)
            ):
                raise ValueError("logical_test_sets.case_count 与对应 case_id 清单不一致")
        return self


def validate_case_collection(cases: list[PlannerEvalCase]) -> None:
    """校验一组样本的跨行边界：case_id 唯一，leakage_group_id 不跨 split。"""
    case_ids: set[str] = set()
    leakage_split: dict[str, CaseSplit] = {}
    for case in cases:
        if case.case_id in case_ids:
            raise ValueError(f"case_id 不能重复：{case.case_id}")
        case_ids.add(case.case_id)

        previous_split = leakage_split.get(case.leakage_group_id)
        if previous_split and previous_split != case.split:
            raise ValueError(
                "同一 leakage_group_id 不能跨 split："
                f"{case.leakage_group_id} 同时属于 {previous_split.value} 和 {case.split.value}"
            )
        leakage_split[case.leakage_group_id] = case.split


def validate_cases_for_sft_export(cases: list[PlannerEvalCase]) -> None:
    """导出 SFT 前的硬边界：test 和 demo_regression 样本不能进入训练数据。"""
    invalid_case_ids = [
        case.case_id
        for case in cases
        if case.split in {CaseSplit.TEST, CaseSplit.DEMO_REGRESSION}
    ]
    if invalid_case_ids:
        raise ValueError("test/demo 样本不能导出训练：" + ", ".join(invalid_case_ids))


def load_planner_cases(path: str | Path, *, allow_empty: bool = False) -> list[PlannerEvalCase]:
    """读取阶段 8 JSONL 样本并执行跨样本校验。空行和 # 注释行会被跳过。"""
    case_path = Path(path)
    cases: list[PlannerEvalCase] = []
    with case_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                cases.append(PlannerEvalCase.model_validate_json(line))
            except Exception as error:
                raise ValueError(f"{case_path}:{line_number} 评测样本非法：{error}") from error
    if not cases and not allow_empty:
        raise ValueError(f"{case_path} 没有可用评测样本")
    validate_case_collection(cases)
    return cases


__all__ = [
    "CaseGroup",
    "CaseSplit",
    "ChunkRelevance",
    "EnvironmentSnapshot",
    "ExpectedBehavior",
    "ExpectedChunk",
    "GoldOrigin",
    "HumanReviewStatus",
    "LabelSource",
    "PlannerEvalCase",
    "PlannerEvalResult",
    "PlannerMode",
    "PrivacyScope",
    "SnapshotChunkIdentity",
    "SnapshotDocument",
    "SplitManifest",
    "load_planner_cases",
    "validate_case_collection",
    "validate_cases_for_sft_export",
]
