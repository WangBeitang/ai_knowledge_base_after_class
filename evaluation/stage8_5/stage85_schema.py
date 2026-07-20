"""阶段 8.5 公开数据处理入口的文件化契约和通用校验工具。

阶段 8.5 的目标不是直接训练 Planner，而是先把公开数据变成可审计的候选输入。
这里的 schema（数据形状和校验规则）服务 `evaluation/stage8_5/` 下的一次性脚本：

- 来源清单必须说明许可证和训练用途边界。
- 故障场景卡片必须把表格、时序、音频或文档片段转成可解释业务语义。
- Planner 候选 case 必须继续复用阶段 8 的 `PlannerEvalCase`，避免训练数据另起一套格式。

说明几个英文术语：
- `source manifest`：来源清单，记录每个公开数据源的来源、许可证和审批状态。
- `fault card`：故障场景卡片，把原始数据解释成“故障现象、原因、排查步骤、处理建议”。
- `approved/review/rejected pool`：通过池、审核池、拒绝池，用于防止未审核数据直接进入训练。
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.rag.evaluation.case_schema import (
    CaseSplit,
    HumanReviewStatus,
    PlannerEvalCase,
    SplitManifest,
    validate_case_collection,
)


T = TypeVar("T", bound=BaseModel)


class Stage85SchemaModel(BaseModel):
    """阶段 8.5 schema 公共基类。

    `extra='forbid'` 的中文含义是“拒绝未知字段”。公开数据处理最怕字段拼错后被静默忽略，
    因此这里延续阶段 8 的严格策略：清单、卡片、报告中出现未知字段时直接报错。
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


class SourceType(str, Enum):
    """公开数据来源类型。

    每个枚举成员都对应一种处理边界：文档类可以直接切 chunk，表格/时序/音频/图片必须先
    解释成故障场景卡片，社区二次数据必须额外检查可信度。
    """

    MANUAL = "manual"  # 设备手册，优先级最高，可直接构建 RAG 文档和 expected_chunks。
    SOP = "sop"  # 维护 SOP，适合生成步骤类问答和安全边界。
    ALARM_CODE = "alarm_code"  # 报警码说明，适合生成设备型号 + 报警码查询。
    TABLE = "table"  # 表格数据，必须先解释字段含义，不能直接塞进向量库。
    TIMESERIES = "timeseries"  # 时序数据，必须先转成故障现象和诊断卡片。
    AUDIO = "audio"  # 音频数据，只能转异常声音现象卡片，不直接当 RAG 文本。
    IMAGE = "image"  # 图像数据，只能转缺陷描述或检查结论卡片。
    COMMUNITY = "community"  # 社区/二次整理数据，许可证和可信度必须更严格检查。
    OTHER = "other"  # 其他来源，默认进入 pending，人工确认后才能 approved。


class ApprovalStatus(str, Enum):
    """来源或样本审批状态。

    这个状态决定数据能否继续向后流动：`approved` 才能进入通过池，`pending` 只能进入审核池，
    `rejected` 必须保留拒绝原因，方便后续复盘为什么不能训练。
    """

    APPROVED = "approved"  # 已确认来源、许可证和训练用途边界，可以进入候选生成。
    PENDING = "pending"  # 信息还不完整，只能保留等待人工审核。
    REJECTED = "rejected"  # 明确不能使用，必须写明 reject_reason。


class IssueSeverity(str, Enum):
    """校验问题严重程度。"""

    ERROR = "error"  # 硬错误：会让当前脚本返回非 0，或让记录进入 rejected。
    WARNING = "warning"  # 警告：记录可保留，但需要人工关注。
    INFO = "info"  # 信息：用于报告统计，不阻断流程。


class Stage85Issue(Stage85SchemaModel):
    """阶段 8.5 校验问题明细。

    每条 issue 都必须能定位到来源、卡片或 case。报告只展示统计是不够的，后续人工复核
    需要知道哪一条记录、哪个字段触发了门禁。
    """

    severity: IssueSeverity = Field(description="问题严重程度，决定脚本是否失败或记录是否进入拒绝池。")
    code: str = Field(min_length=1, description="机器可读错误码，用于测试、报告和后续过滤统计。")
    message: str = Field(min_length=1, description="中文说明，描述问题原因和处理建议。")
    location: str = Field(default="", description="文件行号或字段路径，例如 source_manifest.jsonl:3。")
    source_id: str = Field(default="", description="关联来源 ID，没有来源上下文时为空。")
    case_id: str = Field(default="", description="关联 PlannerEvalCase ID，没有 case 上下文时为空。")
    card_id: str = Field(default="", description="关联故障场景卡片 ID，没有 card 上下文时为空。")


class LicenseRecord(Stage85SchemaModel):
    """许可证清单记录。

    许可证记录用于给多个 source 复用同一套授权判断。字段允许写得比 source 更完整：
    source 可以只填 `license_name`，校验时从这里补齐 training/redistribution/commercial 边界。
    """

    license_id: str = Field(min_length=1, description="许可证稳定 ID，例如 cc-by-4.0。")
    license_name: str = Field(min_length=1, description="许可证展示名，例如 CC BY 4.0。")
    license_url: str = Field(default="", description="许可证说明链接；为空时不能让来源自动 approved。")
    redistribution_allowed: bool | None = Field(default=None, description="是否允许再分发；未知用 null。")
    training_allowed: bool | None = Field(default=None, description="是否允许训练或派生训练数据；未知用 null。")
    commercial_use_allowed: bool | None = Field(default=None, description="是否允许商业使用；未知用 null。")
    notes: str = Field(default="", description="许可证人工备注，例如需要署名或相同方式共享。")


class SourceRecord(Stage85SchemaModel):
    """公开数据来源记录，对应 `source_manifest.jsonl` 的一行。

    它是阶段 8.5 的第一道门禁。后续故障卡片、候选 case、报告都要能回溯到 source_id。
    """

    source_id: str = Field(min_length=1, description="来源稳定 ID，全局唯一，后续卡片和 case 都用它追溯来源。")
    source_type: SourceType = Field(description="来源类型，决定是直接切 chunk 还是先转故障场景卡片。")
    title: str = Field(min_length=1, description="来源标题，报告展示和人工复核使用。")
    publisher: str = Field(default="", description="发布方或维护方，用于判断可信度。")
    url_or_path: str = Field(min_length=1, description="原始 URL 或本地路径；必须可追溯。")
    collected_at: str = Field(default="", description="采集时间，建议 ISO 字符串；为空表示尚未真实采集。")
    source_hash: str = Field(default="", description="原文件 hash，用于发现内容漂移；未下载时可为空。")
    license_name: str = Field(default="", description="许可证名称，可从 license_manifest.jsonl 关联补齐。")
    license_url: str = Field(default="", description="许可证说明链接；approved 来源必须能解析出许可证说明。")
    redistribution_allowed: bool | None = Field(default=None, description="是否允许再分发；未知用 null。")
    training_allowed: bool | None = Field(default=None, description="是否允许训练使用；approved 来源必须为 true。")
    commercial_use_allowed: bool | None = Field(default=None, description="是否允许商业使用；未知不阻断研究，但报告必须记录。")
    approval_status: ApprovalStatus = Field(description="审批状态，只有 approved 可以进入候选生成。")
    reject_reason: str = Field(default="", description="rejected/pending 的原因；approved 时通常为空。")
    notes: str = Field(default="", description="人工备注，例如字段含义、抽样边界或署名要求。")

    @model_validator(mode="after")
    def validate_status_reason(self) -> "SourceRecord":
        if self.approval_status is ApprovalStatus.REJECTED and not self.reject_reason:
            raise ValueError("approval_status=rejected 时必须填写 reject_reason")
        return self


class EvidenceChunkRef(Stage85SchemaModel):
    """故障场景卡片引用的证据 chunk 版本身份。

    这里和阶段 8 的 ExpectedChunk 使用同一个身份原则：必须同时有 document_id、chunk_id、
    index_version，避免文档重建后把旧证据标注套到新版本上。
    """

    document_id: str = Field(min_length=1, description="来源文档 ID，后续会写入 PlannerEvalCase.source_document_ids。")
    chunk_id: str | int = Field(description="chunk 主键，可以是 Milvus int，也可以是 JSON 字符串。")
    index_version: int = Field(ge=0, description="文档索引版本，用于防止证据版本漂移。")
    relevance: str = Field(default="required", description="证据强度：required/supporting/acceptable。")
    answer_point_ids: list[str] = Field(default_factory=list, description="该 chunk 支撑的答案要点 ID。")

    @field_validator("relevance")
    @classmethod
    def validate_relevance(cls, value: str) -> str:
        normalized = value.strip()
        if normalized not in {"required", "supporting", "acceptable"}:
            raise ValueError("relevance 必须是 required/supporting/acceptable")
        return normalized


class FaultScenarioCard(Stage85SchemaModel):
    """故障场景卡片，对应 `fault_scenario_cards.jsonl` 的一行。

    卡片是“原始公开数据”和“PlannerEvalCase”之间的中间层。表格、时序、音频、图片不能
    直接变成训练样本，必须先解释成这些业务字段，人工才能判断是否可信。
    """

    card_id: str = Field(min_length=1, description="场景卡片稳定 ID。")
    source_id: str = Field(min_length=1, description="关联来源 ID，必须出现在 approved source 清单中。")
    source_document_id: str = Field(default="", description="导入 RAG 后的 document_id，尚未导入时可为空。")
    source_section: str = Field(default="", description="来源章节、页码、表格名或字段名，用于泄漏分组。")
    equipment_model: str = Field(default="", description="设备型号，例如 MetroPT-3 APU 或某型号直线轴。")
    component_name: str = Field(default="", description="部件或零件名称，例如 bearing、ball screw、valve。")
    alarm_code: str = Field(default="", description="报警码；没有报警码的数据可为空。")
    symptom: str = Field(min_length=1, description="故障现象，候选问题和答案要点会围绕它生成。")
    possible_causes: list[str] = Field(default_factory=list, description="可能原因列表。")
    diagnostic_steps: list[str] = Field(default_factory=list, description="排查步骤列表。")
    maintenance_actions: list[str] = Field(default_factory=list, description="处理建议列表。")
    safety_notes: list[str] = Field(default_factory=list, description="安全注意事项列表。")
    evidence_text: str = Field(default="", description="可追溯原文证据或字段解释，报告和人工复核使用。")
    evidence_chunk_ids: list[EvidenceChunkRef] = Field(default_factory=list, description="该卡片对应的版本化证据 chunk。")
    quality_flags: list[str] = Field(default_factory=list, description="质量标记，例如 weak_evidence 或 ambiguous_identifier。")
    candidate_queries: list[str] = Field(default_factory=list, description="可生成 PlannerEvalCase 的候选问题。")
    expected_answer_points: list[str] = Field(default_factory=list, description="候选问题的答案要点。")

    @field_validator(
        "possible_causes",
        "diagnostic_steps",
        "maintenance_actions",
        "safety_notes",
        "quality_flags",
        "candidate_queries",
        "expected_answer_points",
    )
    @classmethod
    def normalize_text_lists(cls, values: list[str]) -> list[str]:
        return _normalize_text_list(values)


class RejectedCandidateRecord(Stage85SchemaModel):
    """无法通过阶段 8 `PlannerEvalCase` schema 的候选记录。"""

    raw_payload: dict[str, Any] = Field(default_factory=dict, description="原始 JSON payload，便于人工修复。")
    issues: list[Stage85Issue] = Field(default_factory=list, description="导致拒绝的结构化问题。")


class Stage85QualityReport(Stage85SchemaModel):
    """阶段 8.5 数据处理入口报告。

    这是机器可读 JSON 报告，Markdown 报告会从它生成。它不保存 chunk 正文或答案 Prompt，
    只保存数量、路径、问题和边界信息。
    """

    report_version: str = Field(default="stage8.5-entry-v1", description="报告 schema 版本。")
    generated_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"),
        description="报告生成时间，UTC ISO 字符串。",
    )
    files: dict[str, str] = Field(default_factory=dict, description="本次读取或写入的文件路径。")
    source_counts: dict[str, int] = Field(default_factory=dict, description="来源按 approved/pending/rejected 的数量。")
    card_counts: dict[str, int] = Field(default_factory=dict, description="故障场景卡片数量统计。")
    case_counts: dict[str, int] = Field(default_factory=dict, description="候选 case 的 approved/review/rejected 数量。")
    split_counts: dict[str, int] = Field(default_factory=dict, description="train/dev/test/demo_regression 数量。")
    approved_source_ids: list[str] = Field(default_factory=list, description="通过来源门禁的 source_id 列表。")
    issues: list[Stage85Issue] = Field(default_factory=list, description="所有校验问题。")


def read_jsonl(path: str | Path, model_type: type[T], *, allow_missing: bool = False) -> list[T]:
    """读取 JSONL 并按指定 Pydantic schema 校验。

    空行和 `#` 开头的注释行会跳过，方便模板文件保留中文说明。读取失败时带上行号，
    否则后续人工很难定位是哪条公开来源或卡片有问题。
    """

    jsonl_path = Path(path)
    if allow_missing and not jsonl_path.exists():
        return []
    records: list[T] = []
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                payload = json.loads(line)
                records.append(model_type.model_validate(payload))
            except Exception as error:
                raise ValueError(f"{jsonl_path}:{line_number} JSONL 记录非法：{error}") from error
    return records


def write_jsonl(path: str | Path, records: Iterable[BaseModel | dict[str, Any]]) -> None:
    """写出 JSONL 文件；每条记录一行，便于后续人工 diff 和增量追加。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for record in records:
        payload = record.model_dump(mode="json") if isinstance(record, BaseModel) else record
        lines.append(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_json(path: str | Path, payload: BaseModel | dict[str, Any]) -> None:
    """写出格式化 JSON，主要用于 data_quality_report 和 split_manifest。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_source_records(
        sources: Sequence[SourceRecord],
        licenses: Sequence[LicenseRecord],
) -> Stage85QualityReport:
    """执行来源门禁，确保 approved 来源有明确许可证和训练权限。"""

    issues: list[Stage85Issue] = []
    source_ids = [source.source_id for source in sources]
    duplicate_source_ids = sorted(_duplicates(source_ids))
    for source_id in duplicate_source_ids:
        issues.append(_issue("error", "duplicate_source_id", f"source_id 重复：{source_id}", source_id=source_id))

    licenses_by_name = {license_record.license_name: license_record for license_record in licenses}
    approved_source_ids: list[str] = []
    status_counter = Counter(source.approval_status.value for source in sources)

    for source in sources:
        license_record = licenses_by_name.get(source.license_name)
        if source.approval_status is ApprovalStatus.APPROVED:
            _validate_approved_source(source, license_record, issues)
            if not any(issue.source_id == source.source_id and issue.severity is IssueSeverity.ERROR for issue in issues):
                approved_source_ids.append(source.source_id)
        elif source.approval_status is ApprovalStatus.PENDING and not source.reject_reason and not source.notes:
            issues.append(_issue(
                "warning",
                "pending_without_reason",
                "pending 来源建议写明缺少什么信息，方便后续补齐。",
                source_id=source.source_id,
            ))

    return Stage85QualityReport(
        files={},
        source_counts={
            "total": len(sources),
            "approved": status_counter.get(ApprovalStatus.APPROVED.value, 0),
            "pending": status_counter.get(ApprovalStatus.PENDING.value, 0),
            "rejected": status_counter.get(ApprovalStatus.REJECTED.value, 0),
        },
        approved_source_ids=approved_source_ids,
        issues=issues,
    )


def filter_fault_cards_by_sources(
        cards: Sequence[FaultScenarioCard],
        approved_source_ids: Sequence[str],
) -> tuple[list[FaultScenarioCard], list[Stage85Issue]]:
    """过滤故障场景卡片，只保留来自 approved source 的卡片。"""

    approved_set = set(approved_source_ids)
    accepted_cards: list[FaultScenarioCard] = []
    issues: list[Stage85Issue] = []
    for card in cards:
        if card.source_id not in approved_set:
            issues.append(_issue(
                "error",
                "card_source_not_approved",
                "故障场景卡片关联的 source_id 未通过来源门禁，不能生成训练候选。",
                source_id=card.source_id,
                card_id=card.card_id,
            ))
            continue
        accepted_cards.append(card)
    return accepted_cards, issues


def build_case_payloads_from_cards(
        cards: Sequence[FaultScenarioCard],
        *,
        dataset_id: str,
        owner_user_id: str,
        tenant_id: str,
        split: CaseSplit,
) -> tuple[list[dict[str, Any]], list[RejectedCandidateRecord]]:
    """从故障场景卡片生成阶段 8 PlannerEvalCase payload。

    这里故意只生成 `human_review_status=pending` 的候选样本。自动生成只能进入审核池，
    不能直接成为训练正样本。
    """

    payloads: list[dict[str, Any]] = []
    rejected: list[RejectedCandidateRecord] = []
    for card in cards:
        missing_reasons = _candidate_generation_blockers(card)
        if missing_reasons:
            rejected.append(RejectedCandidateRecord(
                raw_payload=card.model_dump(mode="json"),
                issues=[
                    _issue("error", "card_missing_case_fields", reason, source_id=card.source_id, card_id=card.card_id)
                    for reason in missing_reasons
                ],
            ))
            continue

        for index, query in enumerate(card.candidate_queries, start=1):
            evidence_chunks = [
                {
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id,
                    "index_version": chunk.index_version,
                    "relevance": chunk.relevance,
                    "answer_point_ids": chunk.answer_point_ids,
                }
                for chunk in card.evidence_chunk_ids
            ]
            source_versions = {
                chunk.document_id: chunk.index_version
                for chunk in card.evidence_chunk_ids
            }
            source_document_ids = sorted(source_versions)
            payloads.append({
                "case_id": f"stage85-{card.card_id}-{index:03d}",
                "case_group": "core",
                "split": split.value,
                "leakage_group_id": _leakage_group_id(card),
                "query": query,
                "query_variants": [],
                "dataset_ids": [dataset_id],
                "owner_user_id": owner_user_id,
                "tenant_id": tenant_id,
                "privacy_scope": "public_demo",
                "source_document_ids": source_document_ids,
                "source_index_versions": source_versions,
                "expected_subject_ids": [],
                "expected_subject_names": _normalize_text_list([card.equipment_model, card.component_name]),
                "expected_chunks": evidence_chunks,
                "expected_answer_points": card.expected_answer_points,
                "expected_behavior": {
                    "should_answer": True,
                    "should_refuse": False,
                    "should_ask_clarification": False,
                    "should_call_web": False,
                    "web_required_reason": "",
                    "forbidden_actions": ["web_search"],
                },
                "acceptable_action_paths": [
                    ["local_search", "answer"],
                    ["local_search", "hyde_search", "answer"],
                ],
                "expected_identifiers": _expected_identifiers(card),
                "label_source": "synthetic",
                "human_review_status": HumanReviewStatus.PENDING.value,
                "notes": (
                    f"stage8.5 synthetic candidate; source_id={card.source_id}; "
                    f"card_id={card.card_id}; source_section={card.source_section}"
                ),
            })
    return payloads, rejected


def validate_candidate_payloads(
        payloads: Sequence[dict[str, Any]],
) -> tuple[list[PlannerEvalCase], list[PlannerEvalCase], list[RejectedCandidateRecord], Stage85QualityReport]:
    """校验候选 case，并按 reviewed/pending/rejected 分流。

    `approved` 只表示“schema 合法且人工已复核”，不代表已经进入训练。是否进入 SFT/GRPO 还要
    在后续阶段看 split、Reward v1.1 阈值和隐私边界。
    """

    approved: list[PlannerEvalCase] = []
    review: list[PlannerEvalCase] = []
    rejected: list[RejectedCandidateRecord] = []

    for payload in payloads:
        try:
            case = PlannerEvalCase.model_validate(payload)
        except Exception as error:
            rejected.append(RejectedCandidateRecord(
                raw_payload=payload,
                issues=[
                    _issue(
                        "error",
                        "planner_case_schema_error",
                        f"PlannerEvalCase schema 校验失败：{error}",
                        case_id=str(payload.get("case_id", "")),
                    )
                ],
            ))
            continue

        if case.human_review_status is HumanReviewStatus.REJECTED:
            rejected.append(RejectedCandidateRecord(
                raw_payload=case.model_dump(mode="json"),
                issues=[
                    _issue(
                        "error",
                        "case_human_rejected",
                        "human_review_status=rejected，不能进入训练候选。",
                        case_id=case.case_id,
                    )
                ],
            ))
        elif case.human_review_status is HumanReviewStatus.REVIEWED:
            approved.append(case)
        else:
            review.append(case)

    all_valid_cases = approved + review
    issues: list[Stage85Issue] = [issue for record in rejected for issue in record.issues]
    try:
        validate_case_collection(all_valid_cases)
    except ValueError as error:
        issues.append(_issue("error", "candidate_collection_error", str(error)))

    split_counter = Counter(case.split.value for case in all_valid_cases)
    report = Stage85QualityReport(
        case_counts={
            "approved": len(approved),
            "review": len(review),
            "rejected": len(rejected),
            "valid_total": len(all_valid_cases),
            "total": len(payloads),
        },
        split_counts=dict(sorted(split_counter.items())),
        issues=issues,
    )
    return approved, review, rejected, report


def build_split_manifest(
        cases: Sequence[PlannerEvalCase],
        *,
        manifest_id: str,
        snapshot_id: str,
        notes: str,
) -> SplitManifest:
    """根据已通过 schema 的 case 生成 split_manifest.json。"""

    validate_case_collection(list(cases))
    case_ids_by_split: dict[CaseSplit, list[str]] = {
        CaseSplit.TRAIN: [],
        CaseSplit.DEV: [],
        CaseSplit.TEST: [],
        CaseSplit.DEMO_REGRESSION: [],
    }
    leakage_group_to_split: dict[str, CaseSplit] = {}
    for case in cases:
        case_ids_by_split[case.split].append(case.case_id)
        leakage_group_to_split[case.leakage_group_id] = case.split

    return SplitManifest(
        manifest_id=manifest_id,
        created_at=_utc_now(),
        snapshot_id=snapshot_id,
        train_case_ids=sorted(case_ids_by_split[CaseSplit.TRAIN]),
        dev_case_ids=sorted(case_ids_by_split[CaseSplit.DEV]),
        test_case_ids=sorted(case_ids_by_split[CaseSplit.TEST]),
        demo_regression_case_ids=sorted(case_ids_by_split[CaseSplit.DEMO_REGRESSION]),
        leakage_group_to_split=dict(sorted(leakage_group_to_split.items())),
        notes=notes,
    )


def merge_reports(*reports: Stage85QualityReport, files: dict[str, str] | None = None) -> Stage85QualityReport:
    """合并多个脚本阶段的统计，供 Markdown 报告生成使用。"""

    merged = Stage85QualityReport(files=files or {})
    for report in reports:
        merged.source_counts.update(report.source_counts)
        merged.card_counts.update(report.card_counts)
        merged.case_counts.update(report.case_counts)
        merged.split_counts.update(report.split_counts)
        merged.approved_source_ids.extend(source_id for source_id in report.approved_source_ids if source_id not in merged.approved_source_ids)
        merged.issues.extend(report.issues)
        merged.files.update(report.files)
    return merged


def load_json(path: str | Path, *, allow_missing: bool = False) -> dict[str, Any]:
    """读取普通 JSON 文件；报告脚本使用。"""

    json_path = Path(path)
    if allow_missing and not json_path.exists():
        return {}
    return json.loads(json_path.read_text(encoding="utf-8"))


def _validate_approved_source(
        source: SourceRecord,
        license_record: LicenseRecord | None,
        issues: list[Stage85Issue],
) -> None:
    """approved 来源的硬门禁。"""

    if not source.license_name:
        issues.append(_issue("error", "approved_missing_license", "approved 来源必须填写 license_name。", source_id=source.source_id))
    if source.license_name and license_record is None:
        issues.append(_issue("error", "approved_license_not_in_manifest", "approved 来源的 license_name 必须出现在 license_manifest.jsonl。", source_id=source.source_id))
    license_url = source.license_url or (license_record.license_url if license_record else "")
    if not license_url:
        issues.append(_issue("error", "approved_missing_license_url", "approved 来源必须能追溯许可证说明链接。", source_id=source.source_id))
    if _effective_permission(source.training_allowed, license_record.training_allowed if license_record else None) is not True:
        issues.append(_issue("error", "approved_training_not_allowed", "approved 来源必须明确 training_allowed=true。", source_id=source.source_id))
    if _effective_permission(source.redistribution_allowed, license_record.redistribution_allowed if license_record else None) is False:
        issues.append(_issue("error", "approved_redistribution_forbidden", "禁止再分发的来源不能进入 approved。", source_id=source.source_id))


def _candidate_generation_blockers(card: FaultScenarioCard) -> list[str]:
    """判断一张故障卡片是否足够生成回答型 Planner case。"""

    blockers: list[str] = []
    if not card.candidate_queries:
        blockers.append("candidate_queries 为空，无法生成 query。")
    if not card.expected_answer_points:
        blockers.append("expected_answer_points 为空，无法评价答案要点覆盖。")
    if not card.evidence_chunk_ids:
        blockers.append("evidence_chunk_ids 为空，回答型样本无法生成 expected_chunks。")
    return blockers


def _leakage_group_id(card: FaultScenarioCard) -> str:
    """生成保守泄漏组 ID，保证同源设备/章节/报警码不会跨 split。"""

    parts = [
        card.source_id,
        card.source_section or "unknown_section",
        card.equipment_model or "unknown_equipment",
        card.alarm_code or card.component_name or "unknown_topic",
    ]
    return "-".join(_slug(part) for part in parts if part)


def _expected_identifiers(card: FaultScenarioCard) -> dict[str, list[str]]:
    """把卡片里的设备、报警码和部件名转成阶段 8 expected_identifiers。"""

    identifiers = {
        "equipment_model": _normalize_text_list([card.equipment_model]),
        "alarm_code": _normalize_text_list([card.alarm_code]),
        "part_name": _normalize_text_list([card.component_name]),
    }
    return {key: values for key, values in identifiers.items() if values}


def _issue(
        severity: str,
        code: str,
        message: str,
        *,
        location: str = "",
        source_id: str = "",
        case_id: str = "",
        card_id: str = "",
) -> Stage85Issue:
    """创建统一 issue 对象，避免各脚本手写不同字段名。"""

    return Stage85Issue(
        severity=IssueSeverity(severity),
        code=code,
        message=message,
        location=location,
        source_id=source_id,
        case_id=case_id,
        card_id=card_id,
    )


def _effective_permission(source_value: bool | None, license_value: bool | None) -> bool | None:
    """source 显式值优先，否则使用 license_manifest 中的默认值。"""

    return source_value if source_value is not None else license_value


def _duplicates(values: Iterable[str]) -> set[str]:
    """找出重复值。"""

    seen: set[str] = set()
    repeated: set[str] = set()
    for value in values:
        if value in seen:
            repeated.add(value)
        seen.add(value)
    return repeated


def _normalize_text_list(values: Iterable[str]) -> list[str]:
    """清洗字符串列表，去空值、去重并保留首次出现顺序。"""

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        value = str(raw_value or "").strip()
        if value and value not in seen:
            normalized.append(value)
            seen.add(value)
    return normalized


def _slug(value: str) -> str:
    """生成适合放进 case_id/leakage_group_id 的稳定片段。"""

    return "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-") or "unknown"


def _utc_now() -> str:
    """返回 UTC ISO 时间，保证报告和 manifest 可复现比较。"""

    return datetime.now(UTC).isoformat(timespec="seconds")
