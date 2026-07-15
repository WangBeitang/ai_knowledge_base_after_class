"""
阶段 6 Chunk 管理 API 的数据契约。

本模块只定义 HTTP 入参/出参形状和枚举，不访问 Milvus 或 Mongo。
Chunk 的中文含义是“知识切片”；API schema 是“接口数据结构合同”，后续
service、API endpoint、前端和测试都应复用这里的字段名，避免各层各写一套。
"""

from enum import Enum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ChunkSchemaModel(BaseModel):
    """
    Chunk API schema 公共基类。

    ``extra='forbid'`` 用来拒绝拼错或临时发明的字段，让接口契约在开发期就暴露问题；
    ``str_strip_whitespace`` 会去掉字符串首尾空白，避免 reason/detail 等字段出现
    “看似有值、实际只有空格”的情况。
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class ChunkEnabledFilter(str, Enum):
    """chunk 列表按启停状态过滤的查询枚举。"""

    ALL = "all"  # 查看全部：启用和禁用 chunk 都返回，用于人工质检和恢复误禁用。
    ENABLED = "true"  # 只看有效启用 chunk；这里的 true 对应 API 查询参数字符串。
    DISABLED = "false"  # 只看已禁用 chunk；用于集中复核低质量或误禁用记录。

    def to_bool(self) -> bool | None:
        """转换为 service 层更容易处理的三态值：None 表示不加 enabled 条件。"""
        if self == ChunkEnabledFilter.ALL:
            return None
        return self == ChunkEnabledFilter.ENABLED


class ChunkManualStatus(str, Enum):
    """
    人工覆盖状态。

    路线 B 下 Milvus 保留原始 ``enabled``，Mongo 记录人工覆盖层。manual status
    表示“人工判断层”的当前状态，不等同于 Milvus 里的基础索引字段。
    """

    NONE = "none"  # 没有人工覆盖，最终有效状态只看 Milvus enabled。
    ENABLED = "enabled"  # 人工恢复过；若 Milvus 基础状态为 false，仍不能强行召回。
    DISABLED = "disabled"  # 人工禁用覆盖；查询时必须排除该 chunk。


class ChunkStatusOperation(str, Enum):
    """一次 chunk 状态变更事件的操作类型。"""

    ENABLE = "enable"  # 启用：恢复一个被人工禁用的 chunk。
    DISABLE = "disable"  # 禁用：把低质量或不适合召回的 chunk 加入人工排除层。


class ChunkStatusReasonType(str, Enum):
    """
    chunk 启停的机器可读原因类型。

    reason type 用于审计、统计和阶段 8 数据筛选，不能只依赖中文自由文本原因。
    """

    PARSE_ERROR = "parse_error"  # 解析错误：PDF/Markdown 解析后段落、表格或图片说明错乱。
    HEADER_FOOTER = "header_footer"  # 页眉页脚：页码、版权、重复标题等非知识正文。
    GARBLED_TEXT = "garbled_text"  # 乱码：OCR 或编码问题导致正文不可读。
    OUTDATED_CONTENT = "outdated_content"  # 内容过期：旧版 SOP、旧报警说明或废弃流程。
    HUMAN_MISJUDGMENT = "human_misjudgment"  # 人工误判：记录人判断错误或需要纠正人工标注。
    MANUAL_RESTORE = "manual_restore"  # 人工恢复：恢复一个仍有价值的 chunk。
    OTHER = "other"  # 其他原因：必须填写 reason_detail，避免审计时只看到“其他”。


class ChunkStatusEventSource(str, Enum):
    """
    chunk 状态事件来源。

    source 只说明事件从哪里产生，不代表质量标签可信度；是否人工确认由
    ``human_confirmed`` 单独表达。
    """

    MANUAL = "manual"  # 人工操作：由当前用户通过 API 或前端触发。
    SYSTEM = "system"  # 系统候选：由规则、脚本或后续自动质检流程生成，默认仍需复核。


class ChunkStatusEventSchema(ChunkSchemaModel):
    """一次 chunk 启用/禁用审计事件的 API 表达。"""

    event_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    # chunk_id 当前来自 Milvus auto_id，通常是 int；保留 str 是为了兼容未来应用生成稳定 ID。
    chunk_id: int | str
    dataset_id: str = Field(min_length=1)
    owner_user_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    visibility: str = Field(min_length=1)
    # index_version 是文档级索引产物版本，不是 Milvus 物理索引版本。
    index_version: int = Field(ge=0)
    chunk_index: int = Field(ge=0)
    operator_user_id: str = Field(min_length=1)
    operation: ChunkStatusOperation
    previous_enabled: bool
    enabled: bool
    reason_type: ChunkStatusReasonType
    reason_detail: str = Field(default="", max_length=500)
    source: ChunkStatusEventSource = ChunkStatusEventSource.MANUAL
    # human_confirmed 表示是否人工确认过；即使为 true，也只能作为弱质量信号，
    # 不能未经阶段 8 复核直接作为训练 reward。
    human_confirmed: bool = True
    created_at: str = Field(min_length=1)
    # 从 Retrieval Trace 入口触发时可写入 trace_id；普通文档列表操作为空。
    trace_id: str | None = None

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id(cls, chunk_id: int | str) -> int | str:
        """chunk_id 是 Trace、Citation 和人工标注的关联键，不能为空白字符串。"""
        if isinstance(chunk_id, str) and not chunk_id.strip():
            raise ValueError("chunk_id 不能为空")
        return chunk_id

    @model_validator(mode="after")
    def validate_event_consistency(self) -> Self:
        """校验事件方向、最终状态和原因说明之间不矛盾。"""
        if self.operation == ChunkStatusOperation.DISABLE:
            if self.previous_enabled is not True or self.enabled is not False:
                raise ValueError("disable 事件必须从 enabled=true 变为 enabled=false")
        if self.operation == ChunkStatusOperation.ENABLE:
            if self.previous_enabled is not False or self.enabled is not True:
                raise ValueError("enable 事件必须从 enabled=false 变为 enabled=true")
        if self.reason_type == ChunkStatusReasonType.OTHER and not self.reason_detail:
            raise ValueError("reason_type=other 时必须填写 reason_detail")
        return self


class ChunkListItemSchema(ChunkSchemaModel):
    """chunk 列表单项；只返回正文预览，不作为正文编辑入口。"""

    chunk_id: int | str
    document_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    owner_user_id: str = ""
    tenant_id: str = ""
    visibility: str = ""
    # index_version 是 document 这一版索引产物的版本，用于防止旧标注套到新 chunk。
    index_version: int = Field(ge=0)
    chunk_index: int = Field(ge=0)
    # enabled 是 Milvus 基础索引状态；effective_enabled 是叠加人工覆盖后的实际查询状态。
    enabled: bool = True
    manual_status: ChunkManualStatus = ChunkManualStatus.NONE
    effective_enabled: bool = True
    title: str = ""
    parent_title: str = ""
    source_title: str = ""
    # content_preview 是正文预览，不是完整 chunk 正文，也不是编辑入口。
    content_preview: str = Field(default="", max_length=500)
    content_length: int = Field(default=0, ge=0)
    subject_id: str = ""
    standard_subject_name: str = ""
    equipment_model: str = ""
    alarm_code: str = ""
    part_name: str = ""
    sop_type: str = ""
    safety_level: str = ""
    maintenance_stage: str = ""
    latest_event: ChunkStatusEventSchema | None = None

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id(cls, chunk_id: int | str) -> int | str:
        """列表中的 chunk_id 也必须可用于后续详情、启停和事件查询。"""
        if isinstance(chunk_id, str) and not chunk_id.strip():
            raise ValueError("chunk_id 不能为空")
        return chunk_id

    @model_validator(mode="after")
    def validate_effective_enabled(self) -> Self:
        """人工禁用后 effective_enabled 必须为 false，基础 disabled 也不能被强行打开。"""
        if self.manual_status == ChunkManualStatus.DISABLED and self.effective_enabled:
            raise ValueError("manual_status=disabled 时 effective_enabled 必须为 false")
        if not self.enabled and self.effective_enabled:
            raise ValueError("Milvus enabled=false 时 effective_enabled 不能为 true")
        return self


class ChunkDetailSchema(ChunkListItemSchema):
    """chunk 详情；比列表多返回完整正文，但仍不包含向量和 BM25 原始字段。"""

    content: str = ""


class ChunkListSchema(ChunkSchemaModel):
    """chunk 列表响应。"""

    code: int = 200
    items: list[ChunkListItemSchema] = Field(default_factory=list)


class ChunkStatusChangeRequest(ChunkSchemaModel):
    """启用或禁用 chunk 的请求体。"""

    enabled: bool
    # expected_index_version 是前端看到的 document 索引版本；不一致时 service 应返回 409。
    expected_index_version: int = Field(ge=0)
    reason_type: ChunkStatusReasonType
    reason_detail: str = Field(default="", max_length=500)
    trace_id: str | None = None

    @model_validator(mode="after")
    def validate_reason_detail(self) -> Self:
        """other 原因必须补充说明，保证审计记录后续可读。"""
        if self.reason_type == ChunkStatusReasonType.OTHER and not self.reason_detail:
            raise ValueError("reason_type=other 时必须填写 reason_detail")
        return self


class ChunkStatusChangeResponse(ChunkSchemaModel):
    """启停操作响应；changed=false 表示目标状态相同，未产生新事件。"""

    code: int = 200
    message: str
    changed: bool
    document_id: str = Field(min_length=1)
    chunk_id: int | str
    index_version: int = Field(ge=0)
    enabled: bool
    manual_status: ChunkManualStatus = ChunkManualStatus.NONE
    effective_enabled: bool
    latest_event: ChunkStatusEventSchema | None = None


class ChunkEventListSchema(ChunkSchemaModel):
    """某个 chunk 的启停历史响应。"""

    code: int = 200
    document_id: str = Field(min_length=1)
    chunk_id: int | str
    index_version: int = Field(ge=0)
    items: list[ChunkStatusEventSchema] = Field(default_factory=list)


__all__ = [
    "ChunkDetailSchema",
    "ChunkEnabledFilter",
    "ChunkEventListSchema",
    "ChunkListItemSchema",
    "ChunkListSchema",
    "ChunkManualStatus",
    "ChunkSchemaModel",
    "ChunkStatusChangeRequest",
    "ChunkStatusChangeResponse",
    "ChunkStatusEventSchema",
    "ChunkStatusEventSource",
    "ChunkStatusOperation",
    "ChunkStatusReasonType",
]
