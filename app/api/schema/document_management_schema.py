"""阶段 7 Document 管理 API 契约。"""

from enum import Enum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.api.schema.dataset_schema import DatasetVisibility


class DocumentManagementSchemaModel(BaseModel):
    """Document 管理 schema 公共基类。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


class DocumentRecordKind(str, Enum):
    """文档列表记录类型。"""

    ACTIVE = "active"  # 当前有效记录。
    HISTORY = "history"  # 被后续成功导入替代的旧失败记录。


class AccessSyncStatus(str, Enum):
    """Document 权限字段同步状态。"""

    NONE = "none"  # 不需要同步。
    PENDING = "pending"  # 已创建同步/重建任务，尚未执行。
    PROCESSING = "processing"  # 正在同步。
    COMPLETED = "completed"  # 已同步完成。
    FAILED = "failed"  # 同步失败。


class DocumentAccessUpdateRequest(DocumentManagementSchemaModel):
    """修改 document owner/visibility 的请求体。"""

    owner_user_id: str | None = Field(default=None, min_length=1, max_length=120)
    visibility: DatasetVisibility | None = None
    expected_index_version: int = Field(ge=0)
    sync_mode: str = Field(default="rebuild")

    @model_validator(mode="after")
    def validate_update_target(self) -> Self:
        if self.owner_user_id is None and self.visibility is None:
            raise ValueError("owner_user_id 和 visibility 至少需要提供一个")
        if self.sync_mode != "rebuild":
            raise ValueError("阶段 7 第一版只支持 sync_mode=rebuild")
        return self


class DocumentAccessUpdateResponse(DocumentManagementSchemaModel):
    """document owner/visibility 更新响应。"""

    code: int = 200
    document_id: str
    index_version: int = Field(ge=0)
    owner_user_id: str
    visibility: DatasetVisibility
    access_sync_status: AccessSyncStatus = AccessSyncStatus.NONE
    task_id: str = ""
    requires_rebuild: bool = False


class FailedRecordCleanupRequest(DocumentManagementSchemaModel):
    """清理当前用户失败导入记录的请求体。"""

    dataset_id: str
    document_ids: list[str] = Field(default_factory=list)
    only_superseded: bool = True
    dry_run: bool = False


class FailedRecordCleanupItem(DocumentManagementSchemaModel):
    """失败记录清理结果单项。"""

    document_id: str
    status: str
    hidden_at: str = ""
    cleaned: bool
    reason: str = ""


class FailedRecordCleanupResponse(DocumentManagementSchemaModel):
    """失败记录清理响应。"""

    code: int = 200
    matched_count: int = Field(default=0, ge=0)
    cleaned_count: int = Field(default=0, ge=0)
    items: list[FailedRecordCleanupItem] = Field(default_factory=list)


__all__ = [
    "AccessSyncStatus",
    "DocumentAccessUpdateRequest",
    "DocumentAccessUpdateResponse",
    "DocumentRecordKind",
    "FailedRecordCleanupItem",
    "FailedRecordCleanupRequest",
    "FailedRecordCleanupResponse",
]
