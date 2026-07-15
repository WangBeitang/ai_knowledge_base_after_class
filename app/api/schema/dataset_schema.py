"""阶段 7 Dataset 管理 API 契约。"""

from enum import Enum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DatasetSchemaModel(BaseModel):
    """Dataset API schema 公共基类，拒绝拼错字段进入接口契约。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


class DatasetVisibility(str, Enum):
    """Dataset 可见性。可见性只决定读取范围，不等同于写权限。"""

    PRIVATE = "private"  # 私有：owner 和显式成员可见。
    SHARED = "shared"  # 共享：第一版仍要求显式成员，不再默认租户内所有人可编辑。
    PUBLIC = "public"  # 公开：登录用户可读，但不能因此获得编辑权限。


class DatasetStatus(str, Enum):
    """Dataset 生命周期状态。"""

    ACTIVE = "active"  # 正常可用。
    DELETED = "deleted"  # 软删除，不在默认列表展示。


class DatasetMemberRole(str, Enum):
    """Dataset 成员角色。第一版是轻量角色，不是完整 RBAC 权限系统。"""

    VIEWER = "viewer"  # 查看者：可查询 dataset 和可见 document/chunk。
    EDITOR = "editor"  # 编辑者：可上传、重建、删除自己可写范围内的 document，并启停 chunk。
    ADMIN = "admin"  # 管理员：可修改 dataset 元数据和管理 members。


class DatasetCreateRequest(DatasetSchemaModel):
    """创建 Dataset 的请求体。dataset_id 默认由后端生成。"""

    dataset_id: str | None = Field(default=None, max_length=120)
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=500)
    visibility: DatasetVisibility = DatasetVisibility.PRIVATE

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, dataset_id: str | None) -> str | None:
        if dataset_id is None:
            return None
        normalized = dataset_id.strip()
        if not normalized:
            return None
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
        if any(ch not in allowed for ch in normalized):
            raise ValueError("dataset_id 只能包含字母、数字、下划线和中划线")
        return normalized


class DatasetUpdateRequest(DatasetSchemaModel):
    """更新 Dataset 基础信息、owner 或 visibility 的请求体。"""

    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=500)
    visibility: DatasetVisibility | None = None
    owner_user_id: str | None = Field(default=None, min_length=1, max_length=120)

    @model_validator(mode="after")
    def validate_non_empty_patch(self) -> Self:
        if (
            self.name is None
            and self.description is None
            and self.visibility is None
            and self.owner_user_id is None
        ):
            raise ValueError("至少需要提供一个要更新的字段")
        return self


class DatasetSchema(DatasetSchemaModel):
    """Dataset API 响应对象。current_user_role 表示当前请求用户在该 dataset 中的有效角色。"""

    code: int = 200
    dataset_id: str = Field(min_length=1)
    name: str = ""
    description: str = ""
    owner_user_id: str = ""
    tenant_id: str = ""
    visibility: DatasetVisibility = DatasetVisibility.PRIVATE
    status: DatasetStatus = DatasetStatus.ACTIVE
    current_user_role: DatasetMemberRole | None = None
    document_count: int = Field(default=0, ge=0)
    member_count: int = Field(default=0, ge=0)
    created_by_user_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    deleted_at: str = ""


class DatasetListSchema(DatasetSchemaModel):
    """当前用户可见 Dataset 列表。"""

    code: int = 200
    items: list[DatasetSchema] = Field(default_factory=list)


class DatasetMemberUpsertRequest(DatasetSchemaModel):
    """新增或修改 Dataset 成员。"""

    user_id: str = Field(min_length=1, max_length=120)
    role: DatasetMemberRole


class DatasetMemberSchema(DatasetSchemaModel):
    """Dataset 成员关系响应。"""

    member_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    role: DatasetMemberRole
    added_by_user_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    removed_at: str = ""


class DatasetMemberListSchema(DatasetSchemaModel):
    """Dataset 成员列表响应。"""

    code: int = 200
    dataset_id: str = Field(min_length=1)
    items: list[DatasetMemberSchema] = Field(default_factory=list)


class DatasetMemberDeleteSchema(DatasetSchemaModel):
    """移除 Dataset 成员响应。"""

    code: int = 200
    dataset_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    removed: bool
    removed_at: str = ""


__all__ = [
    "DatasetCreateRequest",
    "DatasetListSchema",
    "DatasetMemberDeleteSchema",
    "DatasetMemberListSchema",
    "DatasetMemberRole",
    "DatasetMemberSchema",
    "DatasetMemberUpsertRequest",
    "DatasetSchema",
    "DatasetStatus",
    "DatasetUpdateRequest",
    "DatasetVisibility",
]
