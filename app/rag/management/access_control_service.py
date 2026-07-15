"""阶段 7 Dataset/Document/Chunk 访问控制。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.infra.persistence.import_metadata_repository import (
    DATASET_ROLE_ADMIN,
    DATASET_ROLE_EDITOR,
    DATASET_ROLE_VIEWER,
    DEFAULT_DATASET_ID,
    DEFAULT_TENANT_ID,
    STATUS_DELETED,
    get_import_metadata_repository,
)


class AccessControlError(ValueError):
    """访问控制业务异常基类。"""


class ResourceNotFoundError(AccessControlError):
    """资源不存在或为了避免泄露私有资源而按不存在处理。"""


class PermissionDeniedError(AccessControlError):
    """当前用户可知道资源存在，但没有执行操作的权限。"""


WRITE_ROLES = {DATASET_ROLE_EDITOR, DATASET_ROLE_ADMIN}


class AccessControlService:
    """集中管理 Dataset、Document 和 Chunk 的轻量权限规则。"""

    def __init__(self, *, metadata_repository=None) -> None:
        self.metadata_repository = metadata_repository or get_import_metadata_repository()

    def get_dataset_role(
            self,
            *,
            dataset_id: str,
            user_id: str,
            tenant_id: str = DEFAULT_TENANT_ID,
    ) -> str | None:
        if not hasattr(self.metadata_repository, "get_dataset"):
            return DATASET_ROLE_VIEWER if dataset_id == DEFAULT_DATASET_ID else None
        if dataset_id == DEFAULT_DATASET_ID and hasattr(self.metadata_repository, "ensure_default_dataset"):
            dataset = self.metadata_repository.ensure_default_dataset()
        else:
            dataset = self.metadata_repository.get_dataset(dataset_id)
        if not dataset:
            return None
        if dataset.get("owner_user_id") == user_id:
            return DATASET_ROLE_ADMIN
        member = {}
        if hasattr(self.metadata_repository, "get_dataset_member"):
            member = self.metadata_repository.get_dataset_member(dataset_id=dataset_id, user_id=user_id)
        if member:
            return str(member.get("role") or DATASET_ROLE_VIEWER)
        if dataset.get("visibility") == "public":
            return DATASET_ROLE_VIEWER
        return None

    def require_dataset_read(
            self,
            *,
            dataset_id: str,
            user_id: str,
            tenant_id: str = DEFAULT_TENANT_ID,
    ) -> tuple[dict[str, Any], str]:
        if not hasattr(self.metadata_repository, "get_dataset"):
            if dataset_id == DEFAULT_DATASET_ID:
                return {"dataset_id": dataset_id, "visibility": "public"}, DATASET_ROLE_VIEWER
            raise ResourceNotFoundError(f"dataset_id={dataset_id} 不存在")
        if dataset_id == DEFAULT_DATASET_ID and hasattr(self.metadata_repository, "ensure_default_dataset"):
            dataset = self.metadata_repository.ensure_default_dataset()
        else:
            dataset = self.metadata_repository.get_dataset(dataset_id)
        if not dataset:
            raise ResourceNotFoundError(f"dataset_id={dataset_id} 不存在")
        role = self.get_dataset_role(dataset_id=dataset_id, user_id=user_id, tenant_id=tenant_id)
        if role is None:
            raise ResourceNotFoundError(f"dataset_id={dataset_id} 不存在")
        return dataset, role

    def require_dataset_write(
            self,
            *,
            dataset_id: str,
            user_id: str,
            tenant_id: str = DEFAULT_TENANT_ID,
            allow_default_upload: bool = False,
    ) -> tuple[dict[str, Any], str]:
        dataset, role = self.require_dataset_read(dataset_id=dataset_id, user_id=user_id, tenant_id=tenant_id)
        # 默认设备运维知识库承担 demo + 私有导入容器双重角色。允许登录用户向默认 dataset
        # 上传自己的 private document，但这不赋予修改他人 document/chunk 的 editor 权限。
        if allow_default_upload and dataset_id == DEFAULT_DATASET_ID:
            return dataset, role
        if role not in WRITE_ROLES:
            raise PermissionDeniedError(f"当前用户没有 dataset_id={dataset_id} 的写权限")
        return dataset, role

    @staticmethod
    def can_read_document(document: Mapping[str, Any], *, user_id: str, dataset_role: str | None) -> bool:
        if not document or document.get("status") == STATUS_DELETED:
            return False
        if document.get("owner_user_id") == user_id:
            return True
        if document.get("visibility") == "public":
            return True
        return dataset_role is not None and document.get("visibility") == "shared"

    @staticmethod
    def can_write_document(document: Mapping[str, Any], *, user_id: str, dataset_role: str | None) -> bool:
        if not document or document.get("status") == STATUS_DELETED:
            return False
        if document.get("owner_user_id") == user_id:
            return True
        return dataset_role in WRITE_ROLES

    def require_document_read(
            self,
            *,
            document_id: str,
            user_id: str,
            tenant_id: str = DEFAULT_TENANT_ID,
    ) -> tuple[dict[str, Any], str | None]:
        if hasattr(self.metadata_repository, "get_visible_document"):
            document = self.metadata_repository.get_visible_document(
                document_id=document_id,
                owner_user_id=user_id,
                tenant_id=tenant_id,
            )
        else:
            document = self.metadata_repository.get_document(document_id, user_id)
        if not document:
            raise ResourceNotFoundError(f"document_id={document_id} 不存在")
        dataset_id = str(document.get("dataset_id") or "")
        role = self.get_dataset_role(dataset_id=dataset_id, user_id=user_id, tenant_id=tenant_id)
        if not self.can_read_document(document, user_id=user_id, dataset_role=role):
            raise ResourceNotFoundError(f"document_id={document_id} 不存在")
        return document, role

    def require_document_write(
            self,
            *,
            document_id: str,
            user_id: str,
            tenant_id: str = DEFAULT_TENANT_ID,
    ) -> tuple[dict[str, Any], str | None]:
        document, role = self.require_document_read(document_id=document_id, user_id=user_id, tenant_id=tenant_id)
        if not self.can_write_document(document, user_id=user_id, dataset_role=role):
            raise PermissionDeniedError(f"当前用户没有 document_id={document_id} 的写权限")
        return document, role

    def require_chunk_operation(
            self,
            *,
            document: Mapping[str, Any],
            user_id: str,
            dataset_role: str | None,
    ) -> None:
        if not self.can_write_document(document, user_id=user_id, dataset_role=dataset_role):
            raise PermissionDeniedError("当前用户只能查看该 chunk，不能执行启停操作")


_access_control_service: AccessControlService | None = None


def get_access_control_service() -> AccessControlService:
    global _access_control_service
    if _access_control_service is None:
        _access_control_service = AccessControlService()
    return _access_control_service
