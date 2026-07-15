"""阶段 7 Dataset 管理业务服务。"""

from __future__ import annotations

import uuid
from typing import Any

from app.infra.persistence.import_metadata_repository import (
    DATASET_ROLE_ADMIN,
    DEFAULT_TENANT_ID,
    get_import_metadata_repository,
)
from app.rag.management.access_control_service import (
    PermissionDeniedError,
    ResourceNotFoundError,
    get_access_control_service,
)


class DatasetManagementService:
    """Dataset 与 members 管理编排。"""

    def __init__(self, *, metadata_repository=None, access_control_service=None) -> None:
        self.metadata_repository = metadata_repository or get_import_metadata_repository()
        self.access = access_control_service or get_access_control_service()

    def _format_dataset(self, dataset: dict[str, Any], *, user_id: str) -> dict[str, Any]:
        dataset_id = str(dataset.get("dataset_id") or "")
        role = self.access.get_dataset_role(dataset_id=dataset_id, user_id=user_id)
        return {
            "dataset_id": dataset_id,
            "name": str(dataset.get("name") or ""),
            "description": str(dataset.get("description") or ""),
            "owner_user_id": str(dataset.get("owner_user_id") or ""),
            "tenant_id": str(dataset.get("tenant_id") or DEFAULT_TENANT_ID),
            "visibility": str(dataset.get("visibility") or "private"),
            "status": str(dataset.get("status") or "active"),
            "current_user_role": role,
            "document_count": self.metadata_repository.count_dataset_documents(dataset_id),
            "member_count": self.metadata_repository.count_dataset_members(dataset_id),
            "created_by_user_id": str(dataset.get("created_by_user_id") or ""),
            "created_at": str(dataset.get("created_at") or ""),
            "updated_at": str(dataset.get("updated_at") or ""),
            "deleted_at": str(dataset.get("deleted_at") or ""),
        }

    def create_dataset(
            self,
            *,
            user_id: str,
            name: str,
            description: str = "",
            visibility: str = "private",
            dataset_id: str | None = None,
            tenant_id: str = DEFAULT_TENANT_ID,
    ) -> dict[str, Any]:
        dataset = self.metadata_repository.create_dataset(
            dataset_id=dataset_id or f"dataset_{uuid.uuid4().hex}",
            owner_user_id=user_id,
            name=name,
            description=description,
            tenant_id=tenant_id,
            visibility=visibility,
        )
        return {"code": 200, **self._format_dataset(dataset, user_id=user_id)}

    def list_datasets(
            self,
            *,
            user_id: str,
            visibility: str | None = None,
            limit: int = 50,
            tenant_id: str = DEFAULT_TENANT_ID,
    ) -> dict[str, Any]:
        datasets = self.metadata_repository.list_visible_datasets(
            user_id=user_id,
            tenant_id=tenant_id,
            visibility=visibility,
            limit=limit,
        )
        return {"code": 200, "items": [self._format_dataset(dataset, user_id=user_id) for dataset in datasets]}

    def get_dataset(self, *, dataset_id: str, user_id: str) -> dict[str, Any]:
        dataset, _role = self.access.require_dataset_read(dataset_id=dataset_id, user_id=user_id)
        return {"code": 200, **self._format_dataset(dataset, user_id=user_id)}

    def update_dataset(
            self,
            *,
            dataset_id: str,
            user_id: str,
            fields: dict[str, Any],
    ) -> dict[str, Any]:
        dataset, role = self.access.require_dataset_read(dataset_id=dataset_id, user_id=user_id)
        owner_change = fields.get("owner_user_id")
        if owner_change and dataset.get("owner_user_id") != user_id:
            raise PermissionDeniedError("只有当前 owner 可以转移 Dataset owner")
        if role != DATASET_ROLE_ADMIN:
            raise PermissionDeniedError("只有 Dataset admin 可以修改 Dataset")
        updated = self.metadata_repository.update_dataset(dataset_id, **fields)
        return {"code": 200, **self._format_dataset(updated, user_id=user_id)}

    def list_members(self, *, dataset_id: str, user_id: str, limit: int = 100) -> dict[str, Any]:
        self.access.require_dataset_read(dataset_id=dataset_id, user_id=user_id)
        members = self.metadata_repository.list_dataset_members(dataset_id=dataset_id, limit=limit)
        return {"code": 200, "dataset_id": dataset_id, "items": members}

    def upsert_member(
            self,
            *,
            dataset_id: str,
            operator_user_id: str,
            member_user_id: str,
            role: str,
    ) -> dict[str, Any]:
        dataset, operator_role = self.access.require_dataset_read(dataset_id=dataset_id, user_id=operator_user_id)
        if operator_role != DATASET_ROLE_ADMIN:
            raise PermissionDeniedError("只有 Dataset admin 可以管理成员")
        if dataset.get("owner_user_id") == member_user_id:
            raise PermissionDeniedError("owner 的 admin 权限不能通过 members API 修改")
        member = self.metadata_repository.upsert_dataset_member(
            dataset_id=dataset_id,
            user_id=member_user_id,
            role=role,
            added_by_user_id=operator_user_id,
        )
        return member

    def remove_member(self, *, dataset_id: str, operator_user_id: str, member_user_id: str) -> dict[str, Any]:
        dataset, operator_role = self.access.require_dataset_read(dataset_id=dataset_id, user_id=operator_user_id)
        if operator_role != DATASET_ROLE_ADMIN:
            raise PermissionDeniedError("只有 Dataset admin 可以移除成员")
        if dataset.get("owner_user_id") == member_user_id:
            raise PermissionDeniedError("不能移除 Dataset owner 的隐式 admin 权限")
        removed = self.metadata_repository.remove_dataset_member(dataset_id=dataset_id, user_id=member_user_id)
        if not removed:
            raise ResourceNotFoundError(f"user_id={member_user_id} 不是 dataset 成员")
        return {
            "code": 200,
            "dataset_id": dataset_id,
            "user_id": member_user_id,
            "removed": True,
            "removed_at": str(removed.get("removed_at") or ""),
        }


_dataset_management_service: DatasetManagementService | None = None


def get_dataset_management_service() -> DatasetManagementService:
    global _dataset_management_service
    if _dataset_management_service is None:
        _dataset_management_service = DatasetManagementService()
    return _dataset_management_service
