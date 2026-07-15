"""阶段 7 Document 管理业务服务。"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app.infra.persistence.import_metadata_repository import (
    DEFAULT_TENANT_ID,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_UPLOADED,
    get_import_metadata_repository,
)
from app.rag.management.access_control_service import (
    PermissionDeniedError,
    ResourceNotFoundError,
    get_access_control_service,
)
from app.shared.utils.path_util import PROJECT_ROOT


class DocumentManagementError(ValueError):
    """Document 管理业务异常基类。"""


class DocumentVersionConflictError(DocumentManagementError):
    """请求基于旧 index_version。"""


class DocumentStateManagementError(DocumentManagementError):
    """document 当前状态不允许目标操作。"""


class DocumentManagementService:
    """Document 列表、失败记录和 access sync 管理。"""

    def __init__(self, *, metadata_repository=None, access_control_service=None) -> None:
        self.metadata_repository = metadata_repository or get_import_metadata_repository()
        self.access = access_control_service or get_access_control_service()

    @staticmethod
    def _history_group_key(document: dict[str, Any]) -> str:
        return "|".join([
            str(document.get("owner_user_id") or ""),
            str(document.get("dataset_id") or ""),
            str(document.get("file_name") or "").strip().lower(),
        ])

    def list_documents(
            self,
            *,
            user_id: str,
            dataset_id: str,
            status: str | None = None,
            keyword: str | None = None,
            fold_history: bool = True,
            limit: int = 20,
            tenant_id: str = DEFAULT_TENANT_ID,
    ) -> list[dict[str, Any]]:
        """返回当前用户可见文档，并给同名旧失败记录打 history 标记。"""
        self.access.require_dataset_read(dataset_id=dataset_id, user_id=user_id, tenant_id=tenant_id)
        if hasattr(self.metadata_repository, "list_visible_documents"):
            documents = self.metadata_repository.list_visible_documents(
                owner_user_id=user_id,
                dataset_id=dataset_id,
                status=status,
                keyword=keyword,
                limit=limit,
            )
        else:
            documents = self.metadata_repository.list_documents(
                owner_user_id=user_id,
                dataset_id=dataset_id,
                status=status,
                keyword=keyword,
                limit=limit,
            )
        latest_completed_by_group: dict[str, str] = {}
        for document in documents:
            if document.get("status") == STATUS_COMPLETED:
                latest_completed_by_group.setdefault(self._history_group_key(document), document.get("document_id", ""))

        items: list[dict[str, Any]] = []
        for document in documents:
            item = dict(document)
            group_key = self._history_group_key(document)
            superseded_by = latest_completed_by_group.get(group_key, "")
            is_history = (
                fold_history
                and document.get("status") == STATUS_FAILED
                and superseded_by
                and superseded_by != document.get("document_id")
            )
            item["record_kind"] = "history" if is_history else "active"
            item["history_group_key"] = group_key
            item["superseded_by_document_id"] = superseded_by if is_history else str(document.get("superseded_by_document_id") or "")
            item["history_record_count"] = 0
            items.append(item)
        return items

    def cleanup_failed_records(
            self,
            *,
            user_id: str,
            dataset_id: str,
            document_ids: list[str],
            only_superseded: bool = True,
            dry_run: bool = False,
    ) -> dict[str, Any]:
        self.access.require_dataset_read(dataset_id=dataset_id, user_id=user_id)
        results = self.metadata_repository.hide_failed_documents(
            owner_user_id=user_id,
            dataset_id=dataset_id,
            document_ids=document_ids,
            only_superseded=only_superseded,
            dry_run=dry_run,
        )
        return {
            "code": 200,
            "matched_count": len(results),
            "cleaned_count": sum(1 for item in results if item.get("cleaned")),
            "items": results,
        }

    def update_document_access(
            self,
            *,
            document_id: str,
            user_id: str,
            expected_index_version: int,
            owner_user_id: str | None,
            visibility: str | None,
            tenant_id: str = DEFAULT_TENANT_ID,
    ) -> dict[str, Any]:
        document, role = self.access.require_document_write(
            document_id=document_id,
            user_id=user_id,
            tenant_id=tenant_id,
        )
        current_index_version = int(document.get("index_version") or 0)
        if int(expected_index_version) != current_index_version:
            raise DocumentVersionConflictError(
                f"expected_index_version={expected_index_version} 与当前 index_version={current_index_version} 不一致"
            )

        next_owner = owner_user_id or str(document.get("owner_user_id") or "")
        next_visibility = visibility or str(document.get("visibility") or "private")
        requires_rebuild = document.get("status") == STATUS_COMPLETED

        if requires_rebuild:
            source_file_path = Path(str(document.get("file_path") or ""))
            if not source_file_path.is_file():
                raise DocumentStateManagementError(f"document_id={document_id} 的原始文件不存在，无法同步权限字段")
            task_id = str(uuid.uuid4())
            local_dir = PROJECT_ROOT / "output" / datetime.now().strftime("%Y%m%d") / task_id
            local_dir.mkdir(parents=True, exist_ok=True)
            self.metadata_repository.update_document_access(
                document_id=document_id,
                owner_user_id=next_owner,
                visibility=next_visibility,
                access_sync_status="pending",
                access_sync_task_id=task_id,
            )
            updated_document, _task = self.metadata_repository.create_rebuild_task_metadata(
                document_id=document_id,
                task_id=task_id,
                owner_user_id=next_owner,
                local_dir=str(local_dir),
            )
            return {
                "code": 200,
                "document_id": document_id,
                "index_version": int(updated_document.get("index_version") or current_index_version + 1),
                "owner_user_id": next_owner,
                "visibility": next_visibility,
                "access_sync_status": "pending",
                "task_id": task_id,
                "requires_rebuild": True,
                "dataset_id": str(updated_document.get("dataset_id") or document.get("dataset_id") or ""),
                "tenant_id": str(updated_document.get("tenant_id") or DEFAULT_TENANT_ID),
                "source_file_path": source_file_path,
                "local_dir": local_dir,
            }

        if document.get("status") not in {STATUS_UPLOADED, STATUS_FAILED}:
            raise DocumentStateManagementError(f"document_id={document_id} 当前状态不允许直接修改权限")
        updated = self.metadata_repository.update_document_access(
            document_id=document_id,
            owner_user_id=next_owner,
            visibility=next_visibility,
            access_sync_status="completed",
        )
        return {
            "code": 200,
            "document_id": document_id,
            "index_version": int(updated.get("index_version") or current_index_version),
            "owner_user_id": next_owner,
            "visibility": next_visibility,
            "access_sync_status": "completed",
            "task_id": "",
            "requires_rebuild": False,
        }


_document_management_service: DocumentManagementService | None = None


def get_document_management_service() -> DocumentManagementService:
    global _document_management_service
    if _document_management_service is None:
        _document_management_service = DocumentManagementService()
    return _document_management_service
