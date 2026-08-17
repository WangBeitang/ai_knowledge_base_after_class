"""
阶段 6 chunk 管理业务服务。

Service（业务服务）负责把 document 可见性、当前索引版本、Milvus chunk 读取、
Mongo 人工覆盖状态和审计事件串成一个一致的启停流程。HTTP API 只做参数解析和
异常映射，Mongo repository 只做持久化，Milvus gateway 只做数据读取。
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from app.infra.persistence.chunk_status_repository import (
    MANUAL_STATUS_DISABLED,
    MANUAL_STATUS_ENABLED,
    MANUAL_STATUS_NONE,
    get_chunk_status_repository,
)
from app.infra.persistence.import_metadata_repository import (
    DATASET_ROLE_ADMIN,
    DATASET_ROLE_EDITOR,
    DEFAULT_TENANT_ID,
    STATUS_DELETED,
    get_import_metadata_repository,
)
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.rag.query.chunk_retrieval_utils import CHUNK_OUTPUT_FIELDS, build_chunk_management_filter
from app.shared.utils.escape_milvus_string_utils import escape_milvus_string


CONTENT_PREVIEW_MAX_LENGTH = 300


class ChunkManagementError(ValueError):
    """chunk 管理业务异常基类。"""


class ChunkNotFoundError(ChunkManagementError):
    """document/chunk 不存在、已删除或当前用户不可见。API 映射为 404。"""


class ChunkPermissionError(ChunkManagementError):
    """当前用户可见但没有启停操作权限。API 映射为 403。"""


class ChunkVersionConflictError(ChunkManagementError):
    """请求基于旧 index_version，API 映射为 409，前端应刷新后重试。"""


class ChunkStateError(ChunkManagementError):
    """当前 chunk 状态无法完成目标操作，API 映射为 409。"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_chunk_id(chunk_id: int | str) -> int | str:
    """
    规范化路径里的 chunk_id。

    当前 Milvus 主键是 INT64 auto_id，HTTP path 传进来时是字符串；若全是数字则转为
    int，方便与 Milvus/Mongo 中的实际主键类型匹配。保留字符串分支是为未来应用生成
    稳定业务 ID 预留。
    """
    if isinstance(chunk_id, bool):
        raise ValueError("chunk_id 必须是字符串或整数")
    if isinstance(chunk_id, int):
        return chunk_id
    normalized = str(chunk_id or "").strip()
    if not normalized:
        raise ValueError("chunk_id 不能为空")
    if normalized.isdigit():
        return int(normalized)
    return normalized


def _chunk_id_filter_clause(chunk_id: int | str) -> str:
    normalized_chunk_id = _normalize_chunk_id(chunk_id)
    if isinstance(normalized_chunk_id, int):
        return f"chunk_id == {normalized_chunk_id}"
    return f'chunk_id == "{escape_milvus_string(normalized_chunk_id)}"'


def _as_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} 必须是整数")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是整数") from exc


def _is_visible_document(document: Mapping[str, Any], *, user_id: str, tenant_id: str) -> bool:
    visibility = str(document.get("visibility") or "private")
    if document.get("owner_user_id") == user_id:
        return True
    if visibility == "public":
        return True
    if visibility == "shared" and document.get("tenant_id") == tenant_id:
        return True
    return False


def _event_value(value: Any) -> str:
    """兼容 API enum 和测试里直接传字符串的写法。"""
    return str(getattr(value, "value", value) or "").strip()


def _dataset_write_role(metadata_repository: Any, *, document: Mapping[str, Any], user_id: str) -> bool:
    """判断当前用户是否是 document 所属 dataset 的 editor/admin。"""
    dataset_id = str(document.get("dataset_id") or "")
    if not dataset_id or not hasattr(metadata_repository, "get_dataset_member"):
        return False
    dataset = metadata_repository.get_dataset(dataset_id) if hasattr(metadata_repository, "get_dataset") else {}
    if dataset and dataset.get("owner_user_id") == user_id:
        return True
    member = metadata_repository.get_dataset_member(dataset_id=dataset_id, user_id=user_id)
    return str(member.get("role") or "") in {DATASET_ROLE_EDITOR, DATASET_ROLE_ADMIN}


class ChunkManagementService:
    """
    chunk 管理业务编排。

    这里是“状态事实”的合并点：Milvus 保存原始 chunk 和基础 enabled，Mongo overrides
    保存人工覆盖层。最终 ``effective_enabled`` 由两者共同决定。
    """

    def __init__(
            self,
            *,
            metadata_repository=None,
            status_repository=None,
            vector_gateway=None,
    ) -> None:
        self.metadata_repository = metadata_repository or get_import_metadata_repository()
        self.status_repository = status_repository or get_chunk_status_repository()
        self.vector_gateway = vector_gateway or milvus_gateway

    def _get_visible_document(self, document_id: str, user_id: str, tenant_id: str) -> dict[str, Any]:
        if hasattr(self.metadata_repository, "get_visible_document"):
            document = self.metadata_repository.get_visible_document(
                document_id=document_id,
                owner_user_id=user_id,
                tenant_id=tenant_id,
            )
        else:
            document = self.metadata_repository.get_document(document_id, user_id)

        if not document or document.get("status") == STATUS_DELETED:
            raise ChunkNotFoundError(f"document_id={document_id} 不存在")
        if not _is_visible_document(document, user_id=user_id, tenant_id=tenant_id):
            raise ChunkNotFoundError(f"document_id={document_id} 不存在")
        return document

    @staticmethod
    def _document_index_version(document: Mapping[str, Any]) -> int:
        return _as_int(document.get("index_version", 0), field_name="index_version")

    def _query_chunks(
            self,
            *,
            document: Mapping[str, Any],
            user_id: str,
            tenant_id: str,
            enabled: bool | None,
            limit: int,
            offset: int | None = None,
            chunk_id: int | str | None = None,
    ) -> list[dict[str, Any]]:
        dataset_id = str(document.get("dataset_id") or "").strip()
        if not dataset_id:
            raise ChunkStateError("document 缺少 dataset_id，无法查询 chunk")

        index_version = self._document_index_version(document)
        filter_expr = build_chunk_management_filter(
            dataset_ids=[dataset_id],
            owner_user_id=user_id,
            tenant_id=tenant_id,
            document_id=str(document.get("document_id") or ""),
            index_version=index_version,
            enabled=enabled,
            # 真分页：把 chunk_index 范围交给 Milvus 在查询阶段裁剪，
            # Milvus query limit 只需 page limit，不拉全量。
            # offset=None（如 chunk_id 详情查询）时不拼范围。
            chunk_index_min=offset if offset is not None else None,
            chunk_index_max=None if offset is None else offset + limit,
        )
        if chunk_id is not None:
            filter_expr = f"{filter_expr} AND {_chunk_id_filter_clause(chunk_id)}"

        chunks = self.vector_gateway.query_entities(
            collection_name=self.vector_gateway.chunk_collection_name,
            filter_expr=filter_expr,
            output_fields=CHUNK_OUTPUT_FIELDS,
            limit=limit,
        )
        chunks = sorted(
            chunks,
            key=lambda chunk: (
                str(chunk.get("document_id") or ""),
                _as_int(chunk.get("index_version", 0), field_name="index_version"),
                _as_int(chunk.get("chunk_index", 0), field_name="chunk_index"),
                str(chunk.get("chunk_id") or ""),
            ),
        )
        return chunks

    def _get_current_chunk(
            self,
            *,
            document: Mapping[str, Any],
            user_id: str,
            tenant_id: str,
            chunk_id: int | str,
    ) -> dict[str, Any]:
        chunks = self._query_chunks(
            document=document,
            user_id=user_id,
            tenant_id=tenant_id,
            enabled=None,
            limit=1,
            chunk_id=chunk_id,
        )
        if not chunks:
            raise ChunkNotFoundError(f"chunk_id={chunk_id} 不存在")
        chunk = chunks[0]
        document_id = str(document.get("document_id") or "")
        index_version = self._document_index_version(document)
        if chunk.get("document_id") != document_id or int(chunk.get("index_version", -1)) != index_version:
            raise ChunkNotFoundError(f"chunk_id={chunk_id} 不存在")
        return chunk

    def _manual_status_for(self, override: Mapping[str, Any] | None) -> str:
        if not override:
            return MANUAL_STATUS_NONE
        manual_status = str(override.get("manual_status") or MANUAL_STATUS_NONE)
        if manual_status not in {MANUAL_STATUS_DISABLED, MANUAL_STATUS_ENABLED, MANUAL_STATUS_NONE}:
            return MANUAL_STATUS_NONE
        return manual_status

    @staticmethod
    def _effective_enabled(*, base_enabled: bool, manual_status: str) -> bool:
        if not base_enabled:
            return False
        return manual_status != MANUAL_STATUS_DISABLED

    def _latest_event(
            self,
            *,
            document_id: str,
            chunk_id: int | str,
            index_version: int,
    ) -> dict[str, Any] | None:
        events = self.status_repository.list_events(
            document_id=document_id,
            chunk_id=chunk_id,
            index_version=index_version,
            limit=1,
        )
        return events[0] if events else None

    def _format_chunk_item(
            self,
            chunk: Mapping[str, Any],
            *,
            override: Mapping[str, Any] | None,
            latest_event: Mapping[str, Any] | None,
            include_content: bool,
    ) -> dict[str, Any]:
        content = str(chunk.get("content") or "")
        base_enabled = bool(chunk.get("enabled", True))
        manual_status = self._manual_status_for(override)
        item = {
            "chunk_id": chunk.get("chunk_id"),
            "document_id": str(chunk.get("document_id") or ""),
            "dataset_id": str(chunk.get("dataset_id") or ""),
            "owner_user_id": str(chunk.get("owner_user_id") or ""),
            "tenant_id": str(chunk.get("tenant_id") or ""),
            "visibility": str(chunk.get("visibility") or ""),
            "index_version": _as_int(chunk.get("index_version", 0), field_name="index_version"),
            "chunk_index": _as_int(chunk.get("chunk_index", 0), field_name="chunk_index"),
            "enabled": base_enabled,
            "manual_status": manual_status,
            "effective_enabled": self._effective_enabled(
                base_enabled=base_enabled,
                manual_status=manual_status,
            ),
            "title": str(chunk.get("title") or ""),
            "parent_title": str(chunk.get("parent_title") or ""),
            "source_title": str(chunk.get("source_title") or chunk.get("file_title") or ""),
            "content_preview": content[:CONTENT_PREVIEW_MAX_LENGTH],
            "content_length": len(content),
            "subject_id": str(chunk.get("subject_id") or ""),
            "standard_subject_name": str(chunk.get("standard_subject_name") or ""),
            "equipment_model": str(chunk.get("equipment_model") or ""),
            "alarm_code": str(chunk.get("alarm_code") or ""),
            "part_name": str(chunk.get("part_name") or ""),
            "sop_type": str(chunk.get("sop_type") or ""),
            "safety_level": str(chunk.get("safety_level") or ""),
            "maintenance_stage": str(chunk.get("maintenance_stage") or ""),
            "latest_event": dict(latest_event) if latest_event else None,
        }
        if include_content:
            item["content"] = content
        return item

    def list_document_chunks(
            self,
            *,
            document_id: str,
            user_id: str,
            tenant_id: str = DEFAULT_TENANT_ID,
            enabled: bool | None = None,
            limit: int = 100,
            offset: int = 0,
    ) -> dict[str, Any]:
        """查看当前用户可见 document 的当前版本 chunk 列表。

        真分页：offset/limit 以 chunk_index 范围（[offset, offset+limit)）拼入 Milvus
        filter，查询阶段裁剪，不拉全量（不修改导入、检索或 Milvus Schema）。
        """
        document = self._get_visible_document(document_id, user_id, tenant_id)
        chunks = self._query_chunks(
            document=document,
            user_id=user_id,
            tenant_id=tenant_id,
            # 路线 B 下 API 的 enabled 语义是“叠加人工覆盖后的最终有效状态”。
            # 如果直接拼 Milvus enabled == false，会漏掉 base enabled=true 但
            # manual_status=disabled 的人工禁用 chunk。
            enabled=None,
            limit=limit,
            offset=offset,
        )
        index_version = self._document_index_version(document)
        overrides = self.status_repository.get_overrides(
            document_id=document_id,
            chunk_ids=[chunk.get("chunk_id") for chunk in chunks],
            index_version=index_version,
        )
        overrides_by_chunk_id = {override.get("chunk_id"): override for override in overrides}

        items = []
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            item = self._format_chunk_item(
                chunk,
                override=overrides_by_chunk_id.get(chunk_id),
                latest_event=self._latest_event(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    index_version=index_version,
                ),
                include_content=False,
            )
            if enabled is None or item["effective_enabled"] == enabled:
                items.append(item)
        return {"code": 200, "items": items[:limit]}

    def list_chunks(
            self,
            *,
            dataset_id: str,
            user_id: str,
            tenant_id: str = DEFAULT_TENANT_ID,
            document_id: str | None = None,
            enabled: bool | None = None,
            limit: int = 100,
            offset: int = 0,
    ) -> dict[str, Any]:
        """
        跨 document 查询当前用户可见 chunk。

        第一版不做全文搜索和复杂排序；如果传 document_id，复用按 document 的稳定列表。
        """
        if document_id:
            return self.list_document_chunks(
                document_id=document_id,
                user_id=user_id,
                tenant_id=tenant_id,
                enabled=enabled,
                limit=limit,
                offset=offset,
            )

        filter_expr = build_chunk_management_filter(
            dataset_ids=[dataset_id],
            owner_user_id=user_id,
            tenant_id=tenant_id,
            enabled=None,
        )
        chunks = self.vector_gateway.query_entities(
            collection_name=self.vector_gateway.chunk_collection_name,
            filter_expr=filter_expr,
            output_fields=CHUNK_OUTPUT_FIELDS,
            limit=limit,
        )
        items: list[dict[str, Any]] = []
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            chunk_document_id = str(chunk.get("document_id") or "")
            index_version = _as_int(chunk.get("index_version", 0), field_name="index_version")
            overrides = self.status_repository.get_overrides(
                document_id=chunk_document_id,
                chunk_ids=[chunk_id],
                index_version=index_version,
            )
            item = self._format_chunk_item(
                chunk,
                override=overrides[0] if overrides else None,
                latest_event=self._latest_event(
                    document_id=chunk_document_id,
                    chunk_id=chunk_id,
                    index_version=index_version,
                ),
                include_content=False,
            )
            if enabled is None or item["effective_enabled"] == enabled:
                items.append(item)
        return {"code": 200, "items": items[:limit]}

    def change_chunks_enabled(
            self,
            *,
            items: list[dict[str, Any]],
            user_id: str,
            enabled: bool,
            reason_type: str,
            reason_detail: str = "",
            tenant_id: str = DEFAULT_TENANT_ID,
    ) -> dict[str, Any]:
        """批量启停 chunk。每个 item 独立返回结果，不做整批事务回滚。"""
        results: list[dict[str, Any]] = []
        changed_count = 0
        failed_count = 0
        for item in items[:100]:
            try:
                result = self.change_chunk_enabled(
                    document_id=str(item.get("document_id") or ""),
                    chunk_id=item.get("chunk_id"),
                    user_id=user_id,
                    tenant_id=tenant_id,
                    enabled=enabled,
                    expected_index_version=_as_int(
                        item.get("expected_index_version"),
                        field_name="expected_index_version",
                    ),
                    reason_type=reason_type,
                    reason_detail=reason_detail,
                )
                changed_count += int(bool(result.get("changed")))
                results.append(result)
            except Exception as error:
                failed_count += 1
                results.append({
                    "document_id": str(item.get("document_id") or ""),
                    "chunk_id": item.get("chunk_id"),
                    "changed": False,
                    "effective_enabled": None,
                    "error": str(error),
                })
        return {
            "code": 200,
            "changed_count": changed_count,
            "failed_count": failed_count,
            "items": results,
        }

    def get_chunk_detail(
            self,
            *,
            document_id: str,
            chunk_id: int | str,
            user_id: str,
            tenant_id: str = DEFAULT_TENANT_ID,
    ) -> dict[str, Any]:
        """查看单个 chunk 详情，返回完整正文但不返回向量字段。"""
        document = self._get_visible_document(document_id, user_id, tenant_id)
        chunk = self._get_current_chunk(
            document=document,
            user_id=user_id,
            tenant_id=tenant_id,
            chunk_id=chunk_id,
        )
        normalized_chunk_id = chunk.get("chunk_id")
        index_version = self._document_index_version(document)
        overrides = self.status_repository.get_overrides(
            document_id=document_id,
            chunk_ids=[normalized_chunk_id],
            index_version=index_version,
        )
        return self._format_chunk_item(
            chunk,
            override=overrides[0] if overrides else None,
            latest_event=self._latest_event(
                document_id=document_id,
                chunk_id=normalized_chunk_id,
                index_version=index_version,
            ),
            include_content=True,
        )

    def change_chunk_enabled(
            self,
            *,
            document_id: str,
            chunk_id: int | str,
            user_id: str,
            enabled: bool,
            expected_index_version: int,
            reason_type: str,
            reason_detail: str = "",
            trace_id: str | None = None,
            tenant_id: str = DEFAULT_TENANT_ID,
    ) -> dict[str, Any]:
        """
        启用或禁用 chunk。

        阶段 6 选择路线 B：不修改 Milvus 原始 row，只写 Mongo override。阶段 7 起，
        document owner 或 dataset editor/admin 可以启停；普通 public/shared 读者只能查看。
        """
        document = self._get_visible_document(document_id, user_id, tenant_id)
        if document.get("owner_user_id") != user_id and not _dataset_write_role(
            self.metadata_repository,
            document=document,
            user_id=user_id,
        ):
            raise ChunkPermissionError("当前用户只能查看该 chunk，不能执行启停操作")

        index_version = self._document_index_version(document)
        if _as_int(expected_index_version, field_name="expected_index_version") != index_version:
            raise ChunkVersionConflictError(
                f"expected_index_version={expected_index_version} 与当前 index_version={index_version} 不一致"
            )

        chunk = self._get_current_chunk(
            document=document,
            user_id=user_id,
            tenant_id=tenant_id,
            chunk_id=chunk_id,
        )
        normalized_chunk_id = chunk.get("chunk_id")
        overrides = self.status_repository.get_overrides(
            document_id=document_id,
            chunk_ids=[normalized_chunk_id],
            index_version=index_version,
        )
        override = overrides[0] if overrides else None
        manual_status = self._manual_status_for(override)
        base_enabled = bool(chunk.get("enabled", True))
        current_effective_enabled = self._effective_enabled(
            base_enabled=base_enabled,
            manual_status=manual_status,
        )

        if enabled and not base_enabled:
            raise ChunkStateError("Milvus enabled=false 的 chunk 不能通过路线 B 人工恢复")

        latest_event = self._latest_event(
            document_id=document_id,
            chunk_id=normalized_chunk_id,
            index_version=index_version,
        )
        if enabled == current_effective_enabled:
            return {
                "code": 200,
                "message": "chunk 状态未变化",
                "changed": False,
                "document_id": document_id,
                "chunk_id": normalized_chunk_id,
                "index_version": index_version,
                "enabled": base_enabled,
                "manual_status": manual_status,
                "effective_enabled": current_effective_enabled,
                "latest_event": latest_event,
            }

        operation = "enable" if enabled else "disable"
        next_manual_status = MANUAL_STATUS_ENABLED if enabled else MANUAL_STATUS_DISABLED
        event = {
            "event_id": f"chunk_evt_{uuid.uuid4().hex}",
            "document_id": document_id,
            "chunk_id": normalized_chunk_id,
            "dataset_id": str(chunk.get("dataset_id") or document.get("dataset_id") or ""),
            "owner_user_id": str(chunk.get("owner_user_id") or document.get("owner_user_id") or ""),
            "tenant_id": str(chunk.get("tenant_id") or document.get("tenant_id") or DEFAULT_TENANT_ID),
            "visibility": str(chunk.get("visibility") or document.get("visibility") or "private"),
            "index_version": index_version,
            "chunk_index": _as_int(chunk.get("chunk_index", 0), field_name="chunk_index"),
            "operator_user_id": user_id,
            "operation": operation,
            "previous_enabled": current_effective_enabled,
            "enabled": enabled,
            "reason_type": _event_value(reason_type),
            "reason_detail": str(reason_detail or ""),
            "source": "manual",
            "human_confirmed": True,
            "created_at": _now_iso(),
            "trace_id": trace_id,
        }
        latest_event = self.status_repository.record_event(event)
        self.status_repository.set_override({
            "document_id": document_id,
            "chunk_id": normalized_chunk_id,
            "index_version": index_version,
            "dataset_id": event["dataset_id"],
            "owner_user_id": event["owner_user_id"],
            "tenant_id": event["tenant_id"],
            "visibility": event["visibility"],
            "manual_status": next_manual_status,
            "latest_event_id": latest_event["event_id"],
        })

        return {
            "code": 200,
            "message": "chunk 已启用" if enabled else "chunk 已禁用",
            "changed": True,
            "document_id": document_id,
            "chunk_id": normalized_chunk_id,
            "index_version": index_version,
            "enabled": base_enabled,
            "manual_status": next_manual_status,
            "effective_enabled": self._effective_enabled(
                base_enabled=base_enabled,
                manual_status=next_manual_status,
            ),
            "latest_event": latest_event,
        }

    def list_chunk_events(
            self,
            *,
            document_id: str,
            chunk_id: int | str,
            user_id: str,
            tenant_id: str = DEFAULT_TENANT_ID,
            limit: int = 20,
    ) -> dict[str, Any]:
        """查看当前可见 chunk 的启停审计历史。"""
        document = self._get_visible_document(document_id, user_id, tenant_id)
        chunk = self._get_current_chunk(
            document=document,
            user_id=user_id,
            tenant_id=tenant_id,
            chunk_id=chunk_id,
        )
        normalized_chunk_id = chunk.get("chunk_id")
        index_version = self._document_index_version(document)
        events = self.status_repository.list_events(
            document_id=document_id,
            chunk_id=normalized_chunk_id,
            index_version=index_version,
            limit=limit,
        )
        return {
            "code": 200,
            "document_id": document_id,
            "chunk_id": normalized_chunk_id,
            "index_version": index_version,
            "items": events,
        }


_chunk_management_service: ChunkManagementService | None = None


def get_chunk_management_service() -> ChunkManagementService:
    """首次进入 chunk 管理 API 时才初始化 Mongo/Milvus 依赖。"""
    global _chunk_management_service
    if _chunk_management_service is None:
        _chunk_management_service = ChunkManagementService()
    return _chunk_management_service
