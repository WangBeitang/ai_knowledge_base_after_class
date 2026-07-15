"""
阶段 6 chunk 启停状态持久化仓储。

本模块只负责 Mongo 读写，不访问 Milvus，不判断当前用户是否有业务操作权限。
Chunk 的当前人工覆盖状态和审计事件分开保存：
- ``chunk_status_events`` 是不可变事件流水，用于审计和后续评测数据筛选。
- ``chunk_status_overrides`` 是路线 B 的当前覆盖状态，用于查询侧快速排除人工禁用 chunk。
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient


load_dotenv()

CHUNK_STATUS_EVENTS_COLLECTION = "chunk_status_events"
CHUNK_STATUS_OVERRIDES_COLLECTION = "chunk_status_overrides"

# manual_status 是“人工覆盖层”的状态，不等同于 Milvus 原始 enabled 字段。
# 路线 B 下 Milvus row 保持不变，Mongo 只记录人工判断是否要额外排除。
MANUAL_STATUS_ENABLED = "enabled"
MANUAL_STATUS_DISABLED = "disabled"
MANUAL_STATUS_NONE = "none"
MANUAL_STATUSES = {MANUAL_STATUS_ENABLED, MANUAL_STATUS_DISABLED, MANUAL_STATUS_NONE}

VISIBILITY_PUBLIC = "public"
VISIBILITY_SHARED = "shared"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _without_mongo_id(document: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None
    result = dict(document)
    result.pop("_id", None)
    return result


def _require_non_empty_string(field_name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} 不能为空")
    return value.strip()


def _validate_chunk_id(chunk_id: Any) -> int | str:
    if isinstance(chunk_id, bool):
        raise ValueError("chunk_id 必须是字符串或整数")
    if isinstance(chunk_id, int):
        return chunk_id
    if isinstance(chunk_id, str) and chunk_id.strip():
        return chunk_id.strip()
    raise ValueError("chunk_id 不能为空")


def _validate_index_version(index_version: Any) -> int:
    if isinstance(index_version, bool) or not isinstance(index_version, int) or index_version < 0:
        raise ValueError("index_version 必须是大于等于 0 的整数")
    return index_version


def _validate_limit(limit: Any) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit 必须是正整数")
    return min(limit, 200)


def _normalize_dataset_ids(dataset_ids: Iterable[str]) -> list[str]:
    if isinstance(dataset_ids, (str, bytes)):
        raise ValueError("dataset_ids 必须是字符串列表")
    normalized = [
        dataset_id.strip()
        for dataset_id in dataset_ids
        if isinstance(dataset_id, str) and dataset_id.strip()
    ]
    if not normalized:
        raise ValueError("dataset_ids 不能为空，禁止退化为全库查询")
    return normalized


def _normalize_chunk_ids(chunk_ids: Iterable[int | str]) -> list[int | str]:
    if isinstance(chunk_ids, (str, bytes)):
        return [_validate_chunk_id(chunk_ids)]
    normalized = [_validate_chunk_id(chunk_id) for chunk_id in chunk_ids]
    return normalized


def _chunk_identity_query(document_id: str, chunk_id: int | str, index_version: int) -> dict[str, Any]:
    return {
        "document_id": _require_non_empty_string("document_id", document_id),
        "chunk_id": _validate_chunk_id(chunk_id),
        "index_version": _validate_index_version(index_version),
    }


class ChunkStatusRepository:
    """
    MongoDB chunk 启停仓储。

    Repository（仓储）是持久层边界：它知道集合名、索引和 Mongo 查询形状，但不应该
    知道 FastAPI、Milvus 网关或 service 权限规则。模块 import 时不会创建 MongoClient；
    只有实例化仓储或调用 ``get_chunk_status_repository`` 时才连接数据库。
    """

    def __init__(
            self,
            *,
            client: MongoClient | None = None,
            mongo_url: str | None = None,
            db_name: str | None = None,
    ) -> None:
        self.mongo_url = mongo_url or os.getenv("MONGO_URL")
        self.db_name = db_name or os.getenv("MONGO_DB_NAME")
        if client is None and not self.mongo_url:
            raise ValueError("缺少 MONGO_URL 配置，无法初始化 chunk status 仓储")
        if not self.db_name:
            raise ValueError("缺少 MONGO_DB_NAME 配置，无法初始化 chunk status 仓储")

        # client 参数用于单元测试注入 fake collection。真实客户端使用短超时，避免
        # 管理侧审计写入问题长时间阻塞导入服务或查询服务。
        self.client = client or MongoClient(self.mongo_url, serverSelectionTimeoutMS=2_000)
        self.db = self.client[self.db_name]
        self.events = self.db[CHUNK_STATUS_EVENTS_COLLECTION]
        self.overrides = self.db[CHUNK_STATUS_OVERRIDES_COLLECTION]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        """
        创建审计追溯和路线 B 查询过滤所需索引。

        events 侧按 event_id 幂等去重，并支持单个 chunk 历史、操作人回溯、原因类型
        统计和阶段 8 高可信弱标签筛选。overrides 侧按 document/chunk/version 保证
        当前覆盖状态唯一，并支持查询侧按 dataset/owner/状态快速取禁用 ID。
        """
        self.events.create_index([("event_id", ASCENDING)], unique=True)
        self.events.create_index([
            ("document_id", ASCENDING),
            ("chunk_id", ASCENDING),
            ("index_version", ASCENDING),
            ("created_at", DESCENDING),
        ])
        self.events.create_index([("operator_user_id", ASCENDING), ("created_at", DESCENDING)])
        self.events.create_index([("reason_type", ASCENDING), ("created_at", DESCENDING)])
        self.events.create_index([
            ("dataset_id", ASCENDING),
            ("human_confirmed", ASCENDING),
            ("reason_type", ASCENDING),
            ("created_at", DESCENDING),
        ])

        self.overrides.create_index([
            ("document_id", ASCENDING),
            ("chunk_id", ASCENDING),
            ("index_version", ASCENDING),
        ], unique=True)
        self.overrides.create_index([("dataset_id", ASCENDING), ("manual_status", ASCENDING)])
        self.overrides.create_index([
            ("owner_user_id", ASCENDING),
            ("dataset_id", ASCENDING),
            ("manual_status", ASCENDING),
        ])
        self.overrides.create_index([
            ("document_id", ASCENDING),
            ("index_version", ASCENDING),
            ("manual_status", ASCENDING),
        ])

    def record_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """
        写入一次启停审计事件。

        event 是不可变流水，调用方应在 service 层先完成状态判断和权限判断。这里做基础
        身份字段校验，避免后续 Trace、Citation 或评测数据无法按 document/chunk/version
        反查。
        """
        event_document = dict(event)
        event_document.pop("_id", None)
        _require_non_empty_string("event_id", event_document.get("event_id"))
        _chunk_identity_query(
            event_document.get("document_id", ""),
            event_document.get("chunk_id"),
            event_document.get("index_version"),
        )
        for field_name in (
                "dataset_id",
                "owner_user_id",
                "tenant_id",
                "visibility",
                "operator_user_id",
                "operation",
                "reason_type",
                "created_at",
        ):
            _require_non_empty_string(field_name, event_document.get(field_name))

        self.events.insert_one(event_document)
        return _without_mongo_id(event_document) or {}

    def list_events(
            self,
            *,
            document_id: str,
            chunk_id: int | str,
            index_version: int,
            limit: int = 20,
    ) -> list[dict[str, Any]]:
        """按 document/chunk/version 查询启停历史，按最新事件优先返回。"""
        query = _chunk_identity_query(document_id, chunk_id, index_version)
        cursor = self.events.find(query).sort("created_at", DESCENDING).limit(_validate_limit(limit))
        return [_without_mongo_id(document) or {} for document in cursor]

    def set_override(self, override: Mapping[str, Any]) -> dict[str, Any]:
        """
        写入或覆盖路线 B 当前人工状态。

        overrides 不是审计流水，只保存某个 document/chunk/index_version 的当前人工覆盖层。
        ``latest_event_id`` 用来从当前状态反查最近一次审计事件；``updated_at`` 由仓储兜底
        写入，避免调用方遗漏更新时间。
        """
        override_document = dict(override)
        override_document.pop("_id", None)
        key = _chunk_identity_query(
            override_document.get("document_id", ""),
            override_document.get("chunk_id"),
            override_document.get("index_version"),
        )

        manual_status = override_document.get("manual_status")
        if manual_status is None and "enabled" in override_document:
            # 兼容阶段 6 早期文档中的 enabled 字段表述；落库仍转成 manual_status，
            # 避免和 Milvus 原始 enabled 混淆。
            if not isinstance(override_document["enabled"], bool):
                raise ValueError("enabled 必须是 bool")
            manual_status = (
                MANUAL_STATUS_ENABLED if override_document["enabled"] else MANUAL_STATUS_DISABLED
            )
            override_document["manual_status"] = manual_status
        if manual_status not in MANUAL_STATUSES:
            raise ValueError("manual_status 必须是 enabled、disabled 或 none")

        for field_name in ("dataset_id", "owner_user_id", "tenant_id", "visibility", "latest_event_id"):
            _require_non_empty_string(field_name, override_document.get(field_name))

        override_document.update(key)
        override_document["updated_at"] = (
            _require_non_empty_string("updated_at", override_document["updated_at"])
            if override_document.get("updated_at")
            else _now_iso()
        )

        self.overrides.update_one(key, {"$set": override_document}, upsert=True)
        stored = self.overrides.find_one(key)
        return _without_mongo_id(stored) or dict(override_document)

    def get_overrides(
            self,
            *,
            document_id: str,
            chunk_ids: Iterable[int | str] | None = None,
            index_version: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        查询某个 document 下的人工覆盖状态。

        chunk_ids 为 None 表示取该 document 的全部覆盖状态；传空列表时返回空列表，
        避免调用方误以为查到了“全部”。
        """
        query: dict[str, Any] = {
            "document_id": _require_non_empty_string("document_id", document_id),
        }
        if index_version is not None:
            query["index_version"] = _validate_index_version(index_version)
        if chunk_ids is not None:
            normalized_chunk_ids = _normalize_chunk_ids(chunk_ids)
            if not normalized_chunk_ids:
                return []
            query["chunk_id"] = {"$in": normalized_chunk_ids}

        cursor = self.overrides.find(query).sort([
            ("index_version", ASCENDING),
            ("chunk_id", ASCENDING),
        ])
        return [_without_mongo_id(document) or {} for document in cursor]

    def list_disabled_chunk_ids(
            self,
            *,
            dataset_ids: Iterable[str],
            owner_user_id: str,
            tenant_id: str,
            document_id: str | None = None,
            index_version: int | None = None,
    ) -> list[int | str]:
        """
        查询当前可见范围内被人工禁用的 chunk_id。

        这个方法服务路线 B 的查询过滤：上层已经决定当前用户可访问哪些 dataset，这里只用
        Mongo 条件把 disabled override 限定在 public/shared/owner 可见范围内，避免把
        其它用户私有文档的禁用 ID 拼入 Milvus filter。
        """
        query: dict[str, Any] = {
            "dataset_id": {"$in": _normalize_dataset_ids(dataset_ids)},
            "manual_status": MANUAL_STATUS_DISABLED,
            "$or": [
                {"visibility": VISIBILITY_PUBLIC},
                {
                    "visibility": VISIBILITY_SHARED,
                    "tenant_id": _require_non_empty_string("tenant_id", tenant_id),
                },
                {"owner_user_id": _require_non_empty_string("owner_user_id", owner_user_id)},
            ],
        }
        if document_id is not None:
            query["document_id"] = _require_non_empty_string("document_id", document_id)
        if index_version is not None:
            query["index_version"] = _validate_index_version(index_version)

        cursor = self.overrides.find(query).sort([
            ("document_id", ASCENDING),
            ("index_version", ASCENDING),
            ("chunk_id", ASCENDING),
        ])
        disabled_chunk_ids: list[int | str] = []
        seen: set[tuple[str, int | str]] = set()
        for document in cursor:
            chunk_id = document.get("chunk_id")
            identity = (type(chunk_id).__name__, chunk_id)
            if chunk_id is not None and identity not in seen:
                disabled_chunk_ids.append(chunk_id)
                seen.add(identity)
        return disabled_chunk_ids


_chunk_status_repository: ChunkStatusRepository | None = None


def get_chunk_status_repository() -> ChunkStatusRepository:
    """首次真正需要读写 chunk 启停状态时才初始化 Mongo。"""
    global _chunk_status_repository
    if _chunk_status_repository is None:
        _chunk_status_repository = ChunkStatusRepository()
    return _chunk_status_repository
