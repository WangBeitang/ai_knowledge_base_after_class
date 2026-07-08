"""
导入元数据持久化仓储。

阶段 3 只负责 dataset/document/task 的管理元数据闭环：
- Mongo 作为长期状态源。
- task_utils 仍负责进程内即时进度。
- 不在这里处理 chunk 幂等、删除文档或重建索引。
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient

from app.shared.runtime.logger import logger


load_dotenv()

DEFAULT_DATASET_ID = "dataset_default_equipment_ops"
DEFAULT_DATASET_NAME = "设备运维知识库"

TASK_TYPE_IMPORT = "import"

STATUS_PENDING = "pending"
STATUS_UPLOADED = "uploaded"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

# 失败时用节点名判断失败阶段：这些节点属于“解析阶段”，其它后续节点默认归为
# “索引阶段”。这样 document.status 负责表达整体成功/失败，parse_status 和
# index_status 负责表达失败落在哪个阶段，便于列表筛选和问题排查。
PARSE_NODE_NAMES = {"upload_file", "node_entry", "node_pdf_to_md", "node_md_img"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _without_mongo_id(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None
    result = dict(document)
    result.pop("_id", None)
    return result


class ImportMetadataRepository:
    """
    MongoDB 导入元数据仓储。

    注意：实例化仓储时才创建 MongoClient 和索引。模块 import 时不会连接 Mongo，
    保证 graph compile 等轻量测试不受 Mongo 可用性影响。
    """

    def __init__(self):
        self.mongo_url = os.getenv("MONGO_URL")
        self.db_name = os.getenv("MONGO_DB_NAME")
        if not self.mongo_url:
            raise ValueError("缺少 MONGO_URL 配置，无法初始化导入元数据仓储")
        if not self.db_name:
            raise ValueError("缺少 MONGO_DB_NAME 配置，无法初始化导入元数据仓储")

        self.client = MongoClient(self.mongo_url, serverSelectionTimeoutMS=2000)
        self.db = self.client[self.db_name]
        self.datasets = self.db["datasets"]
        self.documents = self.db["documents"]
        self.tasks = self.db["tasks"]
        self._ensure_indexes()
        logger.info(f"导入元数据仓储已连接 MongoDB: {self.db_name}")

    def _ensure_indexes(self) -> None:
        self.datasets.create_index([("dataset_id", ASCENDING)], unique=True)
        self.documents.create_index([("document_id", ASCENDING)], unique=True)
        self.documents.create_index([("dataset_id", ASCENDING), ("updated_at", DESCENDING)])
        self.documents.create_index([("dataset_id", ASCENDING), ("status", ASCENDING)])
        self.documents.create_index([("latest_task_id", ASCENDING)])
        self.tasks.create_index([("task_id", ASCENDING)], unique=True)
        self.tasks.create_index([("document_id", ASCENDING), ("created_at", DESCENDING)])
        self.tasks.create_index([("dataset_id", ASCENDING), ("status", ASCENDING)])

    def ensure_default_dataset(self) -> dict[str, Any]:
        now = _now_iso()
        self.datasets.update_one(
            {"dataset_id": DEFAULT_DATASET_ID},
            {
                "$setOnInsert": {
                    "dataset_id": DEFAULT_DATASET_ID,
                    "name": DEFAULT_DATASET_NAME,
                    "description": "默认设备运维知识库容器。",
                    "created_at": now,
                },
                "$set": {"updated_at": now},
            },
            upsert=True,
        )
        return self.get_dataset(DEFAULT_DATASET_ID)

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        dataset = self.datasets.find_one({"dataset_id": dataset_id})
        return _without_mongo_id(dataset) or {}

    def resolve_dataset_for_import(self, dataset_id: str | None = None) -> dict[str, Any]:
        """
        解析导入目标 dataset。

        当前单知识库版本允许不传 dataset_id，此时使用并确保默认 dataset 存在。
        多知识库版本中，上传应该指定已有 dataset_id；导入流程不负责创建非默认 dataset，
        避免每次上传文件时误创建新的知识库容器。
        """
        target_dataset_id = (dataset_id or DEFAULT_DATASET_ID).strip() or DEFAULT_DATASET_ID
        if target_dataset_id == DEFAULT_DATASET_ID:
            return self.ensure_default_dataset()

        dataset = self.get_dataset(target_dataset_id)
        if not dataset:
            raise ValueError(f"dataset_id={target_dataset_id} 不存在，请先创建 dataset")
        return dataset

    def create_import_metadata(
        self,
        *,
        dataset_id: str,
        document_id: str,
        task_id: str,
        file_name: str,
        file_path: str,
        local_dir: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        创建一次导入需要的 document 和 task 记录。

        document 是长期对象，task 是一次执行记录。document 只保存 latest_task_id，
        历史任务通过 tasks.document_id 反查。
        """
        dataset = self.resolve_dataset_for_import(dataset_id)
        dataset_id = dataset["dataset_id"]
        now = _now_iso()

        document = {
            "document_id": document_id,
            "dataset_id": dataset_id,
            "latest_task_id": task_id,
            "file_name": file_name,
            "file_path": file_path,
            "local_dir": local_dir,
            "status": STATUS_UPLOADED,
            "parse_status": STATUS_PENDING,
            "index_status": STATUS_PENDING,
            "chunk_count": 0,
            "subject_id": "",
            "standard_subject_name": "",
            "md_path": "",
            "failed_node": "",
            "error_message": "",
            "created_at": now,
            "updated_at": now,
        }
        task = {
            "task_id": task_id,
            "document_id": document_id,
            "dataset_id": dataset_id,
            "task_type": TASK_TYPE_IMPORT,
            "status": STATUS_PENDING,
            "running_nodes": [],
            "done_nodes": [],
            "failed_node": "",
            "error_message": "",
            "created_at": now,
            "updated_at": now,
        }

        self.documents.insert_one(document)
        self.tasks.insert_one(task)
        return document, task

    def get_document(self, document_id: str) -> dict[str, Any]:
        document = self.documents.find_one({"document_id": document_id})
        return _without_mongo_id(document) or {}

    def list_documents(
        self,
        *,
        dataset_id: str = DEFAULT_DATASET_ID,
        status: str | None = None,
        keyword: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {"dataset_id": dataset_id}
        if status:
            query["status"] = status
        if keyword:
            query["file_name"] = {"$regex": keyword, "$options": "i"}

        cursor = self.documents.find(query).sort("updated_at", DESCENDING).limit(limit)
        return [_without_mongo_id(document) or {} for document in cursor]

    def update_document(self, document_id: str, **fields: Any) -> None:
        if not document_id:
            return
        payload = {key: value for key, value in fields.items() if value is not None}
        if not payload:
            return
        payload["updated_at"] = _now_iso()
        self.documents.update_one({"document_id": document_id}, {"$set": payload})

    def get_task(self, task_id: str) -> dict[str, Any]:
        task = self.tasks.find_one({"task_id": task_id})
        return _without_mongo_id(task) or {}

    def list_tasks(
        self,
        *,
        document_id: str | None = None,
        dataset_id: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {}
        if document_id:
            query["document_id"] = document_id
        if dataset_id:
            query["dataset_id"] = dataset_id
        if status:
            query["status"] = status

        cursor = self.tasks.find(query).sort("created_at", DESCENDING).limit(limit)
        return [_without_mongo_id(task) or {} for task in cursor]

    def update_task_status(self, task_id: str, status: str) -> None:
        payload: dict[str, Any] = {
            "status": status,
            "updated_at": _now_iso(),
        }
        if status == STATUS_COMPLETED:
            payload["completed_at"] = payload["updated_at"]
        if status == STATUS_FAILED:
            payload["failed_at"] = payload["updated_at"]
        self.tasks.update_one({"task_id": task_id}, {"$set": payload})

    def update_task_nodes(self, task_id: str, running_nodes: list[str], done_nodes: list[str]) -> None:
        """
        更新一次导入任务的节点进度快照。

        task_utils 仍然是实时进度的内存来源；这里保存的是同一份 running/done
        快照，目的是让服务重启后仍能通过 task_id 或 document_id 查看历史进度。
        """
        self.tasks.update_one(
            {"task_id": task_id},
            {
                "$set": {
                    "running_nodes": running_nodes,
                    "done_nodes": done_nodes,
                    "updated_at": _now_iso(),
                }
            },
        )

    def mark_import_completed(self, task_id: str) -> None:
        task = self.get_task(task_id)
        if not task:
            return

        now = _now_iso()
        self.tasks.update_one(
            {"task_id": task_id},
            {
                "$set": {
                    "status": STATUS_COMPLETED,
                    "running_nodes": [],
                    "failed_node": "",
                    "error_message": "",
                    "completed_at": now,
                    "updated_at": now,
                }
            },
        )
        self.documents.update_one(
            {"document_id": task.get("document_id", "")},
            {
                "$set": {
                    "status": STATUS_COMPLETED,
                    "parse_status": STATUS_COMPLETED,
                    "index_status": STATUS_COMPLETED,
                    "failed_node": "",
                    "error_message": "",
                    "updated_at": now,
                }
            },
        )

    def mark_import_failed(self, task_id: str, failed_node: str, error_message: str) -> None:
        task = self.get_task(task_id)
        if not task:
            return

        now = _now_iso()
        self.tasks.update_one(
            {"task_id": task_id},
            {
                "$set": {
                    "status": STATUS_FAILED,
                    "running_nodes": [],
                    "failed_node": failed_node,
                    "error_message": error_message,
                    "failed_at": now,
                    "updated_at": now,
                }
            },
        )

        document_payload: dict[str, Any] = {
            "status": STATUS_FAILED,
            "failed_node": failed_node,
            "error_message": error_message,
            "updated_at": now,
        }
        # 文档整体失败时，还需要区分失败阶段：解析失败通常说明文件读取、PDF 转 MD
        # 或图片处理有问题；索引失败通常说明切分、向量化或 Milvus 入库链路有问题。
        if failed_node in PARSE_NODE_NAMES:
            document_payload["parse_status"] = STATUS_FAILED
        else:
            document_payload["index_status"] = STATUS_FAILED

        self.documents.update_one(
            {"document_id": task.get("document_id", "")},
            {"$set": document_payload},
        )


_import_metadata_repository: ImportMetadataRepository | None = None


def get_import_metadata_repository() -> ImportMetadataRepository:
    global _import_metadata_repository
    if _import_metadata_repository is None:
        _import_metadata_repository = ImportMetadataRepository()
    return _import_metadata_repository


def safe_update_document(document_id: str, **fields: Any) -> None:
    if not document_id:
        return
    try:
        get_import_metadata_repository().update_document(document_id, **fields)
    except Exception as e:
        logger.warning(f"更新文档元数据失败，document_id={document_id}, error={e}")


def safe_update_task_status(task_id: str, status: str) -> None:
    try:
        get_import_metadata_repository().update_task_status(task_id, status)
    except Exception as e:
        logger.warning(f"更新任务状态失败，task_id={task_id}, error={e}")


def safe_update_task_nodes(task_id: str, running_nodes: list[str], done_nodes: list[str]) -> None:
    try:
        get_import_metadata_repository().update_task_nodes(task_id, running_nodes, done_nodes)
    except Exception as e:
        logger.warning(f"更新任务节点进度失败，task_id={task_id}, error={e}")


def safe_mark_import_completed(task_id: str) -> None:
    try:
        get_import_metadata_repository().mark_import_completed(task_id)
    except Exception as e:
        logger.warning(f"标记导入完成失败，task_id={task_id}, error={e}")


def safe_mark_import_failed(task_id: str, failed_node: str, error_message: str) -> None:
    try:
        get_import_metadata_repository().mark_import_failed(task_id, failed_node, error_message)
    except Exception as e:
        logger.warning(f"标记导入失败失败，task_id={task_id}, error={e}")
