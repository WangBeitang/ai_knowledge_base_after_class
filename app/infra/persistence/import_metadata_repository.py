"""
导入元数据持久化仓储。

本模块负责 dataset/document/task 的管理元数据闭环：
- Mongo 作为长期状态源。
- task_utils 仍负责进程内即时进度。
- repository 只处理 document 软删除状态和 rebuild task 元数据。
- Milvus、MinIO 和导入图调用由上层 service 编排，不进入 repository。
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient

from app.shared.config.knowledge_base_config import (
    DEFAULT_DATASET_ID,
    DEFAULT_TENANT_ID,
    DEFAULT_VISIBILITY,
)
from app.shared.runtime.logger import logger


load_dotenv()

DEFAULT_DATASET_NAME = "设备运维知识库"
DEFAULT_INDEX_VERSION = 1

TASK_TYPE_IMPORT = "import"
TASK_TYPE_REBUILD_INDEX = "rebuild_index"

STATUS_PENDING = "pending"
STATUS_UPLOADED = "uploaded"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_DELETED = "deleted"

# error_code 是“机器可读错误码”，前端使用它区分“节点自身执行失败”
# 和“后台服务重启导致任务中断”。旧数据没有该字段时按空字符串处理，
# 不需要历史数据迁移。
ERROR_CODE_IMPORT_SERVICE_RESTARTED = "import_service_restarted"
IMPORT_SERVICE_RESTARTED_MESSAGE = (
    "导入服务重启，旧进程中的后台任务已中断，请重新上传或重建索引"
)

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
        self.documents.create_index([("owner_user_id", ASCENDING), ("dataset_id", ASCENDING), ("updated_at", DESCENDING)])
        self.documents.create_index([("owner_user_id", ASCENDING), ("document_id", ASCENDING)])
        self.documents.create_index([("owner_user_id", ASCENDING), ("status", ASCENDING)])
        self.tasks.create_index([("task_id", ASCENDING)], unique=True)
        # 导入服务启动时需要按状态找出 pending/processing 遗留任务。
        # 该索引避免任务历史增长后每次启动都全表扫描。
        self.tasks.create_index([("status", ASCENDING)])
        self.tasks.create_index([("document_id", ASCENDING), ("created_at", DESCENDING)])
        self.tasks.create_index([("dataset_id", ASCENDING), ("status", ASCENDING)])
        self.tasks.create_index([("owner_user_id", ASCENDING), ("task_id", ASCENDING)])
        self.tasks.create_index([("owner_user_id", ASCENDING), ("document_id", ASCENDING), ("created_at", DESCENDING)])

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
        owner_user_id: str,
        file_name: str,
        file_path: str,
        local_dir: str,
        tenant_id: str = DEFAULT_TENANT_ID,
        visibility: str = DEFAULT_VISIBILITY,
        index_version: int = DEFAULT_INDEX_VERSION,
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
            "owner_user_id": owner_user_id,
            "tenant_id": tenant_id,
            "visibility": visibility,
            "latest_task_id": task_id,
            "index_version": index_version,
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
            "image_prefix": "",
            "parse_result_zip_path": "",
            "parse_result_dir": "",
            "deleted_at": "",
            "failed_node": "",
            "error_code": "",
            "error_message": "",
            "created_at": now,
            "updated_at": now,
        }
        task = {
            "task_id": task_id,
            "document_id": document_id,
            "dataset_id": dataset_id,
            "owner_user_id": owner_user_id,
            "tenant_id": tenant_id,
            "task_type": TASK_TYPE_IMPORT,
            "status": STATUS_PENDING,
            "running_nodes": [],
            "done_nodes": [],
            "failed_node": "",
            "error_code": "",
            "error_message": "",
            "created_at": now,
            "updated_at": now,
        }

        self.documents.insert_one(document)
        self.tasks.insert_one(task)
        return document, task

    def create_rebuild_task_metadata(
        self,
        *,
        document_id: str,
        task_id: str,
        owner_user_id: str,
        local_dir: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        为已有 document 创建一次重建索引任务。

        rebuild_index 是任务类型，不是运行状态。它会持久保存在 tasks.task_type
        中，任务进度仍由 tasks.status 表达。
        """
        document = self.get_document(document_id, owner_user_id)
        # 正常流程中 deleted document 不会进入重建索引入口；
        # 这里作为 repository 层兜底，防止绕过 API/前端后复活已删除文档。
        if not document:
            raise ValueError(f"document_id={document_id} 不存在")
        if document.get("status") == STATUS_DELETED:
            raise ValueError(f"document_id={document_id} 已删除，不能重建索引")

        now = _now_iso()
        next_index_version = int(document.get("index_version") or DEFAULT_INDEX_VERSION) + 1
        task = {
            "task_id": task_id,
            "document_id": document_id,
            "dataset_id": document.get("dataset_id", DEFAULT_DATASET_ID),
            "owner_user_id": owner_user_id,
            "tenant_id": document.get("tenant_id", DEFAULT_TENANT_ID),
            "task_type": TASK_TYPE_REBUILD_INDEX,
            "status": STATUS_PENDING,
            "running_nodes": [],
            "done_nodes": [],
            "failed_node": "",
            "error_code": "",
            "error_message": "",
            "created_at": now,
            "updated_at": now,
        }
        document_update = {
            "latest_task_id": task_id,
            "index_version": next_index_version,
            "local_dir": local_dir,
            "status": STATUS_PROCESSING,
            "parse_status": STATUS_PENDING,
            "index_status": STATUS_PENDING,
            "failed_node": "",
            "error_code": "",
            "error_message": "",
            "updated_at": now,
        }

        self.tasks.insert_one(task)
        self.documents.update_one(
            {"document_id": document_id},
            {"$set": document_update},
        )
        updated_document = {**document, **document_update}
        return updated_document, task

    def get_document(self, document_id: str, owner_user_id: str) -> dict[str, Any]:
        document = self.documents.find_one({
            "document_id": document_id,
            "owner_user_id": owner_user_id,
        })
        return _without_mongo_id(document) or {}

    def get_visible_document(
            self,
            *,
            document_id: str,
            owner_user_id: str,
            tenant_id: str = DEFAULT_TENANT_ID,
    ) -> dict[str, Any]:
        """
        按阶段 5/6 的轻量可见性规则读取 document。

        visible 的中文含义是“当前用户可见”。这里不表示可编辑或可启停：
        public/shared/owner 只是读取范围，chunk 启停操作权限仍由上层 service 单独判断。
        """
        document = self.documents.find_one({
            "document_id": document_id,
            "$or": [
                {"visibility": "public"},
                {"visibility": "shared", "tenant_id": tenant_id},
                {"owner_user_id": owner_user_id},
            ],
        })
        return _without_mongo_id(document) or {}

    def list_documents(
        self,
        *,
        owner_user_id: str,
        dataset_id: str = DEFAULT_DATASET_ID,
        status: str | None = None,
        keyword: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {
            "owner_user_id": owner_user_id,
            "dataset_id": dataset_id,
        }
        if status:
            query["status"] = status
        else:
            # 默认文档列表只展示当前有效文档；显式传 status=deleted 时仍可查询
            # 当前用户自己的删除历史。
            query["status"] = {"$ne": STATUS_DELETED}
        if keyword:
            query["file_name"] = {"$regex": keyword, "$options": "i"}

        cursor = self.documents.find(query).sort("updated_at", DESCENDING).limit(limit)
        return [_without_mongo_id(document) or {} for document in cursor]

    def mark_document_deleted(
        self,
        *,
        document_id: str,
        owner_user_id: str,
    ) -> dict[str, Any]:
        """
        将当前用户拥有的 document 标记为已删除。

        这里只更新 Mongo 软删除状态。Milvus chunk 和 MinIO 图片必须由 service
        先清理成功，避免 document 已显示 deleted 但检索资源仍然残留。
        """
        document = self.get_document(document_id, owner_user_id)
        if not document:
            raise ValueError(f"document_id={document_id} 不存在")
        if document.get("status") == STATUS_DELETED:
            return document

        now = _now_iso()
        payload = {
            "status": STATUS_DELETED,
            "deleted_at": now,
            "updated_at": now,
        }
        self.documents.update_one(
            {
                "document_id": document_id,
                "owner_user_id": owner_user_id,
            },
            {"$set": payload},
        )
        return {**document, **payload}

    def update_document(self, document_id: str, **fields: Any) -> None:
        if not document_id:
            return
        payload = {key: value for key, value in fields.items() if value is not None}
        if not payload:
            return
        payload["updated_at"] = _now_iso()
        self.documents.update_one({"document_id": document_id}, {"$set": payload})

    def _get_task_by_id(self, task_id: str) -> dict[str, Any]:
        task = self.tasks.find_one({"task_id": task_id})
        return _without_mongo_id(task) or {}

    def get_task(self, task_id: str, owner_user_id: str) -> dict[str, Any]:
        task = self.tasks.find_one({
            "task_id": task_id,
            "owner_user_id": owner_user_id,
        })
        return _without_mongo_id(task) or {}

    def list_tasks(
        self,
        *,
        owner_user_id: str,
        document_id: str | None = None,
        dataset_id: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {"owner_user_id": owner_user_id}
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

    def reconcile_interrupted_tasks(self) -> dict[str, int]:
        """
        在单进程导入服务启动时收口旧进程遗留的任务。

        FastAPI ``BackgroundTasks`` 属于当前 Python 进程，不是可持久任务队列。
        新服务实例启动时，Mongo 中仍是 pending/processing 的任务已经没有
        执行者，必须转为终态，否则前端会永久显示“处理中”。

        返回值是启动日志使用的计数摘要：
        - examined_task_count：本次检查的非终态 task 数。
        - failed_task_count：确认已中断并转为 failed 的 task 数。
        - completed_task_count：document 已成功、仅 task 终态漏写时修复为 completed 的数量。
        - failed_document_count：仍由中断 task 持有并转为 failed 的 document 数。

        方法只处理 pending/processing，并且更新时再次附带状态条件，因此
        重复启动是幂等的。当前方案仅适用于单进程/单实例导入服务。
        """
        in_flight_statuses = [STATUS_PENDING, STATUS_PROCESSING]
        tasks = list(self.tasks.find({"status": {"$in": in_flight_statuses}}))
        summary = {
            "examined_task_count": len(tasks),
            "failed_task_count": 0,
            "completed_task_count": 0,
            "failed_document_count": 0,
        }

        for task in tasks:
            task_id = str(task.get("task_id") or "")
            document_id = str(task.get("document_id") or "")
            if not task_id:
                # task_id 是任务主键；历史脏数据缺少主键时只记日志，
                # 不使用宽泛条件批量更新其他任务。
                logger.warning("发现缺少 task_id 的非终态导入任务，已跳过启动收口")
                continue

            document = self.documents.find_one({"document_id": document_id}) if document_id else None
            is_latest_task = bool(
                document
                and str(document.get("latest_task_id") or "") == task_id
            )
            now = _now_iso()

            # 最后的 Milvus 入库节点会先把 document/index 写成 completed，
            # invoke_graph 随后才收口 task。如果服务恰好在两步之间退出，
            # 应修复 task 为成功，而不是把已经提交的索引误报为失败。
            if (
                is_latest_task
                and document.get("status") == STATUS_COMPLETED
                and document.get("index_status") == STATUS_COMPLETED
            ):
                result = self.tasks.update_one(
                    {"task_id": task_id, "status": {"$in": in_flight_statuses}},
                    {
                        "$set": {
                            "status": STATUS_COMPLETED,
                            "running_nodes": [],
                            "failed_node": "",
                            "error_code": "",
                            "error_message": "",
                            "completed_at": now,
                            "updated_at": now,
                        }
                    },
                )
                summary["completed_task_count"] += int(result.modified_count > 0)
                continue

            running_nodes = list(task.get("running_nodes") or [])
            failed_node = str(running_nodes[-1]) if running_nodes else ""
            # 只有当前 task 仍是 document.latest_task_id 时才可更新 document。
            # 这个条件防止旧进程遗留的 task 把后续新建、已成功的索引状态覆盖回 failed。
            should_fail_document = is_latest_task and document.get("status") in {
                STATUS_UPLOADED,
                STATUS_PROCESSING,
            }
            if should_fail_document:
                document_payload: dict[str, Any] = {
                    "status": STATUS_FAILED,
                    "failed_node": failed_node,
                    "error_code": ERROR_CODE_IMPORT_SERVICE_RESTARTED,
                    "error_message": IMPORT_SERVICE_RESTARTED_MESSAGE,
                    "updated_at": now,
                }
                if failed_node in PARSE_NODE_NAMES:
                    document_payload["parse_status"] = STATUS_FAILED
                elif failed_node:
                    # 有明确的非解析节点时，延续现有阶段归类，记为索引阶段失败。
                    # 如果尚未进入任何节点，则不猜测阶段，保留原有 pending/processing。
                    document_payload["index_status"] = STATUS_FAILED

                # document 先于 task 收口：如果进程在两次 Mongo 写入之间再次中断，
                # task 仍为非终态，下次启动会重试。反过来先写 task 会导致下次
                # 扫描不到它，从而留下永久 processing 的 document。
                document_result = self.documents.update_one(
                    {
                        "document_id": document_id,
                        "latest_task_id": task_id,
                        "status": {"$in": [STATUS_UPLOADED, STATUS_PROCESSING]},
                    },
                    {"$set": document_payload},
                )
                summary["failed_document_count"] += int(document_result.modified_count > 0)

            result = self.tasks.update_one(
                {"task_id": task_id, "status": {"$in": in_flight_statuses}},
                {
                    "$set": {
                        "status": STATUS_FAILED,
                        "running_nodes": [],
                        "failed_node": failed_node,
                        "error_code": ERROR_CODE_IMPORT_SERVICE_RESTARTED,
                        "error_message": IMPORT_SERVICE_RESTARTED_MESSAGE,
                        "failed_at": now,
                        "updated_at": now,
                    }
                },
            )
            summary["failed_task_count"] += int(result.modified_count > 0)

        return summary

    def mark_import_completed(self, task_id: str) -> None:
        task = self._get_task_by_id(task_id)
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
                    "error_code": "",
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
                    "error_code": "",
                    "error_message": "",
                    "updated_at": now,
                }
            },
        )

    def mark_import_failed(self, task_id: str, failed_node: str, error_message: str) -> None:
        task = self._get_task_by_id(task_id)
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
                    "error_code": "",
                    "error_message": error_message,
                    "failed_at": now,
                    "updated_at": now,
                }
            },
        )

        document_payload: dict[str, Any] = {
            "status": STATUS_FAILED,
            "failed_node": failed_node,
            "error_code": "",
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


def safe_reconcile_interrupted_tasks() -> dict[str, int]:
    """
    安全执行导入服务启动收口。

    这是可观测/状态修复侧路：Mongo 暂时不可用时记录完整日志并返回零计数，
    不阻止 FastAPI 进程启动。后续重启时会再次执行幂等收口。
    """
    empty_summary = {
        "examined_task_count": 0,
        "failed_task_count": 0,
        "completed_task_count": 0,
        "failed_document_count": 0,
    }
    try:
        return get_import_metadata_repository().reconcile_interrupted_tasks()
    except Exception:
        logger.exception("导入服务启动时收口遗留任务失败，本次启动继续")
        return empty_summary
