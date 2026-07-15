"""Mongo Retrieval Trace 仓储；模块导入时不建立数据库连接。"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient


load_dotenv()

RETRIEVAL_TRACE_COLLECTION = "retrieval_traces"


class RetrievalTraceRepository:
    """
    封装 ``retrieval_traces`` collection 的增量写入。

    Repository（仓储）只负责 Mongo 文档更新，不从 LangGraph State 猜业务字段。State 到
    Trace 的隐私裁剪和契约校验由 trace_service 完成，避免持久化层意外保存完整 chunk。
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
        if not self.mongo_url:
            raise ValueError("缺少 MONGO_URL 配置，无法初始化 Retrieval Trace 仓储")
        if not self.db_name:
            raise ValueError("缺少 MONGO_DB_NAME 配置，无法初始化 Retrieval Trace 仓储")

        # client 允许单元测试注入内存 fake；真实客户端使用短连接选择超时，Trace 故障不能
        # 长时间阻塞本来可以正常返回的查询。
        self.client = client or MongoClient(self.mongo_url, serverSelectionTimeoutMS=2_000)
        self.collection = self.client[self.db_name][RETRIEVAL_TRACE_COLLECTION]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        """创建与单条追踪、用户/会话回溯和版本评测对应的索引。"""
        self.collection.create_index([("trace_id", ASCENDING)], unique=True)
        self.collection.create_index([("owner_user_id", ASCENDING), ("started_at", DESCENDING)])
        self.collection.create_index([("session_id", ASCENDING), ("started_at", DESCENDING)])
        self.collection.create_index([
            ("policy_version", ASCENDING),
            ("retrieval_config_version", ASCENDING),
            ("started_at", DESCENDING),
        ])
        self.collection.create_index([
            ("planner_type", ASCENDING),
            ("model_id", ASCENDING),
            ("policy_version", ASCENDING),
            ("started_at", DESCENDING),
        ])

    def create_running(self, trace: Mapping[str, Any]) -> None:
        """幂等创建 running Trace；相同 trace_id 重试时不覆盖已经追加的步骤。"""
        trace_document = dict(trace)
        self.collection.update_one(
            {"trace_id": trace_document["trace_id"]},
            {"$setOnInsert": trace_document},
            upsert=True,
        )

    def append_step(self, trace_id: str, step: Mapping[str, Any]) -> None:
        """追加 Planner 已决定但 Action 尚未完成的 pending step。"""
        step_document = dict(step)
        self.collection.update_one(
            {
                "trace_id": trace_id,
                "planner_steps.step": {"$ne": step_document["step"]},
            },
            {"$push": {"planner_steps": step_document}},
        )

    def complete_step(self, trace_id: str, step: Mapping[str, Any]) -> None:
        """用完整 step 覆盖同一步的 pending 内容，保证重试后结果仍可稳定重放。"""
        step_document = dict(step)
        result = self.collection.update_one(
            {"trace_id": trace_id, "planner_steps.step": step_document["step"]},
            {"$set": {"planner_steps.$": step_document}},
        )
        # 如果 pending 写入曾经失败，而 Mongo 已恢复，允许完成节点补写整步。
        if result.matched_count == 0:
            self.append_step(trace_id, step_document)

    def complete_trace(self, trace_id: str, fields: Mapping[str, Any]) -> None:
        """写入正常终态、配置快照、最终引用和候选摘要。"""
        self.collection.update_one({"trace_id": trace_id}, {"$set": dict(fields)})

    def fail_trace(self, trace_id: str, fields: Mapping[str, Any]) -> None:
        """写入未处理异常终态；fields 只能包含结构化错误码，不能包含异常正文。"""
        self.collection.update_one({"trace_id": trace_id}, {"$set": dict(fields)})

    def get_trace(self, trace_id: str, owner_user_id: str | None = None) -> dict[str, Any]:
        """
        读取单条 Trace。

        owner_user_id 不为空时必须匹配，用于 API 层避免用户通过 trace_id 枚举别人的记录。
        """
        query: dict[str, Any] = {"trace_id": trace_id}
        if owner_user_id:
            query["owner_user_id"] = owner_user_id
        trace = self.collection.find_one(query)
        if not trace:
            return {}
        result = dict(trace)
        result.pop("_id", None)
        return result

    def list_traces(
            self,
            *,
            owner_user_id: str,
            session_id: str | None = None,
            dataset_id: str | None = None,
            execution_source: str | None = None,
            limit: int = 50,
    ) -> list[dict[str, Any]]:
        """按用户列出 Trace 摘要，默认按开始时间倒序。"""
        query: dict[str, Any] = {"owner_user_id": owner_user_id}
        if session_id:
            query["session_id"] = session_id
        if dataset_id:
            query["dataset_ids"] = dataset_id
        if execution_source:
            query["execution_source"] = execution_source
        cursor = self.collection.find(query).sort("started_at", DESCENDING).limit(limit)
        traces: list[dict[str, Any]] = []
        for trace in cursor:
            item = dict(trace)
            item.pop("_id", None)
            traces.append(item)
        return traces


_retrieval_trace_repository: RetrievalTraceRepository | None = None


def get_retrieval_trace_repository() -> RetrievalTraceRepository:
    """首次真正写 Trace 时才初始化 Mongo，保证 graph import/compile 不依赖数据库在线。"""
    global _retrieval_trace_repository
    if _retrieval_trace_repository is None:
        _retrieval_trace_repository = RetrievalTraceRepository()
    return _retrieval_trace_repository
