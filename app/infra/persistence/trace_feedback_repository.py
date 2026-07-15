"""阶段 7 Trace 人工反馈仓储；模块导入时不连接 Mongo。"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient


load_dotenv()

TRACE_FEEDBACK_COLLECTION = "trace_feedbacks"


class TraceFeedbackRepository:
    """封装 ``trace_feedbacks`` collection 的写入和查询。"""

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
            raise ValueError("缺少 MONGO_URL 配置，无法初始化 Trace Feedback 仓储")
        if not self.db_name:
            raise ValueError("缺少 MONGO_DB_NAME 配置，无法初始化 Trace Feedback 仓储")

        self.client = client or MongoClient(self.mongo_url, serverSelectionTimeoutMS=2_000)
        self.collection = self.client[self.db_name][TRACE_FEEDBACK_COLLECTION]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self.collection.create_index([("feedback_id", ASCENDING)], unique=True)
        self.collection.create_index([("trace_id", ASCENDING), ("created_at", DESCENDING)])
        self.collection.create_index([("operator_user_id", ASCENDING), ("created_at", DESCENDING)])
        self.collection.create_index([("dataset_ids", ASCENDING), ("created_at", DESCENDING)])

    @staticmethod
    def _without_mongo_id(document: dict[str, Any] | None) -> dict[str, Any]:
        if not document:
            return {}
        result = dict(document)
        result.pop("_id", None)
        return result

    def create_feedback(self, feedback: Mapping[str, Any]) -> dict[str, Any]:
        document = dict(feedback)
        self.collection.insert_one(document)
        return self._without_mongo_id(document)

    def list_feedbacks(self, *, trace_id: str, limit: int = 50) -> list[dict[str, Any]]:
        cursor = self.collection.find({"trace_id": trace_id}).sort("created_at", DESCENDING).limit(limit)
        return [self._without_mongo_id(document) for document in cursor]


_trace_feedback_repository: TraceFeedbackRepository | None = None


def get_trace_feedback_repository() -> TraceFeedbackRepository:
    """首次真正使用反馈 API 时才初始化 Mongo。"""
    global _trace_feedback_repository
    if _trace_feedback_repository is None:
        _trace_feedback_repository = TraceFeedbackRepository()
    return _trace_feedback_repository
