"""阶段 7 Trace 读取和人工反馈服务。"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from app.infra.persistence.retrieval_trace_repository import get_retrieval_trace_repository
from app.infra.persistence.trace_feedback_repository import get_trace_feedback_repository


class TraceNotFoundError(ValueError):
    """Trace 不存在或当前用户不可见。"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def project_trace_summary(trace: dict[str, Any]) -> dict[str, Any]:
    """裁剪 Trace 响应，避免对外返回大正文或内部异常细节。"""
    return {
        "trace_id": str(trace.get("trace_id") or ""),
        "session_id": str(trace.get("session_id") or ""),
        "owner_user_id": str(trace.get("owner_user_id") or ""),
        "dataset_ids": list(trace.get("dataset_ids") or []),
        "original_query": str(trace.get("original_query") or ""),
        "status": str(trace.get("status") or ""),
        "planner_type": str(trace.get("planner_type") or ""),
        "policy_version": str(trace.get("policy_version") or ""),
        "retrieval_config_version": str(trace.get("retrieval_config_version") or ""),
        "planner_steps": list(trace.get("planner_steps") or []),
        "channel_hits": list(trace.get("channel_hits") or []),
        "final_citations": list(trace.get("final_citations") or []),
        "total_duration_ms": int(trace.get("total_duration_ms") or 0),
        "terminal_action": trace.get("terminal_action"),
        "terminal_reason_code": trace.get("terminal_reason_code"),
        "execution_source": str(trace.get("execution_source") or "chat"),
    }


class TraceFeedbackService:
    """Trace 可见性校验、读取和反馈写入。"""

    def __init__(self, *, trace_repository=None, feedback_repository=None) -> None:
        self.trace_repository = trace_repository or get_retrieval_trace_repository()
        self.feedback_repository = feedback_repository or get_trace_feedback_repository()

    def get_trace(self, *, trace_id: str, user_id: str) -> dict[str, Any]:
        trace = self.trace_repository.get_trace(trace_id, owner_user_id=user_id)
        if not trace:
            raise TraceNotFoundError(f"trace_id={trace_id} 不存在")
        return project_trace_summary(trace)

    def list_traces(
            self,
            *,
            user_id: str,
            session_id: str | None = None,
            dataset_id: str | None = None,
            execution_source: str | None = None,
            limit: int = 50,
    ) -> dict[str, Any]:
        traces = self.trace_repository.list_traces(
            owner_user_id=user_id,
            session_id=session_id,
            dataset_id=dataset_id,
            execution_source=execution_source,
            limit=limit,
        )
        return {"code": 200, "items": [project_trace_summary(trace) for trace in traces]}

    def create_feedback(self, *, trace_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        trace = self.trace_repository.get_trace(trace_id, owner_user_id=user_id)
        if not trace:
            raise TraceNotFoundError(f"trace_id={trace_id} 不存在")
        now = _now_iso()
        feedback = {
            "feedback_id": f"feedback_{uuid.uuid4().hex}",
            "trace_id": trace_id,
            "operator_user_id": user_id,
            "dataset_ids": list(trace.get("dataset_ids") or []),
            **payload,
            "created_at": now,
            "updated_at": now,
        }
        return {"code": 200, **self.feedback_repository.create_feedback(feedback)}

    def list_feedbacks(self, *, trace_id: str, user_id: str, limit: int = 50) -> dict[str, Any]:
        self.get_trace(trace_id=trace_id, user_id=user_id)
        return {
            "code": 200,
            "trace_id": trace_id,
            "items": self.feedback_repository.list_feedbacks(trace_id=trace_id, limit=limit),
        }


_trace_feedback_service: TraceFeedbackService | None = None


def get_trace_feedback_service() -> TraceFeedbackService:
    global _trace_feedback_service
    if _trace_feedback_service is None:
        _trace_feedback_service = TraceFeedbackService()
    return _trace_feedback_service
