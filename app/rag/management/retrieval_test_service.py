"""阶段 7 Retrieval Test 服务。"""

from __future__ import annotations

import uuid
from typing import Any, Callable

from app.infra.persistence.retrieval_trace_repository import get_retrieval_trace_repository
from app.rag.management.trace_feedback_service import project_trace_summary


class RetrievalTestError(ValueError):
    """Retrieval Test 业务异常。"""


def run_retrieval_test(
        *,
        query_graph_runner: Callable[..., dict[str, Any]],
        user_id: str,
        query: str,
        dataset_ids: list[str],
        planner_mode: str = "rule",
        retrieval_mode: str | None = None,
        web_fallback_enabled: bool | None = None,
) -> dict[str, Any]:
    """复用真实查询图执行检索测试，但关闭聊天历史写入。"""
    if planner_mode != "rule":
        raise RetrievalTestError(f"planner_mode={planner_mode} 尚未注册或未启用")
    session_id = f"retrieval_test_{uuid.uuid4().hex}"
    result = query_graph_runner(
        session_id=session_id,
        query=query,
        is_stream=False,
        owner_user_id=user_id,
        dataset_ids=dataset_ids,
        history_persistence_enabled=False,
        execution_source="retrieval_test",
        retrieval_mode=retrieval_mode,
        web_fallback_enabled=web_fallback_enabled,
    )
    trace_id = str(result.get("trace_id") or "")
    trace = get_retrieval_trace_repository().get_trace(trace_id, owner_user_id=user_id) if trace_id else {}
    return {
        "code": 200,
        "trace_id": trace_id,
        "session_id": session_id,
        "answer": str(result.get("answer") or ""),
        "citations": list(result.get("citations") or []),
        "trace": project_trace_summary(trace) if trace else None,
    }
