"""
查询侧 chunk 启停过滤服务。

本模块只负责把当前 QueryGraphState 的用户、tenant 和 dataset 范围传给
ChunkStatusRepository，读取路线 B 中 ``manual_status=disabled`` 的 chunk_id 列表。
真正的 Milvus expr 拼接仍放在 ``chunk_retrieval_utils``，避免 repository 层知道
Milvus 表达式语法。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.infra.persistence.chunk_status_repository import get_chunk_status_repository


def get_disabled_chunk_ids_for_query(
        state: Mapping[str, Any],
        *,
        status_repository=None,
) -> list[int | str]:
    """
    读取当前查询范围内的人工禁用 chunk_id。

    ``chunk_status_filter_enabled`` 是查询入口对路线 B 的显式开关。真实 HTTP 查询会开启；
    单元测试、离线 Planner 重放或没有 Mongo 的纯算法测试默认关闭，避免无意访问外部
    数据库。开启后如果 Mongo 配置错误或读取失败，应让查询失败，而不是静默召回已禁用
    chunk。
    """
    if not state.get("chunk_status_filter_enabled"):
        return []

    repository = status_repository or get_chunk_status_repository()
    return repository.list_disabled_chunk_ids(
        dataset_ids=state.get("dataset_ids") or [],
        owner_user_id=state.get("owner_user_id") or "",
        tenant_id=state.get("tenant_id") or "",
    )
