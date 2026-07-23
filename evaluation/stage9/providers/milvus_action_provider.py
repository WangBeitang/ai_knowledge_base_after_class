"""阶段 9.2 真实 ActionProvider（动作执行器）。

MilvusActionProvider（Milvus 动作执行器）是 9.2 主路线：GPU（显卡算力）服务器和
本地开发环境使用同一套业务检索代码。它把 OfflineState（离线运行状态）适配成
QueryGraphState（查询图状态），再调用现有 local_search（本地检索）、HyDE（假设式
改写检索）和 Web（网页检索）节点。
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from app.process.query.agent.state import QueryGraphState, create_query_default_state
from app.rag.evaluation.offline_environment import OfflineState
from app.rag.query.config import RETRIEVAL_DEFAULT_MODE, normalize_retrieval_mode
from app.rag.query.contracts import PlannerDecision, QueryAction, RetrievalCandidate


QueryNode = Callable[[QueryGraphState], Mapping[str, Any]]


class MilvusActionProvider:
    """
    真实检索 ActionProvider（动作执行器）。

    默认函数会懒加载现有业务服务，只有真正执行对应 Action（动作）时才连接 Milvus（向量
    数据库）、LLM（大语言模型）或 Web（网页检索）工具。测试可以注入轻量函数，验证
    State（运行状态）转换和候选契约，而不需要外部服务。
    """

    provider_name = "milvus_action_provider"

    def __init__(
            self,
            *,
            local_search_fn: QueryNode | None = None,
            hyde_search_fn: QueryNode | None = None,
            web_search_fn: QueryNode | None = None,
            chunk_status_filter_enabled: bool = True,
    ) -> None:
        self.local_search_fn = local_search_fn or _default_local_search
        self.hyde_search_fn = hyde_search_fn or _default_hyde_search
        self.web_search_fn = web_search_fn or _default_web_search
        # chunk_status_filter_enabled（禁用切片过滤开关）默认开启，表示真实环境读取 Mongo
        # 中人工禁用 chunk。测试或纯离线烟测可关闭，避免无意连接外部数据库。
        self.chunk_status_filter_enabled = bool(chunk_status_filter_enabled)

    def local_search(self, state: OfflineState, decision: PlannerDecision) -> list[RetrievalCandidate]:
        """执行 local_search（本地检索），返回 Milvus（向量数据库）本地候选。"""
        graph_state = self._query_graph_state(state, decision)
        result_state = self.local_search_fn(graph_state)
        return _candidate_list(result_state, "embedding_chunks")

    def hyde_search(self, state: OfflineState, decision: PlannerDecision) -> list[RetrievalCandidate]:
        """执行 hyde_search（假设式改写检索），返回本地候选。"""
        graph_state = self._query_graph_state(state, decision)
        result_state = self.hyde_search_fn(graph_state)
        return _candidate_list(result_state, "hyde_embedding_chunks")

    def web_search(self, state: OfflineState, decision: PlannerDecision) -> list[RetrievalCandidate]:
        """执行 web_search（网页检索），返回 Web（网页）候选。"""
        if not state.web_search_allowed:
            raise ValueError("当前 State（运行状态）不允许 Web（网页检索）")
        graph_state = self._query_graph_state(state, decision)
        result_state = self.web_search_fn(graph_state)
        return _candidate_list(result_state, "web_search_docs")

    def _query_graph_state(self, state: OfflineState, decision: PlannerDecision) -> QueryGraphState:
        """
        把 OfflineState（离线运行状态）投影成 QueryGraphState（查询图状态）。

        该转换只复制当前 Action（动作）需要的身份、权限、快照和 Planner（规划器）事实，
        不读取聊天历史，不写 Mongo Trace（追踪记录），也不把评测 State（运行状态）原样
        暴露给真实检索服务。
        """
        retrieval_mode = normalize_retrieval_mode(
            state.retrieval_config_snapshot.get("retrieval_mode") or RETRIEVAL_DEFAULT_MODE
        )
        return create_query_default_state(
            session_id=state.session_id,
            original_query=state.original_query,
            is_stream=False,
            owner_user_id=state.owner_user_id,
            tenant_id=state.tenant_id,
            dataset_ids=list(state.dataset_ids),
            query_started_at=datetime.now(UTC).isoformat(timespec="seconds"),
            rewritten_query=decision.query or state.current_query,
            subject_ids=list(state.subject_ids),
            standard_subject_names=list(state.standard_subject_names),
            subject_resolution_status=state.subject_resolution_status,
            subject_candidates=[],
            clarification_question=(
                state.latest_observation.clarification_question
                if state.latest_observation
                else None
            ),
            query_identifiers=copy.deepcopy(state.query_identifiers),
            history=[],
            trace_id=state.run_id,
            planner_step=state.planner_step,
            policy_version=state.policy_version,
            current_planner_decision=decision,
            planner_action_history=list(state.action_history),
            planner_type=state.planner_mode,
            planner_runtime_metadata={
                "provider": self.provider_name,
                "snapshot_id": state.snapshot_id,
            },
            web_search_allowed=state.web_search_allowed,
            safe_guard_triggered=bool(state.errors),
            planner_max_steps=state.planner_max_steps,
            retrieval_observation=state.latest_observation,
            retrieval_mode=retrieval_mode.value,
            retrieval_config_version=state.retrieval_config_version,
            retrieval_config_snapshot=copy.deepcopy(state.retrieval_config_snapshot),
            # 真实训练/评测默认启用人工禁用 chunk 覆盖；本字段可在测试中关闭。
            chunk_status_filter_enabled=self.chunk_status_filter_enabled,
            disabled_chunk_ids=list(state.disabled_chunk_ids),
            trace_persistence_enabled=False,
            history_persistence_enabled=False,
            execution_source="retrieval_test",
            config_match_status=state.config_match_status,
            corpus_match_status=state.corpus_match_status,
        )


# RealActionProvider（真实动作执行器）是语义别名，便于文档和训练脚本表达 9.2 主路线。
RealActionProvider = MilvusActionProvider


def _candidate_list(result_state: Mapping[str, Any], field_name: str) -> list[RetrievalCandidate]:
    candidates = result_state.get(field_name) or []
    return [RetrievalCandidate.model_validate(candidate) for candidate in candidates]


def _default_local_search(state: QueryGraphState) -> Mapping[str, Any]:
    from app.rag.query.embedding_search_service import search_by_embedding

    return search_by_embedding(state)


def _default_hyde_search(state: QueryGraphState) -> Mapping[str, Any]:
    from app.rag.query.hyde_search_service import search_by_hyde

    return search_by_hyde(state)


def _default_web_search(state: QueryGraphState) -> Mapping[str, Any]:
    from app.rag.query.web_search_service import search_by_web

    return search_by_web(state)
