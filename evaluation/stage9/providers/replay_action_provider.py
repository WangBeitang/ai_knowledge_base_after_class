"""阶段 9.2 最小 ReplayActionProvider（回放动作执行器）。"""

from __future__ import annotations

import copy
import re
from pathlib import Path

from app.rag.evaluation.offline_environment import OfflineState
from app.rag.query.contracts import PlannerDecision, QueryAction, RetrievalCandidate
from evaluation.stage9.providers.recording_action_provider import (
    ProviderObservationRecord,
    read_provider_observation_records,
)


class ReplayActionProvider:
    """
    最小回放 ActionProvider（动作执行器）。

    Replay（回放）不是阶段 9.2 主训练环境，只用于 smoke test（冒烟测试）、regression test
    （回归测试）和单条异常轨迹复现。缺少 case/action/query 时必须明确报错，不能用空候选
    静默通过。
    """

    provider_name = "replay_action_provider"

    def __init__(self, records_path: str | Path) -> None:
        self.records_path = Path(records_path)
        self.records = read_provider_observation_records(self.records_path)
        self._record_by_key = _record_index(self.records)

    def local_search(self, state: OfflineState, decision: PlannerDecision) -> list[RetrievalCandidate]:
        """回放 local_search（本地检索）候选。"""
        return self._candidates_for(state, decision, QueryAction.LOCAL_SEARCH)

    def hyde_search(self, state: OfflineState, decision: PlannerDecision) -> list[RetrievalCandidate]:
        """回放 hyde_search（假设式改写检索）候选。"""
        return self._candidates_for(state, decision, QueryAction.HYDE_SEARCH)

    def web_search(self, state: OfflineState, decision: PlannerDecision) -> list[RetrievalCandidate]:
        """回放 web_search（网页检索）候选。"""
        return self._candidates_for(state, decision, QueryAction.WEB_SEARCH)

    def _candidates_for(
            self,
            state: OfflineState,
            decision: PlannerDecision,
            expected_action: QueryAction,
    ) -> list[RetrievalCandidate]:
        if decision.action != expected_action:
            raise ValueError(f"Replay 期望 Action={expected_action.value}，实际为 {decision.action.value}")
        key = _record_key(
            snapshot_id=state.snapshot_id,
            case_id=state.case_id,
            action=decision.action,
            query=decision.query,
        )
        record = self._record_by_key.get(key)
        if record is None:
            raise KeyError(
                "Replay 缺少对应记录："
                f"snapshot_id={state.snapshot_id}, case_id={state.case_id}, "
                f"action={decision.action.value}, query={decision.query}"
            )
        return [
            copy.deepcopy(RetrievalCandidate.model_validate(_candidate_contract_payload(candidate)))
            for candidate in record.candidates
        ]


def _record_index(records: list[ProviderObservationRecord]) -> dict[tuple[str, str, str, str], ProviderObservationRecord]:
    index: dict[tuple[str, str, str, str], ProviderObservationRecord] = {}
    for record in records:
        key = _record_key(
            snapshot_id=record.snapshot_id,
            case_id=record.case_id,
            action=record.action,
            query=record.query,
        )
        if key in index:
            raise ValueError(
                "Replay 记录存在重复键："
                f"snapshot_id={record.snapshot_id}, case_id={record.case_id}, "
                f"action={record.action.value}, query={record.query}"
            )
        index[key] = record
    return index


def _record_key(
        *,
        snapshot_id: str,
        case_id: str,
        action: QueryAction,
        query: str,
) -> tuple[str, str, str, str]:
    return (
        snapshot_id,
        case_id,
        action.value,
        _normalize_query(query),
    )


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", str(query or "").strip())


def _candidate_contract_payload(candidate: dict) -> dict:
    payload = dict(candidate)
    # content_truncated（正文已截断）是记录文件的审计元数据，不属于 RetrievalCandidate
    #（检索候选）契约；回放执行时必须剥离，避免污染真实候选 schema（数据结构）。
    payload.pop("content_truncated", None)
    return payload
