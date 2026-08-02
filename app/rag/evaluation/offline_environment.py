"""
阶段 8 离线 RAG Environment。

OfflineRagEnvironment 的职责不是线上聊天，也不是直接训练模型，而是在固定
EnvironmentSnapshot（环境快照）下执行 Planner Action，产出可复现的 State、Observation
和 JSON Trace。阶段 9 做 SFT/GRPO 时，Policy 可以变化，轨迹中的 State 也会随着 Action
变化，但 snapshot、Action 合法转移、检索配置和语料边界必须固定。

本模块第一版把真实检索执行抽象成 OfflineActionProvider。默认 provider 返回空结果，
后续 baseline runner 可以注入真实 Milvus/Web/答案执行器；单元测试也可以注入 fake
provider，避免离线环境本身在 import 或测试阶段连接 Mongo、Milvus、聊天历史或 Web。
"""

from __future__ import annotations

import copy
import time
import uuid
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.rag.evaluation.case_schema import (
    EnvironmentSnapshot,
    PlannerEvalCase,
    PlannerMode,
)
from app.rag.query.config import PLANNER_MAX_STEPS
from app.rag.query.contracts import (
    Citation,
    DEFAULT_EVIDENCE_EXCERPT_CHARS,
    EvidenceSourceType,
    EvidenceSummary,
    IdentifierResolutionStatus,
    ObservationStatus,
    PlannerContext,
    PlannerDecision,
    PlannerExecutionStatus,
    PlannerHistoryItem,
    PlannerReasonCode,
    QueryAction,
    RetrievalCandidate,
    RetrievalChannel,
    RetrievalObservation,
    SubjectResolutionStatus,
    TraceStepStatus,
)
from app.rag.query.planner import QueryPlanner


TERMINAL_ACTIONS = {
    QueryAction.ANSWER,
    QueryAction.ASK_CLARIFICATION,
    QueryAction.REFUSE,
}
RETRIEVAL_ACTIONS = {
    QueryAction.LOCAL_SEARCH,
    QueryAction.HYDE_SEARCH,
    QueryAction.WEB_SEARCH,
}
FIRST_ACTIONS = {
    QueryAction.LOCAL_SEARCH,
    QueryAction.WEB_SEARCH,
    QueryAction.ASK_CLARIFICATION,
    QueryAction.REFUSE,
}
VALID_TRANSITIONS = {
    QueryAction.LOCAL_SEARCH: {
        QueryAction.HYDE_SEARCH,
        QueryAction.WEB_SEARCH,
        QueryAction.ANSWER,
        QueryAction.ASK_CLARIFICATION,
        QueryAction.REFUSE,
    },
    QueryAction.HYDE_SEARCH: {
        QueryAction.WEB_SEARCH,
        QueryAction.ANSWER,
        QueryAction.ASK_CLARIFICATION,
        QueryAction.REFUSE,
    },
    QueryAction.WEB_SEARCH: {
        QueryAction.ANSWER,
        QueryAction.ASK_CLARIFICATION,
        QueryAction.REFUSE,
    },
}


class OfflineEnvironmentModel(BaseModel):
    """离线环境内部 schema 公共基类，拒绝未知字段，避免 Trace 悄悄丢失关键信息。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class OfflineTrajectoryStatus(str, Enum):
    """离线轨迹终态。completed 表示正常到达 answer/refuse/ask，failed 表示环境执行失败。"""

    COMPLETED = "completed"  # 正常终止：到达 answer、ask_clarification 或 refuse。
    FAILED = "failed"  # 执行失败：非法路径、Planner 输出非法或超过最大步数。


class OfflineError(OfflineEnvironmentModel):
    """
    离线执行错误。

    Error 是面向评测/Reward 的结构化错误，不保存异常堆栈或模型私有思维链。后续 Reward
    可以按 code 直接扣 R_format、R_behavior 或 R_cost。
    """

    # 机器可读错误码，例如 illegal_action_transition、planner_output_invalid。
    code: str = Field(min_length=1)
    # 中文可读错误说明，用于报告和调试。
    message: str = Field(min_length=1)
    # 发生错误的 step，从 1 开始；reset 阶段或全局错误为 0。
    step: int = Field(default=0, ge=0)
    # 触发错误的 Action。解析失败或全局错误时为空。
    action: QueryAction | None = None


class OfflineTraceStep(OfflineEnvironmentModel):
    """
    JSON Trace 中的一步。

    Trace 的中文含义是“轨迹记录”。它记录 Planner 做决定前看到的 Observation、实际
    Decision、Action 执行后的 Observation 和错误码，不记录聊天历史或私有思维链。
    """

    # 从 1 开始的步骤号。
    step: int = Field(ge=1)
    # Planner 做决定前看到的上一轮 Observation；第一步通常为空。
    input_observation: RetrievalObservation | None = None
    # Planner 或固定 action path 给出的结构化决策。
    decision: PlannerDecision
    # 本步执行状态。非法 Action 会标记 failed。
    execution_status: TraceStepStatus
    # 检索类 Action 的输出 Observation；终止 Action 通常为空。
    output_observation: RetrievalObservation | None = None
    # 本步耗时，单位毫秒。
    duration_ms: int = Field(default=0, ge=0)
    # 本步错误。正常执行为空。
    error: OfflineError | None = None


class OfflineState(OfflineEnvironmentModel):
    """
    一条离线轨迹的可变运行 State。

    State 会随着 Action 变化；EnvironmentSnapshot 不变。每条轨迹从同一 snapshot 创建
    独立 State 副本，避免 local_search、HyDE、Web 或终态答案互相污染。
    """

    # 本次离线运行 ID。用于 session_id/trace_id 关联，不写入线上聊天历史。
    run_id: str = Field(min_length=1)
    # 当前评测样本 ID。
    case_id: str = Field(min_length=1)
    # 当前环境快照 ID。
    snapshot_id: str = Field(min_length=1)
    # 当前轨迹绑定的环境快照。该字段只供离线执行期校验 corpus，不输出到 JSON Trace，
    # 避免每条轨迹重复保存完整 environment_snapshot.json。
    snapshot: EnvironmentSnapshot = Field(repr=False, exclude=True)
    # 当前评测 Planner 模式，例如 rule/api/local_base。
    planner_mode: str = Field(min_length=1)
    # 离线 session ID。只用于 Trace 区分，不进入真实 conversation。
    session_id: str = Field(min_length=1)
    # 固定测试用户，来自 case.owner_user_id。
    owner_user_id: str = Field(min_length=1)
    # 固定租户，来自 case.tenant_id。
    tenant_id: str = Field(min_length=1)
    # 本 case 允许访问的 dataset 范围。
    dataset_ids: list[str] = Field(min_length=1)
    # 用户原始问题。
    original_query: str = Field(min_length=1)
    # 当前 Action 使用的查询文本；HyDE/Web 可以在后续扩展中改写。
    current_query: str = Field(min_length=1)
    # 固定主体 ID，优先来自 case.expected_subject_ids。
    subject_ids: list[str] = Field(default_factory=list)
    # 固定主体展示名，来自 case.expected_subject_names。
    standard_subject_names: list[str] = Field(default_factory=list)
    # 主体确认状态。阶段 8 主评测默认用标注固定主体，减少主体 LLM 波动。
    subject_resolution_status: SubjectResolutionStatus
    # 设备型号、报警码等结构化标识，来自 case.expected_identifiers。
    query_identifiers: dict[str, list[str]] = Field(default_factory=dict)
    # 检索配置版本，必须来自 snapshot。
    retrieval_config_version: str = Field(min_length=1)
    # 检索配置真实参数快照，必须来自 snapshot。
    retrieval_config_snapshot: dict[str, Any] = Field(default_factory=dict)
    # Planner 策略版本，必须来自 snapshot 或当前 planner。
    policy_version: str = Field(min_length=1)
    # 是否允许 Web。默认由 snapshot 的 web_fallback_enabled 和 case.expected_behavior 共同约束。
    web_search_allowed: bool = False
    # 人工禁用 chunk 快照，只能来自 EnvironmentSnapshot.disabled_chunks。
    disabled_chunk_ids: list[str | int] = Field(default_factory=list)
    # 最大 Planner 步数，防止短轨迹任务因状态错误陷入循环。
    planner_max_steps: int = Field(default=PLANNER_MAX_STEPS, ge=1)
    # 已完成 Planner 决策步数。
    planner_step: int = Field(default=0, ge=0)
    # 已执行 Action 历史，提供给 Planner 防循环。
    action_history: list[PlannerHistoryItem] = Field(default_factory=list)
    # 当前 Planner 可见的最近 Observation。
    latest_observation: RetrievalObservation | None = None
    # 按来源保存检索候选，供 Trace 和 Reward 统计。
    retrieval_candidates: dict[str, list[RetrievalCandidate]] = Field(default_factory=dict)
    # 累计融合后的候选。第一版按执行顺序去重保留，不做真实 RRF。
    accumulated_candidates: list[RetrievalCandidate] = Field(default_factory=list)
    # 最终 answer 使用的 citation。
    citations: list[Citation] = Field(default_factory=list)
    # 最终交付文本。可能是答案、追问或拒答。
    answer: str = ""
    # 终止 Action。
    terminal_action: QueryAction | None = None
    # 终止原因码。
    terminal_reason_code: PlannerReasonCode | None = None
    # JSON Trace 步骤。
    trace_steps: list[OfflineTraceStep] = Field(default_factory=list)
    # 结构化错误列表。
    errors: list[OfflineError] = Field(default_factory=list)
    # 配置是否与 snapshot 匹配。reset 默认根据输入快照计算。
    config_match_status: str = "unknown"
    # 语料是否与 snapshot 匹配。候选越界或 case 引用不在快照时为 mismatch。
    corpus_match_status: str = "unknown"
    # 离线评测不写聊天历史。
    history_persistence_enabled: bool = False
    # 第一版离线环境只生成 JSON Trace，不写 Mongo Trace。
    trace_persistence_enabled: bool = False
    # 执行来源固定为 evaluation。
    execution_source: str = "evaluation"

    @model_validator(mode="after")
    def validate_offline_boundaries(self) -> "OfflineState":
        if self.history_persistence_enabled:
            raise ValueError("OfflineState 不能开启 history_persistence_enabled")
        if self.execution_source != "evaluation":
            raise ValueError("OfflineState.execution_source 必须是 evaluation")
        return self


class OfflineStepResult(OfflineEnvironmentModel):
    """一次 step 的执行结果。state 是执行后的新 State，原 State 不会被就地修改。"""

    state: OfflineState
    decision: PlannerDecision
    observation: RetrievalObservation | None = None
    terminal: bool = False
    error: OfflineError | None = None


class OfflineTrajectoryResult(OfflineEnvironmentModel):
    """
    一条离线轨迹的最终结果。

    Trajectory 的中文含义是“轨迹”。它和 PlannerEvalResult 的区别是：这里保存完整
    step/Observation/citation 细节；后续 baseline runner 可以再投影成 PlannerEvalResult。
    """

    run_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    snapshot_id: str = Field(min_length=1)
    planner_mode: str = Field(min_length=1)
    status: OfflineTrajectoryStatus
    action_path: list[QueryAction] = Field(default_factory=list)
    terminal_action: QueryAction | None = None
    terminal_reason_code: PlannerReasonCode | None = None
    config_match_status: str = "unknown"
    corpus_match_status: str = "unknown"
    trace_steps: list[OfflineTraceStep] = Field(default_factory=list)
    retrieved_candidates: list[RetrievalCandidate] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    answer: str = ""
    errors: list[OfflineError] = Field(default_factory=list)
    final_state: OfflineState

    def to_json_trace(self) -> dict[str, Any]:
        """生成可写文件的 JSON Trace，不包含聊天历史，不包含模型私有思维链。"""
        return {
            "run_id": self.run_id,
            "case_id": self.case_id,
            "snapshot_id": self.snapshot_id,
            "planner_mode": self.planner_mode,
            "status": self.status.value,
            "action_path": [action.value for action in self.action_path],
            "terminal_action": self.terminal_action.value if self.terminal_action else None,
            "terminal_reason_code": (
                self.terminal_reason_code.value if self.terminal_reason_code else None
            ),
            "config_match_status": self.config_match_status,
            "corpus_match_status": self.corpus_match_status,
            "steps": [step.model_dump(mode="json") for step in self.trace_steps],
            "citations": [citation.model_dump(mode="json") for citation in self.citations],
            "errors": [error.model_dump(mode="json") for error in self.errors],
            "answer": self.answer,
        }


class OfflineActionProvider(Protocol):
    """
    离线 Action 执行 provider。

    Provider 的中文含义是“执行提供者”。Environment 负责状态机和快照边界，Provider 负责
    具体检索/Web 返回哪些候选。这样阶段 8.4 不强依赖真实 Mongo/Milvus，8.6 可以注入
    真实 provider，测试可以注入 fake provider。
    """

    def local_search(self, state: OfflineState, decision: PlannerDecision) -> list[RetrievalCandidate]:
        """执行 local_search，返回本地候选。"""

    def hyde_search(self, state: OfflineState, decision: PlannerDecision) -> list[RetrievalCandidate]:
        """执行 hyde_search，返回本地候选。"""

    def web_search(self, state: OfflineState, decision: PlannerDecision) -> list[RetrievalCandidate]:
        """执行 web_search，返回 Web 候选。"""


class EmptyOfflineActionProvider:
    """默认空 provider：不连接外部系统，所有检索 Action 都返回空候选。"""

    def local_search(self, state: OfflineState, decision: PlannerDecision) -> list[RetrievalCandidate]:
        return []

    def hyde_search(self, state: OfflineState, decision: PlannerDecision) -> list[RetrievalCandidate]:
        return []

    def web_search(self, state: OfflineState, decision: PlannerDecision) -> list[RetrievalCandidate]:
        return []


class OfflineRagEnvironment:
    """
    固定 snapshot 下的离线 RAG 执行器。

    Environment 本身保存“规则和语料边界”，每条轨迹的 State 由 reset 创建并在 step 中
    复制更新。不要把 OfflineRagEnvironment 理解成一条轨迹中的 S0、S1、S2；真正变化的
    是 OfflineState。
    """

    def __init__(
            self,
            *,
            snapshot: EnvironmentSnapshot | None = None,
            action_provider: OfflineActionProvider | None = None,
            planner_mode: PlannerMode | str = PlannerMode.RULE,
            run_id_prefix: str = "stage8_eval",
            max_steps: int = PLANNER_MAX_STEPS,
    ) -> None:
        self.snapshot = snapshot
        self.action_provider = action_provider or EmptyOfflineActionProvider()
        self.planner_mode = _normalize_planner_mode(planner_mode)
        self.run_id_prefix = _require_text(run_id_prefix, field_name="run_id_prefix")
        if max_steps <= 0:
            raise ValueError("max_steps 必须大于 0")
        self.max_steps = int(max_steps)

    def reset(
            self,
            case: PlannerEvalCase,
            snapshot: EnvironmentSnapshot | None = None,
            *,
            run_id: str | None = None,
            planner_mode: PlannerMode | str | None = None,
    ) -> OfflineState:
        """根据评测样本和环境快照创建一份独立初始 State。"""
        active_snapshot = snapshot or self.snapshot
        if active_snapshot is None:
            raise ValueError("reset 必须传入 snapshot，或在 Environment 初始化时提供 snapshot")
        normalized_planner_mode = _normalize_planner_mode(planner_mode or self.planner_mode)
        normalized_run_id = _require_text(
            run_id or f"{self.run_id_prefix}_{uuid.uuid4().hex[:12]}",
            field_name="run_id",
        )
        retrieval_config_snapshot = dict(active_snapshot.retrieval_config_snapshot)
        web_search_allowed = bool(
            retrieval_config_snapshot.get("web_fallback_enabled", False)
            and case.expected_behavior.should_call_web
        )
        subject_resolution_status = (
            SubjectResolutionStatus.CONFIRMED
            if case.expected_subject_ids
            else (
                SubjectResolutionStatus.AMBIGUOUS
                if case.expected_behavior.should_ask_clarification
                else SubjectResolutionStatus.NO_MENTION
            )
        )
        disabled_chunk_ids = [chunk.chunk_id for chunk in active_snapshot.disabled_chunks]
        corpus_match_status = self._case_corpus_match_status(case, active_snapshot)
        return OfflineState(
            run_id=normalized_run_id,
            case_id=case.case_id,
            snapshot_id=active_snapshot.snapshot_id,
            snapshot=active_snapshot,
            planner_mode=normalized_planner_mode,
            session_id=f"eval_{normalized_run_id}_{case.case_id}_{normalized_planner_mode}",
            owner_user_id=case.owner_user_id,
            tenant_id=case.tenant_id,
            dataset_ids=list(case.dataset_ids),
            original_query=case.query,
            current_query=case.query,
            subject_ids=list(case.expected_subject_ids),
            standard_subject_names=list(case.expected_subject_names),
            subject_resolution_status=subject_resolution_status,
            query_identifiers=copy.deepcopy(case.expected_identifiers),
            retrieval_config_version=active_snapshot.retrieval_config_version,
            retrieval_config_snapshot=retrieval_config_snapshot,
            policy_version=active_snapshot.policy_version,
            web_search_allowed=web_search_allowed,
            disabled_chunk_ids=disabled_chunk_ids,
            planner_max_steps=self.max_steps,
            config_match_status=self._config_match_status(active_snapshot, retrieval_config_snapshot),
            corpus_match_status=corpus_match_status,
        )

    def step(self, state: OfflineState, decision: PlannerDecision) -> OfflineStepResult:
        """执行一个 PlannerDecision，返回新的 State 和 Observation。"""
        next_state = state.model_copy(deep=True)
        start = time.monotonic()
        step_number = next_state.planner_step + 1

        if next_state.terminal_action is not None:
            error = OfflineError(
                code="terminal_state_already_reached",
                message="轨迹已经终止，不能继续执行新的 Action",
                step=step_number,
                action=decision.action,
            )
            return self._failed_step(next_state, decision, error, start)

        transition_error = self._validate_transition(next_state, decision, step_number)
        if transition_error is not None:
            return self._failed_step(next_state, decision, transition_error, start)

        if decision.action in RETRIEVAL_ACTIONS:
            return self._execute_retrieval_action(next_state, decision, start)
        return self._execute_terminal_action(next_state, decision, start)

    def run_action_path(
            self,
            case: PlannerEvalCase,
            actions: list[PlannerDecision | QueryAction | str],
            *,
            snapshot: EnvironmentSnapshot | None = None,
            run_id: str | None = None,
            planner_mode: PlannerMode | str | None = None,
    ) -> OfflineTrajectoryResult:
        """按外部指定 Action 序列执行一条离线轨迹。"""
        state = self.reset(case, snapshot, run_id=run_id, planner_mode=planner_mode)
        for raw_action in actions:
            decision_result = self._coerce_decision(raw_action, state)
            if isinstance(decision_result, OfflineError):
                state.errors.append(decision_result)
                return self._trajectory_result(state)
            step_result = self.step(state, decision_result)
            state = step_result.state
            if step_result.error is not None or step_result.terminal:
                break
        if state.terminal_action is None and not state.errors:
            state.errors.append(OfflineError(
                code="no_terminal_action",
                message="Action 序列没有到达 answer/refuse/ask_clarification 终态",
                step=state.planner_step,
            ))
        return self._trajectory_result(state)

    def run_planner(
            self,
            case: PlannerEvalCase,
            planner: QueryPlanner,
            *,
            snapshot: EnvironmentSnapshot | None = None,
            run_id: str | None = None,
            planner_mode: PlannerMode | str | None = None,
    ) -> OfflineTrajectoryResult:
        """让 Planner 基于每一步 Observation 自动决策，直到终态或失败。"""
        state = self.reset(case, snapshot, run_id=run_id, planner_mode=planner_mode)
        state.policy_version = _require_text(
            getattr(planner, "policy_version", state.policy_version),
            field_name="planner.policy_version",
        )
        while state.terminal_action is None and state.planner_step < state.planner_max_steps:
            try:
                decision = planner.plan(self._planner_context(state))
                if not isinstance(decision, PlannerDecision):
                    decision = PlannerDecision.model_validate(decision)
            except Exception as exc:
                state.errors.append(OfflineError(
                    code="planner_output_invalid",
                    message=f"Planner 输出无法解析为 PlannerDecision：{exc}",
                    step=state.planner_step + 1,
                ))
                break
            step_result = self.step(state, decision)
            state = step_result.state
            if step_result.error is not None:
                break
        if state.terminal_action is None and not state.errors:
            state.errors.append(OfflineError(
                code="max_steps_exceeded",
                message="Planner 未在最大步数内到达终态",
                step=state.planner_step,
            ))
        return self._trajectory_result(state)

    def _execute_retrieval_action(
            self,
            state: OfflineState,
            decision: PlannerDecision,
            start: float,
    ) -> OfflineStepResult:
        provider_method = {
            QueryAction.LOCAL_SEARCH: self.action_provider.local_search,
            QueryAction.HYDE_SEARCH: self.action_provider.hyde_search,
            QueryAction.WEB_SEARCH: self.action_provider.web_search,
        }[decision.action]

        try:
            raw_candidates = provider_method(state.model_copy(deep=True), decision)
            candidates = [self._normalize_candidate(candidate) for candidate in raw_candidates]
            observation = self._observation_from_candidates(state, decision, candidates)
            execution_status = (
                PlannerExecutionStatus.FAILED
                if observation.status == ObservationStatus.FAILED
                else PlannerExecutionStatus.COMPLETED
            )
            error = self._validate_candidates_against_snapshot(state, candidates, decision)
        except Exception as exc:
            observation = self._failed_observation(decision, error_code="action_provider_failed")
            execution_status = PlannerExecutionStatus.FAILED
            error = OfflineError(
                code="action_provider_failed",
                message=f"{decision.action.value} 执行失败：{exc}",
                step=state.planner_step + 1,
                action=decision.action,
            )
            candidates = []

        duration_ms = _elapsed_ms(start)
        recording_error = _notify_provider_observation(
            self.action_provider,
            state=state,
            decision=decision,
            candidates=candidates,
            observation=observation,
            error=error,
            duration_ms=duration_ms,
        )
        if error is None and recording_error is not None:
            error = recording_error
        state.planner_step += 1
        state.action_history.append(PlannerHistoryItem(
            step=state.planner_step,
            decision=decision,
            execution_status=execution_status,
        ))
        state.latest_observation = observation
        state.retrieval_candidates[decision.action.value] = candidates
        state.accumulated_candidates = _merge_candidates(state.accumulated_candidates, candidates)
        state.trace_steps.append(OfflineTraceStep(
            step=state.planner_step,
            input_observation=self._previous_observation_before_current_step(state),
            decision=decision,
            execution_status=(
                TraceStepStatus.FAILED
                if error is not None or observation.status == ObservationStatus.FAILED
                else TraceStepStatus.COMPLETED
            ),
            output_observation=observation,
            duration_ms=duration_ms,
            error=error,
        ))
        if error is not None:
            state.errors.append(error)
        return OfflineStepResult(
            state=state,
            decision=decision,
            observation=observation,
            terminal=False,
            error=error,
        )

    def _execute_terminal_action(
            self,
            state: OfflineState,
            decision: PlannerDecision,
            start: float,
    ) -> OfflineStepResult:
        state.planner_step += 1
        state.action_history.append(PlannerHistoryItem(
            step=state.planner_step,
            decision=decision,
            execution_status=PlannerExecutionStatus.COMPLETED,
        ))
        state.terminal_action = decision.action
        state.terminal_reason_code = decision.reason_code
        if decision.action == QueryAction.ANSWER:
            state.citations = self._citations_from_candidates(state.accumulated_candidates)
            state.answer = self._build_offline_answer(state)
        elif decision.action == QueryAction.ASK_CLARIFICATION:
            state.answer = self._build_clarification_text(state)
        else:
            state.answer = "当前离线评测环境判定为拒答，未生成设备处置答案。"
        state.trace_steps.append(OfflineTraceStep(
            step=state.planner_step,
            input_observation=state.latest_observation,
            decision=decision,
            execution_status=TraceStepStatus.COMPLETED,
            output_observation=None,
            duration_ms=_elapsed_ms(start),
        ))
        return OfflineStepResult(
            state=state,
            decision=decision,
            observation=None,
            terminal=True,
        )

    def _failed_step(
            self,
            state: OfflineState,
            decision: PlannerDecision,
            error: OfflineError,
            start: float,
    ) -> OfflineStepResult:
        state.errors.append(error)
        state.trace_steps.append(OfflineTraceStep(
            step=max(1, error.step),
            input_observation=state.latest_observation,
            decision=decision,
            execution_status=TraceStepStatus.FAILED,
            output_observation=None,
            duration_ms=_elapsed_ms(start),
            error=error,
        ))
        return OfflineStepResult(state=state, decision=decision, error=error)

    def _validate_transition(
            self,
            state: OfflineState,
            decision: PlannerDecision,
            step_number: int,
    ) -> OfflineError | None:
        if decision.action not in self._allowed_actions(state):
            return OfflineError(
                code="action_not_allowed",
                message=f"Action={decision.action.value} 不在当前离线环境允许列表中",
                step=step_number,
                action=decision.action,
            )
        previous_action = state.action_history[-1].decision.action if state.action_history else None
        if previous_action is None:
            if decision.action not in FIRST_ACTIONS:
                return OfflineError(
                    code="illegal_action_transition",
                    message=f"第一步不能执行 Action={decision.action.value}",
                    step=step_number,
                    action=decision.action,
                )
            return None
        if previous_action in TERMINAL_ACTIONS:
            return OfflineError(
                code="terminal_state_already_reached",
                message="终态 Action 之后不能继续执行",
                step=step_number,
                action=decision.action,
            )
        if decision.action not in VALID_TRANSITIONS.get(previous_action, set()):
            return OfflineError(
                code="illegal_action_transition",
                message=(
                    f"非法 Action 转移：{previous_action.value} -> {decision.action.value}"
                ),
                step=step_number,
                action=decision.action,
            )
        executed_retrieval_actions = {
            item.decision.action
            for item in state.action_history
            if item.decision.action in RETRIEVAL_ACTIONS
        }
        if decision.action in executed_retrieval_actions:
            return OfflineError(
                code="duplicated_retrieval_action",
                message=f"同一条轨迹中不能重复执行 {decision.action.value}",
                step=step_number,
                action=decision.action,
            )
        return None

    def _planner_context(self, state: OfflineState) -> PlannerContext:
        return PlannerContext(
            original_query=state.original_query,
            current_query=state.current_query,
            subject_resolution_status=state.subject_resolution_status,
            subject_ids=list(state.subject_ids),
            subject_candidates=[],
            clarification_question=None,
            query_identifiers=copy.deepcopy(state.query_identifiers),
            latest_observation=state.latest_observation,
            action_history=list(state.action_history),
            web_search_allowed=state.web_search_allowed,
            safe_guard_triggered=bool(state.errors),
            planner_step=state.planner_step,
            max_steps=state.planner_max_steps,
            allowed_actions=self._allowed_actions(state),
        )

    def _allowed_actions(self, state: OfflineState) -> list[QueryAction]:
        actions: list[QueryAction] = []
        # 真实本地/HyDE Provider（动作执行器）要求 subject_ids（主体 ID）非空，禁止
        # 无主体时退化为全库检索。Environment（环境）必须只向 Planner（规划器）暴露
        # Provider 实际可执行的 Action（动作），否则合法 JSON 也会在执行层必然失败。
        if state.subject_ids:
            actions.extend((QueryAction.LOCAL_SEARCH, QueryAction.HYDE_SEARCH))
        if state.web_search_allowed:
            actions.append(QueryAction.WEB_SEARCH)
        actions.extend((
            QueryAction.ANSWER,
            QueryAction.ASK_CLARIFICATION,
            QueryAction.REFUSE,
        ))
        return actions

    def _observation_from_candidates(
            self,
            state: OfflineState,
            decision: PlannerDecision,
            candidates: list[RetrievalCandidate],
    ) -> RetrievalObservation:
        if not candidates:
            return self._empty_observation(state, decision)

        evidence_summaries = [
            _candidate_to_evidence_summary(candidate, state.query_identifiers)
            for candidate in candidates[:5]
        ]
        matched_identifiers = _matched_identifiers(candidates, state.query_identifiers)
        identifier_status, clarification_question = _identifier_status(
            requested_identifiers=state.query_identifiers,
            matched_identifiers=matched_identifiers,
            has_candidates=bool(candidates),
        )
        scores = [
            candidate.rerank_score
            if candidate.rerank_score is not None
            else min(1.0, float(candidate.retrieval_score))
            for candidate in candidates
        ]
        return RetrievalObservation(
            action=decision.action,
            status=ObservationStatus.SUCCESS,
            channel_counts={decision.action.value: len(candidates)},
            candidate_count=len(candidates),
            reranked_count=len(candidates),
            top_rerank_score=max(scores),
            requested_identifiers=copy.deepcopy(state.query_identifiers),
            matched_identifiers=matched_identifiers,
            identifier_resolution_status=identifier_status,
            citation_count=0,
            evidence_summaries=evidence_summaries,
            clarification_question=clarification_question,
            used_structured_filter=bool(state.query_identifiers),
        )

    def _empty_observation(self, state: OfflineState, decision: PlannerDecision) -> RetrievalObservation:
        identifier_status, clarification_question = _identifier_status(
            requested_identifiers=state.query_identifiers,
            matched_identifiers={},
            has_candidates=False,
        )
        return RetrievalObservation(
            action=decision.action,
            status=ObservationStatus.EMPTY,
            candidate_count=0,
            reranked_count=0,
            requested_identifiers=copy.deepcopy(state.query_identifiers),
            matched_identifiers={},
            identifier_resolution_status=identifier_status,
            clarification_question=clarification_question,
            used_structured_filter=bool(state.query_identifiers),
        )

    @staticmethod
    def _failed_observation(decision: PlannerDecision, *, error_code: str) -> RetrievalObservation:
        return RetrievalObservation(
            action=decision.action,
            status=ObservationStatus.FAILED,
            candidate_count=0,
            reranked_count=0,
            identifier_resolution_status=IdentifierResolutionStatus.NOT_APPLICABLE,
            error_code=error_code,
        )

    def _validate_candidates_against_snapshot(
            self,
            state: OfflineState,
            candidates: list[RetrievalCandidate],
            decision: PlannerDecision,
    ) -> OfflineError | None:
        enabled_identity = _snapshot_enabled_identity(state.snapshot)
        disabled_identity = _snapshot_disabled_identity(state.snapshot)
        for candidate in candidates:
            if candidate.source_type == EvidenceSourceType.WEB:
                continue
            identity = (
                str(candidate.document_id),
                candidate.chunk_id,
                int(candidate.index_version or 0),
            )
            if identity in disabled_identity:
                state.corpus_match_status = "mismatch"
                return OfflineError(
                    code="candidate_disabled_by_snapshot",
                    message=(
                        "Action 返回了 snapshot 中被人工禁用的 chunk："
                        f"document_id={candidate.document_id}, "
                        f"chunk_id={candidate.chunk_id}, "
                        f"index_version={candidate.index_version}"
                    ),
                    step=state.planner_step + 1,
                    action=decision.action,
                )
            if identity not in enabled_identity:
                state.corpus_match_status = "mismatch"
                return OfflineError(
                    code="candidate_not_in_snapshot",
                    message=(
                        "Action 返回了不在 snapshot.enabled_chunks 中的本地候选："
                        f"document_id={candidate.document_id}, "
                        f"chunk_id={candidate.chunk_id}, "
                        f"index_version={candidate.index_version}"
                    ),
                    step=state.planner_step + 1,
                    action=decision.action,
                )
        return None

    def _citations_from_candidates(self, candidates: list[RetrievalCandidate]) -> list[Citation]:
        citations: list[Citation] = []
        for candidate in candidates[:3]:
            if candidate.source_type == EvidenceSourceType.LOCAL:
                citations.append(Citation(
                    document_id=candidate.document_id,
                    chunk_id=candidate.chunk_id,
                    title=candidate.title,
                    source=candidate.source_title or candidate.title,
                    score=float(candidate.rerank_score or candidate.retrieval_score),
                    source_type=EvidenceSourceType.LOCAL,
                ))
            else:
                citations.append(Citation(
                    title=candidate.title,
                    source=str(candidate.url),
                    score=float(candidate.rerank_score or candidate.retrieval_score),
                    source_type=EvidenceSourceType.WEB,
                ))
        return citations

    @staticmethod
    def _build_offline_answer(state: OfflineState) -> str:
        if not state.citations:
            return "离线评测已到达 answer 终态，但当前轨迹没有形成可引用证据。"
        titles = "、".join(citation.title for citation in state.citations[:3])
        return f"离线评测基于 {titles} 形成 answer 终态。"

    @staticmethod
    def _build_clarification_text(state: OfflineState) -> str:
        if state.latest_observation and state.latest_observation.clarification_question:
            return state.latest_observation.clarification_question
        return "请补充设备型号、报警码或更具体的文档范围后再继续。"

    @staticmethod
    def _previous_observation_before_current_step(state: OfflineState) -> RetrievalObservation | None:
        if not state.trace_steps:
            return None
        for step in reversed(state.trace_steps):
            if step.output_observation is not None:
                return step.output_observation
        return None

    def _coerce_decision(
            self,
            raw_action: PlannerDecision | QueryAction | str,
            state: OfflineState,
    ) -> PlannerDecision | OfflineError:
        if isinstance(raw_action, PlannerDecision):
            return raw_action
        try:
            action = raw_action if isinstance(raw_action, QueryAction) else QueryAction(str(raw_action))
        except ValueError:
            return OfflineError(
                code="unknown_action",
                message=f"未知 Action：{raw_action}",
                step=state.planner_step + 1,
            )
        return PlannerDecision(
            action=action,
            query=state.current_query,
            reason_code=_default_reason_for_action(action),
        )

    def _trajectory_result(self, state: OfflineState) -> OfflineTrajectoryResult:
        status = (
            OfflineTrajectoryStatus.COMPLETED
            if state.terminal_action is not None and not state.errors
            else OfflineTrajectoryStatus.FAILED
        )
        return OfflineTrajectoryResult(
            run_id=state.run_id,
            case_id=state.case_id,
            snapshot_id=state.snapshot_id,
            planner_mode=state.planner_mode,
            status=status,
            action_path=[step.decision.action for step in state.trace_steps],
            terminal_action=state.terminal_action,
            terminal_reason_code=state.terminal_reason_code,
            config_match_status=state.config_match_status,
            corpus_match_status=state.corpus_match_status,
            trace_steps=list(state.trace_steps),
            retrieved_candidates=list(state.accumulated_candidates),
            citations=list(state.citations),
            answer=state.answer,
            errors=list(state.errors),
            final_state=state,
        )

    @staticmethod
    def _config_match_status(
            snapshot: EnvironmentSnapshot,
            retrieval_config_snapshot: dict[str, Any],
    ) -> str:
        return (
            "match"
            if dict(snapshot.retrieval_config_snapshot) == dict(retrieval_config_snapshot)
            else "mismatch"
        )

    @staticmethod
    def _case_corpus_match_status(case: PlannerEvalCase, snapshot: EnvironmentSnapshot) -> str:
        snapshot_dataset_ids = set(snapshot.dataset_ids)
        if not set(case.dataset_ids).issubset(snapshot_dataset_ids):
            return "mismatch"
        document_versions = {
            document.document_id: document.index_version
            for document in snapshot.documents
        }
        enabled_identity = _snapshot_enabled_identity(snapshot)
        disabled_identity = _snapshot_disabled_identity(snapshot)
        for document_id, index_version in case.source_index_versions.items():
            if document_versions.get(document_id) != index_version:
                return "mismatch"
        for chunk in case.expected_chunks:
            identity = (chunk.document_id, chunk.chunk_id, chunk.index_version)
            if identity in disabled_identity or identity not in enabled_identity:
                return "mismatch"
        return "match"

    def _normalize_candidate(self, candidate: RetrievalCandidate | dict[str, Any]) -> RetrievalCandidate:
        if isinstance(candidate, RetrievalCandidate):
            return candidate
        return RetrievalCandidate.model_validate(candidate)


def _candidate_to_evidence_summary(
        candidate: RetrievalCandidate,
        requested_identifiers: dict[str, list[str]],
) -> EvidenceSummary:
    return EvidenceSummary(
        document_id=candidate.document_id,
        chunk_id=candidate.chunk_id,
        title=candidate.title,
        source_type=candidate.source_type,
        rerank_score=candidate.rerank_score,
        matched_identifiers=_matched_identifiers([candidate], requested_identifiers),
        content_excerpt=candidate.content[:DEFAULT_EVIDENCE_EXCERPT_CHARS],
    )


def _matched_identifiers(
        candidates: list[RetrievalCandidate],
        requested_identifiers: dict[str, list[str]],
) -> dict[str, list[str]]:
    matched: dict[str, list[str]] = {}
    for identifier_type, values in requested_identifiers.items():
        normalized_values = [str(value).strip() for value in values if str(value).strip()]
        hits: list[str] = []
        for candidate in candidates:
            candidate_value = getattr(candidate, identifier_type, None)
            haystack = f"{candidate_value or ''}\n{candidate.title}\n{candidate.content}".lower()
            for value in normalized_values:
                if value.lower() in haystack and value not in hits:
                    hits.append(value)
        if hits:
            matched[identifier_type] = hits
    return matched


def _identifier_status(
        *,
        requested_identifiers: dict[str, list[str]],
        matched_identifiers: dict[str, list[str]],
        has_candidates: bool,
) -> tuple[IdentifierResolutionStatus, str | None]:
    if not requested_identifiers:
        return IdentifierResolutionStatus.NOT_APPLICABLE, None
    for identifier_type, requested_values in requested_identifiers.items():
        matched_values = set(matched_identifiers.get(identifier_type, []))
        if not set(requested_values).issubset(matched_values):
            question = (
                "请确认要查询的设备型号、报警码或零件编号；"
                f"当前检索{'有候选但未命中' if has_candidates else '没有命中'} {identifier_type}。"
            )
            return IdentifierResolutionStatus.NOT_FOUND, question
    return IdentifierResolutionStatus.EXACT_MATCH, None


def _merge_candidates(
        existing: list[RetrievalCandidate],
        new_candidates: list[RetrievalCandidate],
) -> list[RetrievalCandidate]:
    merged = list(existing)
    seen = {_candidate_identity(candidate) for candidate in merged}
    for candidate in new_candidates:
        identity = _candidate_identity(candidate)
        if identity in seen:
            continue
        merged.append(candidate)
        seen.add(identity)
    return merged


def _candidate_identity(candidate: RetrievalCandidate) -> tuple[str, str | int | None, int | None, str | None]:
    if candidate.source_type == EvidenceSourceType.WEB:
        return ("web", None, None, candidate.url)
    return (
        str(candidate.document_id),
        candidate.chunk_id,
        candidate.index_version,
        None,
    )


def _snapshot_enabled_identity(snapshot: EnvironmentSnapshot) -> set[tuple[str, str | int, int]]:
    document_versions = {
        document.document_id: document.index_version
        for document in snapshot.documents
    }
    identities: set[tuple[str, str | int, int]] = set()
    for document_id, chunk_ids in snapshot.enabled_chunks.items():
        index_version = document_versions.get(document_id)
        if index_version is None:
            continue
        for chunk_id in chunk_ids:
            identities.add((document_id, chunk_id, index_version))
    return identities


def _snapshot_disabled_identity(snapshot: EnvironmentSnapshot) -> set[tuple[str, str | int, int]]:
    return {
        (chunk.document_id, chunk.chunk_id, chunk.index_version)
        for chunk in snapshot.disabled_chunks
    }


def _default_reason_for_action(action: QueryAction) -> PlannerReasonCode:
    return {
        QueryAction.LOCAL_SEARCH: PlannerReasonCode.INITIAL_LOCAL_SEARCH,
        QueryAction.HYDE_SEARCH: PlannerReasonCode.LOCAL_LOW_SCORE,
        QueryAction.WEB_SEARCH: PlannerReasonCode.REALTIME_QUERY,
        QueryAction.ANSWER: PlannerReasonCode.LOCAL_EVIDENCE_SUFFICIENT,
        QueryAction.ASK_CLARIFICATION: PlannerReasonCode.SUBJECT_REQUIRED,
        QueryAction.REFUSE: PlannerReasonCode.SAFE_GUARD_TRIGGERED,
    }[action]


def _normalize_planner_mode(value: PlannerMode | str) -> str:
    if isinstance(value, PlannerMode):
        return value.value
    text = str(value or "").strip()
    if not text:
        raise ValueError("planner_mode 不能为空")
    return text


def _require_text(value: Any, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} 不能为空")
    return text


def _elapsed_ms(start: float) -> int:
    return max(0, int((time.monotonic() - start) * 1000))


def _notify_provider_observation(
        provider: OfflineActionProvider,
        *,
        state: OfflineState,
        decision: PlannerDecision,
        candidates: list[RetrievalCandidate],
        observation: RetrievalObservation,
        error: OfflineError | None,
        duration_ms: int,
) -> OfflineError | None:
    """
    通知可选的 RecordingActionProvider（记录型动作执行器）保存 Observation（观察结果）。

    OfflineActionProvider（离线动作执行器）协议仍然只要求返回候选；记录能力是阶段 9.2
    的可选扩展。这样旧测试和旧 provider 不需要改动，真实训练时又能把每次 Action（动作）
    看到的 Observation（观察结果）写成审计日志。
    """
    record_observation = getattr(provider, "record_observation", None)
    if record_observation is None:
        return None
    try:
        record_observation(
            state=state.model_copy(deep=True),
            decision=decision.model_copy(deep=True),
            candidates=[candidate.model_copy(deep=True) for candidate in candidates],
            observation=observation.model_copy(deep=True),
            error=error.model_copy(deep=True) if error is not None else None,
            duration_ms=duration_ms,
        )
    except Exception as exc:
        return OfflineError(
            code="provider_recording_failed",
            message=f"Provider Observation 记录失败：{exc}",
            step=state.planner_step + 1,
            action=decision.action,
        )
    return None


__all__ = [
    "EmptyOfflineActionProvider",
    "OfflineActionProvider",
    "OfflineError",
    "OfflineRagEnvironment",
    "OfflineState",
    "OfflineStepResult",
    "OfflineTraceStep",
    "OfflineTrajectoryResult",
    "OfflineTrajectoryStatus",
]
