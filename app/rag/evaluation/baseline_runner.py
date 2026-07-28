"""
阶段 8.6 Planner baseline runner。

baseline runner 的中文含义是“基线评测跑批器”。它不训练模型，也不计算 GRPO 的
组内 advantage；它只在固定 case、固定 EnvironmentSnapshot 和固定 Reward 版本下，
让一个或多个 Planner 跑完整离线轨迹，再把 Trace、Reward、配置和用量写成稳定 JSON。

当前第一版只真正执行 RuleBasedPlanner。API Planner 和本地零样本 Planner 尚未接入
结构化模型适配器时，会被明确标记为 skipped，不能伪造 Reward 或假装评测成功。
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.rag.evaluation.case_schema import (
    CaseSplit,
    EnvironmentSnapshot,
    PlannerEvalCase,
    PlannerEvalResult,
    PlannerMode,
    SnapshotDocument,
    load_planner_cases,
)
from app.rag.evaluation.offline_environment import (
    OfflineActionProvider,
    OfflineRagEnvironment,
    OfflineState,
    OfflineTrajectoryResult,
)
from app.rag.evaluation.reward import (
    REWARD_VERSION,
    RewardConfig,
    TrajectoryReward,
    score_trajectory,
)
from app.rag.query.config import RERANK_EVIDENCE_THRESHOLD, RETRIEVAL_CONFIG_VERSION
from app.rag.query.contracts import (
    EvidenceSourceType,
    PlannerDecision,
    RetrievalCandidate,
    RetrievalChannel,
    UsageMetrics,
)
from app.rag.query.planner import RuleBasedPlanner, RuleBasedPlannerConfig


EXECUTABLE_PLANNER_MODES = {PlannerMode.RULE, PlannerMode.API, PlannerMode.LOCAL_BASE}
RUNNER_VERSION = "stage8-baseline-runner-v1"
SNAPSHOT_EXPECTED_PROVIDER_NAME = "snapshot_expected_chunks"


# 第一部分：runner 输出结构。先固定结果文件形状，后续报告和阶段 9 对比才能稳定读取。
class BaselineRunnerModel(BaseModel):
    """baseline runner 内部和输出 schema 公共基类，拒绝未知字段以避免报告格式漂移。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


class BaselinePlannerSummary(BaselineRunnerModel):
    """
    单个 Planner 在本次跑批中的聚合摘要。

    status 使用字符串而不另建枚举，是为了输出文件对报告脚本更直接：completed 表示
    至少跑完了请求 split 的所有 case；skipped 表示当前环境没有这个 planner 的可执行配置。
    """

    # 本次摘要对应的 Planner 模式，例如 rule/api/local_base。
    planner_mode: PlannerMode
    # 运行状态：completed 或 skipped。失败 case 不会让整个 planner 伪装成 skipped。
    status: str = Field(min_length=1)
    # 当前 Planner 的版本、阈值、provider、模型等配置快照。
    config: dict[str, Any] = Field(default_factory=dict)
    # token、耗时、成本等聚合用量。规则 Planner 没有模型调用，token/cost 为 0。
    usage: dict[str, Any] = Field(default_factory=dict)
    # 聚合 Reward 摘要，只保存均值和计数；单条分项明细在 PlannerEvalResult.reward 中。
    reward: dict[str, Any] = Field(default_factory=dict)
    # 该 planner 实际参与评测的 case 数。
    case_count: int = Field(default=0, ge=0)
    # 正常完成离线轨迹的 case 数。
    completed_case_count: int = Field(default=0, ge=0)
    # 环境执行失败、非法 Action 或无终态的 case 数。
    failed_case_count: int = Field(default=0, ge=0)
    # 全局跳过原因。只有 status=skipped 时应有值。
    skip_reason: str = ""

    @model_validator(mode="after")
    def validate_skip_reason(self) -> "BaselinePlannerSummary":
        """防止 skipped 摘要没有原因，报告里看不出为什么少了某个 baseline。"""
        if self.status == "skipped" and not self.skip_reason:
            raise ValueError("status=skipped 时必须提供 skip_reason")
        return self


class BaselineEvalOutput(BaselineRunnerModel):
    """
    baseline runner 的完整 JSON 输出。

    results 保存逐 case 结果；planner_summaries 保存每个 planner 的聚合统计和跳过原因。
    这种结构能同时服务阶段 8 报告和阶段 9 微调前后对比。
    """

    # 本次跑批 ID。一个 run_id 可以包含多个 planner 和多个 case。
    run_id: str = Field(min_length=1)
    # runner 代码版本。runner 行为变更时要升级，避免新旧输出混用。
    runner_version: str = RUNNER_VERSION
    # 创建时间，UTC ISO 字符串，便于跨机器对齐。
    created_at: str = Field(min_length=1)
    # 本次读取的 case split，例如 dev/test。
    split: CaseSplit
    # 固定环境快照 ID。
    snapshot_id: str = Field(min_length=1)
    # Reward 版本。阶段 9 对比时必须使用同一版本。
    reward_version: str = Field(min_length=1)
    # 请求运行的 planner 列表，按命令行顺序保留。
    requested_planners: list[PlannerMode] = Field(min_length=1)
    # 当前候选 provider 类型。第一版使用 snapshot_expected_chunks，不伪装真实 Milvus。
    action_provider: str = Field(min_length=1)
    # 本次 split 下参与评测的 case 数。
    case_count: int = Field(ge=0)
    # 每个 planner 的聚合摘要和 skipped 原因。
    planner_summaries: list[BaselinePlannerSummary] = Field(default_factory=list)
    # 逐 case 的评测结果。跳过的 planner 不生成假 PlannerEvalResult。
    results: list[PlannerEvalResult] = Field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        """返回可直接写入磁盘的 JSON 字典。"""
        return self.model_dump(mode="json")


# 第二部分：离线候选 provider。它让 runner 在没有 Milvus 的环境中仍能验证完整评测管线。
class SnapshotExpectedChunkActionProvider(OfflineActionProvider):
    """
    使用 case 中冻结的 expected evidence 构造确定性候选的离线 provider。

    这个 provider 的边界很重要：它不是线上检索质量评测，也不声称模拟 Milvus 召回。
    本地证据来自 ``expected_chunks``，Web 证据来自 ``expected_web_evidence``；两者都
    只证明 Planner 状态机、Reward 和引用契约可执行，不证明真实 Milvus/Web 召回质量。
    输出里会记录 action_provider=snapshot_expected_chunks，后续接真实 Milvus provider 时
    可以直接替换，不改变 runner 主流程。
    """

    def __init__(self, cases: Iterable[PlannerEvalCase], snapshot: EnvironmentSnapshot) -> None:
        self.case_by_id = {case.case_id: case for case in cases}
        self.document_by_id = {document.document_id: document for document in snapshot.documents}
        self.enabled_chunks = snapshot.enabled_chunks

    def local_search(self, state: OfflineState, decision: PlannerDecision) -> list[RetrievalCandidate]:
        """本地检索 Action：回答型样本返回 expected_chunks，非回答型样本返回空。"""
        case = self.case_by_id[state.case_id]
        return self._local_candidates_for_case(case, RetrievalChannel.ORIGINAL)

    def hyde_search(self, state: OfflineState, decision: PlannerDecision) -> list[RetrievalCandidate]:
        """HyDE 检索 Action：只在 Planner 真的进入 HyDE 时返回同一批期望本地候选。"""
        case = self.case_by_id[state.case_id]
        return self._local_candidates_for_case(case, RetrievalChannel.HYDE)

    def web_search(self, state: OfflineState, decision: PlannerDecision) -> list[RetrievalCandidate]:
        """
        Web 检索 Action：不联网，只把 case 中已冻结的网页证据投影成高分候选。

        没有 ``expected_web_evidence`` 的历史实时 case 仍返回低分占位候选，保持旧的
        web_search -> refuse 行为。新的 Web 回答型 Gold 则能离线验证
        web_search -> answer、URL Citation 和 Reward 契约。
        """
        case = self.case_by_id[state.case_id]
        if not case.expected_behavior.should_call_web:
            return []
        if case.expected_web_evidence:
            content = "；".join(case.expected_answer_points) or case.query
            return [
                RetrievalCandidate(
                    title=evidence.source_title,
                    source_title=evidence.source_title,
                    content=content,
                    equipment_model=_first_identifier(case, "equipment_model"),
                    alarm_code=_first_identifier(case, "alarm_code"),
                    source_type=EvidenceSourceType.WEB,
                    retrieval_channels=[RetrievalChannel.WEB],
                    retrieval_rank=rank,
                    retrieval_score=max(0.0, 0.95 - (rank - 1) * 0.03),
                    rerank_score=max(0.0, 0.95 - (rank - 1) * 0.03),
                    url=evidence.url,
                )
                for rank, evidence in enumerate(
                    case.expected_web_evidence,
                    start=1,
                )
            ]
        return [
            RetrievalCandidate(
                title=f"{case.query} 的离线 Web 占位候选",
                content=f"离线评测未连接真实 Web；仅记录该问题需要外部实时信息：{case.query}",
                equipment_model=_first_identifier(case, "equipment_model"),
                alarm_code=_first_identifier(case, "alarm_code"),
                source_type=EvidenceSourceType.WEB,
                retrieval_channels=[RetrievalChannel.WEB],
                retrieval_rank=1,
                retrieval_score=0.20,
                rerank_score=0.20,
                url=f"https://example.invalid/stage8/{case.case_id}",
            )
        ]

    def _local_candidates_for_case(
            self,
            case: PlannerEvalCase,
            retrieval_channel: RetrievalChannel,
    ) -> list[RetrievalCandidate]:
        if not case.expected_behavior.should_answer:
            return []
        candidates: list[RetrievalCandidate] = []
        for rank, expected_chunk in enumerate(case.expected_chunks, start=1):
            document = self.document_by_id.get(expected_chunk.document_id)
            if document is None:
                continue
            candidates.append(_candidate_from_expected_chunk(
                case=case,
                document=document,
                chunk_id=expected_chunk.chunk_id,
                chunk_index=_chunk_index(
                    self.enabled_chunks.get(expected_chunk.document_id, []),
                    expected_chunk.chunk_id,
                ),
                rank=rank,
                retrieval_channel=retrieval_channel,
            ))
        return candidates


# 第三部分：runner 主流程。输入 case/snapshot/planner 列表，输出可落盘的 BaselineEvalOutput。
def run_baseline_evaluation(
        *,
        cases: list[PlannerEvalCase],
        snapshot: EnvironmentSnapshot,
        split: CaseSplit | str,
        planners: Iterable[PlannerMode | str] | str,
        reward_config: RewardConfig | None = None,
        run_id: str | None = None,
        action_provider: OfflineActionProvider | None = None,
) -> BaselineEvalOutput:
    """
    执行一次 baseline 评测跑批。

    数据流：cases 先按 split 过滤；每个可用 Planner 生成 trajectory；Reward v1 对
    trajectory 打分；最后把每条结果投影成 PlannerEvalResult，并汇总 planner summary。
    """
    active_split = _normalize_split(split)
    selected_cases = [case for case in cases if case.split == active_split]
    if not selected_cases:
        raise ValueError(f"没有 split={active_split.value} 的可评测 case")

    planner_modes = parse_planner_modes(planners)
    active_reward_config = reward_config or RewardConfig()
    normalized_run_id = run_id or f"stage8_eval_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    provider_name = SNAPSHOT_EXPECTED_PROVIDER_NAME if action_provider is None else action_provider.__class__.__name__
    provider = action_provider or SnapshotExpectedChunkActionProvider(selected_cases, snapshot)

    all_results: list[PlannerEvalResult] = []
    summaries: list[BaselinePlannerSummary] = []
    for planner_mode in planner_modes:
        skip_reason = _planner_skip_reason(planner_mode, snapshot)
        if skip_reason:
            summaries.append(_skipped_summary(planner_mode, skip_reason))
            continue
        planner_summary, planner_results = _run_single_planner(
            cases=selected_cases,
            snapshot=snapshot,
            planner_mode=planner_mode,
            reward_config=active_reward_config,
            run_id=normalized_run_id,
            action_provider=provider,
            action_provider_name=provider_name,
        )
        summaries.append(planner_summary)
        all_results.extend(planner_results)

    return BaselineEvalOutput(
        run_id=normalized_run_id,
        created_at=datetime.now(UTC).isoformat(),
        split=active_split,
        snapshot_id=snapshot.snapshot_id,
        reward_version=active_reward_config.reward_version,
        requested_planners=planner_modes,
        action_provider=provider_name,
        case_count=len(selected_cases),
        planner_summaries=summaries,
        results=all_results,
    )


def parse_planner_modes(planners: Iterable[PlannerMode | str] | str) -> list[PlannerMode]:
    """解析命令行传入的 planner 列表，支持 'rule,api' 和 ['rule', 'api'] 两种形式。"""
    raw_values: Iterable[PlannerMode | str]
    if isinstance(planners, str):
        raw_values = planners.split(",")
    else:
        raw_values = planners

    modes: list[PlannerMode] = []
    seen: set[PlannerMode] = set()
    for raw_value in raw_values:
        if isinstance(raw_value, PlannerMode):
            mode = raw_value
        else:
            normalized_value = str(raw_value).strip()
            if not normalized_value:
                continue
            mode = PlannerMode(normalized_value)
        if mode not in EXECUTABLE_PLANNER_MODES:
            supported = ", ".join(sorted(mode.value for mode in EXECUTABLE_PLANNER_MODES))
            raise ValueError(f"阶段 8.6 只支持 {supported}，当前收到 {mode.value}")
        if mode not in seen:
            modes.append(mode)
            seen.add(mode)
    if not modes:
        raise ValueError("planners 不能为空")
    return modes


# 第四部分：文件读写。CLI 和测试共用这些函数，避免脚本里散落 JSON 细节。
def load_environment_snapshot(path: str | Path) -> EnvironmentSnapshot:
    """读取 environment_snapshot.json 并校验 EnvironmentSnapshot schema。"""
    snapshot_path = Path(path)
    return EnvironmentSnapshot.model_validate_json(snapshot_path.read_text(encoding="utf-8"))


def run_baseline_evaluation_from_files(
        *,
        cases_path: str | Path,
        snapshot_path: str | Path,
        split: CaseSplit | str,
        planners: Iterable[PlannerMode | str] | str,
        reward_version: str = REWARD_VERSION,
        output_path: str | Path | None = None,
) -> BaselineEvalOutput:
    """从磁盘读取 case/snapshot，执行评测，并在传入 output_path 时写入 JSON。"""
    cases = load_planner_cases(cases_path)
    snapshot = load_environment_snapshot(snapshot_path)
    output = run_baseline_evaluation(
        cases=cases,
        snapshot=snapshot,
        split=split,
        planners=planners,
        reward_config=RewardConfig(reward_version=reward_version),
    )
    if output_path is not None:
        write_baseline_eval_output(output, output_path)
    return output


def write_baseline_eval_output(output: BaselineEvalOutput, path: str | Path) -> None:
    """把 BaselineEvalOutput 写成 UTF-8 JSON，父目录不存在时自动创建。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output.to_json_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# 第五部分：内部执行工具。放在文件末尾，保持上方主流程按调用顺序阅读。
def _run_single_planner(
        *,
        cases: list[PlannerEvalCase],
        snapshot: EnvironmentSnapshot,
        planner_mode: PlannerMode,
        reward_config: RewardConfig,
        run_id: str,
        action_provider: OfflineActionProvider,
        action_provider_name: str,
) -> tuple[BaselinePlannerSummary, list[PlannerEvalResult]]:
    if planner_mode != PlannerMode.RULE:
        return _skipped_summary(planner_mode, f"{planner_mode.value} Planner 适配器尚未实现"), []

    planner = _build_rule_planner(snapshot)
    environment = OfflineRagEnvironment(
        snapshot=snapshot,
        action_provider=action_provider,
        planner_mode=planner_mode,
        run_id_prefix=run_id,
    )
    planner_results: list[PlannerEvalResult] = []
    rewards: list[TrajectoryReward] = []
    start = time.monotonic()

    for case in cases:
        trajectory = environment.run_planner(
            case,
            planner,
            run_id=f"{run_id}_{planner_mode.value}_{case.case_id}",
            planner_mode=planner_mode,
        )
        reward = score_trajectory(case, trajectory, reward_config)
        planner_results.append(_planner_eval_result_from_trajectory(
            run_id=run_id,
            case=case,
            planner_mode=planner_mode,
            trajectory=trajectory,
            reward=reward,
        ))
        rewards.append(reward)

    duration_ms = _elapsed_ms(start)
    completed_count = sum(
        1
        for result in planner_results
        if not result.errors
    )
    summary = BaselinePlannerSummary(
        planner_mode=planner_mode,
        status="completed",
        config=_planner_config(planner_mode, planner, snapshot, action_provider_name),
        usage=_aggregate_usage(planner_results, duration_ms=duration_ms),
        reward=_aggregate_rewards(rewards),
        case_count=len(cases),
        completed_case_count=completed_count,
        failed_case_count=len(cases) - completed_count,
    )
    return summary, planner_results


def _planner_eval_result_from_trajectory(
        *,
        run_id: str,
        case: PlannerEvalCase,
        planner_mode: PlannerMode,
        trajectory: OfflineTrajectoryResult,
        reward: TrajectoryReward,
) -> PlannerEvalResult:
    usage = _trajectory_usage(trajectory)
    return PlannerEvalResult(
        run_id=run_id,
        case_id=case.case_id,
        split=case.split,
        planner_mode=planner_mode,
        snapshot_id=trajectory.snapshot_id,
        reward_version=reward.reward_version,
        trace_id=trajectory.run_id,
        action_path=trajectory.action_path,
        terminal_action=trajectory.terminal_action,
        terminal_reason_code=(
            trajectory.terminal_reason_code.value
            if trajectory.terminal_reason_code is not None
            else ""
        ),
        retrieved_chunk_ids=[
            candidate.chunk_id
            for candidate in trajectory.retrieved_candidates
            if candidate.source_type == EvidenceSourceType.LOCAL and candidate.chunk_id is not None
        ],
        citation_chunk_ids=[
            citation.chunk_id
            for citation in trajectory.citations
            if citation.source_type == EvidenceSourceType.LOCAL and citation.chunk_id is not None
        ],
        metrics=_metrics_from_reward(reward),
        reward=reward.to_json_dict(),
        usage=usage,
        errors=[error.model_dump(mode="json") for error in trajectory.errors],
    )


def _build_rule_planner(snapshot: EnvironmentSnapshot) -> RuleBasedPlanner:
    retrieval_snapshot = dict(snapshot.retrieval_config_snapshot)
    threshold = float(retrieval_snapshot.get("evidence_threshold", RERANK_EVIDENCE_THRESHOLD))
    retrieval_config_version = str(snapshot.retrieval_config_version or RETRIEVAL_CONFIG_VERSION)
    return RuleBasedPlanner(
        config=RuleBasedPlannerConfig(
            rerank_evidence_threshold=threshold,
            retrieval_config_version=retrieval_config_version,
        ),
        policy_version=snapshot.policy_version,
    )


def _planner_skip_reason(planner_mode: PlannerMode, snapshot: EnvironmentSnapshot) -> str:
    registry_entry = _planner_registry_entry(planner_mode, snapshot)
    if registry_entry and not bool(registry_entry.get("enabled_for_eval", False)):
        return str(registry_entry.get("unavailable_reason") or f"{planner_mode.value} 未启用离线评测")
    if planner_mode == PlannerMode.RULE:
        return ""
    if planner_mode == PlannerMode.API:
        return "API Planner provider 未配置，已跳过，不影响 rule baseline"
    if planner_mode == PlannerMode.LOCAL_BASE:
        return "本地零样本 Planner 模型未配置，已跳过，不伪装评测结果"
    return f"{planner_mode.value} 不属于阶段 8.6 可执行 baseline"


def _skipped_summary(planner_mode: PlannerMode, reason: str) -> BaselinePlannerSummary:
    return BaselinePlannerSummary(
        planner_mode=planner_mode,
        status="skipped",
        config={"enabled_for_eval": False, "skip_reason": reason},
        usage=UsageMetrics().model_dump(mode="json"),
        reward={"average_total_reward": None, "scored_case_count": 0},
        skip_reason=reason,
    )


def _planner_registry_entry(planner_mode: PlannerMode, snapshot: EnvironmentSnapshot) -> dict[str, Any] | None:
    for entry in snapshot.planner_registry:
        if str(entry.get("planner_mode", "")).strip() == planner_mode.value:
            return entry
    return None


def _planner_config(
        planner_mode: PlannerMode,
        planner: RuleBasedPlanner,
        snapshot: EnvironmentSnapshot,
        action_provider_name: str,
) -> dict[str, Any]:
    return {
        "planner_mode": planner_mode.value,
        "policy_version": planner.policy_version,
        "retrieval_config_version": planner.config.retrieval_config_version,
        "rerank_evidence_threshold": planner.config.rerank_evidence_threshold,
        "snapshot_policy_version": snapshot.policy_version,
        "action_provider": action_provider_name,
    }


def _aggregate_rewards(rewards: list[TrajectoryReward]) -> dict[str, Any]:
    if not rewards:
        return {"average_total_reward": None, "scored_case_count": 0}
    component_names = sorted(rewards[0].components)
    return {
        "average_total_reward": _mean(reward.total_reward for reward in rewards),
        "average_raw_total_reward": _mean(reward.raw_total_reward for reward in rewards),
        "scored_case_count": len(rewards),
        "format_valid_rate": _mean(1.0 if reward.format_valid else 0.0 for reward in rewards),
        "component_average_scores": {
            name: _mean(reward.components[name].score for reward in rewards)
            for name in component_names
        },
    }


def _aggregate_usage(results: list[PlannerEvalResult], *, duration_ms: int) -> dict[str, Any]:
    planner_calls = sum(int(result.usage.get("planner_calls", 0)) for result in results)
    failed_count = sum(1 for result in results if result.errors)
    return {
        "planner_calls": planner_calls,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "duration_ms": duration_ms,
        "estimated_cost": 0.0,
        "currency": "CNY",
        "failed_case_count": failed_count,
    }


def _trajectory_usage(trajectory: OfflineTrajectoryResult) -> dict[str, Any]:
    return {
        "planner_calls": len(trajectory.trace_steps),
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "duration_ms": sum(step.duration_ms for step in trajectory.trace_steps),
        "estimated_cost": 0.0,
        "currency": "CNY",
        "trajectory_status": trajectory.status.value,
        "config_match_status": trajectory.config_match_status,
        "corpus_match_status": trajectory.corpus_match_status,
    }


def _metrics_from_reward(reward: TrajectoryReward) -> dict[str, float | int | bool | None]:
    retrieval_details = reward.components["retrieval"].details
    citation_details = reward.components["citation"].details
    answer_details = reward.components["answer"].details
    behavior_details = reward.components["behavior"].details
    return {
        "total_reward": reward.total_reward,
        "raw_total_reward": reward.raw_total_reward,
        "format_valid": reward.format_valid,
        "recall_at_k": retrieval_details.get("recall_at_k"),
        "mrr": retrieval_details.get("mrr"),
        "ndcg_at_k": retrieval_details.get("ndcg_at_k"),
        "citation_hit_rate": citation_details.get("citation_hit_rate"),
        "answer_point_coverage": answer_details.get("answer_point_coverage"),
        "path_match": behavior_details.get("path_match"),
    }


def _candidate_from_expected_chunk(
        *,
        case: PlannerEvalCase,
        document: SnapshotDocument,
        chunk_id: str | int,
        chunk_index: int,
        rank: int,
        retrieval_channel: RetrievalChannel,
) -> RetrievalCandidate:
    title = case.expected_subject_names[0] if case.expected_subject_names else case.query[:80]
    content = "；".join(case.expected_answer_points) or case.query
    return RetrievalCandidate(
        document_id=document.document_id,
        chunk_id=chunk_id,
        dataset_id=document.dataset_id,
        index_version=document.index_version,
        chunk_index=chunk_index,
        enabled=True,
        title=title,
        source_title=document.document_id,
        subject_id=case.expected_subject_ids[0] if case.expected_subject_ids else None,
        standard_subject_name=case.expected_subject_names[0] if case.expected_subject_names else None,
        content=content,
        equipment_model=_first_identifier(case, "equipment_model"),
        alarm_code=_first_identifier(case, "alarm_code"),
        part_name=_first_identifier(case, "part_name"),
        sop_type=_first_identifier(case, "sop_type"),
        safety_level=_first_identifier(case, "safety_level"),
        maintenance_stage=_first_identifier(case, "maintenance_stage"),
        source_type=EvidenceSourceType.LOCAL,
        retrieval_channels=[retrieval_channel],
        retrieval_rank=rank,
        retrieval_score=max(0.0, 0.95 - (rank - 1) * 0.03),
        rerank_score=max(0.0, 0.95 - (rank - 1) * 0.03),
    )


def _chunk_index(enabled_chunk_ids: list[str | int], chunk_id: str | int) -> int:
    for index, enabled_chunk_id in enumerate(enabled_chunk_ids):
        if str(enabled_chunk_id) == str(chunk_id):
            return index
    return 0


def _first_identifier(case: PlannerEvalCase, identifier_type: str) -> str | None:
    values = case.expected_identifiers.get(identifier_type, [])
    return values[0] if values else None


def _normalize_split(split: CaseSplit | str) -> CaseSplit:
    return split if isinstance(split, CaseSplit) else CaseSplit(str(split).strip())


def _mean(values: Iterable[float]) -> float:
    numbers = [float(value) for value in values]
    return sum(numbers) / len(numbers) if numbers else 0.0


def _elapsed_ms(start: float) -> int:
    return max(0, int((time.monotonic() - start) * 1000))


__all__ = [
    "BaselineEvalOutput",
    "BaselinePlannerSummary",
    "RUNNER_VERSION",
    "SNAPSHOT_EXPECTED_PROVIDER_NAME",
    "SnapshotExpectedChunkActionProvider",
    "load_environment_snapshot",
    "parse_planner_modes",
    "run_baseline_evaluation",
    "run_baseline_evaluation_from_files",
    "write_baseline_eval_output",
]
