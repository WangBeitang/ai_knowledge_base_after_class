"""
阶段 8.5 Reward v1.1。

Reward 的中文含义是“奖励/评分信号”。阶段 8 中它用于离线评测，阶段 9 中同一套
score_trajectory 会作为 GRPO 的训练奖励。v1.1 仍使用可执行、可复现的规则指标，但按
Planner 可控性重标定六路权重：行为分和成本分承担主要检索路线信号，检索分和引用分
保留为低权重终局质量约束，避免 chunk 排名这类间接指标主导 Planner 训练。
"""

from __future__ import annotations

from typing import Any, Callable
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.rag.evaluation.case_schema import PlannerEvalCase
from app.rag.evaluation.metrics import (
    action_values,
    answer_point_coverage,
    candidate_evidence_keys,
    expected_evidence_keys,
    identifier_hit_rate,
    matches_any_action_path,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    shortest_acceptable_path_length,
)
from app.rag.evaluation.offline_environment import OfflineTrajectoryResult, OfflineTrajectoryStatus
from app.rag.query.contracts import EvidenceSourceType, QueryAction
from app.rag.query.rrf_service import canonicalize_web_url


# 第一部分：版本和硬边界常量。score_trajectory 会先用这些常量判断轨迹是否可训练。
REWARD_VERSION = "reward-v1.1"  # Reward 版本必须写入结果文件；权重或规则变化时要升级。
TERMINAL_ACTIONS = {QueryAction.ANSWER, QueryAction.ASK_CLARIFICATION, QueryAction.REFUSE}  # 合法终态集合。
FORMAT_ERROR_CODES = {  # 以下错误表示 Planner 输出格式或 Action 执行路径不合法，会触发总分上限。
    "unknown_action",                 # Planner 输出了系统无法识别或未定义的 Action。
    "planner_output_invalid",         # Planner 输出内容不符合规定格式，无法被正确解析。
    "action_not_allowed",             # 当前 Action 不在系统允许执行的 Action 集合中。
    "illegal_action_transition",      # Action 的状态流转不合法，不允许从上一 Action 转到当前 Action。
    "duplicated_retrieval_action",    # 重复执行了相同的检索 Action，造成无效或冗余检索。
    "terminal_state_already_reached", # 已经到达终止状态后，仍然继续输出或执行 Action。
    "no_terminal_action",             # 整条 Action 路径中没有生成规定的终止 Action。
    "max_steps_exceeded",             # 执行步骤数超过系统允许的最大步数。
}


# 第二部分：Reward 输出 schema。先定义数据形状，再定义评分入口。
class RewardModel(BaseModel):
    """Reward schema 公共基类，拒绝未知字段，避免评测结果悄悄漂移。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)  # 修改字段时也触发 Pydantic 校验。


class RewardComponent(RewardModel):
    """一个可解释 Reward 分项，例如 format、retrieval、citation、answer、behavior、cost。"""

    name: str = Field(min_length=1, description="分项名称，必须和聚合权重中的 key 对齐。")
    score: float = Field(ge=0, le=1, description="0～1 原始分项分，尚未除以总权重。")
    weight: float = Field(ge=0, description="该分项参与总分的权重；允许为 0 以便临时关闭。")
    weighted_score: float = Field(ge=0, description="score * weight，聚合总分时直接相加。")
    details: dict[str, Any] = Field(default_factory=dict, description="机器可读明细，用于报告和排查。")
    reasons: list[str] = Field(default_factory=list, description="中文解释，说明扣分或特殊处理原因。")


class RewardWeights(RewardModel):
    """
    Reward v1.1 分项权重。

    Planner 的中文含义是“检索与终态规划器”。它直接决定 Action 路线、fallback（证据
    不足后的升级或终止选择）和回答时机，但不直接决定某个 chunk 的底层排名。因此 v1.1
    把 behavior + cost 作为主要检索路线分，把 retrieval + citation 降为低权重终局质量
    约束，避免阶段 9 训练过度追逐固定检索器产生的 chunk 排名。
    """

    format: float = Field(default=0.15, ge=0, description="结构化格式和 Action 合法性权重。")
    retrieval: float = Field(default=0.12, ge=0, description="expected chunk 召回、排序和标识命中权重；v1.1 中属于低权重终局质量约束。")
    citation: float = Field(default=0.08, ge=0, description="最终 Citation 是否引用期望/有效 chunk 的权重；v1.1 中不作为主训练信号。")
    answer: float = Field(default=0.15, ge=0, description="答案要点覆盖或拒答/追问终态正确性的权重。")
    behavior: float = Field(default=0.35, ge=0, description="是否该答、该拒、该追问、该 Web 的行为权重；v1.1 中是检索路线分的主体。")
    cost: float = Field(default=0.15, ge=0, description="不必要 HyDE/Web、额外步骤和耗时的成本权重；与 behavior 合起来约束检索路线。")

    def as_dict(self) -> dict[str, float]:
        """按 score 函数名称返回权重，score_trajectory 用它统一聚合。"""
        return {  # dict 的 key 与 components 的 key 保持一致，避免聚合时手写多套名字。
            "format": self.format,
            "retrieval": self.retrieval,
            "citation": self.citation,
            "answer": self.answer,
            "behavior": self.behavior,
            "cost": self.cost,
        }


class RewardConfig(RewardModel):
    """Reward v1.1 运行配置，集中保存版本、权重和硬上限。"""

    reward_version: str = Field(default=REWARD_VERSION, description="Reward 版本，写入每条评测结果。")
    weights: RewardWeights = Field(default_factory=RewardWeights, description="六个分项的聚合权重；v1.1 默认让 behavior + cost 高于 retrieval + citation。")
    retrieval_top_k: int = Field(default=10, ge=1, description="recall@k 和 nDCG@k 使用的 k。")
    invalid_format_cap: float = Field(default=0.20, ge=0, le=1, description="格式非法时总分上限。")
    failed_trajectory_cap: float = Field(default=0.30, ge=0, le=1, description="环境执行失败但非格式错误时总分上限。")
    unnecessary_hyde_penalty: float = Field(default=0.20, ge=0, le=1, description="不必要 HyDE 的成本扣分。")
    unnecessary_web_penalty: float = Field(default=0.30, ge=0, le=1, description="不必要 Web 的成本扣分。")
    extra_step_penalty: float = Field(default=0.08, ge=0, le=1, description="每个额外 Action 步骤的成本扣分。")

    @model_validator(mode="after")
    def validate_weights(self) -> "RewardConfig":
        """保证至少有一个分项参与总分，否则 GRPO 没有可优化目标。"""
        if sum(self.weights.as_dict().values()) <= 0:  # 权重全为 0 会导致除以 0，也没有训练意义。
            raise ValueError("Reward 权重总和必须大于 0")
        return self


class TrajectoryReward(RewardModel):
    """一条轨迹的最终 Reward 输出；total_reward 是 GRPO 真正使用的标量。"""

    reward_version: str = Field(description="本次评分使用的 Reward 版本。")
    total_reward: float = Field(ge=0, le=1, description="应用格式/失败上限后的最终奖励。")
    raw_total_reward: float = Field(ge=0, le=1, description="应用硬上限前的加权总分。")
    capped_by: str | None = Field(default=None, description="触发的总分上限原因，例如 invalid_format。")
    format_valid: bool = Field(description="Planner 输出和 Action 路径是否通过格式校验。")
    components: dict[str, RewardComponent] = Field(description="所有分项奖励，必须随总分一起保存。")
    errors: list[dict[str, Any]] = Field(default_factory=list, description="离线环境返回的结构化错误。")

    def to_json_dict(self) -> dict[str, Any]:
        """返回可直接写入 PlannerEvalResult.reward 的 JSON 字典。"""
        return self.model_dump(mode="json")  # mode=json 会把 Enum 等对象转成可序列化值。


# 第三部分：主入口。阶段 8 评测和阶段 9 GRPO 都应该只从 score_trajectory 进入。
ScoreFn = Callable[[PlannerEvalCase, OfflineTrajectoryResult, RewardConfig], RewardComponent]


def score_trajectory(
        case: PlannerEvalCase,
        trajectory: OfflineTrajectoryResult,
        config: RewardConfig | None = None,
) -> TrajectoryReward:
    """对一条 OfflineTrajectoryResult 计算 Reward v1.1。"""
    active_config = config or RewardConfig()  # 调用方不传配置时使用 v1.1 默认权重。
    score_functions: tuple[ScoreFn, ...] = (  # 按实际评估顺序排列，方便阅读和排查。
        score_format,
        score_retrieval,
        score_citation,
        score_answer,
        score_behavior,
        score_cost,
    )
    components = {  # 逐个执行分项评分，分项名称来自 RewardComponent.name。
        component.name: component
        for component in (fn(case, trajectory, active_config) for fn in score_functions)
    }
    raw_total = _weighted_average(components, active_config.weights.as_dict())  # 先算不带硬上限的加权分。
    format_valid = not _has_format_error(trajectory)  # 格式是否有效会决定是否触发 invalid_format_cap。
    total, capped_by = _apply_total_cap(raw_total, format_valid, trajectory, active_config)  # 应用总分硬上限。

    return TrajectoryReward(  # 输出对象必须能 JSON 序列化，并保留所有分项明细。
        reward_version=active_config.reward_version,
        total_reward=total,
        raw_total_reward=raw_total,
        capped_by=capped_by,
        format_valid=format_valid,
        components=components,
        errors=[error.model_dump(mode="json") for error in trajectory.errors],
    )


# 第四部分：分项 1，格式分。先检查格式，是为了尽早发现 GRPO 可能学会绕过 schema 的问题。
def score_format(case: PlannerEvalCase, trajectory: OfflineTrajectoryResult, config: RewardConfig) -> RewardComponent:
    """评分结构化格式和 Action 合法性，不判断答案内容。"""
    _ = case  # 格式分只依赖实际轨迹；保留 case 参数是为了所有 score 函数签名一致。
    score = 1.0  # 默认满分，再按错误扣分。
    reasons: list[str] = []  # 扣分原因写入报告，便于定位非法 Action。
    format_error_codes = _error_codes(trajectory, FORMAT_ERROR_CODES)  # 只抽取会触发格式上限的错误码。

    if trajectory.terminal_action not in TERMINAL_ACTIONS:  # 没有合法终态，说明轨迹不能作为训练正样本。
        score -= 0.35
        reasons.append("轨迹没有到达 answer/refuse/ask_clarification 终态")
    if format_error_codes:  # 出现非法 Action、解析失败、重复检索等问题。
        score -= 0.65
        reasons.append(f"存在格式或 Action 合法性错误：{format_error_codes}")
    if not trajectory.action_path:  # 空路径无法复盘，也无法计算行为和成本。
        score -= 0.20
        reasons.append("Action 路径为空，无法复盘 Planner 决策")

    return _component(  # 使用统一构造器，保证分数被 clamp 到 0～1。
        name="format",
        score=score,
        weight=config.weights.format,
        details={
            "terminal_action": _action_name(trajectory.terminal_action),
            "action_path": action_values(trajectory.action_path),
            "format_error_codes": format_error_codes,
        },
        reasons=reasons,
    )


# 第五部分：分项 2，检索分。回答型 case 才要求命中冻结的本地或 Web 证据。
def score_retrieval(case: PlannerEvalCase, trajectory: OfflineTrajectoryResult, config: RewardConfig) -> RewardComponent:
    """评分 expected evidence（期望证据）召回、排序质量和结构化标识命中。"""
    if not case.expected_behavior.should_answer:  # 拒答/追问样本没有证据命中要求。
        return _component(
            "retrieval",
            1.0,
            config.weights.retrieval,
            {"not_applicable": True},
            ["非回答型样本不要求命中 expected evidence"],
        )

    expected_keys = expected_evidence_keys(
        case.expected_chunks,
        case.expected_web_evidence,
    )
    retrieved_keys = candidate_evidence_keys(trajectory.retrieved_candidates)
    recall = recall_at_k(retrieved_keys, expected_keys, k=config.retrieval_top_k)
    mrr = reciprocal_rank(retrieved_keys, expected_keys)
    ndcg = ndcg_at_k(retrieved_keys, expected_keys, k=config.retrieval_top_k)
    identifier_rate, identifier_hits, identifier_misses = identifier_hit_rate(  # 设备型号/报警码命中情况。
        trajectory.retrieved_candidates,
        case.expected_identifiers,
    )
    score = 0.40 * recall + 0.25 * mrr + 0.25 * ndcg + 0.10 * identifier_rate  # 保持 v1 核心加权公式不变。
    reasons: list[str] = []  # 检索扣分原因。

    if recall < 1:  # 有 required/supporting chunk 没被找出来。
        reasons.append("未完全召回 expected_chunks/expected_web_evidence")
    if identifier_misses:  # 标识没命中通常意味着设备型号或报警码风险。
        reasons.append("部分设备型号、报警码或部件标识未命中")
    if trajectory.corpus_match_status != "match":  # 语料快照不一致时，检索结果不可完全信任。
        score = min(score, 0.40)
        reasons.append("轨迹语料与 environment snapshot 不匹配，检索分设置上限")

    return _component(
        name="retrieval",
        score=score,
        weight=config.weights.retrieval,
        details={
            "recall_at_k": recall,
            "mrr": mrr,
            "ndcg_at_k": ndcg,
            "identifier_hit_rate": identifier_rate,
            "identifier_hits": identifier_hits,
            "identifier_misses": identifier_misses,
            "expected_chunk_count": len(expected_keys),
            "expected_local_chunk_count": len(case.expected_chunks),
            "expected_web_evidence_count": len(case.expected_web_evidence),
            "retrieved_evidence_count": len(retrieved_keys),
            # 保留旧字段，兼容历史报告读取；Web case 中它表示统一证据数量。
            "retrieved_chunk_count": len(retrieved_keys),
            "top_k": config.retrieval_top_k,
        },
        reasons=reasons,
    )


# 第六部分：分项 3，引用分。答案不仅要检索对，还要引用对。
def score_citation(case: PlannerEvalCase, trajectory: OfflineTrajectoryResult, config: RewardConfig) -> RewardComponent:
    """评分最终 Citation 是否指向 expected chunk 或冻结的 Web URL。"""
    if not case.expected_behavior.should_answer:  # 非回答型样本通常不应产生引用。
        score = 1.0 if not trajectory.citations else 0.40
        reasons = [] if not trajectory.citations else ["非回答型样本不应生成最终引用"]
        return _component(
            "citation",
            score,
            config.weights.citation,
            {"citation_count": len(trajectory.citations), "not_applicable": True},
            reasons,
        )

    hit_rate, expected_count, citation_count, invalid_count = _citation_stats(case, trajectory)  # 统计引用命中和无效引用。
    score = min(hit_rate, 0.60) if invalid_count else hit_rate  # 出现无法映射版本的引用时设置上限。
    reasons: list[str] = []  # 引用扣分原因。
    if hit_rate < 1:
        reasons.append("最终 Citation 没有完全覆盖 expected evidence")
    if invalid_count:
        reasons.append("存在无法映射到本地 chunk/index_version 或冻结 Web URL 的引用")

    return _component(
        name="citation",
        score=score,
        weight=config.weights.citation,
        details={
            "citation_hit_rate": hit_rate,
            "expected_chunk_count": expected_count,
            "expected_local_chunk_count": len(case.expected_chunks),
            "expected_web_evidence_count": len(case.expected_web_evidence),
            "citation_chunk_count": citation_count,
            "invalid_citation_count": invalid_count,
        },
        reasons=reasons,
    )


# 第七部分：分项 4，答案分。拒答/追问样本只看终态，不看答案要点覆盖。
def score_answer(case: PlannerEvalCase, trajectory: OfflineTrajectoryResult, config: RewardConfig) -> RewardComponent:
    """评分最终交付文本或拒答/追问终态。"""
    if not case.expected_behavior.should_answer:  # 拒答和追问没有 expected_answer_points，不应套回答型评分。
        expected_terminal = _expected_terminal_action(case)
        score = 1.0 if trajectory.terminal_action == expected_terminal else 0.0
        return _component(
            name="answer",
            score=score,
            weight=config.weights.answer,
            details={"expected_terminal": expected_terminal.value, "not_applicable_answer_points": True},
            reasons=[] if score == 1.0 else ["非回答型样本的终态行为不符合 expected_behavior"],
        )

    coverage, hit_points, missing_points = answer_point_coverage(trajectory.answer, case.expected_answer_points)  # 文本要点覆盖率。
    score = coverage if trajectory.terminal_action == QueryAction.ANSWER else 0.0  # 没到 answer 终态时答案分归零。
    reasons = [] if not missing_points else ["答案缺少部分 expected_answer_points"]
    if trajectory.terminal_action != QueryAction.ANSWER:
        reasons.append("回答型样本没有到达 answer 终态")

    return _component(
        name="answer",
        score=score,
        weight=config.weights.answer,
        details={
            "answer_point_coverage": coverage,
            "hit_points": hit_points,
            "missing_points": missing_points,
            "answer_length": len(trajectory.answer),
        },
        reasons=reasons,
    )


# 第八部分：分项 5，行为分。判断路线是否符合人工标注的业务预期。
def score_behavior(case: PlannerEvalCase, trajectory: OfflineTrajectoryResult, config: RewardConfig) -> RewardComponent:
    """评分该答/拒/追问/Web 的行为是否符合 expected_behavior。"""
    actual_path = action_values(trajectory.action_path)  # 实际路径转成字符串，便于集合和报告处理。
    expected_terminal = _expected_terminal_action(case)  # 从 should_answer/refuse/ask 映射出期望终态。
    terminal_match = trajectory.terminal_action == expected_terminal  # 终态是否正确。
    path_match = matches_any_action_path(trajectory.action_path, case.acceptable_action_paths)  # 路径是否属于人工可接受路线。
    forbidden_used = sorted(  # 实际使用的禁用 Action。
        set(action_values(case.expected_behavior.forbidden_actions)).intersection(actual_path)
    )
    used_web = QueryAction.WEB_SEARCH.value in actual_path  # 是否使用 Web fallback。
    score = (  # 保持 v1 行为分公式。
        (0.60 if terminal_match else 0.0)
        + (0.20 if path_match else 0.0)
        + (0.10 if not forbidden_used else 0.0)
        + (0.10 if used_web == case.expected_behavior.should_call_web else 0.0)
    )
    reasons: list[str] = []  # 行为扣分原因。

    if not terminal_match:
        reasons.append(
            "终态行为不符合 expected_behavior："
            f"expected={expected_terminal.value}, actual={_action_name(trajectory.terminal_action)}"
        )
    if not path_match:
        reasons.append("实际 Action 路径不在 acceptable_action_paths 中")
    if forbidden_used:
        reasons.append(f"使用了 forbidden_actions：{forbidden_used}")
    if used_web != case.expected_behavior.should_call_web:
        reasons.append("Web fallback 使用状态与 expected_behavior.should_call_web 不一致")

    return _component(
        name="behavior",
        score=score,
        weight=config.weights.behavior,
        details={
            "expected_terminal": expected_terminal.value,
            "actual_terminal": _action_name(trajectory.terminal_action),
            "path_match": path_match,
            "forbidden_used": forbidden_used,
            "should_call_web": case.expected_behavior.should_call_web,
            "used_web": used_web,
        },
        reasons=reasons,
    )


# 第九部分：分项 6，成本分。防止模型为了保险而永远多走 HyDE/Web。
def score_cost(case: PlannerEvalCase, trajectory: OfflineTrajectoryResult, config: RewardConfig) -> RewardComponent:
    """评分不必要 HyDE/Web、额外步骤和耗时。"""
    actual_path = action_values(trajectory.action_path)  # 成本分只关心实际执行过哪些 Action。
    acceptable_values = [action_values(path) for path in case.acceptable_action_paths]  # 人工可接受路径转成字符串。
    shortest_len = shortest_acceptable_path_length(case.acceptable_action_paths)  # 最短可接受路径作为额外步骤基准。
    extra_steps = max(0, len(actual_path) - shortest_len) if shortest_len else 0  # 没有基准时不按步数扣分。
    score = 1.0  # 成本默认满分，再按不必要动作扣分。
    reasons: list[str] = []  # 成本扣分原因。

    if extra_steps:
        score -= min(0.40, extra_steps * config.extra_step_penalty)  # 额外步数扣分设置上限，避免成本分吞掉全部质量信号。
        reasons.append(f"Action 步数比最短可接受路径多 {extra_steps} 步")
    if QueryAction.HYDE_SEARCH.value in actual_path and not any(
        QueryAction.HYDE_SEARCH.value in path
        for path in acceptable_values
    ):
        score -= config.unnecessary_hyde_penalty
        reasons.append("使用了不必要的 HyDE 检索")
    if QueryAction.WEB_SEARCH.value in actual_path and not case.expected_behavior.should_call_web:
        score -= config.unnecessary_web_penalty
        reasons.append("使用了不必要的 Web 检索")

    return _component(
        name="cost",
        score=score,
        weight=config.weights.cost,
        details={
            "action_count": len(actual_path),
            "shortest_acceptable_path_length": shortest_len,
            "extra_steps": extra_steps,
            "used_hyde": QueryAction.HYDE_SEARCH.value in actual_path,
            "used_web": QueryAction.WEB_SEARCH.value in actual_path,
            "total_duration_ms": sum(step.duration_ms for step in trajectory.trace_steps),
        },
        reasons=reasons,
    )


# 第十部分：内部工具。放在最后，保持主入口和六个分项的阅读顺序连贯。
def _weighted_average(components: dict[str, RewardComponent], weights: dict[str, float]) -> float:
    """按权重聚合所有分项，返回应用硬上限前的 raw_total_reward。"""
    total_weight = sum(weights.values())  # 权重总和已在 RewardConfig 中保证大于 0。
    weighted_sum = sum(components[name].weighted_score for name in weights)  # 只聚合权重表声明的分项。
    return _clamp01(weighted_sum / total_weight)  # 归一化到 0～1，便于 GRPO 直接使用。


def _apply_total_cap(
        raw_total: float,
        format_valid: bool,
        trajectory: OfflineTrajectoryResult,
        config: RewardConfig,
) -> tuple[float, str | None]:
    """应用格式非法和环境失败的总分上限。"""
    if not format_valid:  # 格式非法优先级最高，防止模型通过非法输出拿高分。
        return min(raw_total, config.invalid_format_cap), "invalid_format"
    if trajectory.status == OfflineTrajectoryStatus.FAILED:  # 非格式错误的执行失败也不能拿满分。
        return min(raw_total, config.failed_trajectory_cap), "failed_trajectory"
    return raw_total, None  # 正常完成时不设置 capped_by。


def _expected_terminal_action(case: PlannerEvalCase) -> QueryAction:
    """把 expected_behavior 三选一终态映射成 QueryAction。"""
    if case.expected_behavior.should_answer:
        return QueryAction.ANSWER
    if case.expected_behavior.should_refuse:
        return QueryAction.REFUSE
    return QueryAction.ASK_CLARIFICATION


def _has_format_error(trajectory: OfflineTrajectoryResult) -> bool:
    """判断是否存在会触发 invalid_format_cap 的格式错误。"""
    return trajectory.terminal_action not in TERMINAL_ACTIONS or bool(_error_codes(trajectory, FORMAT_ERROR_CODES))


def _error_codes(trajectory: OfflineTrajectoryResult, selected_codes: set[str]) -> list[str]:
    """从 OfflineError 列表中抽取指定错误码。"""
    return [error.code for error in trajectory.errors if error.code in selected_codes]


def _citation_stats(case: PlannerEvalCase, trajectory: OfflineTrajectoryResult) -> tuple[float, int, int, int]:
    """统计 Citation 命中率、期望数量、有效引用数量和无效引用数量。"""
    expected_keys = set(
        expected_evidence_keys(case.expected_chunks, case.expected_web_evidence)
    )
    citation_keys: set[tuple[str, ...]] = set()
    invalid_count = 0

    for citation in trajectory.citations:
        if citation.source_type == EvidenceSourceType.WEB:
            parsed = urlsplit(citation.source)
            if (
                parsed.scheme.lower() not in {"http", "https"}
                or not parsed.netloc
            ):
                invalid_count += 1
            else:
                citation_keys.add(("web", canonicalize_web_url(citation.source)))
            continue
        if citation.document_id is None or citation.chunk_id is None:
            invalid_count += 1
            continue
        index_version = case.source_index_versions.get(citation.document_id)  # Citation 无 index_version，需要从 case 文档版本补齐。
        if index_version is None:
            invalid_count += 1
            continue
        citation_keys.add(
            (
                "local",
                citation.document_id,
                str(citation.chunk_id),
                str(int(index_version)),
            )
        )

    hit_count = sum(1 for key in expected_keys if key in citation_keys)  # expected chunk 被引用覆盖的数量。
    hit_rate = 1.0 if not expected_keys else hit_count / len(expected_keys)  # 没有 expected chunk 时不在这里扣分。
    return hit_rate, len(expected_keys), len(citation_keys), invalid_count


def _component(name: str, score: float, weight: float, details: dict[str, Any], reasons: list[str]) -> RewardComponent:
    """统一构造 RewardComponent，避免每个分项重复 clamp 和 weighted_score 逻辑。"""
    normalized_score = _clamp01(score)  # 扣分后可能小于 0，或者分项加分后超过 1，都要收敛。
    return RewardComponent(
        name=name,
        score=normalized_score,
        weight=weight,
        weighted_score=normalized_score * weight,
        details=details,
        reasons=reasons,
    )


def _action_name(action: QueryAction | None) -> str | None:
    """把可选 Action 转成可写 JSON 的字符串。"""
    return action.value if action else None


def _clamp01(value: float) -> float:
    """把任意浮点分数限制在 0～1。"""
    return max(0.0, min(1.0, float(value)))
