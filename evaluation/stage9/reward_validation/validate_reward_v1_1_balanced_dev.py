"""任务 9.3.15A：在 balanced dev 上独立验证冻结的 Reward v1.1。

本模块只运行固定 Action path（动作路线）和离线 Reward 评分，不加载 Planner 模型，
不运行 heldout test，也不写 Reward profile。验证前后会复核 v1.1 profile 与 Reward
实现文件的 SHA256，防止“边验证边调权”。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

from app.rag.evaluation.baseline_runner import load_environment_snapshot
from app.rag.evaluation.case_schema import PlannerEvalCase, load_planner_cases
from app.rag.evaluation.reward import REWARD_VERSION, RewardConfig, RewardWeights
from evaluation.stage9.model_planner.audit_eval_route_coverage import RouteBucket
from evaluation.stage9.reward_calibration.run_reward_calibration import (
    CalibrationPathResult,
    run_reward_calibration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
VALIDATION_VERSION = "stage9-reward-v1.1-balanced-dev-validation-v1"
FIXED_VALIDATED_AT = "2026-07-29T05:20:00+00:00"
FIXED_RUN_ID = "stage9_reward_v1_1_balanced_dev_validation_20260729"
ROUTE_SCORE_COMPONENTS = ("format", "behavior", "cost")
EVIDENCE_SCORE_COMPONENTS = ("retrieval", "citation")
# v1.1 已冻结；同名版本的 profile 或实现内容发生变化时必须停止，而不是更新这里后继续跑。
EXPECTED_PROFILE_SHA256 = "2d4fc0f92c51beaf655fed57d2f5d3b43abaa50eeb4117ddb66b694f901ccf4e"
EXPECTED_REWARD_IMPLEMENTATION_SHA256 = (
    "1413877cd57a5000c23516359467f67c6170fbce141128a8ed0a1b60a3ed1f72"
)

DEFAULT_CASES = PROJECT_ROOT / "evaluation/stage8/cases/planner_cases.jsonl"
DEFAULT_SNAPSHOT = (
    PROJECT_ROOT
    / "evaluation/stage9/artifacts/heldout_route_test/environment_snapshot.json"
)
DEFAULT_PROFILE = (
    PROJECT_ROOT / "evaluation/stage9/configs/reward_v1_1_training_profile.json"
)
DEFAULT_ROUTE_MATRIX = (
    PROJECT_ROOT / "evaluation/stage9/configs/planner_eval_route_matrix_v1.json"
)
DEFAULT_REWARD_IMPLEMENTATION = PROJECT_ROOT / "app/rag/evaluation/reward.py"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "evaluation/stage9/artifacts/reward/reward_v1_1_balanced_dev_validation.json"
)
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "evaluation/stage9/artifacts/reports/阶段9-Reward-v1.1-balanced-dev验证报告.md"
)


class ValidationModel(BaseModel):
    """9.3.15A 输出 schema（数据结构）公共基类；禁止未声明字段静默进入冻结产物。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class FrozenInput(ValidationModel):
    """一个验证输入的路径和内容身份；SHA256 用于证明本次使用的是哪一版文件。"""

    path: str = Field(min_length=1, description="相对项目根目录的输入路径。")
    sha256: str = Field(min_length=64, max_length=64, description="验证开始时计算的文件哈希。")


class TrajectoryValidation(ValidationModel):
    """单条固定 Action path 的 v1.1 评分和诊断投影。"""

    path_id: str = Field(min_length=1, description="动作路线套件中的稳定路线 ID。")
    action_path: list[str] = Field(min_length=1, description="实际离线执行的 Action 序列。")
    path_match: bool = Field(description="该路线是否属于 case 冻结的 acceptable_action_paths。")
    terminal_action: str | None = Field(description="离线轨迹最终终态；无终态时为空。")
    total_reward: float = Field(ge=0, le=1, description="未改动的 Reward v1.1 最终总分。")
    planner_route_score: float = Field(
        ge=0,
        le=1,
        description="仅重汇总 format/behavior/cost 的诊断分；不替代也不修改 total_reward。",
    )
    evidence_contract_score: float = Field(
        ge=0,
        le=1,
        description="仅重汇总 retrieval/citation 的离线契约分；不代表真实检索质量。",
    )
    answer_quality_score: float = Field(
        ge=0,
        le=1,
        description="原始 answer 分项；回答型 case 当前来自占位 answer executor。",
    )
    component_scores: dict[str, float] = Field(
        description="Reward v1.1 六个原始分项分，不改变权重或组件语义。"
    )
    anti_pattern_flags: list[str] = Field(
        default_factory=list,
        description="旧校准器识别出的乱检索、过早停止等反模式标记。",
    )


class CaseValidation(ValidationModel):
    """一个 reviewed dev case 的正负轨迹区分结果。"""

    case_id: str = Field(min_length=1)
    route_bucket: RouteBucket = Field(description="五类业务路线桶之一。")
    answer_execution_mode: str = Field(
        description="回答型 case 为 offline_placeholder；拒答/澄清为 terminal_only。"
    )
    correct_trajectory_count: int = Field(ge=1)
    incorrect_trajectory_count: int = Field(ge=1)
    correct_average_total_reward: float = Field(ge=0, le=1)
    incorrect_average_total_reward: float = Field(ge=0, le=1)
    minimum_correct_total_reward: float = Field(ge=0, le=1)
    hardest_negative_total_reward: float = Field(ge=0, le=1)
    minimum_total_margin: float = Field(
        description="最低正确轨迹分减最高错误轨迹分；大于 0 表示严格无反超。"
    )
    minimum_route_margin: float = Field(
        description="同一比较在诊断 planner_route_score 上的结果。"
    )
    inversion_count: int = Field(
        ge=0,
        description="错误轨迹分大于或等于任一正确轨迹分的配对数量。",
    )
    attribution: str = Field(
        description="本 case 的诊断归因；正常时为 no_issue。"
    )
    trajectories: list[TrajectoryValidation]


class BucketValidation(ValidationModel):
    """一个 route bucket 的正负轨迹汇总和 separation margin（区分间隔）。"""

    route_bucket: RouteBucket
    case_count: int = Field(ge=1)
    correct_trajectory_count: int = Field(ge=1)
    incorrect_trajectory_count: int = Field(ge=1)
    correct_average_total_reward: float = Field(ge=0, le=1)
    incorrect_average_total_reward: float = Field(ge=0, le=1)
    minimum_case_margin: float
    average_case_margin: float
    minimum_route_margin: float
    inversion_count: int = Field(ge=0)


class ValidationSummary(ValidationModel):
    """9.3.15A 的宏平均、硬门禁和人工讨论结论。"""

    decision: str = Field(
        description="只能是 pass_keep_v1_1 或三个需要暂停讨论/修复的门禁结论之一。"
    )
    decision_reasons: list[str]
    case_count: int = Field(ge=1)
    trajectory_count: int = Field(ge=1)
    route_bucket_count: int = Field(ge=1)
    macro_correct_average_total_reward: float = Field(ge=0, le=1)
    macro_incorrect_average_total_reward: float = Field(ge=0, le=1)
    macro_average_case_margin: float
    minimum_case_margin: float
    minimum_route_margin: float
    inversion_count: int = Field(ge=0)
    answer_placeholder_case_count: int = Field(ge=0)
    component_average_scores_correct: dict[str, float]
    component_average_scores_incorrect: dict[str, float]


class RewardV11BalancedDevValidation(ValidationModel):
    """9.3.15A 完整冻结产物；不包含训练结果、模型推理或 heldout 分数。"""

    validation_version: str = VALIDATION_VERSION
    validated_at: str
    run_id: str
    reward_version: str
    reward_profile_name: str
    selected_split: str = "dev"
    action_provider: str
    action_path_suite_version: str
    inputs: dict[str, FrozenInput]
    snapshot_id: str
    balanced_dev_case_ids: list[str] = Field(
        min_length=25,
        max_length=25,
        description="真正进入评分循环的 25 条 dev case；test/heldout case 不在此列表。",
    )
    balanced_dev_canonical_sha256: str = Field(
        min_length=64,
        max_length=64,
        description="25 条 dev case 规范化 JSON 内容哈希，不受同文件 test case 数量影响。",
    )
    non_dev_case_count_ignored: int = Field(
        ge=0,
        description="输入 registry 中未进入评分循环的 train/test case 数。",
    )
    profile_weights: dict[str, float]
    reward_config: dict[str, Any]
    profile_mutation_performed: bool = False
    reward_mutation_performed: bool = False
    model_execution_performed: bool = False
    heldout_inference_result_count: int = 0
    answer_executor_mode: str = "offline_placeholder_for_answer_cases"
    diagnostic_score_contract: dict[str, Any]
    cases: list[CaseValidation]
    buckets: list[BucketValidation]
    summary: ValidationSummary


def run_validation(
    *,
    cases_path: Path = DEFAULT_CASES,
    snapshot_path: Path = DEFAULT_SNAPSHOT,
    profile_path: Path = DEFAULT_PROFILE,
    route_matrix_path: Path = DEFAULT_ROUTE_MATRIX,
    reward_implementation_path: Path = DEFAULT_REWARD_IMPLEMENTATION,
    validated_at: str = FIXED_VALIDATED_AT,
    run_id: str = FIXED_RUN_ID,
) -> RewardV11BalancedDevValidation:
    """运行 9.3.15A；只读取输入和执行离线路线，不写 profile 或修改 Reward。"""

    input_paths = {
        "planner_cases": cases_path,
        "environment_snapshot": snapshot_path,
        "reward_profile_v1_1": profile_path,
        "route_matrix": route_matrix_path,
        "reward_implementation": reward_implementation_path,
    }
    for name, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"9.3.15A 输入不存在：{name}={path}")
    before_hashes = {name: _sha256(path) for name, path in input_paths.items()}
    _validate_frozen_v1_1_identity(before_hashes)

    profile = _read_json(profile_path)
    reward_config = _reward_config_from_profile(profile)
    matrix = _read_json(route_matrix_path)
    cases = load_planner_cases(cases_path)
    selected_cases = _select_balanced_dev(cases, matrix=matrix)
    snapshot = load_environment_snapshot(snapshot_path)
    _validate_snapshot_binding(
        snapshot_source_hashes=snapshot.source_hashes,
        cases_path=cases_path,
        cases_sha256=before_hashes["planner_cases"],
    )

    calibration = run_reward_calibration(
        cases=selected_cases,
        snapshot=snapshot,
        split="dev",
        case_path=_logical(cases_path),
        snapshot_path=_logical(snapshot_path),
        reward_config=reward_config,
        profile_name=str(profile["profile_name"]),
        run_id=run_id,
    )
    result_by_case: dict[str, list[CalibrationPathResult]] = defaultdict(list)
    for result in calibration.results:
        result_by_case[result.case_id].append(result)

    case_by_id = {case.case_id: case for case in selected_cases}
    case_validations = [
        _case_validation(case_by_id[case_id], result_by_case[case_id], reward_config)
        for case_id in sorted(result_by_case)
    ]
    bucket_validations = _bucket_validations(case_validations)
    summary = _summary(case_validations, bucket_validations)

    after_hashes = {name: _sha256(path) for name, path in input_paths.items()}
    if after_hashes != before_hashes:
        changed = sorted(
            name
            for name in before_hashes
            if before_hashes[name] != after_hashes[name]
        )
        raise RuntimeError(f"9.3.15A 运行期间输入发生变化：{changed}")

    return RewardV11BalancedDevValidation(
        validated_at=validated_at,
        run_id=run_id,
        reward_version=reward_config.reward_version,
        reward_profile_name=str(profile["profile_name"]),
        action_provider=calibration.action_provider,
        action_path_suite_version=calibration.path_suite_version,
        inputs={
            name: FrozenInput(path=_logical(path), sha256=before_hashes[name])
            for name, path in input_paths.items()
        },
        snapshot_id=snapshot.snapshot_id,
        balanced_dev_case_ids=[case.case_id for case in selected_cases],
        balanced_dev_canonical_sha256=_canonical_cases_sha256(selected_cases),
        non_dev_case_count_ignored=len(cases) - len(selected_cases),
        profile_weights=reward_config.weights.as_dict(),
        reward_config=reward_config.model_dump(mode="json"),
        diagnostic_score_contract={
            "planner_route_score_components": list(ROUTE_SCORE_COMPONENTS),
            "evidence_contract_score_components": list(EVIDENCE_SCORE_COMPONENTS),
            "answer_quality_score_component": "answer",
            "aggregation": "weighted_average_within_selected_existing_v1_1_components",
            "changes_total_reward": False,
            "evidence_boundary": (
                "snapshot_expected_chunks 只证明离线 State/Reward/Citation 契约，"
                "不证明真实 Milvus 或 Web 召回质量。"
            ),
        },
        cases=case_validations,
        buckets=bucket_validations,
        summary=summary,
    )


def _select_balanced_dev(
    cases: Iterable[PlannerEvalCase],
    *,
    matrix: dict[str, Any],
) -> list[PlannerEvalCase]:
    """冻结 25 条 reviewed dev，并验证五路线各 5 条；test case 不进入评分循环。"""

    selected = [case for case in cases if case.split.value == "dev"]
    if len(selected) != 25:
        raise ValueError(f"9.3.15A 要求恰好 25 条 dev，当前为 {len(selected)}")
    pending = [
        case.case_id
        for case in selected
        if case.human_review_status.value != "reviewed"
    ]
    if pending:
        raise ValueError(f"balanced dev 含未 reviewed case：{pending}")
    counts = Counter(_route_bucket(case) for case in selected)
    minimum = int(
        matrix["evaluation_sets"]["balanced_dev"]["minimum_reviewed_cases_per_bucket"]
    )
    expected = {bucket: minimum for bucket in RouteBucket}
    if counts != expected:
        raise ValueError(
            "balanced dev 路线桶分布不符合冻结矩阵："
            f"actual={dict(counts)}, expected={expected}"
        )
    return selected


def _route_bucket(case: PlannerEvalCase) -> RouteBucket:
    """按冻结 expected_behavior 和 acceptable_action_paths 映射五路线桶。"""

    behavior = case.expected_behavior
    if behavior.should_call_web:
        return RouteBucket.WEB_REQUIRED
    if behavior.should_ask_clarification:
        return RouteBucket.ASK_CLARIFICATION
    if behavior.should_refuse:
        return RouteBucket.SAFE_REFUSE
    if case.acceptable_action_paths and all(
        any(action.value == "hyde_search" for action in path)
        for path in case.acceptable_action_paths
    ):
        return RouteBucket.HYDE_FALLBACK
    return RouteBucket.LOCAL_ANSWER


def _reward_config_from_profile(profile: dict[str, Any]) -> RewardConfig:
    """以冻结 profile 权重创建 v1.1 配置；不把任何运行结果反写到 profile。"""

    if profile.get("reward_version") != REWARD_VERSION:
        raise ValueError(
            f"9.3.15A 只允许 {REWARD_VERSION}，当前为 {profile.get('reward_version')}"
        )
    if profile.get("decision") != "frozen":
        raise ValueError("Reward v1.1 profile 尚未 frozen，不能进入 9.3.15A")
    weights = RewardWeights.model_validate(profile.get("weights"))
    return RewardConfig(reward_version=REWARD_VERSION, weights=weights)


def _validate_frozen_v1_1_identity(input_hashes: dict[str, str]) -> None:
    """拒绝同名 v1.1 profile 或实现被静默改写；发生变化应新建 Reward 版本。"""

    if input_hashes["reward_profile_v1_1"] != EXPECTED_PROFILE_SHA256:
        raise ValueError(
            "Reward v1.1 profile SHA256 已变化；禁止以同名版本继续 9.3.15A"
        )
    if (
        input_hashes["reward_implementation"]
        != EXPECTED_REWARD_IMPLEMENTATION_SHA256
    ):
        raise ValueError(
            "Reward v1.1 实现 SHA256 已变化；必须先审查并升级版本，不能静默验证"
        )


def _validate_snapshot_binding(
    *,
    snapshot_source_hashes: dict[str, str],
    cases_path: Path,
    cases_sha256: str,
) -> None:
    """确认 snapshot 绑定当前 case registry；避免用旧 case 和新标签混跑。"""

    logical_path = _logical(cases_path)
    if snapshot_source_hashes.get(logical_path) != cases_sha256:
        raise ValueError(
            "EnvironmentSnapshot 未绑定当前 planner_cases SHA256："
            f"path={logical_path}"
        )


def _case_validation(
    case: PlannerEvalCase,
    results: list[CalibrationPathResult],
    reward_config: RewardConfig,
) -> CaseValidation:
    """将旧校准器结果投影成 9.3.15A 的正负轨迹 margin 和 answer 边界。"""

    trajectories = [
        _trajectory_validation(result, reward_config)
        for result in sorted(results, key=lambda item: item.path_id)
    ]
    correct = [row for row in trajectories if row.path_match]
    incorrect = [row for row in trajectories if not row.path_match]
    if not correct or not incorrect:
        raise ValueError(f"case 缺少正轨迹或负轨迹：{case.case_id}")

    minimum_correct = min(row.total_reward for row in correct)
    hardest_negative = max(row.total_reward for row in incorrect)
    minimum_route_correct = min(row.planner_route_score for row in correct)
    hardest_route_negative = max(row.planner_route_score for row in incorrect)
    inversion_count = sum(
        1
        for negative in incorrect
        for positive in correct
        if negative.total_reward >= positive.total_reward
    )
    total_margin = minimum_correct - hardest_negative
    route_margin = minimum_route_correct - hardest_route_negative
    attribution = _case_attribution(
        total_margin=total_margin,
        route_margin=route_margin,
        answer_placeholder=case.expected_behavior.should_answer,
    )
    return CaseValidation(
        case_id=case.case_id,
        route_bucket=_route_bucket(case),
        answer_execution_mode=(
            "offline_placeholder"
            if case.expected_behavior.should_answer
            else "terminal_only"
        ),
        correct_trajectory_count=len(correct),
        incorrect_trajectory_count=len(incorrect),
        correct_average_total_reward=mean(row.total_reward for row in correct),
        incorrect_average_total_reward=mean(row.total_reward for row in incorrect),
        minimum_correct_total_reward=minimum_correct,
        hardest_negative_total_reward=hardest_negative,
        minimum_total_margin=total_margin,
        minimum_route_margin=route_margin,
        inversion_count=inversion_count,
        attribution=attribution,
        trajectories=trajectories,
    )


def _trajectory_validation(
    result: CalibrationPathResult,
    reward_config: RewardConfig,
) -> TrajectoryValidation:
    components = result.reward["components"]
    return TrajectoryValidation(
        path_id=result.path_id,
        action_path=[action.value for action in result.action_path],
        path_match=result.path_match,
        terminal_action=(
            result.terminal_action.value if result.terminal_action else None
        ),
        total_reward=float(result.reward["total_reward"]),
        planner_route_score=_selected_component_score(
            components,
            reward_config,
            ROUTE_SCORE_COMPONENTS,
        ),
        evidence_contract_score=_selected_component_score(
            components,
            reward_config,
            EVIDENCE_SCORE_COMPONENTS,
        ),
        answer_quality_score=float(components["answer"]["score"]),
        component_scores=dict(result.component_scores),
        anti_pattern_flags=list(result.anti_pattern_flags),
    )


def _selected_component_score(
    components: dict[str, dict[str, Any]],
    reward_config: RewardConfig,
    names: tuple[str, ...],
) -> float:
    """只对既有 v1.1 分项做诊断重汇总；不会进入 total_reward 或写回 Reward。"""

    weights = reward_config.weights.as_dict()
    total_weight = sum(weights[name] for name in names)
    return sum(float(components[name]["score"]) * weights[name] for name in names) / total_weight


def _case_attribution(
    *,
    total_margin: float,
    route_margin: float,
    answer_placeholder: bool,
) -> str:
    if total_margin > 0 and route_margin > 0:
        return "no_issue"
    if route_margin > 0 and answer_placeholder:
        return "evaluator_limitation"
    return "reward_misranking"


def _bucket_validations(cases: list[CaseValidation]) -> list[BucketValidation]:
    output: list[BucketValidation] = []
    for bucket in RouteBucket:
        bucket_cases = [case for case in cases if case.route_bucket == bucket]
        correct = [
            trajectory
            for case in bucket_cases
            for trajectory in case.trajectories
            if trajectory.path_match
        ]
        incorrect = [
            trajectory
            for case in bucket_cases
            for trajectory in case.trajectories
            if not trajectory.path_match
        ]
        output.append(
            BucketValidation(
                route_bucket=bucket,
                case_count=len(bucket_cases),
                correct_trajectory_count=len(correct),
                incorrect_trajectory_count=len(incorrect),
                correct_average_total_reward=mean(
                    row.total_reward for row in correct
                ),
                incorrect_average_total_reward=mean(
                    row.total_reward for row in incorrect
                ),
                minimum_case_margin=min(
                    case.minimum_total_margin for case in bucket_cases
                ),
                average_case_margin=mean(
                    case.minimum_total_margin for case in bucket_cases
                ),
                minimum_route_margin=min(
                    case.minimum_route_margin for case in bucket_cases
                ),
                inversion_count=sum(
                    case.inversion_count for case in bucket_cases
                ),
            )
        )
    return output


def _summary(
    cases: list[CaseValidation],
    buckets: list[BucketValidation],
) -> ValidationSummary:
    all_trajectories = [
        trajectory for case in cases for trajectory in case.trajectories
    ]
    correct = [row for row in all_trajectories if row.path_match]
    incorrect = [row for row in all_trajectories if not row.path_match]
    component_names = sorted(correct[0].component_scores)
    attributions = Counter(case.attribution for case in cases)
    if attributions.get("reward_misranking"):
        decision = "reward_change_needs_discussion"
        reasons = [
            f"{attributions['reward_misranking']} 个 case 出现 Reward 或路线诊断错误排序；"
            "必须暂停讨论，不能自动修改 v1.1。"
        ]
    elif attributions.get("evaluator_limitation"):
        decision = "evaluator_limitation_needs_discussion"
        reasons = [
            f"{attributions['evaluator_limitation']} 个回答型 case 的 total_reward 排序受占位 "
            "answer 限制，但 planner_route_score 仍正确；必须暂停讨论。"
        ]
    else:
        decision = "pass_keep_v1_1"
        reasons = [
            "25 条 reviewed balanced dev 的最低正确轨迹均严格高于最高错误轨迹。",
            "五个路线桶均无错误反超，保留 Reward v1.1；不需要执行 9.3.15B。",
        ]
    return ValidationSummary(
        decision=decision,
        decision_reasons=reasons,
        case_count=len(cases),
        trajectory_count=len(all_trajectories),
        route_bucket_count=len(buckets),
        macro_correct_average_total_reward=mean(
            bucket.correct_average_total_reward for bucket in buckets
        ),
        macro_incorrect_average_total_reward=mean(
            bucket.incorrect_average_total_reward for bucket in buckets
        ),
        macro_average_case_margin=mean(
            bucket.average_case_margin for bucket in buckets
        ),
        minimum_case_margin=min(case.minimum_total_margin for case in cases),
        minimum_route_margin=min(case.minimum_route_margin for case in cases),
        inversion_count=sum(case.inversion_count for case in cases),
        answer_placeholder_case_count=sum(
            case.answer_execution_mode == "offline_placeholder"
            for case in cases
        ),
        component_average_scores_correct={
            name: mean(row.component_scores[name] for row in correct)
            for name in component_names
        },
        component_average_scores_incorrect={
            name: mean(row.component_scores[name] for row in incorrect)
            for name in component_names
        },
    )


def render_report(output: RewardV11BalancedDevValidation) -> str:
    """将机器可读验证结果渲染为人工讨论用 Markdown，不重新计算结论。"""

    lines = [
        "# 阶段 9 Reward v1.1 balanced dev 独立验证报告",
        "",
        "## 结论",
        "",
        f"- 9.3.15A 决定：`{output.summary.decision}`。",
        f"- Reward 版本：`{output.reward_version}`；profile：`{output.reward_profile_name}`。",
        f"- balanced dev：{output.summary.case_count} 条；固定 Action 轨迹："
        f"{output.summary.trajectory_count} 条。",
        f"- 最小 total Reward margin：`{output.summary.minimum_case_margin:.4f}`；"
        f"最小 Planner route margin：`{output.summary.minimum_route_margin:.4f}`。",
        f"- 错误反超数量：`{output.summary.inversion_count}`。",
        "",
        *[f"- {reason}" for reason in output.summary.decision_reasons],
        "",
        "## 不可变边界",
        "",
        f"- Reward profile mutation（配置修改）：`{str(output.profile_mutation_performed).lower()}`。",
        "- Reward implementation mutation（实现修改）："
        f"`{str(output.reward_mutation_performed).lower()}`。",
        f"- 模型执行：`{str(output.model_execution_performed).lower()}`。",
        f"- heldout 推理结果数：`{output.heldout_inference_result_count}`。",
        f"- ActionProvider：`{output.action_provider}`；只证明离线契约，不证明真实检索。",
        f"- 实际评分范围：25 条 dev（SHA256 `{output.balanced_dev_canonical_sha256}`）；"
        f"同 registry 中其余 {output.non_dev_case_count_ignored} 条非 dev case 未进入评分循环。",
        "- 回答型 case 使用占位 answer executor；原始 answer 分仍保留在 v1.1 总分中，"
        "同时单独报告 Planner route score，未调整权重。",
        "",
        "| 输入 | 路径 | SHA256 |",
        "|---|---|---|",
    ]
    for name, record in output.inputs.items():
        lines.append(f"| `{name}` | `{record.path}` | `{record.sha256}` |")
    lines.extend(
        [
            "",
            "## 五路线桶 separation margin",
            "",
            "| route bucket | case | 正轨迹 | 负轨迹 | 正轨迹均分 | 负轨迹均分 | 最小 margin | route margin | 反超 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for bucket in output.buckets:
        lines.append(
            f"| `{bucket.route_bucket.value}` | {bucket.case_count} | "
            f"{bucket.correct_trajectory_count} | {bucket.incorrect_trajectory_count} | "
            f"{bucket.correct_average_total_reward:.4f} | "
            f"{bucket.incorrect_average_total_reward:.4f} | "
            f"{bucket.minimum_case_margin:.4f} | "
            f"{bucket.minimum_route_margin:.4f} | {bucket.inversion_count} |"
        )
    lines.extend(
        [
            "",
            "## Reward component（奖励分项）均值",
            "",
            "| component | 正确轨迹 | 错误轨迹 |",
            "|---|---:|---:|",
        ]
    )
    for name in sorted(output.summary.component_average_scores_correct):
        lines.append(
            f"| `{name}` | "
            f"{output.summary.component_average_scores_correct[name]:.4f} | "
            f"{output.summary.component_average_scores_incorrect[name]:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 逐 case 最难负样本",
            "",
            "| case_id | route bucket | 最低正轨迹 | 最高负轨迹 | margin | route margin | attribution |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for case in output.cases:
        lines.append(
            f"| `{case.case_id}` | `{case.route_bucket.value}` | "
            f"{case.minimum_correct_total_reward:.4f} | "
            f"{case.hardest_negative_total_reward:.4f} | "
            f"{case.minimum_total_margin:.4f} | "
            f"{case.minimum_route_margin:.4f} | `{case.attribution}` |"
        )
    lines.extend(
        [
            "",
            "## Answer 与证据边界",
            "",
            f"- 占位 answer case：{output.summary.answer_placeholder_case_count} 条。",
            "- `planner_route_score` 只重汇总 v1.1 已有的 format、behavior、cost 分项，"
            "仅用于诊断，不进入 Reward 总分。",
            "- `evidence_contract_score` 只重汇总 retrieval、citation；由于 provider 为"
            " `snapshot_expected_chunks`，不能解释为真实 Milvus/Web 质量。",
            "- 9.3.15A 没有写入或重建 Reward profile，也没有生成 v1.2。",
            "",
            "## 下一步门禁",
            "",
            (
                "- 当前结论为 `pass_keep_v1_1`：保留 v1.1，跳过 9.3.15B；"
                "在人工确认本报告后才进入 9.3.16。"
                if output.summary.decision == "pass_keep_v1_1"
                else "- 当前结论需要人工讨论；未获得用户明确授权，不得执行 9.3.15B 或进入 9.3.16。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    output: RewardV11BalancedDevValidation,
    *,
    output_path: Path = DEFAULT_OUTPUT,
    report_path: Path = DEFAULT_REPORT,
    overwrite: bool = False,
) -> dict[str, Any]:
    """原子边界内写 JSON 与报告；默认拒绝覆盖已有冻结产物。"""

    for path in (output_path, report_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"9.3.15A 输出已存在，拒绝静默覆盖：{path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    json_text = (
        json.dumps(output.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n"
    )
    report_text = render_report(output)
    output_path.write_text(json_text, encoding="utf-8")
    report_path.write_text(report_text, encoding="utf-8")
    return {
        "output": _logical(output_path),
        "output_sha256": hashlib.sha256(json_text.encode("utf-8")).hexdigest(),
        "report": _logical(report_path),
        "report_sha256": hashlib.sha256(report_text.encode("utf-8")).hexdigest(),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_cases_sha256(cases: Iterable[PlannerEvalCase]) -> str:
    """只哈希实际评分的 dev case，避免同 registry 中 heldout 状态影响 A 的数据身份。"""

    payload = [
        case.model_dump(mode="json")
        for case in sorted(cases, key=lambda item: item.case_id)
    ]
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _logical(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--route-matrix", type=Path, default=DEFAULT_ROUTE_MATRIX)
    parser.add_argument("--reward-implementation", type=Path, default=DEFAULT_REWARD_IMPLEMENTATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output = run_validation(
        cases_path=args.cases,
        snapshot_path=args.snapshot,
        profile_path=args.profile,
        route_matrix_path=args.route_matrix,
        reward_implementation_path=args.reward_implementation,
    )
    files = write_outputs(
        output,
        output_path=args.output,
        report_path=args.report,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "decision": output.summary.decision,
                "case_count": output.summary.case_count,
                "trajectory_count": output.summary.trajectory_count,
                "minimum_case_margin": output.summary.minimum_case_margin,
                "minimum_route_margin": output.summary.minimum_route_margin,
                "inversion_count": output.summary.inversion_count,
                **files,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
