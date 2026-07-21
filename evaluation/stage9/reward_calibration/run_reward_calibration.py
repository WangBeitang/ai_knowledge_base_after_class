"""执行阶段 9 Reward v1.1 dev 多轨迹校准。

本脚本不训练模型，也不调用真实 Planner。它在同一批 dev case 和同一个
EnvironmentSnapshot 下，对每个 case 执行多条固定 Action path，然后用 Reward v1.1
打分，检查 Reward 是否会错误鼓励乱 HyDE、乱 Web、过早拒答或过早回答。
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pvariance
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.evaluation.baseline_runner import (  # noqa: E402
    SNAPSHOT_EXPECTED_PROVIDER_NAME,
    SnapshotExpectedChunkActionProvider,
    load_environment_snapshot,
)
from app.rag.evaluation.case_schema import (  # noqa: E402
    CaseSplit,
    EnvironmentSnapshot,
    PlannerEvalCase,
    load_planner_cases,
)
from app.rag.evaluation.metrics import action_values, matches_any_action_path  # noqa: E402
from app.rag.evaluation.offline_environment import OfflineRagEnvironment  # noqa: E402
from app.rag.evaluation.reward import RewardConfig, RewardWeights, score_trajectory  # noqa: E402
from app.rag.query.contracts import QueryAction  # noqa: E402
from evaluation.stage9.reward_calibration.action_path_suite import (  # noqa: E402
    ACTION_PATH_SUITE_VERSION,
    CalibrationPathSpec,
    build_action_path_suite,
)


CALIBRATION_RUNNER_VERSION = "stage9-reward-calibration-runner-v1"
DEFAULT_PROFILE_NAME = "planner-training-v1"
CRITICAL_ANTI_PATTERN_FLAGS = {
    "unnecessary_hyde_wins",
    "unnecessary_web_wins",
    "premature_refuse_wins",
    "premature_answer_wins",
}


class RewardCalibrationModel(BaseModel):
    """Reward 校准输出 schema 公共基类；拒绝未知字段，保证产物可长期读取。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


class CalibrationPathResult(RewardCalibrationModel):
    """
    单个 case 的单条 Action 路线执行与评分结果。

    route_rank 是同一 case 内按 total_reward 从高到低排序后的名次；anti_pattern_flags
    记录这条路线是否触发“乱搜索高分”等反模式，供冻结 profile 时做硬判断。
    """

    case_id: str = Field(min_length=1)
    path_id: str = Field(min_length=1)
    route_family: str = Field(min_length=1)
    action_path: list[QueryAction] = Field(min_length=1)
    terminal_action: QueryAction | None = None
    terminal_reason_code: str = ""
    status: str = Field(min_length=1)
    route_rank: int = Field(ge=1)
    path_match: bool
    reward: dict[str, Any]
    component_scores: dict[str, float]
    errors: list[dict[str, Any]] = Field(default_factory=list)
    anti_pattern_flags: list[str] = Field(default_factory=list)


class CalibrationSummary(RewardCalibrationModel):
    """整次 Reward 校准的聚合摘要。"""

    component_average_scores: dict[str, float] = Field(default_factory=dict)
    component_variance: dict[str, float] = Field(default_factory=dict)
    case_count: int = Field(ge=0)
    path_count: int = Field(ge=0)
    min_paths_per_case: int = Field(ge=0)
    max_paths_per_case: int = Field(ge=0)
    anti_pattern_counts: dict[str, int] = Field(default_factory=dict)
    route_ordering_violations: list[dict[str, Any]] = Field(default_factory=list)
    unnecessary_search_wins: list[dict[str, Any]] = Field(default_factory=list)
    premature_refusal_wins: list[dict[str, Any]] = Field(default_factory=list)
    premature_answer_wins: list[dict[str, Any]] = Field(default_factory=list)
    no_component_variance_cases: list[str] = Field(default_factory=list)
    freeze_decision: str = Field(min_length=1)
    freeze_reasons: list[str] = Field(default_factory=list)


class RewardTrainingProfile(RewardCalibrationModel):
    """
    阶段 9 训练用 Reward profile。

    profile 的中文含义是“配置档案”。它把 Reward 版本、六路权重、来源 dev case、
    snapshot 和冻结结论固化成机器可读文件，后续 SFT/GRPO 配置只能引用该文件，不能
    在训练脚本里临时改权重。
    """

    profile_name: str = Field(min_length=1)
    reward_version: str = Field(min_length=1)
    weights: dict[str, float]
    frozen_at: str = Field(min_length=1)
    source_dev_cases: str = Field(min_length=1)
    snapshot_id: str = Field(min_length=1)
    snapshot_path: str = Field(min_length=1)
    action_path_suite_version: str = Field(min_length=1)
    calibration_run_id: str = Field(min_length=1)
    runner_version: str = CALIBRATION_RUNNER_VERSION
    decision: str = Field(min_length=1)
    decision_reasons: list[str] = Field(default_factory=list)
    note: str = ""

    @model_validator(mode="after")
    def validate_profile_weights(self) -> "RewardTrainingProfile":
        if sum(self.weights.values()) <= 0:
            raise ValueError("Reward profile 权重总和必须大于 0")
        return self


class RewardCalibrationOutput(RewardCalibrationModel):
    """Reward v1.1 dev 多轨迹校准完整输出。"""

    run_id: str = Field(min_length=1)
    runner_version: str = CALIBRATION_RUNNER_VERSION
    created_at: str = Field(min_length=1)
    reward_version: str = Field(min_length=1)
    reward_profile: str = Field(min_length=1)
    snapshot_id: str = Field(min_length=1)
    source_case_path: str = Field(min_length=1)
    snapshot_path: str = Field(min_length=1)
    split: CaseSplit
    action_provider: str = Field(min_length=1)
    path_suite_version: str = ACTION_PATH_SUITE_VERSION
    case_count: int = Field(ge=0)
    path_count: int = Field(ge=0)
    results: list[CalibrationPathResult]
    summary: CalibrationSummary
    training_profile: RewardTrainingProfile

    def to_json_dict(self) -> dict[str, Any]:
        """返回可直接写入 UTF-8 JSON 文件的 dict。"""
        return self.model_dump(mode="json")


def run_reward_calibration(
        *,
        cases: list[PlannerEvalCase],
        snapshot: EnvironmentSnapshot,
        split: CaseSplit | str,
        case_path: str | Path,
        snapshot_path: str | Path,
        reward_config: RewardConfig | None = None,
        profile_name: str = DEFAULT_PROFILE_NAME,
        run_id: str | None = None,
) -> RewardCalibrationOutput:
    """执行 dev 多轨迹 Reward 校准并返回结构化结果。"""
    active_split = CaseSplit(split)
    selected_cases = [case for case in cases if case.split == active_split]
    if not selected_cases:
        raise ValueError(f"没有 split={active_split.value} 的校准 case")

    active_reward_config = reward_config or RewardConfig()
    normalized_run_id = run_id or f"stage9_reward_calibration_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    provider = SnapshotExpectedChunkActionProvider(selected_cases, snapshot)
    environment = OfflineRagEnvironment(
        snapshot=snapshot,
        action_provider=provider,
        planner_mode="reward_calibration",
        run_id_prefix=normalized_run_id,
    )

    raw_results_by_case: dict[str, list[CalibrationPathResult]] = {}
    for case in selected_cases:
        case_results: list[CalibrationPathResult] = []
        for path_spec in build_action_path_suite(case):
            trajectory = environment.run_action_path(
                case,
                list(path_spec.action_path),
                run_id=f"{normalized_run_id}_{case.case_id}_{path_spec.path_id}",
                planner_mode="reward_calibration",
            )
            reward = score_trajectory(case, trajectory, active_reward_config)
            case_results.append(_path_result(
                case=case,
                path_spec=path_spec,
                trajectory_status=trajectory.status.value,
                terminal_action=trajectory.terminal_action,
                terminal_reason_code=(
                    trajectory.terminal_reason_code.value
                    if trajectory.terminal_reason_code is not None
                    else ""
                ),
                reward=reward.to_json_dict(),
                errors=[error.model_dump(mode="json") for error in trajectory.errors],
            ))
        raw_results_by_case[case.case_id] = _rank_and_flag_case_results(case, case_results)

    results = [
        result
        for case_id in sorted(raw_results_by_case)
        for result in raw_results_by_case[case_id]
    ]
    summary = _build_summary(raw_results_by_case)
    created_at = datetime.now(UTC).isoformat(timespec="seconds")
    profile = RewardTrainingProfile(
        profile_name=profile_name,
        reward_version=active_reward_config.reward_version,
        weights=active_reward_config.weights.as_dict(),
        frozen_at=created_at,
        source_dev_cases=str(case_path),
        snapshot_id=snapshot.snapshot_id,
        snapshot_path=str(snapshot_path),
        action_path_suite_version=ACTION_PATH_SUITE_VERSION,
        calibration_run_id=normalized_run_id,
        decision=summary.freeze_decision,
        decision_reasons=summary.freeze_reasons,
        note=(
            "基于当前 dev 多轨迹校准自动冻结；"
            "如果后续新增独立 dev Gold，应重新生成 profile。"
            if summary.freeze_decision == "frozen"
            else "发现关键 Reward 反模式，禁止直接进入正式 SFT/GRPO。"
        ),
    )
    return RewardCalibrationOutput(
        run_id=normalized_run_id,
        created_at=created_at,
        reward_version=active_reward_config.reward_version,
        reward_profile=profile_name,
        snapshot_id=snapshot.snapshot_id,
        source_case_path=str(case_path),
        snapshot_path=str(snapshot_path),
        split=active_split,
        action_provider=SNAPSHOT_EXPECTED_PROVIDER_NAME,
        case_count=len(selected_cases),
        path_count=len(results),
        results=results,
        summary=summary,
        training_profile=profile,
    )


def write_reward_calibration_output(output: RewardCalibrationOutput, path: str | Path) -> None:
    """写入校准 JSON。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output.to_json_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_reward_training_profile(profile: RewardTrainingProfile, path: str | Path) -> None:
    """写入训练用 Reward profile。"""
    profile_path = Path(path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(
        json.dumps(profile.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _path_result(
        *,
        case: PlannerEvalCase,
        path_spec: CalibrationPathSpec,
        trajectory_status: str,
        terminal_action: QueryAction | None,
        terminal_reason_code: str,
        reward: dict[str, Any],
        errors: list[dict[str, Any]],
) -> CalibrationPathResult:
    components = reward["components"]
    component_scores = {
        name: float(component["score"])
        for name, component in components.items()
    }
    return CalibrationPathResult(
        case_id=case.case_id,
        path_id=path_spec.path_id,
        route_family=path_spec.route_family,
        action_path=list(path_spec.action_path),
        terminal_action=terminal_action,
        terminal_reason_code=terminal_reason_code,
        status=trajectory_status,
        route_rank=1,
        path_match=matches_any_action_path(path_spec.action_path, case.acceptable_action_paths),
        reward=reward,
        component_scores=component_scores,
        errors=errors,
        anti_pattern_flags=[],
    )


def _rank_and_flag_case_results(
        case: PlannerEvalCase,
        results: list[CalibrationPathResult],
) -> list[CalibrationPathResult]:
    ranked = sorted(
        results,
        key=lambda item: (-float(item.reward["total_reward"]), item.path_id),
    )
    best_acceptable = _best_acceptable_reward(results)
    best_total = float(ranked[0].reward["total_reward"]) if ranked else 0.0
    component_variance_low = _case_component_variance_is_low(results)

    output: list[CalibrationPathResult] = []
    for rank, result in enumerate(ranked, start=1):
        flags = list(result.anti_pattern_flags)
        if component_variance_low:
            flags.append("no_component_variance")
        flags.extend(_anti_pattern_flags(case, result, best_acceptable, best_total))
        output.append(result.model_copy(update={
            "route_rank": rank,
            "anti_pattern_flags": sorted(set(flags)),
        }))
    return sorted(output, key=lambda item: item.path_id)


def _best_acceptable_reward(results: list[CalibrationPathResult]) -> float | None:
    acceptable_rewards = [
        float(result.reward["total_reward"])
        for result in results
        if result.path_match
    ]
    return max(acceptable_rewards) if acceptable_rewards else None


def _anti_pattern_flags(
        case: PlannerEvalCase,
        result: CalibrationPathResult,
        best_acceptable: float | None,
        best_total: float,
        *,
        eps: float = 1e-9,
) -> list[str]:
    flags: list[str] = []
    if best_acceptable is None:
        return flags

    total = float(result.reward["total_reward"])
    wins_against_acceptable = total > best_acceptable + eps
    ties_best = abs(total - best_total) <= eps
    actual_path = set(action_values(result.action_path))
    acceptable_values = [
        set(action_values(path))
        for path in case.acceptable_action_paths
    ]

    hyde_unnecessary = (
        QueryAction.HYDE_SEARCH.value in actual_path
        and not any(QueryAction.HYDE_SEARCH.value in path for path in acceptable_values)
    )
    if hyde_unnecessary and wins_against_acceptable:
        flags.append("unnecessary_hyde_wins")

    web_unnecessary = (
        QueryAction.WEB_SEARCH.value in actual_path
        and not case.expected_behavior.should_call_web
    )
    if web_unnecessary and wins_against_acceptable:
        flags.append("unnecessary_web_wins")

    if (
        result.terminal_action == QueryAction.REFUSE
        and case.expected_behavior.should_answer
        and (wins_against_acceptable or ties_best)
    ):
        flags.append("premature_refuse_wins")

    if (
        result.terminal_action == QueryAction.ANSWER
        and not case.expected_behavior.should_answer
        and (wins_against_acceptable or ties_best)
    ):
        flags.append("premature_answer_wins")

    return flags


def _case_component_variance_is_low(
        results: list[CalibrationPathResult],
        *,
        eps: float = 1e-8,
) -> bool:
    if len(results) < 2:
        return True
    component_names = sorted(results[0].component_scores)
    return all(
        pvariance(result.component_scores[name] for result in results) < eps
        for name in component_names
    )


def _build_summary(results_by_case: dict[str, list[CalibrationPathResult]]) -> CalibrationSummary:
    all_results = [
        result
        for case_results in results_by_case.values()
        for result in case_results
    ]
    path_counts = [len(results) for results in results_by_case.values()]
    component_names = sorted(all_results[0].component_scores) if all_results else []
    anti_pattern_counts: Counter[str] = Counter(
        flag
        for result in all_results
        for flag in result.anti_pattern_flags
    )
    route_ordering_violations = _route_ordering_violations(results_by_case)
    unnecessary_search_wins = [
        _flag_record(result)
        for result in all_results
        if {"unnecessary_hyde_wins", "unnecessary_web_wins"}.intersection(result.anti_pattern_flags)
    ]
    premature_refusal_wins = [
        _flag_record(result)
        for result in all_results
        if "premature_refuse_wins" in result.anti_pattern_flags
    ]
    premature_answer_wins = [
        _flag_record(result)
        for result in all_results
        if "premature_answer_wins" in result.anti_pattern_flags
    ]
    no_component_variance_cases = sorted({
        result.case_id
        for result in all_results
        if "no_component_variance" in result.anti_pattern_flags
    })
    critical_flags = {
        flag
        for flag in anti_pattern_counts
        if flag in CRITICAL_ANTI_PATTERN_FLAGS
    }
    freeze_decision = "needs_reward_review" if critical_flags else "frozen"
    freeze_reasons = (
        [f"发现关键 Reward 反模式：{', '.join(sorted(critical_flags))}"]
        if critical_flags
        else ["未发现乱 HyDE、乱 Web、过早拒答或过早回答系统性胜出的关键反模式。"]
    )
    if no_component_variance_cases:
        freeze_reasons.append(
            f"{len(no_component_variance_cases)} 个 case 的六路分项方差过低，需要后续扩充 dev 多样性。"
        )
    return CalibrationSummary(
        component_average_scores={
            name: mean(result.component_scores[name] for result in all_results)
            for name in component_names
        },
        component_variance={
            name: pvariance(result.component_scores[name] for result in all_results)
            for name in component_names
        },
        case_count=len(results_by_case),
        path_count=len(all_results),
        min_paths_per_case=min(path_counts) if path_counts else 0,
        max_paths_per_case=max(path_counts) if path_counts else 0,
        anti_pattern_counts=dict(sorted(anti_pattern_counts.items())),
        route_ordering_violations=route_ordering_violations,
        unnecessary_search_wins=unnecessary_search_wins,
        premature_refusal_wins=premature_refusal_wins,
        premature_answer_wins=premature_answer_wins,
        no_component_variance_cases=no_component_variance_cases,
        freeze_decision=freeze_decision,
        freeze_reasons=freeze_reasons,
    )


def _route_ordering_violations(
        results_by_case: dict[str, list[CalibrationPathResult]],
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for case_id, results in results_by_case.items():
        acceptable = [
            result
            for result in results
            if result.path_match
        ]
        unacceptable = [
            result
            for result in results
            if not result.path_match
        ]
        if not acceptable or not unacceptable:
            continue
        best_acceptable = max(acceptable, key=lambda item: float(item.reward["total_reward"]))
        best_unacceptable = max(unacceptable, key=lambda item: float(item.reward["total_reward"]))
        if float(best_unacceptable.reward["total_reward"]) > float(best_acceptable.reward["total_reward"]):
            violations.append({
                "case_id": case_id,
                "best_acceptable_path_id": best_acceptable.path_id,
                "best_acceptable_reward": best_acceptable.reward["total_reward"],
                "best_unacceptable_path_id": best_unacceptable.path_id,
                "best_unacceptable_reward": best_unacceptable.reward["total_reward"],
            })
    return violations


def _flag_record(result: CalibrationPathResult) -> dict[str, Any]:
    return {
        "case_id": result.case_id,
        "path_id": result.path_id,
        "action_path": action_values(result.action_path),
        "total_reward": result.reward["total_reward"],
        "flags": result.anti_pattern_flags,
    }


def load_reward_calibration_output(path: str | Path) -> RewardCalibrationOutput:
    """读取 Reward 校准 JSON 并校验 schema。"""
    return RewardCalibrationOutput.model_validate_json(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""
    args = _build_parser().parse_args(argv)
    cases = load_planner_cases(args.cases)
    snapshot = load_environment_snapshot(args.snapshot)
    output = run_reward_calibration(
        cases=cases,
        snapshot=snapshot,
        split=args.split,
        case_path=args.cases,
        snapshot_path=args.snapshot,
        reward_config=RewardConfig(
            reward_version=args.reward_version,
            weights=RewardWeights(),
        ),
        profile_name=args.profile_name,
    )
    write_reward_calibration_output(output, args.output)
    write_reward_training_profile(output.training_profile, args.profile_output)
    print(f"run_id={output.run_id}")
    print(f"snapshot_id={output.snapshot_id}")
    print(f"split={output.split.value}, case_count={output.case_count}, path_count={output.path_count}")
    print(f"freeze_decision={output.summary.freeze_decision}")
    print(f"output={args.output}")
    print(f"profile={args.profile_output}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="阶段 9 Reward v1.1 dev 多轨迹校准。")
    parser.add_argument("--cases", required=True, type=Path, help="PlannerEvalCase JSONL 文件。")
    parser.add_argument("--snapshot", required=True, type=Path, help="EnvironmentSnapshot JSON 文件。")
    parser.add_argument("--split", default=CaseSplit.DEV.value, choices=[split.value for split in CaseSplit])
    parser.add_argument("--reward-version", default="reward-v1.1")
    parser.add_argument("--profile-name", default=DEFAULT_PROFILE_NAME)
    parser.add_argument("--output", required=True, type=Path, help="校准 JSON 输出路径。")
    parser.add_argument("--profile-output", required=True, type=Path, help="训练 Reward profile 输出路径。")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
