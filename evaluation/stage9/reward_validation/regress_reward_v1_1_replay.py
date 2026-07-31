"""任务 9.3.19：使用冻结 Replay Observation 回归 Reward v1.1。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import Field

from app.rag.evaluation.action_providers import ReplayActionProvider
from app.rag.evaluation.baseline_runner import load_environment_snapshot
from app.rag.evaluation.case_schema import PlannerEvalCase, load_planner_cases
from app.rag.evaluation.metrics import matches_any_action_path
from app.rag.evaluation.offline_environment import OfflineRagEnvironment
from app.rag.evaluation.reward import score_trajectory
from app.rag.query.contracts import QueryAction
from evaluation.stage9.providers.record_expanded_dev_observations import (
    DEFAULT_RECORDS,
    retrieval_actions_for_case,
)
from evaluation.stage9.providers.validate_expanded_dev_replay import (
    DEFAULT_CONTRACT_OUTPUT,
    ExpandedDevReplayContract,
    validate_expanded_dev_replay,
)
from evaluation.stage9.reward_calibration.action_path_suite import (
    ACTION_PATH_SUITE_VERSION,
    build_action_path_suite,
)
from evaluation.stage9.reward_calibration.run_reward_calibration import (
    CRITICAL_ANTI_PATTERN_FLAGS,
    CalibrationPathResult,
    _path_result,
    _rank_and_flag_case_results,
)
from evaluation.stage9.reward_validation.validate_reward_v1_1_balanced_dev import (
    DEFAULT_CASES,
    DEFAULT_PROFILE,
    DEFAULT_REWARD_IMPLEMENTATION,
    DEFAULT_ROUTE_MATRIX,
    DEFAULT_SNAPSHOT,
    BucketValidation,
    CaseValidation,
    FrozenInput,
    ValidationModel,
    ValidationSummary,
    _bucket_validations,
    _canonical_cases_sha256,
    _case_validation,
    _logical,
    _read_json,
    _reward_config_from_profile,
    _select_balanced_dev,
    _sha256,
    _summary,
    _validate_frozen_v1_1_identity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REGRESSION_VERSION = "stage9-reward-v1.1-replay-regression-v1"
FIXED_REGRESSED_AT = "2026-07-30T15:25:00+00:00"
FIXED_RUN_ID = "stage9_reward_v1_1_replay_regression_20260730"
EXPECTED_SOURCE_TRAJECTORY_COUNT = 231
MISSING_REPLAY_MARKER = "Replay 缺少对应记录"

DEFAULT_BASELINE_VALIDATION = (
    PROJECT_ROOT
    / "evaluation/stage9/artifacts/reward/reward_v1_1_balanced_dev_validation.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "evaluation/stage9/artifacts/reward/reward_v1_1_replay_regression.json"
)
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "evaluation/stage9/artifacts/reports/阶段9-Reward-v1.1-Replay回归报告.md"
)


class SkippedReplayPath(ValidationModel):
    """因未冻结非必要负路线 Observation 而不进入 Reward 排序的一条路径。"""

    case_id: str = Field(min_length=1)
    path_id: str = Field(min_length=1)
    route_family: str = Field(min_length=1)
    action_path: list[QueryAction] = Field(min_length=1)
    missing_action: QueryAction
    reason: str = Field(min_length=1)


class RewardV11ReplayRegression(ValidationModel):
    """9.3.19 的完整机器可读回归产物。"""

    regression_version: str = REGRESSION_VERSION
    regressed_at: str
    run_id: str
    reward_version: str
    reward_profile_name: str
    selected_split: str = "dev"
    action_provider: str = "replay_action_provider"
    action_path_suite_version: str
    inputs: dict[str, FrozenInput]
    snapshot_id: str
    replay_contract_version: str
    provider_records_sha256: str = Field(min_length=64, max_length=64)
    balanced_dev_case_ids: list[str] = Field(min_length=25, max_length=25)
    balanced_dev_canonical_sha256: str = Field(min_length=64, max_length=64)
    non_dev_case_count_ignored: int = Field(ge=0)
    source_trajectory_count: int = Field(ge=1)
    scored_trajectory_count: int = Field(ge=1)
    skipped_missing_observation_count: int = Field(ge=0)
    skipped_paths: list[SkippedReplayPath]
    profile_weights: dict[str, float]
    reward_config: dict[str, Any]
    profile_mutation_performed: bool = False
    reward_mutation_performed: bool = False
    model_execution_performed: bool = False
    heldout_inference_result_count: int = 0
    answer_executor_mode: str = "offline_placeholder_for_answer_cases"
    critical_anti_pattern_counts: dict[str, int]
    diagnostic_score_contract: dict[str, Any]
    baseline_comparison: dict[str, Any]
    cases: list[CaseValidation]
    buckets: list[BucketValidation]
    summary: ValidationSummary


def run_regression(
    *,
    cases_path: Path = DEFAULT_CASES,
    snapshot_path: Path = DEFAULT_SNAPSHOT,
    profile_path: Path = DEFAULT_PROFILE,
    route_matrix_path: Path = DEFAULT_ROUTE_MATRIX,
    reward_implementation_path: Path = DEFAULT_REWARD_IMPLEMENTATION,
    provider_records_path: Path = DEFAULT_RECORDS,
    replay_contract_path: Path = DEFAULT_CONTRACT_OUTPUT,
    baseline_validation_path: Path = DEFAULT_BASELINE_VALIDATION,
    regressed_at: str = FIXED_REGRESSED_AT,
    run_id: str = FIXED_RUN_ID,
) -> RewardV11ReplayRegression:
    """运行 9.3.19；只读取冻结输入，不连接检索服务、不修改 Reward。"""

    input_paths = {
        "planner_cases": cases_path,
        "environment_snapshot": snapshot_path,
        "reward_profile_v1_1": profile_path,
        "route_matrix": route_matrix_path,
        "reward_implementation": reward_implementation_path,
        "provider_records_9_3_18": provider_records_path,
        "replay_contract_9_3_18": replay_contract_path,
        "baseline_validation_9_3_15a": baseline_validation_path,
    }
    for name, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"9.3.19 输入不存在：{name}={path}")
    before_hashes = {name: _sha256(path) for name, path in input_paths.items()}
    _validate_frozen_v1_1_identity(before_hashes)

    profile = _read_json(profile_path)
    reward_config = _reward_config_from_profile(profile)
    matrix = _read_json(route_matrix_path)
    all_cases = load_planner_cases(cases_path)
    selected_cases = _select_balanced_dev(all_cases, matrix=matrix)
    snapshot = load_environment_snapshot(snapshot_path)
    replay_contract = _validate_frozen_replay_contract(
        cases_path=cases_path,
        snapshot_path=snapshot_path,
        provider_records_path=provider_records_path,
        replay_contract_path=replay_contract_path,
    )
    baseline = _read_json(baseline_validation_path)
    _validate_baseline_identity(baseline, selected_cases)

    ranked_results, skipped_paths, source_trajectory_count = _run_replay_paths(
        cases=selected_cases,
        snapshot=snapshot,
        provider_records_path=provider_records_path,
        reward_config=reward_config,
        run_id=run_id,
    )
    if source_trajectory_count != EXPECTED_SOURCE_TRAJECTORY_COUNT:
        raise ValueError(
            "9.3.19 Action path suite 数量漂移："
            f"{source_trajectory_count} != {EXPECTED_SOURCE_TRAJECTORY_COUNT}"
        )

    case_by_id = {case.case_id: case for case in selected_cases}
    case_validations = [
        _case_validation(case_by_id[case_id], ranked_results[case_id], reward_config)
        for case_id in sorted(ranked_results)
    ]
    bucket_validations = _bucket_validations(case_validations)
    summary = _summary(case_validations, bucket_validations)
    critical_counts = Counter(
        flag
        for results in ranked_results.values()
        for result in results
        for flag in result.anti_pattern_flags
        if flag in CRITICAL_ANTI_PATTERN_FLAGS
    )
    if critical_counts and summary.decision == "pass_keep_v1_1":
        summary = summary.model_copy(
            update={
                "decision": "reward_change_needs_discussion",
                "decision_reasons": [
                    f"Replay 回归出现关键 Reward 反模式：{dict(critical_counts)}；"
                    "必须暂停，不能自动修改 v1.1。"
                ],
            }
        )

    after_hashes = {name: _sha256(path) for name, path in input_paths.items()}
    if after_hashes != before_hashes:
        changed = sorted(
            name
            for name in before_hashes
            if before_hashes[name] != after_hashes[name]
        )
        raise RuntimeError(f"9.3.19 运行期间输入发生变化：{changed}")

    scored_trajectory_count = sum(len(rows) for rows in ranked_results.values())
    if scored_trajectory_count + len(skipped_paths) != source_trajectory_count:
        raise RuntimeError("9.3.19 路线计数不守恒")
    return RewardV11ReplayRegression(
        regressed_at=regressed_at,
        run_id=run_id,
        reward_version=reward_config.reward_version,
        reward_profile_name=str(profile["profile_name"]),
        action_path_suite_version=ACTION_PATH_SUITE_VERSION,
        inputs={
            name: FrozenInput(path=_logical(path), sha256=before_hashes[name])
            for name, path in input_paths.items()
        },
        snapshot_id=snapshot.snapshot_id,
        replay_contract_version=replay_contract.contract_version,
        provider_records_sha256=replay_contract.records_sha256,
        balanced_dev_case_ids=[case.case_id for case in selected_cases],
        balanced_dev_canonical_sha256=_canonical_cases_sha256(selected_cases),
        non_dev_case_count_ignored=len(all_cases) - len(selected_cases),
        source_trajectory_count=source_trajectory_count,
        scored_trajectory_count=scored_trajectory_count,
        skipped_missing_observation_count=len(skipped_paths),
        skipped_paths=skipped_paths,
        profile_weights=reward_config.weights.as_dict(),
        reward_config=reward_config.model_dump(mode="json"),
        critical_anti_pattern_counts=dict(critical_counts),
        diagnostic_score_contract={
            "provider_observation_scope": (
                "只使用 9.3.18 冻结的 32 条真实 Provider Observation。"
            ),
            "skipped_path_policy": (
                "缺少非必要负路线 Replay Observation 的轨迹不评分；"
                "不得用空候选或 action_provider_failed 伪造低分。"
            ),
            "required_case_gate": (
                "每个 case 必须至少保留一条正确轨迹和一条错误轨迹。"
            ),
            "planner_route_score_components": ["format", "behavior", "cost"],
            "evidence_contract_score_components": ["retrieval", "citation"],
            "answer_quality_score_component": "answer",
            "changes_total_reward": False,
            "evidence_boundary": (
                "Replay 证明冻结 Observation 下的 Reward 排序，"
                "不代表当前线上检索或模型质量。"
            ),
        },
        baseline_comparison={
            "baseline_task": "9.3.15A",
            "baseline_action_provider": baseline["action_provider"],
            "baseline_trajectory_count": baseline["summary"]["trajectory_count"],
            "baseline_inversion_count": baseline["summary"]["inversion_count"],
            "baseline_minimum_case_margin": baseline["summary"]["minimum_case_margin"],
            "baseline_minimum_route_margin": baseline["summary"]["minimum_route_margin"],
            "case_id_set_unchanged": True,
        },
        cases=case_validations,
        buckets=bucket_validations,
        summary=summary,
    )


def _validate_frozen_replay_contract(
    *,
    cases_path: Path,
    snapshot_path: Path,
    provider_records_path: Path,
    replay_contract_path: Path,
) -> ExpandedDevReplayContract:
    """复算 9.3.18 契约并与冻结文件对齐，忽略非身份字段 created_at。"""

    frozen = ExpandedDevReplayContract.model_validate_json(
        replay_contract_path.read_text(encoding="utf-8")
    )
    regenerated = validate_expanded_dev_replay(
        cases_path=cases_path,
        snapshot_path=snapshot_path,
        records_path=provider_records_path,
    )
    frozen_payload = frozen.model_dump(mode="json", exclude={"created_at"})
    regenerated_payload = regenerated.model_dump(mode="json", exclude={"created_at"})
    if frozen_payload != regenerated_payload:
        raise ValueError("9.3.18 Replay contract 与当前 case/records 不一致")
    if not frozen.ok:
        raise ValueError("9.3.18 Replay contract 未通过，不能进入 9.3.19")
    return frozen


def _validate_baseline_identity(
    baseline: dict[str, Any],
    selected_cases: list[PlannerEvalCase],
) -> None:
    """确认比较对象确实是同 Reward、同路线套件和同一组 25 个 case ID。"""

    if baseline.get("reward_version") != "reward-v1.1":
        raise ValueError("9.3.15A baseline 不是 Reward v1.1")
    if baseline.get("action_path_suite_version") != ACTION_PATH_SUITE_VERSION:
        raise ValueError("9.3.15A 与 9.3.19 Action path suite 版本不一致")
    baseline_ids = set(baseline.get("balanced_dev_case_ids") or [])
    current_ids = {case.case_id for case in selected_cases}
    if baseline_ids != current_ids:
        raise ValueError("9.3.15A 与 9.3.19 balanced dev case_id 集合不一致")


def _run_replay_paths(
    *,
    cases: list[PlannerEvalCase],
    snapshot: Any,
    provider_records_path: Path,
    reward_config: Any,
    run_id: str,
) -> tuple[dict[str, list[CalibrationPathResult]], list[SkippedReplayPath], int]:
    """执行路线套件；只排除缺少非必要 Replay Observation 的负路线。"""

    environment = OfflineRagEnvironment(
        snapshot=snapshot,
        action_provider=ReplayActionProvider(provider_records_path),
        planner_mode="reward_regression",
        run_id_prefix=run_id,
    )
    raw_results_by_case: dict[str, list[CalibrationPathResult]] = defaultdict(list)
    skipped_paths: list[SkippedReplayPath] = []
    source_trajectory_count = 0

    for case in cases:
        required_actions = set(retrieval_actions_for_case(case))
        for path_spec in build_action_path_suite(case):
            source_trajectory_count += 1
            trajectory = environment.run_action_path(
                case,
                list(path_spec.action_path),
                run_id=f"{run_id}_{case.case_id}_{path_spec.path_id}",
                planner_mode="reward_regression",
            )
            provider_errors = [
                error
                for error in trajectory.errors
                if error.code == "action_provider_failed"
            ]
            if provider_errors:
                if len(provider_errors) != 1:
                    raise ValueError(f"{case.case_id}/{path_spec.path_id} Provider 错误数量异常")
                error = provider_errors[0]
                if MISSING_REPLAY_MARKER not in error.message:
                    raise ValueError(
                        f"{case.case_id}/{path_spec.path_id} 出现非预期 Provider 错误："
                        f"{error.message}"
                    )
                if matches_any_action_path(
                    path_spec.action_path,
                    case.acceptable_action_paths,
                ):
                    raise ValueError(
                        f"{case.case_id}/{path_spec.path_id} 可接受路线缺少 Replay Observation"
                    )
                if error.action is None or error.action in required_actions:
                    raise ValueError(
                        f"{case.case_id}/{path_spec.path_id} 缺失的是必要 Replay Action"
                    )
                skipped_paths.append(
                    SkippedReplayPath(
                        case_id=case.case_id,
                        path_id=path_spec.path_id,
                        route_family=path_spec.route_family,
                        action_path=list(path_spec.action_path),
                        missing_action=error.action,
                        reason="missing_non_required_replay_observation",
                    )
                )
                continue

            unexpected_errors = [
                error
                for error in trajectory.errors
                if error.code != "action_not_allowed"
            ]
            if unexpected_errors:
                raise ValueError(
                    f"{case.case_id}/{path_spec.path_id} 出现非预期环境错误："
                    f"{[error.code for error in unexpected_errors]}"
                )
            reward = score_trajectory(case, trajectory, reward_config)
            raw_results_by_case[case.case_id].append(
                _path_result(
                    case=case,
                    path_spec=path_spec,
                    trajectory_status=trajectory.status.value,
                    terminal_action=trajectory.terminal_action,
                    terminal_reason_code=(
                        trajectory.terminal_reason_code.value
                        if trajectory.terminal_reason_code
                        else ""
                    ),
                    reward=reward.to_json_dict(),
                    errors=[
                        error.model_dump(mode="json")
                        for error in trajectory.errors
                    ],
                )
            )

    ranked_results = {
        case.case_id: _rank_and_flag_case_results(
            case,
            raw_results_by_case[case.case_id],
        )
        for case in cases
    }
    return ranked_results, skipped_paths, source_trajectory_count


def render_report(output: RewardV11ReplayRegression) -> str:
    """渲染 9.3.19 中文审计报告。"""

    skip_counts = Counter(path.path_id for path in output.skipped_paths)
    lines = [
        "# 阶段 9 Reward v1.1 Replay 回归报告",
        "",
        "## 结论",
        "",
        f"- 9.3.19 决定：`{output.summary.decision}`。",
        f"- Reward：`{output.reward_version}`；权重和实现均未修改。",
        f"- balanced dev：{output.summary.case_count} 条；旧路线框架 "
        f"{output.source_trajectory_count} 条。",
        f"- 使用冻结 Replay 实际评分：{output.scored_trajectory_count} 条；"
        f"排除缺少非必要 Observation 的负路线："
        f"{output.skipped_missing_observation_count} 条。",
        f"- 错误反超：{output.summary.inversion_count}；关键反模式："
        f"`{output.critical_anti_pattern_counts}`。",
        f"- 最小 total Reward margin：`{output.summary.minimum_case_margin:.4f}`；"
        f"最小 Planner route margin：`{output.summary.minimum_route_margin:.4f}`。",
        f"- Provider records SHA256：`{output.provider_records_sha256}`。",
        "",
        *[f"- {reason}" for reason in output.summary.decision_reasons],
        "",
        "## Replay 与输入边界",
        "",
        f"- ActionProvider：`{output.action_provider}`；snapshot：`{output.snapshot_id}`。",
        "- 仅使用 9.3.18 冻结的 32 条真实 Observation；本任务没有调用 Milvus、Web 或 LLM。",
        "- 52 条未评分路线缺少的是非必要负路线 Observation；没有用空候选或 "
        "`action_provider_failed` 伪造低分。",
        "- 每个 case 都保留至少一条正确轨迹和一条错误轨迹。",
        f"- 模型执行：`{str(output.model_execution_performed).lower()}`；"
        f"heldout 推理：{output.heldout_inference_result_count}。",
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
            "## 未冻结负路线 Observation",
            "",
            "| path_id | 排除数量 |",
            "|---|---:|",
        ]
    )
    for path_id, count in sorted(skip_counts.items()):
        lines.append(f"| `{path_id}` | {count} |")
    lines.extend(
        [
            "",
            "完整逐 case 排除记录保存在机器可读 JSON 的 `skipped_paths`。",
            "",
            "## 边界与下一步",
            "",
            "- 回答型 case 仍使用占位 answer executor；本结论只说明冻结 Replay 下 "
            "Reward v1.1 没有错误鼓励错误路线。",
            "- 本报告不代表 SFT v1 已通过，也不代表 heldout 泛化。",
            (
                "- 当前结论允许保留 Reward v1.1；待 9.3.17 环境 freeze 闭环后，"
                "由用户确认是否进入 9.3.20 重跑 SFT v1。"
                if output.summary.decision == "pass_keep_v1_1"
                else "- 当前结论需要人工讨论；不得自动调整 Reward 权重或进入 9.3.20。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    output: RewardV11ReplayRegression,
    *,
    output_path: Path = DEFAULT_OUTPUT,
    report_path: Path = DEFAULT_REPORT,
    overwrite: bool = False,
) -> dict[str, Any]:
    """写出 9.3.19 JSON 和报告；默认拒绝覆盖已有产物。"""

    for path in (output_path, report_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"9.3.19 输出已存在，拒绝静默覆盖：{path}")
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--route-matrix", type=Path, default=DEFAULT_ROUTE_MATRIX)
    parser.add_argument(
        "--reward-implementation",
        type=Path,
        default=DEFAULT_REWARD_IMPLEMENTATION,
    )
    parser.add_argument("--provider-records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument(
        "--replay-contract",
        type=Path,
        default=DEFAULT_CONTRACT_OUTPUT,
    )
    parser.add_argument(
        "--baseline-validation",
        type=Path,
        default=DEFAULT_BASELINE_VALIDATION,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output = run_regression(
        cases_path=args.cases,
        snapshot_path=args.snapshot,
        profile_path=args.profile,
        route_matrix_path=args.route_matrix,
        reward_implementation_path=args.reward_implementation,
        provider_records_path=args.provider_records,
        replay_contract_path=args.replay_contract,
        baseline_validation_path=args.baseline_validation,
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
                "ok": output.summary.decision == "pass_keep_v1_1",
                "decision": output.summary.decision,
                "case_count": output.summary.case_count,
                "source_trajectory_count": output.source_trajectory_count,
                "scored_trajectory_count": output.scored_trajectory_count,
                "skipped_missing_observation_count": (
                    output.skipped_missing_observation_count
                ),
                "inversion_count": output.summary.inversion_count,
                "minimum_case_margin": output.summary.minimum_case_margin,
                "minimum_route_margin": output.summary.minimum_route_margin,
                **files,
            },
            ensure_ascii=False,
        )
    )
    return 0 if output.summary.decision == "pass_keep_v1_1" else 2


if __name__ == "__main__":
    raise SystemExit(main())
