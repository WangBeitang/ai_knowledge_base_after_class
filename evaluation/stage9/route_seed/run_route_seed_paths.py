"""执行阶段 9 route seed 固定 Action path 并计算 Reward。"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.evaluation.baseline_runner import (  # noqa: E402
    SNAPSHOT_EXPECTED_PROVIDER_NAME,
    BaselineEvalOutput,
    BaselinePlannerSummary,
    SnapshotExpectedChunkActionProvider,
    load_environment_snapshot,
    write_baseline_eval_output,
)
from app.rag.evaluation.case_schema import CaseSplit, PlannerEvalCase, PlannerEvalResult, PlannerMode, load_planner_cases  # noqa: E402
from app.rag.evaluation.offline_environment import OfflineRagEnvironment, OfflineTrajectoryResult  # noqa: E402
from app.rag.evaluation.reward import RewardConfig, TrajectoryReward, score_trajectory  # noqa: E402
from app.rag.query.contracts import EvidenceSourceType, UsageMetrics  # noqa: E402
from evaluation.stage9.route_seed.build_route_seed_cases import (  # noqa: E402
    DEFAULT_OUTPUT,
    DEFAULT_PATHS,
    RouteSeedActionPath,
    read_route_seed_paths,
)


ROUTE_SEED_RUNNER_VERSION = "stage9-route-seed-runner-v1"
DEFAULT_SNAPSHOT = PROJECT_ROOT / "evaluation/stage8_5/artifacts/intermediate/sft_seed/environment_snapshot_training_v2.json"
DEFAULT_BASELINE = PROJECT_ROOT / "evaluation/stage9/artifacts/route_seed/route_seed_baseline_train.json"


def run_route_seed_paths(
        *,
        cases: list[PlannerEvalCase],
        paths: list[RouteSeedActionPath],
        snapshot_path: str | Path,
        output_path: str | Path | None = None,
        run_id: str | None = None,
) -> BaselineEvalOutput:
    """按 route_seed_action_paths.jsonl 执行固定路线，并输出 BaselineEvalOutput 兼容 JSON。"""
    snapshot = load_environment_snapshot(snapshot_path)
    path_by_case = _path_map(paths)
    selected_cases = [case for case in cases if case.case_id in path_by_case]
    if len(selected_cases) != len(path_by_case):
        missing = sorted(set(path_by_case) - {case.case_id for case in selected_cases})
        raise ValueError(f"paths 引用了不存在的 case_id：{missing}")
    if any(case.split != CaseSplit.TRAIN for case in selected_cases):
        raise ValueError("route seed 只能执行 train split case")

    normalized_run_id = run_id or f"stage9_route_seed_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    provider = SnapshotExpectedChunkActionProvider(selected_cases, snapshot)
    environment = OfflineRagEnvironment(
        snapshot=snapshot,
        action_provider=provider,
        planner_mode="route_seed",
        run_id_prefix=normalized_run_id,
    )
    results: list[PlannerEvalResult] = []
    rewards: list[TrajectoryReward] = []
    for case in selected_cases:
        path = path_by_case[case.case_id]
        trajectory = environment.run_action_path(
            case,
            list(path.action_path),
            run_id=f"{normalized_run_id}_{case.case_id}_{path.path_id}",
            planner_mode="route_seed",
        )
        reward = score_trajectory(case, trajectory, RewardConfig())
        results.append(_result_from_trajectory(
            run_id=normalized_run_id,
            case=case,
            path=path,
            trajectory=trajectory,
            reward=reward,
        ))
        rewards.append(reward)

    output = BaselineEvalOutput(
        run_id=normalized_run_id,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        split=CaseSplit.TRAIN,
        snapshot_id=snapshot.snapshot_id,
        reward_version=RewardConfig().reward_version,
        requested_planners=[PlannerMode.RULE],
        action_provider=SNAPSHOT_EXPECTED_PROVIDER_NAME,
        case_count=len(selected_cases),
        planner_summaries=[_summary(results, rewards)],
        results=results,
    )
    if output_path is not None:
        write_baseline_eval_output(output, output_path)
    return output


def _result_from_trajectory(
        *,
        run_id: str,
        case: PlannerEvalCase,
        path: RouteSeedActionPath,
        trajectory: OfflineTrajectoryResult,
        reward: TrajectoryReward,
) -> PlannerEvalResult:
    return PlannerEvalResult(
        run_id=run_id,
        case_id=case.case_id,
        split=case.split,
        # PlannerMode 仍使用 rule 以兼容既有 SFT exporter；usage 中显式写明 label_source_override。
        planner_mode=PlannerMode.RULE,
        snapshot_id=trajectory.snapshot_id,
        reward_version=reward.reward_version,
        trace_id=trajectory.run_id,
        action_path=trajectory.action_path,
        terminal_action=trajectory.terminal_action,
        terminal_reason_code=trajectory.terminal_reason_code.value if trajectory.terminal_reason_code else "",
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
        usage={
            "planner_calls": len(trajectory.trace_steps),
            "duration_ms": sum(step.duration_ms for step in trajectory.trace_steps),
            "trajectory_status": trajectory.status.value,
            "config_match_status": trajectory.config_match_status,
            "corpus_match_status": trajectory.corpus_match_status,
            "route_family": path.route_family,
            "route_path_id": path.path_id,
            "label_source_override": path.label_source,
            "runner_version": ROUTE_SEED_RUNNER_VERSION,
        },
        errors=[error.model_dump(mode="json") for error in trajectory.errors],
    )


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


def _summary(results: list[PlannerEvalResult], rewards: list[TrajectoryReward]) -> BaselinePlannerSummary:
    family_counts = Counter(str(result.usage.get("route_family", "")) for result in results)
    failed_count = sum(1 for result in results if result.errors)
    return BaselinePlannerSummary(
        planner_mode=PlannerMode.RULE,
        status="completed",
        config={
            "runner_version": ROUTE_SEED_RUNNER_VERSION,
            "route_family_counts": dict(sorted(family_counts.items())),
            "action_provider": SNAPSHOT_EXPECTED_PROVIDER_NAME,
            "label_source": "manual_route_seed",
        },
        usage={
            **UsageMetrics().model_dump(mode="json"),
            "planner_calls": sum(int(result.usage.get("planner_calls", 0)) for result in results),
            "failed_case_count": failed_count,
        },
        reward={
            "average_total_reward": _mean(reward.total_reward for reward in rewards),
            "scored_case_count": len(rewards),
            "component_average_scores": {
                name: _mean(reward.components[name].score for reward in rewards)
                for name in sorted(rewards[0].components)
            } if rewards else {},
        },
        case_count=len(results),
        completed_case_count=len(results) - failed_count,
        failed_case_count=failed_count,
    )


def _path_map(paths: list[RouteSeedActionPath]) -> dict[str, RouteSeedActionPath]:
    mapping: dict[str, RouteSeedActionPath] = {}
    for path in paths:
        if not path.export_to_sft or path.review_status != "reviewed":
            continue
        if path.case_id in mapping:
            raise ValueError(f"每个 route seed case 第一版只能绑定一条目标 path：{path.case_id}")
        mapping[path.case_id] = path
    if not mapping:
        raise ValueError("没有可执行的 reviewed route seed path")
    return mapping


def _mean(values) -> float | None:
    numbers = [float(value) for value in values]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output = run_route_seed_paths(
        cases=load_planner_cases(args.cases),
        paths=read_route_seed_paths(args.paths),
        snapshot_path=args.snapshot,
        output_path=args.output,
    )
    print(f"run_id={output.run_id}")
    print(f"snapshot_id={output.snapshot_id}")
    print(f"case_count={output.case_count}")
    print(f"output={args.output}")
    print(json.dumps(output.planner_summaries[0].config["route_family_counts"], ensure_ascii=False, sort_keys=True))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="执行阶段 9 route seed 固定 Action path。")
    parser.add_argument("--cases", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--paths", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_BASELINE)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())

