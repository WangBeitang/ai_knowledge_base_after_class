"""阶段 9 ModelPlanner（模型规划器）离线评测入口。"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
from app.rag.evaluation.offline_environment import EmptyOfflineActionProvider, OfflineActionProvider, OfflineRagEnvironment, OfflineTrajectoryResult  # noqa: E402
from app.rag.evaluation.reward import RewardConfig, TrajectoryReward, score_trajectory  # noqa: E402
from app.rag.query.contracts import EvidenceSourceType, UsageMetrics  # noqa: E402
from app.rag.query.model_planner import ModelPlanner  # noqa: E402


MODEL_PLANNER_EVAL_VERSION = "stage9-model-planner-eval-v1"
REPLAY_PROVIDER_NAME = "replay"


def run_model_planner_eval(
        *,
        checkpoint_dir: str | Path,
        cases: list[PlannerEvalCase],
        snapshot_path: str | Path,
        split: CaseSplit | str,
        output_path: str | Path | None = None,
        provider_name: str = SNAPSHOT_EXPECTED_PROVIDER_NAME,
        provider_records_path: str | Path | None = None,
        reward_config: RewardConfig | None = None,
        max_cases: int | None = None,
        run_id: str | None = None,
) -> BaselineEvalOutput:
    """在固定 snapshot（快照）下执行 SFT（监督微调）Planner 离线评测。"""

    active_reward_config = reward_config or RewardConfig()
    active_split = CaseSplit(split)
    selected_cases = [case for case in cases if case.split == active_split]
    if max_cases is not None:
        selected_cases = selected_cases[:max_cases]
    if not selected_cases:
        raise ValueError(f"没有 split={active_split.value} 的可评测 case")

    snapshot = load_environment_snapshot(snapshot_path)
    provider = _build_provider(
        provider_name,
        selected_cases,
        snapshot,
        provider_records_path=provider_records_path,
    )
    normalized_run_id = run_id or f"stage9_sft_eval_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    planner = ModelPlanner.from_checkpoint(checkpoint_dir)
    environment = OfflineRagEnvironment(
        snapshot=snapshot,
        action_provider=provider,
        planner_mode=PlannerMode.SFT,
        run_id_prefix=normalized_run_id,
    )

    results: list[PlannerEvalResult] = []
    rewards: list[TrajectoryReward] = []
    start = time.monotonic()
    for case_index, case in enumerate(selected_cases, start=1):
        case_start = time.monotonic()
        print(
            f"[dev_eval] case={case_index}/{len(selected_cases)} "
            f"case_id={case.case_id} status=running",
            flush=True,
        )
        trajectory = environment.run_planner(
            case,
            planner,
            run_id=f"{normalized_run_id}_sft_{case.case_id}",
            planner_mode=PlannerMode.SFT,
        )
        reward = score_trajectory(case, trajectory, active_reward_config)
        results.append(_result_from_trajectory(
            run_id=normalized_run_id,
            case=case,
            trajectory=trajectory,
            reward=reward,
        ))
        rewards.append(reward)
        print(
            f"[dev_eval] case={case_index}/{len(selected_cases)} "
            f"case_id={case.case_id} status=completed "
            f"duration_ms={_elapsed_ms(case_start)} "
            f"action_path={' -> '.join(action.value for action in trajectory.action_path)}",
            flush=True,
        )

    output = BaselineEvalOutput(
        run_id=normalized_run_id,
        runner_version=MODEL_PLANNER_EVAL_VERSION,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        split=active_split,
        snapshot_id=snapshot.snapshot_id,
        reward_version=active_reward_config.reward_version,
        requested_planners=[PlannerMode.SFT],
        action_provider=provider.__class__.__name__,
        case_count=len(selected_cases),
        planner_summaries=[_summary(
            results,
            rewards,
            planner=planner,
            checkpoint_dir=checkpoint_dir,
            provider_name=provider_name,
            duration_ms=_elapsed_ms(start),
        )],
        results=results,
    )
    if output_path is not None:
        write_baseline_eval_output(output, output_path)
    return output


def _build_provider(
        provider_name: str,
        cases: list[PlannerEvalCase],
        snapshot: Any,
        *,
        provider_records_path: str | Path | None = None,
) -> OfflineActionProvider:
    if provider_name == SNAPSHOT_EXPECTED_PROVIDER_NAME:
        return SnapshotExpectedChunkActionProvider(cases, snapshot)
    if provider_name == "empty":
        return EmptyOfflineActionProvider()
    if provider_name == "milvus":
        from app.rag.evaluation.action_providers import MilvusActionProvider

        return MilvusActionProvider()
    if provider_name == REPLAY_PROVIDER_NAME:
        if provider_records_path is None:
            raise ValueError(
                "provider=replay（回放执行器）必须提供 provider_records_path（真实动作记录文件）"
            )
        from app.rag.evaluation.action_providers import ReplayActionProvider

        return ReplayActionProvider(provider_records_path)
    raise ValueError(f"未知 action provider：{provider_name}")


def _result_from_trajectory(
        *,
        run_id: str,
        case: PlannerEvalCase,
        trajectory: OfflineTrajectoryResult,
        reward: TrajectoryReward,
) -> PlannerEvalResult:
    return PlannerEvalResult(
        run_id=run_id,
        case_id=case.case_id,
        split=case.split,
        planner_mode=PlannerMode.SFT,
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
            **UsageMetrics().model_dump(mode="json"),
            "planner_calls": len(trajectory.trace_steps),
            "duration_ms": sum(step.duration_ms for step in trajectory.trace_steps),
            "trajectory_status": trajectory.status.value,
            "config_match_status": trajectory.config_match_status,
            "corpus_match_status": trajectory.corpus_match_status,
        },
        errors=[error.model_dump(mode="json") for error in trajectory.errors],
    )


def _summary(
        results: list[PlannerEvalResult],
        rewards: list[TrajectoryReward],
        *,
        planner: ModelPlanner,
        checkpoint_dir: str | Path,
        provider_name: str,
        duration_ms: int,
) -> BaselinePlannerSummary:
    failed_count = sum(1 for result in results if result.errors)
    path_counts = Counter(" -> ".join(action.value for action in result.action_path) for result in results)
    return BaselinePlannerSummary(
        planner_mode=PlannerMode.SFT,
        status="completed",
        config={
            "runner_version": MODEL_PLANNER_EVAL_VERSION,
            "policy_version": planner.policy_version,
            "checkpoint": str(checkpoint_dir),
            "action_provider": provider_name,
            "path_counts": dict(sorted(path_counts.items())),
        },
        usage={
            **UsageMetrics().model_dump(mode="json"),
            "planner_calls": sum(int(result.usage.get("planner_calls", 0)) for result in results),
            "duration_ms": duration_ms,
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


def _mean(values) -> float | None:
    numbers = [float(value) for value in values]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _elapsed_ms(start: float) -> int:
    return max(0, int((time.monotonic() - start) * 1000))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段 9 ModelPlanner 离线评测。")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--provider",
        default=SNAPSHOT_EXPECTED_PROVIDER_NAME,
        choices=[SNAPSHOT_EXPECTED_PROVIDER_NAME, "empty", "milvus", REPLAY_PROVIDER_NAME],
    )
    parser.add_argument(
        "--provider-records",
        type=Path,
        help="provider=replay 时必填；记录来自真实 Provider（动作执行器）的 JSONL 文件。",
    )
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args(argv)
    output = run_model_planner_eval(
        checkpoint_dir=args.checkpoint,
        cases=load_planner_cases(args.cases),
        snapshot_path=args.snapshot,
        split=args.split,
        output_path=args.output,
        provider_name=args.provider,
        provider_records_path=args.provider_records,
        max_cases=args.max_cases,
    )
    print(f"run_id={output.run_id}")
    print(f"checkpoint={args.checkpoint}")
    print(f"case_count={output.case_count}")
    print(f"output={args.output}")
    print(json.dumps(output.planner_summaries[0].reward, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
