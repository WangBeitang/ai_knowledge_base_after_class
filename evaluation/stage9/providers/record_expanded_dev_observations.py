"""为 expanded dev（扩展开发集）录制真实检索动作的 Observation（观察结果）。"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.evaluation.action_providers import (  # noqa: E402
    MilvusActionProvider,
    RecordingActionProvider,
)
from app.rag.evaluation.baseline_runner import load_environment_snapshot  # noqa: E402
from app.rag.evaluation.case_schema import (  # noqa: E402
    CaseSplit,
    HumanReviewStatus,
    PlannerEvalCase,
    load_planner_cases,
)
from app.rag.evaluation.offline_environment import (  # noqa: E402
    OfflineActionProvider,
    OfflineRagEnvironment,
)
from app.rag.query.contracts import (  # noqa: E402
    PlannerDecision,
    PlannerReasonCode,
    QueryAction,
)


DEFAULT_CASES = PROJECT_ROOT / "evaluation/stage8/cases/planner_cases.jsonl"
DEFAULT_SNAPSHOT = (
    PROJECT_ROOT
    / "evaluation/stage9/artifacts/heldout_route_test/environment_snapshot.json"
)
DEFAULT_RECORDS = (
    PROJECT_ROOT
    / "evaluation/stage9/artifacts/provider_records/expanded_dev_provider_observations.jsonl"
)


def record_expanded_dev_observations(
        *,
        cases_path: str | Path,
        snapshot_path: str | Path,
        output_path: str | Path,
        chunk_status_filter_enabled: bool,
        case_ids: set[str] | None = None,
        max_cases: int | None = None,
        max_candidate_content_chars: int | None = None,
        overwrite: bool = False,
        action_provider: OfflineActionProvider | None = None,
) -> Counter[str]:
    """
    为每条 reviewed dev（已审核开发集）录制完整检索动作集合。

    每个 case 都录制 local_search -> hyde_search；允许 Web（网页检索）的 case 再录制
    web_search。这样模型即使选择了错误检索路线，Replay（回放）也能返回真实结果，
    不会把“缺少回放记录”误判为模型路线错误。终态 answer/refuse/ask 不依赖 Provider，
    因此不在本文件中录制。
    """

    if max_cases is not None and max_cases <= 0:
        raise ValueError("max_cases 必须大于 0")
    output = Path(output_path)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"真实 Provider 记录已存在；如需覆盖请显式传 --overwrite：{output}")
        output.unlink()

    cases = _reviewed_dev_cases(cases_path)
    available_ids = {case.case_id for case in cases}
    if case_ids:
        unknown_ids = sorted(case_ids - available_ids)
        if unknown_ids:
            raise ValueError(f"未找到 reviewed dev case（已审核开发集样本）：{unknown_ids}")
        cases = [case for case in cases if case.case_id in case_ids]
    if max_cases is not None:
        cases = cases[:max_cases]
    if not cases:
        raise ValueError("没有可录制的 reviewed dev case（已审核开发集样本）")

    snapshot = load_environment_snapshot(snapshot_path)
    provider = RecordingActionProvider(
        action_provider
        or MilvusActionProvider(
            chunk_status_filter_enabled=chunk_status_filter_enabled,
        ),
        output_path=output,
        max_candidate_content_chars=max_candidate_content_chars,
    )
    environment = OfflineRagEnvironment(
        snapshot=snapshot,
        action_provider=provider,
        planner_mode="real_provider",
        run_id_prefix="stage9_expanded_dev_provider_record",
        max_steps=4,
    )

    counts: Counter[str] = Counter()
    for case_index, case in enumerate(cases, start=1):
        print(
            f"[expanded_dev_provider] case={case_index}/{len(cases)} "
            f"case_id={case.case_id} status=running",
            flush=True,
        )
        state = environment.reset(
            case,
            run_id=f"stage9_expanded_dev_provider_record_{case.case_id}",
            planner_mode="real_provider",
        )
        for decision in _retrieval_decisions(case):
            step = environment.step(state, decision)
            state = step.state
            counts["record_count"] += 1
            counts[f"action:{decision.action.value}"] += 1
            if step.error is not None:
                counts["error_count"] += 1
                counts[f"error:{step.error.code}"] += 1
        counts["case_count"] += 1
        print(
            f"[expanded_dev_provider] case={case_index}/{len(cases)} "
            f"case_id={case.case_id} status=completed "
            f"error_count={len(state.errors)}",
            flush=True,
        )
    return counts


def _reviewed_dev_cases(path: str | Path) -> list[PlannerEvalCase]:
    """读取并冻结 25 条 reviewed dev；train/test 不进入真实动作录制。"""

    cases = sorted(
        (
            case
            for case in load_planner_cases(path)
            if case.split == CaseSplit.DEV
            and case.human_review_status == HumanReviewStatus.REVIEWED
        ),
        key=lambda case: case.case_id,
    )
    if len(cases) != 25:
        raise ValueError(f"expanded dev 必须恰好包含 25 条 reviewed dev，实际为 {len(cases)}")
    return cases


def _retrieval_decisions(case: PlannerEvalCase) -> list[PlannerDecision]:
    """
    生成动作覆盖计划；这里只决定“要录哪些动作”，不根据正确答案构造候选。

    Query（查询文本）固定使用 case.query，真实候选完全由 Milvus/Web Provider 返回。
    """

    decisions = [
        PlannerDecision(
            action=QueryAction.LOCAL_SEARCH,
            query=case.query,
            reason_code=PlannerReasonCode.INITIAL_LOCAL_SEARCH,
        ),
        PlannerDecision(
            action=QueryAction.HYDE_SEARCH,
            query=case.query,
            reason_code=PlannerReasonCode.LOCAL_LOW_SCORE,
        ),
    ]
    if case.expected_behavior.should_call_web:
        decisions.append(
            PlannerDecision(
                action=QueryAction.WEB_SEARCH,
                query=case.query,
                reason_code=PlannerReasonCode.REALTIME_QUERY,
            )
        )
    return decisions


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="录制 expanded dev 的真实 Provider（动作执行器）观察结果。",
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="只录制指定 case_id；可重复传入。完整冻结时不要使用该选项。",
    )
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--max-candidate-content-chars",
        type=int,
        default=None,
        help="可选正文截断上限；正式 9.3.18 默认不截断，保证 Replay 与真实候选一致。",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--disable-chunk-status-filter",
        action="store_true",
        help="关闭 Mongo 禁用 chunk 覆盖读取；只允许在无 Mongo 的受控探针环境使用。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    counts = record_expanded_dev_observations(
        cases_path=args.cases,
        snapshot_path=args.snapshot,
        output_path=args.output,
        chunk_status_filter_enabled=not args.disable_chunk_status_filter,
        case_ids=set(args.case_id) if args.case_id else None,
        max_cases=args.max_cases,
        max_candidate_content_chars=args.max_candidate_content_chars,
        overwrite=args.overwrite,
    )
    print(f"output={args.output}")
    for key, value in sorted(counts.items()):
        print(f"{key}={value}")
    return 2 if counts["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
