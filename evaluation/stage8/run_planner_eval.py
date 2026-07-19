"""阶段 8.6 Planner baseline 一键评测入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# 允许用户从仓库根目录直接执行：
# uv run python evaluation/stage8/run_planner_eval.py ...
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.evaluation.baseline_runner import (  # noqa: E402
    parse_planner_modes,
    run_baseline_evaluation_from_files,
)
from app.rag.evaluation.case_schema import CaseSplit  # noqa: E402
from app.rag.evaluation.reward import REWARD_VERSION  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """解析命令行参数，执行 baseline runner，并打印每个 planner 的摘要。"""
    args = _build_parser().parse_args(argv)
    output = run_baseline_evaluation_from_files(
        cases_path=args.cases,
        snapshot_path=args.snapshot,
        split=args.split,
        planners=args.planners,
        reward_version=args.reward_version,
        output_path=args.output,
    )
    print(f"run_id={output.run_id}")
    print(f"snapshot_id={output.snapshot_id}")
    print(f"split={output.split.value}, case_count={output.case_count}")
    print(f"output={args.output}")
    for summary in output.planner_summaries:
        average_reward = summary.reward.get("average_total_reward")
        reward_text = "n/a" if average_reward is None else f"{average_reward:.4f}"
        if summary.status == "skipped":
            print(f"- {summary.planner_mode.value}: skipped, reason={summary.skip_reason}")
        else:
            print(
                f"- {summary.planner_mode.value}: "
                f"cases={summary.case_count}, "
                f"failed={summary.failed_case_count}, "
                f"avg_reward={reward_text}"
            )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "阶段 8 Planner baseline 评测。"
            "固定 case/snapshot/reward 后运行 rule/api/local_base，并输出 JSON 结果。"
        )
    )
    parser.add_argument(
        "--cases",
        required=True,
        help="PlannerEvalCase JSONL 文件路径，例如 evaluation/stage8/cases/planner_cases.jsonl。",
    )
    parser.add_argument(
        "--snapshot",
        required=True,
        help="EnvironmentSnapshot JSON 文件路径，例如 evaluation/stage8/snapshots/environment_snapshot.json。",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=[
            CaseSplit.DEV.value,
            CaseSplit.TEST.value,
            CaseSplit.TRAIN.value,
            CaseSplit.DEMO_REGRESSION.value,
        ],
        help="本次评测使用的 split。阶段 9 前后对比通常使用 dev/test。",
    )
    parser.add_argument(
        "--planners",
        required=True,
        type=_parse_planners,
        help="逗号分隔的 planner 列表，例如 rule 或 rule,api,local_base。",
    )
    parser.add_argument(
        "--reward-version",
        default=REWARD_VERSION,
        help=f"Reward 版本，默认 {REWARD_VERSION}。",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="评测结果 JSON 输出路径。",
    )
    return parser


def _parse_planners(value: str):
    """让 argparse 在入口处尽早校验 planner 名称。"""
    try:
        return parse_planner_modes(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


if __name__ == "__main__":
    raise SystemExit(main())
