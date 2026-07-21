"""阶段 8.7 Planner SFT 数据导出入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# 允许从仓库根目录直接执行：
# uv run python evaluation/stage8/export_sft_data.py ...
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.evaluation.sft_exporter import (  # noqa: E402
    DEFAULT_REWARD_THRESHOLD,
    SftArtifactStatus,
    SftExportConfig,
    export_sft_samples_from_files,
    parse_allowed_splits,
)


def main(argv: list[str] | None = None) -> int:
    """解析命令行参数，导出 Planner SFT JSONL 和 manifest。"""
    args = _build_parser().parse_args(argv)
    config = SftExportConfig(
        reward_threshold=args.reward_threshold,
        allowed_splits=parse_allowed_splits(args.allowed_splits),
        require_private_review=not args.allow_unreviewed_private,
        artifact_status=SftArtifactStatus(args.artifact_status),
    )
    result = export_sft_samples_from_files(
        eval_result_path=args.eval_result,
        cases_path=args.cases,
        output_path=args.output,
        manifest_path=args.manifest,
        config=config,
    )
    manifest = result.manifest
    print(f"manifest_id={manifest.manifest_id}")
    print(f"source_run_id={manifest.source_run_id}")
    print(f"sample_count={manifest.sample_count}")
    print(f"artifact_status={manifest.artifact_status.value}")
    print(f"exported_trajectory_count={manifest.exported_trajectory_count}")
    print(f"source_counts={manifest.source_counts}")
    print(f"filter_counts={manifest.filter_counts}")
    print(f"output={args.output}")
    print(f"manifest={args.manifest}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "阶段 8 Planner SFT 数据导出。"
            "从 baseline runner 评测结果中筛选高质量轨迹，并拆成单步 PlannerDecision 样本。"
        )
    )
    parser.add_argument(
        "--eval-result",
        required=True,
        help="8.6 baseline runner 输出 JSON，例如 evaluation/stage8/results/planner_eval_train.json。",
    )
    parser.add_argument(
        "--cases",
        required=True,
        help="PlannerEvalCase JSONL 文件路径，例如 evaluation/stage8/cases/planner_cases.jsonl。",
    )
    parser.add_argument(
        "--reward-threshold",
        type=float,
        default=DEFAULT_REWARD_THRESHOLD,
        help=f"自动导出最低 Reward 阈值，默认 {DEFAULT_REWARD_THRESHOLD}。",
    )
    parser.add_argument(
        "--allowed-splits",
        default="train,dev",
        help="允许进入 SFT 的 split，默认 train,dev；test/demo 会在 schema 层被拒绝。",
    )
    parser.add_argument(
        "--allow-unreviewed-private",
        action="store_true",
        help="允许未 reviewed 的 private_user 样本导出。默认关闭，正常不建议使用。",
    )
    parser.add_argument(
        "--artifact-status",
        choices=[status.value for status in SftArtifactStatus],
        default=SftArtifactStatus.CANDIDATE.value,
        help=(
            "导出审批级别。默认 candidate；只有 train + reviewed + 明确 Gold 来源的轨迹"
            "才能使用 approved_training_seed。"
        ),
    )
    parser.add_argument(
        "--output",
        default="evaluation/stage8/results/sft_planner_train.jsonl",
        help="SFT JSONL 输出路径。",
    )
    parser.add_argument(
        "--manifest",
        default="evaluation/stage8/results/sft_manifest.json",
        help="SFT manifest 输出路径。",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
