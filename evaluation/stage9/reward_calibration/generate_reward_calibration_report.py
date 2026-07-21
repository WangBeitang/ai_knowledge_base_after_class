"""根据 Reward v1.1 多轨迹校准 JSON 生成 Markdown 报告。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.stage9.reward_calibration.run_reward_calibration import (  # noqa: E402
    RewardCalibrationOutput,
    load_reward_calibration_output,
)


COMPONENT_ORDER = ("format", "retrieval", "citation", "answer", "behavior", "cost")


def build_markdown_report(output: RewardCalibrationOutput) -> str:
    """把校准 JSON 转成可读 Markdown。"""
    lines = [
        "# 阶段 9 Reward v1.1 dev 多轨迹校准报告",
        "",
        "## 结论",
        "",
        f"- 冻结结论：`{output.summary.freeze_decision}`。",
        f"- Reward 版本：`{output.reward_version}`。",
        f"- Reward profile：`{output.reward_profile}`。",
        f"- EnvironmentSnapshot：`{output.snapshot_id}`。",
        f"- ActionProvider：`{output.action_provider}`。",
        f"- dev case 数：`{output.case_count}`。",
        f"- Action 路线总数：`{output.path_count}`。",
        f"- 每个 case 路线数：`{output.summary.min_paths_per_case}` ~ `{output.summary.max_paths_per_case}`。",
        "",
        "冻结理由：",
        "",
        *[f"- {reason}" for reason in output.summary.freeze_reasons],
        "",
        "## Reward 分项统计",
        "",
        _component_table(output),
        "",
        "## 反模式统计",
        "",
        _anti_pattern_table(output.summary.anti_pattern_counts),
        "",
        "## 路线排序异常",
        "",
        _violation_table(output.summary.route_ordering_violations),
        "",
        "## 各 case 最优路线",
        "",
        _case_best_table(output),
        "",
        "## 使用边界",
        "",
        "- 本报告只用于 Reward v1.1 训练前校准，不是正式 held-out test 结论。",
        "- 当前 provider 若为 `snapshot_expected_chunks`，检索和引用分主要证明离线契约可执行，不代表真实 Milvus/Web 质量。",
        "- 如果后续新增独立真实文档 dev Gold，应重新运行本校准并生成新的 Reward profile。",
        "",
    ]
    return "\n".join(lines)


def write_report(report: str, path: str | Path) -> None:
    """写入 Markdown 报告。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


def _component_table(output: RewardCalibrationOutput) -> str:
    lines = [
        "| 分项 | 平均分 | 方差 |",
        "|---|---:|---:|",
    ]
    for name in COMPONENT_ORDER:
        average = output.summary.component_average_scores.get(name)
        variance = output.summary.component_variance.get(name)
        lines.append(f"| `{name}` | {_fmt(average)} | {_fmt(variance)} |")
    return "\n".join(lines)


def _anti_pattern_table(counts: dict[str, int]) -> str:
    if not counts:
        return "未发现反模式标记。"
    lines = [
        "| 反模式 | 数量 |",
        "|---|---:|",
    ]
    for name, count in sorted(counts.items()):
        lines.append(f"| `{name}` | {count} |")
    return "\n".join(lines)


def _violation_table(violations: list[dict[str, Any]]) -> str:
    if not violations:
        return "未发现不可接受路线高于可接受路线的排序异常。"
    lines = [
        "| case_id | best acceptable | reward | best unacceptable | reward |",
        "|---|---|---:|---|---:|",
    ]
    for row in violations:
        lines.append(
            f"| `{row['case_id']}` | `{row['best_acceptable_path_id']}` | "
            f"{_fmt(row['best_acceptable_reward'])} | `{row['best_unacceptable_path_id']}` | "
            f"{_fmt(row['best_unacceptable_reward'])} |"
        )
    return "\n".join(lines)


def _case_best_table(output: RewardCalibrationOutput) -> str:
    best_by_case: dict[str, Any] = {}
    for result in output.results:
        current = best_by_case.get(result.case_id)
        if current is None or float(result.reward["total_reward"]) > float(current.reward["total_reward"]):
            best_by_case[result.case_id] = result
    lines = [
        "| case_id | best path | action_path | reward | flags |",
        "|---|---|---|---:|---|",
    ]
    for case_id in sorted(best_by_case):
        result = best_by_case[case_id]
        action_path = " -> ".join(action.value for action in result.action_path)
        flags = ", ".join(result.anti_pattern_flags) or "-"
        lines.append(
            f"| `{case_id}` | `{result.path_id}` | `{action_path}` | "
            f"{_fmt(result.reward['total_reward'])} | {flags} |"
        )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output = load_reward_calibration_output(args.input)
    report = build_markdown_report(output)
    write_report(report, args.output)
    print(f"wrote={args.output}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成阶段 9 Reward v1.1 多轨迹校准报告。")
    parser.add_argument("--input", required=True, type=Path, help="run_reward_calibration.py 输出 JSON。")
    parser.add_argument("--output", required=True, type=Path, help="Markdown 报告输出路径。")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())

