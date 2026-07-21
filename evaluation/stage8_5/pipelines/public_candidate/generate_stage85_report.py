"""阶段 8.5 数据处理入口 Markdown 报告生成脚本。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.stage8_5.pipelines.common.paths import stage85_layout  # noqa: E402
from evaluation.stage8_5.pipelines.common.stage85_schema import load_json  # noqa: E402


_LAYOUT = stage85_layout()
DEFAULT_REPORT_JSON = _LAYOUT.public_intermediate / "data_quality_report.json"
DEFAULT_OUTPUT = _LAYOUT.reports / "阶段8.5数据处理报告.md"


def main(argv: list[str] | None = None) -> int:
    """读取机器可读质量报告，生成方便人工复核的 Markdown。"""

    args = _build_parser().parse_args(argv)
    report = load_json(args.report, allow_missing=True)
    markdown = build_markdown_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"wrote={args.output}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成阶段 8.5 数据处理 Markdown 报告。")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_JSON, help="data_quality_report.json 路径。")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Markdown 输出路径。")
    return parser


def build_markdown_report(report: dict[str, Any]) -> str:
    """把质量报告 JSON 转成 Markdown。"""

    lines = [
        "# 阶段 8.5 数据处理报告",
        "",
        "## 历史边界",
        "",
        "- 阶段 8 报告使用 `reward-v1`，只作为历史结果保留。",
        "- 阶段 8.5 起新增评测、筛选和训练导出使用 `reward-v1.1` 或后续版本。",
        "- 本报告不生成训练数据，只记录来源、卡片、候选 case 和 split 门禁结果。",
        "",
        "## 文件",
        "",
        _dict_table(report.get("files", {}), "文件类型", "路径"),
        "",
        "## 来源统计",
        "",
        _dict_table(report.get("source_counts", {}), "状态", "数量"),
        "",
        "## 卡片统计",
        "",
        _dict_table(report.get("card_counts", {}), "状态", "数量"),
        "",
        "## 候选 case 统计",
        "",
        _dict_table(report.get("case_counts", {}), "状态", "数量"),
        "",
        "## split 统计",
        "",
        _dict_table(report.get("split_counts", {}), "split", "数量"),
        "",
        "## 问题明细",
        "",
        _issues_table(report.get("issues", [])),
        "",
    ]
    return "\n".join(lines)


def _dict_table(values: dict[str, Any], key_title: str, value_title: str) -> str:
    if not values:
        return "暂无数据。"
    lines = [f"| {key_title} | {value_title} |", "|---|---:|"]
    for key, value in values.items():
        lines.append(f"| `{key}` | {value} |")
    return "\n".join(lines)


def _issues_table(issues: list[dict[str, Any]]) -> str:
    if not issues:
        return "暂无错误或警告。"
    lines = [
        "| 严重程度 | code | source_id | card_id | case_id | 说明 |",
        "|---|---|---|---|---|---|",
    ]
    for issue in issues:
        lines.append(
            f"| {issue.get('severity', '')} | `{issue.get('code', '')}` | "
            f"`{issue.get('source_id', '')}` | `{issue.get('card_id', '')}` | "
            f"`{issue.get('case_id', '')}` | {issue.get('message', '')} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
