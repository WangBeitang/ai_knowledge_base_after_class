"""生成阶段 9 SFT 路线覆盖报告。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.evaluation.baseline_runner import BaselineEvalOutput  # noqa: E402
from app.rag.evaluation.case_schema import load_planner_cases  # noqa: E402
from app.rag.evaluation.sft_exporter import SftExportManifest, SftPlannerSample  # noqa: E402
from evaluation.stage9.route_seed.build_route_seed_cases import DEFAULT_OUTPUT, DEFAULT_PATHS, read_route_seed_paths  # noqa: E402
from evaluation.stage9.route_seed.export_route_sft_data import (  # noqa: E402
    DEFAULT_MERGED_MANIFEST,
    DEFAULT_MERGED_SFT,
    DEFAULT_ROUTE_MANIFEST,
    DEFAULT_ROUTE_SFT,
    Stage9SftMergeManifest,
)
from evaluation.stage9.route_seed.run_route_seed_paths import DEFAULT_BASELINE  # noqa: E402


DEFAULT_REPORT = PROJECT_ROOT / "evaluation/stage9/artifacts/reports/阶段9-SFT路线覆盖报告.md"


def build_route_seed_report(
        *,
        cases_path: str | Path,
        paths_path: str | Path,
        baseline_path: str | Path,
        route_sft_path: str | Path,
        route_manifest_path: str | Path,
        merged_sft_path: str | Path,
        merged_manifest_path: str | Path,
) -> str:
    """汇总 route seed 和合并后 SFT 数据，生成 Markdown 报告。"""
    cases = load_planner_cases(cases_path)
    paths = read_route_seed_paths(paths_path)
    baseline = BaselineEvalOutput.model_validate_json(Path(baseline_path).read_text(encoding="utf-8"))
    route_manifest = SftExportManifest.model_validate_json(Path(route_manifest_path).read_text(encoding="utf-8"))
    route_samples = _load_samples(route_sft_path)
    merged_manifest = Stage9SftMergeManifest.model_validate_json(Path(merged_manifest_path).read_text(encoding="utf-8"))
    merged_samples = _load_samples(merged_sft_path)
    path_by_case = {path.case_id: path for path in paths}

    lines = [
        "# 阶段 9 SFT 路线覆盖报告",
        "",
        "## 结论",
        "",
        "- 9.1 route seed 已生成、执行、导出并与阶段 8.5 curated seed 合并。",
        f"- route seed case 数：`{len(cases)}`。",
        f"- route seed SFT 样本数：`{route_manifest.sample_count}`。",
        f"- 合并后 SFT 样本数：`{merged_manifest.sample_count}`。",
        f"- 合并后来源 case 数：`{merged_manifest.source_case_count}`。",
        f"- baseline run：`{baseline.run_id}`。",
        f"- snapshot：`{baseline.snapshot_id}`。",
        f"- Reward：`{baseline.reward_version}`。",
        "",
        "## Route Seed 分布",
        "",
        _counter_table("路线家族", Counter(path.route_family for path in paths)),
        "",
        _counter_table("目标终态", Counter(path.action_path[-1].value for path in paths)),
        "",
        "## 合并后 Action 分布",
        "",
        _dict_table("Action", merged_manifest.action_counts),
        "",
        "## 合并后路线家族分布",
        "",
        _dict_table("路线家族", merged_manifest.route_family_counts),
        "",
        "## 导出边界",
        "",
        _dict_table("Gold 来源", merged_manifest.gold_origin_counts),
        "",
        _dict_table("Label 来源", merged_manifest.label_source_counts),
        "",
        _dict_table("Review 状态", merged_manifest.review_status_counts),
        "",
        "## Route Seed 明细",
        "",
        _case_table(cases, path_by_case),
        "",
        "## 使用边界",
        "",
        "- route seed 全部为 `train + reviewed + route_seed_gold + approved_training_seed`。",
        "- `route_seed_gold` 只表示阶段 9 人工路线种子，不是独立 held-out test。",
        "- 当前 Web 路线在离线 provider 下以 `web_search -> refuse` 或 `local_search -> web_search -> refuse` 训练 Web 识别和安全收口；Web answer 需要真实 Web/replay provider 与 Web 证据 schema 后再补。",
        "- 当前 route seed 使用 `snapshot_expected_chunks` provider 验证流程和 Reward，不代表真实 Milvus/Web 质量。",
        "",
    ]
    _ = route_samples, merged_samples  # 触发读取校验，报告主表使用 manifest 聚合。
    return "\n".join(lines)


def _load_samples(path: str | Path) -> list[SftPlannerSample]:
    samples: list[SftPlannerSample] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                samples.append(SftPlannerSample.model_validate_json(line))
    if not samples:
        raise ValueError(f"{path} 没有 SFT 样本")
    return samples


def _counter_table(label: str, counter: Counter[str]) -> str:
    return _dict_table(label, dict(sorted(counter.items())))


def _dict_table(label: str, values: dict[str, int]) -> str:
    lines = [
        f"| {label} | 数量 |",
        "|---|---:|",
    ]
    for key, value in sorted(values.items()):
        lines.append(f"| `{key}` | {value} |")
    return "\n".join(lines)


def _case_table(cases, path_by_case) -> str:
    lines = [
        "| case_id | route_family | action_path |",
        "|---|---|---|",
    ]
    for case in sorted(cases, key=lambda item: item.case_id):
        path = path_by_case[case.case_id]
        lines.append(
            f"| `{case.case_id}` | `{path.route_family}` | "
            f"`{' -> '.join(action.value for action in path.action_path)}` |"
        )
    return "\n".join(lines)


def write_report(report: str, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = build_route_seed_report(
        cases_path=args.cases,
        paths_path=args.paths,
        baseline_path=args.baseline,
        route_sft_path=args.route_sft,
        route_manifest_path=args.route_manifest,
        merged_sft_path=args.merged_sft,
        merged_manifest_path=args.merged_manifest,
    )
    write_report(report, args.output)
    print(f"wrote={args.output}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成阶段 9 SFT 路线覆盖报告。")
    parser.add_argument("--cases", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--paths", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--route-sft", type=Path, default=DEFAULT_ROUTE_SFT)
    parser.add_argument("--route-manifest", type=Path, default=DEFAULT_ROUTE_MANIFEST)
    parser.add_argument("--merged-sft", type=Path, default=DEFAULT_MERGED_SFT)
    parser.add_argument("--merged-manifest", type=Path, default=DEFAULT_MERGED_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())

