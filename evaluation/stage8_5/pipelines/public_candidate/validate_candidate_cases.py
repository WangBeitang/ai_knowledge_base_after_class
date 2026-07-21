"""阶段 8.5 PlannerEvalCase 候选校验和分流入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.stage8_5.pipelines.common.paths import stage85_layout  # noqa: E402
from evaluation.stage8_5.pipelines.common.stage85_schema import (  # noqa: E402
    IssueSeverity,
    validate_candidate_payloads,
    write_json,
    write_jsonl,
)


_LAYOUT = stage85_layout()
DEFAULT_CANDIDATES = _LAYOUT.public_intermediate / "planner_case_candidates.jsonl"
DEFAULT_APPROVED = _LAYOUT.public_review / "schema_approved_cases.jsonl"
DEFAULT_REVIEW = _LAYOUT.public_review / "review_queue.jsonl"
DEFAULT_REJECTED = _LAYOUT.public_review / "rejected_cases.jsonl"
DEFAULT_REPORT = _LAYOUT.public_intermediate / "data_quality_report.json"


def main(argv: list[str] | None = None) -> int:
    """复用阶段 8 schema 校验候选 case，并输出基础门禁通过池、审核池和拒绝池。"""

    args = _build_parser().parse_args(argv)
    payloads = _read_payloads(args.input, allow_missing=args.allow_missing)
    approved, review, rejected, report = validate_candidate_payloads(payloads)
    report.files.update({
        "input": str(args.input),
        "approved": str(args.approved),
        "review": str(args.review),
        "rejected": str(args.rejected),
    })
    write_jsonl(args.approved, approved)
    write_jsonl(args.review, review)
    write_jsonl(args.rejected, rejected)
    write_json(args.report, report)
    error_count = sum(1 for issue in report.issues if issue.severity is IssueSeverity.ERROR)
    print(
        f"cases={len(payloads)}, approved={len(approved)}, "
        f"review={len(review)}, rejected={len(rejected)}, errors={error_count}, report={args.report}"
    )
    return 1 if error_count else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="阶段 8.5 候选 PlannerEvalCase schema 校验和分流。")
    parser.add_argument("--input", type=Path, default=DEFAULT_CANDIDATES, help="候选 case JSONL 输入。")
    parser.add_argument("--approved", type=Path, default=DEFAULT_APPROVED, help="已复核通过 case 输出。")
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW, help="待审核 case 输出。")
    parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED, help="拒绝 case 输出。")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="质量报告 JSON 输出。")
    parser.add_argument("--allow-missing", action="store_true", help="允许候选文件暂时不存在，输出空分流结果。")
    return parser


def _read_payloads(path: Path, *, allow_missing: bool) -> list[dict]:
    if allow_missing and not path.exists():
        return []
    payloads: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                payloads.append(json.loads(line))
            except json.JSONDecodeError as error:
                payloads.append({"case_id": f"invalid-json-line-{line_number}", "_json_error": str(error)})
    return payloads


if __name__ == "__main__":
    raise SystemExit(main())
