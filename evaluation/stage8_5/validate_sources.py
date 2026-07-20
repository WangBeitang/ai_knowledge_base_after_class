"""阶段 8.5 来源清单校验入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.stage8_5.stage85_schema import (  # noqa: E402
    IssueSeverity,
    LicenseRecord,
    SourceRecord,
    read_jsonl,
    validate_source_records,
    write_json,
)


DEFAULT_SOURCES = PROJECT_ROOT / "evaluation/stage8_5/sources/source_manifest.jsonl"
DEFAULT_LICENSES = PROJECT_ROOT / "evaluation/stage8_5/sources/license_manifest.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "evaluation/stage8_5/results/data_quality_report.json"


def main(argv: list[str] | None = None) -> int:
    """读取来源和许可证清单，输出机器可读质量报告。"""

    args = _build_parser().parse_args(argv)
    sources = read_jsonl(args.sources, SourceRecord, allow_missing=args.allow_missing)
    licenses = read_jsonl(args.licenses, LicenseRecord, allow_missing=True)
    report = validate_source_records(sources, licenses)
    report.files.update({
        "sources": str(args.sources),
        "licenses": str(args.licenses),
        "report": str(args.report),
    })
    write_json(args.report, report)

    error_count = sum(1 for issue in report.issues if issue.severity is IssueSeverity.ERROR)
    print(
        "sources="
        f"{report.source_counts.get('total', 0)}, "
        f"approved={report.source_counts.get('approved', 0)}, "
        f"pending={report.source_counts.get('pending', 0)}, "
        f"rejected={report.source_counts.get('rejected', 0)}, "
        f"errors={error_count}, report={args.report}"
    )
    return 1 if error_count else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="阶段 8.5 来源清单和许可证门禁校验。")
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES, help="source_manifest.jsonl 路径。")
    parser.add_argument("--licenses", type=Path, default=DEFAULT_LICENSES, help="license_manifest.jsonl 路径。")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="质量报告 JSON 输出路径。")
    parser.add_argument("--allow-missing", action="store_true", help="允许来源清单暂时不存在，输出空报告。")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())

