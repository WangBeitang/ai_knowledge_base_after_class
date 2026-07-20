"""阶段 8.5 故障场景卡片校验和过滤入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.stage8_5.stage85_schema import (  # noqa: E402
    FaultScenarioCard,
    Stage85QualityReport,
    filter_fault_cards_by_sources,
    load_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


DEFAULT_INPUT_CARDS = PROJECT_ROOT / "evaluation/stage8_5/processed/fault_scenario_cards.raw.jsonl"
DEFAULT_OUTPUT_CARDS = PROJECT_ROOT / "evaluation/stage8_5/processed/fault_scenario_cards.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "evaluation/stage8_5/results/data_quality_report.json"


def main(argv: list[str] | None = None) -> int:
    """校验故障场景卡片，并只输出来源已 approved 的卡片。"""

    args = _build_parser().parse_args(argv)
    report_payload = load_json(args.source_report, allow_missing=True)
    approved_source_ids = report_payload.get("approved_source_ids", [])
    cards = read_jsonl(args.input, FaultScenarioCard, allow_missing=args.allow_missing)
    accepted_cards, issues = filter_fault_cards_by_sources(cards, approved_source_ids)
    write_jsonl(args.output, accepted_cards)

    report = Stage85QualityReport(
        files={
            "input_cards": str(args.input),
            "output_cards": str(args.output),
            "source_report": str(args.source_report),
        },
        card_counts={
            "total": len(cards),
            "accepted": len(accepted_cards),
            "rejected": len(cards) - len(accepted_cards),
        },
        approved_source_ids=list(approved_source_ids),
        issues=issues,
    )
    write_json(args.report, report)
    print(
        f"cards={len(cards)}, accepted={len(accepted_cards)}, "
        f"rejected={len(cards) - len(accepted_cards)}, output={args.output}"
    )
    return 1 if issues else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="阶段 8.5 故障场景卡片门禁。")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_CARDS, help="待校验 fault_scenario_cards.raw.jsonl。")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CARDS, help="通过门禁后的卡片 JSONL。")
    parser.add_argument("--source-report", type=Path, default=DEFAULT_REPORT, help="validate_sources 输出的报告。")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="本步骤质量报告 JSON。")
    parser.add_argument("--allow-missing", action="store_true", help="允许卡片文件暂时不存在，输出空结果。")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
