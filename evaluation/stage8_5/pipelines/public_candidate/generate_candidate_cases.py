"""阶段 8.5 从故障场景卡片生成 PlannerEvalCase 候选入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.evaluation.case_schema import CaseSplit  # noqa: E402
from app.shared.config.knowledge_base_config import DEFAULT_DATASET_ID, DEFAULT_TENANT_ID  # noqa: E402
from evaluation.stage8_5.pipelines.common.paths import stage85_layout  # noqa: E402
from evaluation.stage8_5.pipelines.common.stage85_schema import (  # noqa: E402
    FaultScenarioCard,
    Stage85QualityReport,
    build_case_payloads_from_cards,
    read_jsonl,
    write_json,
    write_jsonl,
)


_LAYOUT = stage85_layout()
DEFAULT_CARDS = _LAYOUT.public_intermediate / "fault_scenario_cards.jsonl"
DEFAULT_CANDIDATES = _LAYOUT.public_intermediate / "planner_case_candidates.jsonl"
DEFAULT_REJECTED = _LAYOUT.public_review / "rejected_cases.jsonl"
DEFAULT_REPORT = _LAYOUT.public_intermediate / "data_quality_report.json"


def main(argv: list[str] | None = None) -> int:
    """生成 pending 状态的 PlannerEvalCase 候选样本。"""

    args = _build_parser().parse_args(argv)
    cards = read_jsonl(args.cards, FaultScenarioCard, allow_missing=args.allow_missing)
    payloads, rejected_cards = build_case_payloads_from_cards(
        cards,
        dataset_id=args.dataset_id,
        owner_user_id=args.owner_user_id,
        tenant_id=args.tenant_id,
        split=CaseSplit(args.split),
    )
    write_jsonl(args.output, payloads)
    write_jsonl(args.rejected, rejected_cards)
    report = Stage85QualityReport(
        files={
            "cards": str(args.cards),
            "candidates": str(args.output),
            "rejected": str(args.rejected),
        },
        card_counts={
            "total": len(cards),
            "generated": len(cards) - len(rejected_cards),
            "rejected": len(rejected_cards),
        },
        case_counts={
            "generated": len(payloads),
            "rejected": len(rejected_cards),
        },
        issues=[issue for record in rejected_cards for issue in record.issues],
    )
    write_json(args.report, report)
    print(f"cards={len(cards)}, generated_cases={len(payloads)}, rejected_cards={len(rejected_cards)}, output={args.output}")
    return 1 if rejected_cards else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="阶段 8.5 从故障场景卡片生成 PlannerEvalCase 候选。")
    parser.add_argument("--cards", type=Path, default=DEFAULT_CARDS, help="fault_scenario_cards.jsonl 路径。")
    parser.add_argument("--output", type=Path, default=DEFAULT_CANDIDATES, help="候选 PlannerEvalCase JSONL 输出。")
    parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED, help="无法生成 case 的卡片输出。")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="质量报告 JSON 输出。")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID, help="候选 case 默认 dataset_id。")
    parser.add_argument("--owner-user-id", default="eval_demo_user", help="候选 case 固定测试用户。")
    parser.add_argument("--tenant-id", default=DEFAULT_TENANT_ID, help="候选 case 固定租户。")
    parser.add_argument(
        "--split",
        default=CaseSplit.TRAIN.value,
        choices=[CaseSplit.TRAIN.value, CaseSplit.DEV.value, CaseSplit.TEST.value],
        help="自动候选默认 split；第一批通常先放 train 或 dev，test 需要人工冻结后再生成。",
    )
    parser.add_argument("--allow-missing", action="store_true", help="允许卡片文件暂时不存在，输出空候选。")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
