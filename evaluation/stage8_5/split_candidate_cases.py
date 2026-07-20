"""阶段 8.5 候选 case split manifest 生成入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.evaluation.case_schema import load_planner_cases  # noqa: E402
from evaluation.stage8_5.stage85_schema import build_split_manifest, write_json  # noqa: E402


DEFAULT_CASES = PROJECT_ROOT / "evaluation/stage8_5/candidates/approved_cases.jsonl"
DEFAULT_MANIFEST = PROJECT_ROOT / "evaluation/stage8_5/candidates/split_manifest.json"


def main(argv: list[str] | None = None) -> int:
    """从已通过 schema 的候选 case 生成 split manifest。"""

    args = _build_parser().parse_args(argv)
    cases = load_planner_cases(args.cases, allow_empty=args.allow_empty)
    manifest = build_split_manifest(
        cases,
        manifest_id=args.manifest_id,
        snapshot_id=args.snapshot_id,
        notes=args.notes,
    )
    write_json(args.output, manifest)
    print(
        f"cases={len(cases)}, train={len(manifest.train_case_ids)}, "
        f"dev={len(manifest.dev_case_ids)}, test={len(manifest.test_case_ids)}, output={args.output}"
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="阶段 8.5 split manifest 生成。")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="已通过候选 case JSONL。")
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST, help="split_manifest.json 输出路径。")
    parser.add_argument("--manifest-id", default="stage85-split-manifest-v1", help="manifest 版本 ID。")
    parser.add_argument("--snapshot-id", default="", help="关联环境快照 ID；公开数据导入前可为空。")
    parser.add_argument("--notes", default="阶段 8.5 公开数据候选 split 清单。", help="拆分说明和例外原因。")
    parser.add_argument("--allow-empty", action="store_true", help="允许当前 approved case 为空。")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())

