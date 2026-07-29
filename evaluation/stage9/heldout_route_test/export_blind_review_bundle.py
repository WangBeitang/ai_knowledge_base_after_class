"""导出 9.3.14 round2 的 clean blind review（干净盲审）输入包。

审核包只包含 round1 后实质变更的 5 条 pending heldout 候选、冻结来源、路线规则和
最小泄漏参考，不包含任何 round1 decision、reviewer、审核备注或 heldout 推理结果。
round1 的 20 条 approved case 已由构建脚本按 fingerprint 继续保留，不重复送审。
round2 结束后使用送审时冻结的 queue 重建审核包，不读取已清空的当前 pending queue。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.stage9.balanced_dev.export_blind_review_bundle import (
    export_blind_review_bundle,
)
from evaluation.stage9.heldout_route_test.build_heldout_route_test import (
    CASE_SPECS,
    HYDE_PROBES,
    PROJECT_ROOT,
)


BUNDLE_VERSION = "stage9-heldout-route-test-blind-review-bundle-v2"
DEFAULT_QUEUE = (
    PROJECT_ROOT
    / "evaluation/stage9/artifacts/heldout_route_test/independent_review_round2"
    / "review_round2_clean_queue.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation/stage9/artifacts/heldout_route_test/blind_review_bundle_round2"
)
ROUND1_OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation/stage9/artifacts/heldout_route_test/blind_review_bundle_v1"
)
DEFAULT_LOCAL_SOURCE_MANIFEST = (
    PROJECT_ROOT
    / "evaluation/stage9/configs/heldout_route_test_source_manifest_v1.json"
)
DEFAULT_LOCAL_EVIDENCE_MANIFEST = (
    PROJECT_ROOT
    / "evaluation/stage9/artifacts/heldout_route_test/source_import_manifest.json"
)
DEFAULT_WEB_SOURCE_MANIFEST = (
    PROJECT_ROOT
    / "evaluation/stage9/configs/heldout_route_test_web_source_manifest_v1.json"
)
DEFAULT_WEB_EVIDENCE_MANIFEST = (
    PROJECT_ROOT
    / "evaluation/stage9/artifacts/heldout_route_test/web_evidence_manifest.json"
)


def export_heldout_blind_review_bundle(
    *,
    queue_path: Path = DEFAULT_QUEUE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    overwrite: bool = False,
) -> dict[str, object]:
    """导出并立即执行字段污染、文件集合、SHA256 和 fingerprint 校验。"""

    return export_blind_review_bundle(
        queue_path=queue_path,
        local_source_manifest_path=DEFAULT_LOCAL_SOURCE_MANIFEST,
        local_evidence_manifest_path=DEFAULT_LOCAL_EVIDENCE_MANIFEST,
        web_source_manifest_path=DEFAULT_WEB_SOURCE_MANIFEST,
        web_evidence_manifest_path=DEFAULT_WEB_EVIDENCE_MANIFEST,
        output_dir=output_dir,
        generated_at="2026-07-29T02:40:00+00:00",
        overwrite=overwrite,
        case_specs=CASE_SPECS,
        hyde_probes=HYDE_PROBES,
        expected_route_counts={
            "hyde_fallback": 4,
            "ask_clarification": 1,
        },
        bundle_version=BUNDLE_VERSION,
        bundle_id_prefix="stage9-heldout-route-test-round2-clean",
        task_label="heldout route test round2",
        # round2 已结束；这里从冻结 queue 复建历史包。通用导出器仍默认只允许 pending。
        required_review_status=None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_heldout_blind_review_bundle(
        queue_path=args.queue,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
