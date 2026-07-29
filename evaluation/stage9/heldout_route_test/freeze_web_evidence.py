"""冻结任务 9.3.14 的官方 Web heldout 证据，不运行任何 Planner。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.stage9.balanced_dev.freeze_web_evidence import (
    freeze_web_evidence as freeze_shared_web_evidence,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FREEZE_VERSION = "stage9-heldout-route-test-web-evidence-freeze-v1"
DEFAULT_SOURCE_MANIFEST = (
    PROJECT_ROOT
    / "evaluation/stage9/configs/heldout_route_test_web_source_manifest_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "evaluation/stage9/artifacts/heldout_route_test/web_evidence_manifest.json"
)


def freeze_web_evidence(
    *,
    source_manifest_path: Path = DEFAULT_SOURCE_MANIFEST,
    output_path: Path = DEFAULT_OUTPUT,
    timeout_seconds: float = 30.0,
) -> dict:
    """复用已验证的抓取与短语核验逻辑，只替换 heldout 的路径和版本身份。"""

    return freeze_shared_web_evidence(
        source_manifest_path=source_manifest_path,
        output_path=output_path,
        timeout_seconds=timeout_seconds,
        freeze_version=FREEZE_VERSION,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = freeze_web_evidence(
        source_manifest_path=args.source_manifest,
        output_path=args.output,
        timeout_seconds=args.timeout_seconds,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output),
                "source_count": result["source_count"],
                "captured_at": result["captured_at"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
