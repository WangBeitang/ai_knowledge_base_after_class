"""阶段 9 Planner（规划器）checkpoint（检查点）推理 CLI 入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.query.model_planner.checkpoint_runtime import (  # noqa: E402
    CheckpointManifest,
    PlannerCheckpointRuntime,
    PlannerInferenceResult,
    TrainingBackend,
    load_checkpoint_manifest,
    load_checkpoint_runtime,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段 9 Planner checkpoint 推理。")
    parser.add_argument("--checkpoint", required=True, type=Path, help="checkpoint 目录。")
    parser.add_argument("--context-json", required=True, type=Path, help="PlannerContext 或 input_context JSON 文件。")
    args = parser.parse_args(argv)
    runtime = load_checkpoint_runtime(args.checkpoint)
    context_payload = json.loads(args.context_json.read_text(encoding="utf-8"))
    result = runtime.predict(context_payload)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0 if result.success else 2


__all__ = [
    "CheckpointManifest",
    "PlannerCheckpointRuntime",
    "PlannerInferenceResult",
    "TrainingBackend",
    "load_checkpoint_manifest",
    "load_checkpoint_runtime",
]


if __name__ == "__main__":
    raise SystemExit(main())
