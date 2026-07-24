"""使用真实 Provider（执行器）录制阶段 9.2 Action（动作）观察记录。"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.evaluation.baseline_runner import load_environment_snapshot  # noqa: E402
from app.rag.evaluation.case_schema import load_planner_cases  # noqa: E402
from app.rag.evaluation.offline_environment import OfflineRagEnvironment  # noqa: E402
from app.rag.evaluation.action_providers import MilvusActionProvider, RecordingActionProvider  # noqa: E402
from evaluation.stage9.route_seed.build_route_seed_cases import (  # noqa: E402
    DEFAULT_OUTPUT,
    DEFAULT_PATHS,
    read_route_seed_paths,
)
from evaluation.stage9.route_seed.run_route_seed_paths import DEFAULT_SNAPSHOT  # noqa: E402


DEFAULT_RECORDS = PROJECT_ROOT / "evaluation/stage9/artifacts/provider_records/stage9_provider_observations.jsonl"


def record_provider_observations(
        *,
        cases_path: str | Path,
        paths_path: str | Path,
        snapshot_path: str | Path,
        output_path: str | Path,
        chunk_status_filter_enabled: bool,
) -> Counter[str]:
    """
    按 route seed（路线种子）Action path（动作路径）调用真实 Provider（执行器）并记录观察结果。

    本函数会直接调用 Milvus（向量数据库）、HyDE（假设式改写检索）和 Web（网页检索）
    相关服务；只有在 GPU（显卡算力）服务器或本地真实环境配置完成后才应该运行。
    """
    cases = load_planner_cases(cases_path)
    paths = read_route_seed_paths(paths_path)
    path_by_case = {path.case_id: path for path in paths if path.export_to_sft and path.review_status == "reviewed"}
    snapshot = load_environment_snapshot(snapshot_path)
    provider = RecordingActionProvider(
        MilvusActionProvider(chunk_status_filter_enabled=chunk_status_filter_enabled),
        output_path=output_path,
    )
    environment = OfflineRagEnvironment(
        snapshot=snapshot,
        action_provider=provider,
        planner_mode="real_provider",
        run_id_prefix="stage9_real_provider_record",
    )
    counts: Counter[str] = Counter()
    for case in cases:
        path = path_by_case.get(case.case_id)
        if path is None:
            continue
        trajectory = environment.run_action_path(
            case,
            list(path.action_path),
            run_id=f"stage9_real_provider_record_{case.case_id}_{path.path_id}",
            planner_mode="real_provider",
        )
        counts["case_count"] += 1
        counts[f"status:{trajectory.status.value}"] += 1
        counts[f"route_family:{path.route_family}"] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    counts = record_provider_observations(
        cases_path=args.cases,
        paths_path=args.paths,
        snapshot_path=args.snapshot,
        output_path=args.output,
        chunk_status_filter_enabled=not args.disable_chunk_status_filter,
    )
    print(f"output={args.output}")
    for key, value in sorted(counts.items()):
        print(f"{key}={value}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="录制阶段 9.2 真实 Provider 观察记录。")
    parser.add_argument("--cases", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--paths", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument(
        "--disable-chunk-status-filter",
        action="store_true",
        help="关闭 Mongo 禁用 chunk 覆盖读取，仅用于无 Mongo 的本地烟测。",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
