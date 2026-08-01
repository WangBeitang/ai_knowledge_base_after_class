"""从冻结的 SFT V2 reviewed_75_v1 导出 GRPO train-only case。

本入口不生成 rollout（采样轨迹）、不调用 Provider（动作执行器/环境结果提供器）、
不计算 Reward（奖励分数），也不执行训练。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.rag.evaluation.case_schema import PlannerEvalCase
from app.rag.evaluation.grpo_case_exporter import (
    GRPO_CASE_EXPORT_VERSION,
    GrpoCaseExportManifest,
    build_grpo_training_cases,
    validate_grpo_training_cases,
    write_grpo_training_cases,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SFT_FREEZE_DIR = PROJECT_ROOT / "evaluation/stage9/artifacts/sft_v2/frozen_reviewed_75_v1"
SFT_FREEZE_MANIFEST = SFT_FREEZE_DIR / "sft_v2_freeze_manifest.json"
SFT_TRAJECTORY_INDEX = SFT_FREEZE_DIR / "sft_v2_trajectory_index.jsonl"
SFT_NEW_TRAJECTORIES = SFT_FREEZE_DIR / "sft_v2_approved_new_trajectories.jsonl"
OLD_CURATED_CASES = (
    PROJECT_ROOT
    / "evaluation/stage8_5/artifacts/intermediate/sft_seed/curated_seed_train_cases.jsonl"
)
OLD_ROUTE_CASES = PROJECT_ROOT / "evaluation/stage9/artifacts/route_seed/route_seed_cases.jsonl"
REWARD_PROFILE = PROJECT_ROOT / "evaluation/stage9/configs/reward_v1_1_training_profile.json"
ENVIRONMENT_SNAPSHOT = SFT_FREEZE_DIR / "sft_v2_environment_snapshot.json"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "evaluation/stage9/artifacts/grpo/sft_v2_reviewed_75_v1"
)

DATASET_VERSION = "grpo-cases-from-sft-v2-reviewed-75-v1"
CASE_FILE = "grpo_train_cases.jsonl"
MANIFEST_FILE = "grpo_case_manifest.json"
REPORT_FILE = "grpo_case_report.md"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} 不是 JSON 对象")
            rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _logical_path(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_case_sources(
    trajectory_index: list[dict[str, Any]],
) -> tuple[dict[str, PlannerEvalCase], set[str]]:
    required_ids = {str(row["source_case_id"]) for row in trajectory_index}
    source_rows = [*_read_jsonl(OLD_CURATED_CASES), *_read_jsonl(OLD_ROUTE_CASES)]
    old_cases = {
        row["case_id"]: PlannerEvalCase.model_validate(row)
        for row in source_rows
        if row.get("case_id") in required_ids
    }

    new_trajectory_rows = _read_jsonl(SFT_NEW_TRAJECTORIES)
    new_cases = {
        row["candidate_id"]: PlannerEvalCase.model_validate(row["case_contract"])
        for row in new_trajectory_rows
    }
    overlap = set(old_cases) & set(new_cases)
    if overlap:
        raise ValueError(f"新旧 case_id 重复：{sorted(overlap)}")
    case_sources = {**old_cases, **new_cases}
    if set(case_sources) != required_ids:
        raise ValueError(
            "75 条 case 来源不闭环："
            f"missing={sorted(required_ids - set(case_sources))}, "
            f"extra={sorted(set(case_sources) - required_ids)}"
        )
    return case_sources, set(new_cases)


def _verify_frozen_inputs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(SFT_FREEZE_MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest.get("freeze_version") != "sft-v2-reviewed-75-v1"
        or manifest.get("trajectory_count") != 75
        or manifest.get("action_sample_count") != 163
        or manifest.get("formal_dataset_frozen") is not True
    ):
        raise ValueError("SFT V2 冻结版本身份不符合 GRPO 导出要求")
    for name, metadata in manifest["files"].items():
        path = SFT_FREEZE_DIR / name
        if _sha256(path) != metadata["sha256"] or path.stat().st_size != metadata["bytes"]:
            raise ValueError(f"SFT V2 冻结文件漂移：{name}")
    trajectory_index = _read_jsonl(SFT_TRAJECTORY_INDEX)
    if len(trajectory_index) != 75:
        raise ValueError("SFT V2 冻结轨迹索引不是 75 条")
    return manifest, trajectory_index


def _report(manifest: GrpoCaseExportManifest) -> str:
    return f"""# GRPO train-only case 导出报告

- 数据版本：`{manifest.dataset_version}`。
- 来源：冻结的 `sft-v2-reviewed-75-v1`。
- case（案例）数量：{manifest.case_count}，唯一 query（问题）数量：{manifest.unique_query_count}。
- 全部属于 train-only（仅训练）、全部 reviewed（已审核）。
- 每条参考轨迹均在 `acceptable_action_paths`（可接受动作路线）中。
- 绑定 Reward profile（奖励配置）：`{manifest.reward_profile_path}`。
- 本轮没有生成 rollout（采样轨迹），没有调用 Provider（动作执行器/环境结果提供器），没有执行 GRPO（组相对策略优化）训练。

该文件只为训练阶段提供问题、事实、证据、行为边界和参考轨迹。训练阶段仍须由当前策略实时采样多条轨迹，并在真实或冻结回放环境中评分。
"""


def build_grpo_case_dataset(output_dir: Path = DEFAULT_OUTPUT_DIR) -> GrpoCaseExportManifest:
    if output_dir.exists():
        raise FileExistsError(f"GRPO case 目录已存在，拒绝覆盖：{output_dir}")
    sft_manifest, trajectory_index = _verify_frozen_inputs()
    case_sources, externally_reviewed_ids = _load_case_sources(trajectory_index)
    cases = build_grpo_training_cases(
        trajectory_index=trajectory_index,
        case_sources=case_sources,
        externally_reviewed_case_ids=externally_reviewed_ids,
        dataset_version=DATASET_VERSION,
        source_dataset_fingerprint=sft_manifest["dataset_fingerprint"],
    )
    validation = validate_grpo_training_cases(cases)
    if validation["case_count"] != 75:
        raise ValueError("GRPO case 导出不是 75 条")

    input_paths = (
        SFT_FREEZE_MANIFEST,
        SFT_TRAJECTORY_INDEX,
        SFT_NEW_TRAJECTORIES,
        OLD_CURATED_CASES,
        OLD_ROUTE_CASES,
        REWARD_PROFILE,
        ENVIRONMENT_SNAPSHOT,
    )
    input_hashes = {_logical_path(path): _sha256(path) for path in input_paths}
    created_at = datetime.now(UTC).isoformat(timespec="seconds")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".grpo-case-export-", dir=output_dir.parent) as temp:
        work = Path(temp)
        write_grpo_training_cases(work / CASE_FILE, cases)
        case_hash = _sha256(work / CASE_FILE)
        preliminary = GrpoCaseExportManifest(
            created_at=created_at,
            dataset_version=DATASET_VERSION,
            source_dataset_fingerprint=sft_manifest["dataset_fingerprint"],
            source_sft_freeze_status=sft_manifest["freeze_status"],
            source_sft_legacy_exceptions=sft_manifest["validation"],
            **validation,
            input_file_sha256=input_hashes,
            output_file_sha256={CASE_FILE: case_hash},
            reward_profile_path=_logical_path(REWARD_PROFILE),
            reward_profile_sha256=_sha256(REWARD_PROFILE),
            environment_snapshot_path=_logical_path(ENVIRONMENT_SNAPSHOT),
            environment_snapshot_sha256=_sha256(ENVIRONMENT_SNAPSHOT),
            rollout_source="training_time_policy_sampling",
            rollout_generated=False,
            training_performed=False,
        )
        (work / REPORT_FILE).write_text(_report(preliminary), encoding="utf-8")
        output_hashes = {
            CASE_FILE: case_hash,
            REPORT_FILE: _sha256(work / REPORT_FILE),
        }
        manifest = preliminary.model_copy(update={"output_file_sha256": output_hashes})
        _write_json(work / MANIFEST_FILE, manifest.model_dump(mode="json"))
        work.replace(output_dir)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest = build_grpo_case_dataset(args.output_dir)
    print(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
