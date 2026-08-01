"""合并并冻结当前已审核的 75 条 SFT V2（监督微调第二版）轨迹。

本入口只接纳两类来源：37 条旧 reviewed（已审核）保留轨迹，以及 round3
（第三轮审核）明确 approve（批准）的 38 条新轨迹。它不会读取抢救候选作为训练
输入，也不会运行训练。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from app.rag.evaluation.sft_exporter import SftArtifactStatus, SftPlannerSample
from evaluation.stage9.sft_v2.build_sft_v2_candidates import (
    CandidateTrajectory,
    _char_ngrams,
    _jaccard,
    _nontrain_queries_and_chunks,
    _normalized_text,
)
from evaluation.stage9.sft_v2.repair_sft_v2_candidates import (
    SOURCE_CASES,
    SOURCE_DECISIONS,
    SOURCE_POOL,
    SOURCE_RECORDS,
    SOURCE_TRAJECTORIES,
    SOURCE_WEB,
    _original_fingerprint,
    audit_round3_lock,
    _raw_pool_parts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = PROJECT_ROOT / "evaluation/stage9/artifacts/sft_v2"
DEFAULT_OUTPUT_DIR = ARTIFACT_DIR / "frozen_reviewed_75_v1"
SOURCE_SNAPSHOT = ARTIFACT_DIR / "sft_v2_environment_snapshot.json"
SALVAGE_DECISIONS = ARTIFACT_DIR / "independent_review_salvage_v1/review_decisions.jsonl"
FREEZE_VERSION = "sft-v2-reviewed-75-v1"

TRAIN_FILE = "sft_v2_train.jsonl"
TRAJECTORY_INDEX_FILE = "sft_v2_trajectory_index.jsonl"
NEW_TRAJECTORIES_FILE = "sft_v2_approved_new_trajectories.jsonl"
PROVIDER_FILE = "sft_v2_provider_observations.jsonl"
WEB_FILE = "sft_v2_web_evidence_manifest.json"
SNAPSHOT_FILE = "sft_v2_environment_snapshot.json"
REPORT_FILE = "sft_v2_freeze_report.md"
MANIFEST_FILE = "sft_v2_freeze_manifest.json"

EXPECTED_OLD_TRAJECTORIES = 37
EXPECTED_NEW_TRAJECTORIES = 38
EXPECTED_TOTAL_TRAJECTORIES = 75
EXPECTED_OLD_SAMPLES = 84
EXPECTED_NEW_SAMPLES = 79
EXPECTED_TOTAL_SAMPLES = 163

# 这 37 条旧轨迹已被用户明确要求保持不变。冻结时仍报告旧数据里按当前算法命中的
# 历史近义模板和元问题，不把它们伪装成 0；只允许这里列出的已知旧数据例外。
ALLOWED_LEGACY_NEAR_DUPLICATES = {
    frozenset({"stage85-gold-ai4i-osf-rule-002", "stage9-route-hyde-answer-008"}),
    frozenset({"stage85-gold-ai4i-rnf-rule-002", "stage9-route-hyde-answer-010"}),
    frozenset({
        "stage85-gold-hydraulic-cooler-profile-002",
        "stage85-gold-hydraulic-valve-profile-002",
    }),
}
ALLOWED_LEGACY_META_QUERY_IDS = {"stage9-route-multi-fallback-001"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} 不是 JSON 对象")
            rows.append(value)
    return rows


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _logical_path(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def _source_hashes() -> dict[str, str]:
    paths = (
        SOURCE_POOL,
        SOURCE_CASES,
        SOURCE_TRAJECTORIES,
        SOURCE_RECORDS,
        SOURCE_WEB,
        SOURCE_SNAPSHOT,
        SOURCE_DECISIONS,
        SALVAGE_DECISIONS,
    )
    return {_logical_path(path): _sha256_file(path) for path in paths}


def _approved_new_samples(
    approved_ids: set[str],
    approved_fingerprints: dict[str, str],
) -> tuple[bytes, list[SftPlannerSample], list[SftPlannerSample]]:
    old_raw, approved_lines = _raw_pool_parts(approved_ids)
    old_samples = [
        SftPlannerSample.model_validate(json.loads(line))
        for line in old_raw.splitlines()
        if line.strip()
    ]
    if len(old_samples) != EXPECTED_OLD_SAMPLES:
        raise ValueError("旧保留动作样本不是 84 条")
    if any(
        sample.review_status != "reviewed"
        or sample.artifact_status != SftArtifactStatus.APPROVED_TRAINING_SEED
        for sample in old_samples
    ):
        raise ValueError("旧保留动作样本存在未审核或非正式训练状态")

    new_samples: list[SftPlannerSample] = []
    for case_id in sorted(approved_ids):
        fingerprint = approved_fingerprints[case_id]
        for line in approved_lines[case_id]:
            raw = json.loads(line)
            reward_summary = {
                "content_fingerprint": fingerprint,
                "evaluated": False,
                "generation_gate_passed": True,
                "review_decision": "approve",
                "review_round": "round3",
                "reward_version": "not_scored-independent-review-selection",
                "selected_by": "independent_review_round3",
            }
            raw.update({
                "artifact_status": SftArtifactStatus.APPROVED_TRAINING_SEED.value,
                "review_status": "reviewed",
                "reward_summary": reward_summary,
            })
            new_samples.append(SftPlannerSample.model_validate(raw))
    if len(new_samples) != EXPECTED_NEW_SAMPLES:
        raise ValueError("38 条新通过轨迹没有完整导出为 79 条动作样本")
    return bytes(old_raw), old_samples, new_samples


def _group_samples(
    samples: list[SftPlannerSample],
) -> dict[str, list[SftPlannerSample]]:
    grouped: dict[str, list[SftPlannerSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.source_trace_id].append(sample)
    for trace_id, rows in grouped.items():
        rows.sort(key=lambda row: row.turn_index)
        if [row.turn_index for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError(f"轨迹 turn_index 不连续：{trace_id}")
        if len({row.sample_id for row in rows}) != len(rows):
            raise ValueError(f"轨迹 sample_id 重复：{trace_id}")
    return grouped


def _trajectory_index(
    old_samples: list[SftPlannerSample],
    new_samples: list[SftPlannerSample],
    new_trajectories: dict[str, CandidateTrajectory],
    decisions_sha256: str,
) -> list[dict[str, Any]]:
    old_groups = _group_samples(old_samples)
    new_groups = _group_samples(new_samples)
    if len(old_groups) != EXPECTED_OLD_TRAJECTORIES:
        raise ValueError("旧保留来源轨迹不是 37 条")
    if len(new_groups) != EXPECTED_NEW_TRAJECTORIES:
        raise ValueError("新通过来源轨迹不是 38 条")

    rows: list[dict[str, Any]] = []
    for trace_id, samples in sorted(old_groups.items()):
        payload = [sample.model_dump(mode="json") for sample in samples]
        rows.append({
            "action_sample_count": len(samples),
            "content_fingerprint": _sha256_bytes(_canonical_bytes(payload)),
            "device_family": None,
            "leakage_group_id": None,
            "legacy_metadata_missing": True,
            "origin": "old_retained_reviewed",
            "query": samples[0].input_context["query"],
            "question_family": None,
            "review_decision": "approved_training_seed",
            "review_evidence": "embedded_reviewed_status",
            "route": [sample.target_decision["action"] for sample in samples],
            "sample_ids": [sample.sample_id for sample in samples],
            "source_case_id": samples[0].source_case_id,
            "source_trace_id": trace_id,
            "split": samples[0].split.value,
        })

    for case_id, trajectory in sorted(new_trajectories.items()):
        samples = new_groups.get(trajectory.source_trace_id)
        if not samples:
            raise ValueError(f"新通过轨迹缺少动作样本：{case_id}")
        route = [sample.target_decision["action"] for sample in samples]
        if route != trajectory.route:
            raise ValueError(f"新通过轨迹与动作样本路线不一致：{case_id}")
        rows.append({
            "action_sample_count": len(samples),
            "content_fingerprint": trajectory.content_fingerprint,
            "device_family": trajectory.device_family,
            "leakage_group_id": trajectory.leakage_group_id,
            "legacy_metadata_missing": False,
            "origin": "round3_approved",
            "provider_record_ids": trajectory.provider_record_ids,
            "query": trajectory.query,
            "question_family": trajectory.question_family,
            "review_decision": "approve",
            "review_evidence": _logical_path(SOURCE_DECISIONS),
            "review_file_sha256": decisions_sha256,
            "route": route,
            "sample_ids": [sample.sample_id for sample in samples],
            "source_case_id": case_id,
            "source_evidence_ids": [item.source_id for item in trajectory.source_evidence],
            "source_trace_id": trajectory.source_trace_id,
            "split": trajectory.split,
        })
    if len(rows) != EXPECTED_TOTAL_TRAJECTORIES:
        raise ValueError("冻结轨迹索引不是 75 条")
    return rows


def _duplicate_and_leakage_report(
    trajectory_index: list[dict[str, Any]],
    new_trajectories: list[CandidateTrajectory],
) -> dict[str, Any]:
    exact_seen: dict[str, str] = {}
    exact_duplicates: list[list[str]] = []
    for row in trajectory_index:
        normalized = _normalized_text(row["query"])
        if normalized in exact_seen:
            exact_duplicates.append([exact_seen[normalized], row["source_case_id"]])
        exact_seen[normalized] = row["source_case_id"]

    near_duplicates: list[dict[str, Any]] = []
    for left_index, left in enumerate(trajectory_index):
        for right in trajectory_index[left_index + 1:]:
            jaccard_score = _jaccard(
                _char_ngrams(left["query"]),
                _char_ngrams(right["query"]),
            )
            sequence_score = SequenceMatcher(
                None,
                left["query"],
                right["query"],
            ).ratio()
            score = max(jaccard_score, sequence_score)
            if score >= 0.82:
                near_duplicates.append({
                    "case_ids": [left["source_case_id"], right["source_case_id"]],
                    "jaccard_score": round(jaccard_score, 4),
                    "score": round(score, 4),
                    "sequence_score": round(sequence_score, 4),
                })

    actual_legacy_pairs = {
        frozenset(item["case_ids"])
        for item in near_duplicates
        if all(
            next(row for row in trajectory_index if row["source_case_id"] == case_id)["origin"]
            == "old_retained_reviewed"
            for case_id in item["case_ids"]
        )
    }
    unexpected_near = [
        item for item in near_duplicates
        if frozenset(item["case_ids"]) not in ALLOWED_LEGACY_NEAR_DUPLICATES
    ]
    if actual_legacy_pairs != ALLOWED_LEGACY_NEAR_DUPLICATES or unexpected_near:
        raise ValueError(f"近义重复集合发生变化：unexpected={unexpected_near}")

    banned_meta = ("本地检索", "目标路线", "hyde_search", "web_search", "action（动作）")
    meta_query_ids = {
        row["source_case_id"]
        for row in trajectory_index
        if any(token.lower() in row["query"].lower() for token in banned_meta)
    }
    if meta_query_ids != ALLOWED_LEGACY_META_QUERY_IDS:
        raise ValueError(f"元问题集合发生变化：{sorted(meta_query_ids)}")

    nontrain_queries, nontrain_chunks = _nontrain_queries_and_chunks()
    split_query_leaks: list[dict[str, Any]] = []
    for row in trajectory_index:
        query_grams = _char_ngrams(row["query"])
        for nontrain_case_id, nontrain_query in nontrain_queries:
            score = _jaccard(query_grams, _char_ngrams(nontrain_query))
            if (
                _normalized_text(row["query"]) == _normalized_text(nontrain_query)
                or score >= 0.78
            ):
                split_query_leaks.append({
                    "source_case_id": row["source_case_id"],
                    "nontrain_case_id": nontrain_case_id,
                    "score": round(score, 4),
                })
    split_chunk_leaks = sorted({
        str(evidence.chunk_id)
        for trajectory in new_trajectories
        for evidence in trajectory.source_evidence
        if evidence.source_type == "local"
        and evidence.chunk_id is not None
        and str(evidence.chunk_id) in nontrain_chunks
    })
    if exact_duplicates or split_query_leaks or split_chunk_leaks:
        raise ValueError(
            "冻结数据存在完全重复或 split 泄漏："
            f"exact={exact_duplicates}, query={split_query_leaks}, chunk={split_chunk_leaks}"
        )
    return {
        "exact_duplicate_count": 0,
        "legacy_meta_query_count": len(meta_query_ids),
        "legacy_meta_query_ids": sorted(meta_query_ids),
        "legacy_near_duplicate_count": len(near_duplicates),
        "legacy_near_duplicates": near_duplicates,
        "new_or_cross_generation_near_duplicate_count": 0,
        "split_chunk_leak_count": 0,
        "split_query_leak_count": 0,
    }


def _selected_provider_records(
    trajectories: list[CandidateTrajectory],
) -> list[dict[str, Any]]:
    required_ids = {
        record_id for trajectory in trajectories for record_id in trajectory.provider_record_ids
    }
    source_rows = _read_jsonl(SOURCE_RECORDS)
    source_by_id = {row["record_id"]: row for row in source_rows}
    if len(source_by_id) != len(source_rows):
        raise ValueError("Provider（动作执行器）记录 ID 不唯一")
    if not required_ids.issubset(source_by_id):
        raise ValueError(f"缺少 Provider（动作执行器）记录：{sorted(required_ids - source_by_id)}")
    selected = [source_by_id[record_id] for record_id in sorted(required_ids)]
    approved_ids = {trajectory.candidate_id for trajectory in trajectories}
    if any(row["case_id"] not in approved_ids for row in selected):
        raise ValueError("Provider（动作执行器）记录引用未批准 case")
    return selected


def _selected_web_manifest(trajectories: list[CandidateTrajectory]) -> dict[str, Any]:
    source = json.loads(SOURCE_WEB.read_text(encoding="utf-8"))
    web_case_ids = {
        trajectory.candidate_id
        for trajectory in trajectories
        if any(item.source_type == "web" for item in trajectory.source_evidence)
    }
    candidate_facts = {
        case_id: source["candidate_facts"][case_id]
        for case_id in sorted(web_case_ids)
    }
    used_source_ids = {value["source_id"] for value in candidate_facts.values()}
    captured_sources = [
        row for row in source["captured_sources"] if row["source_id"] in used_source_ids
    ]
    if {row["source_id"] for row in captured_sources} != used_source_ids:
        raise ValueError("冻结网页来源与候选事实不闭环")
    return {
        "candidate_facts": candidate_facts,
        "captured_sources": captured_sources,
        "freeze_version": FREEZE_VERSION,
        "generation_batch": source["generation_batch"],
        "manifest_version": "sft-v2-frozen-web-evidence-v1",
    }


def _route_counts(index_rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(" -> ".join(row["route"]) for row in index_rows).items()))


def _report_markdown(manifest: dict[str, Any]) -> str:
    validation = manifest["validation"]
    return f"""# SFT V2 reviewed_75_v1 冻结报告

## 冻结结论

- 冻结状态：`frozen_with_legacy_exceptions`（带旧数据例外冻结）。
- 正式训练轨迹：{manifest['trajectory_count']} 条，其中旧保留 {manifest['old_trajectory_count']} 条、round3（第三轮审核）通过 {manifest['round3_approved_trajectory_count']} 条。
- 逐步 `Action`（动作）样本：{manifest['action_sample_count']} 条。
- round3（第三轮审核）拒绝 87 条、抢救二审拒绝 8 条均未进入训练文件。
- 本轮没有运行训练。

## 口径调整

原方案目标是 150 条轨迹和 17 条完整路线满配。用户已于 2026-08-01 明确停止扩充，
因此本版本只冻结现有 75 条已审核轨迹，不声明完成原 150 条配额。

## 验证结果

- `Schema`（数据结构约束）：通过；163 条均可按 `SftPlannerSample`（监督微调规划器样本）读取，且全部为 `reviewed/approved_training_seed`（已审核/正式训练种子）。
- 审核绑定：37 条继承旧正式训练种子状态；38 条逐项绑定 round3（第三轮审核）批准决定和原内容指纹。
- `Provider`（动作执行器/环境结果提供器）记录：{manifest['provider_observation_count']} 条，与38条新通过轨迹引用闭环。
- 完全重复：0；新数据或跨代近义重复：0。
- `split`（数据切分）query 泄漏：0；chunk 泄漏：0。
- 旧数据例外：当前算法命中 {validation['legacy_near_duplicate_count']} 组历史近义模板，另有 {validation['legacy_meta_query_count']} 条历史路线元问题。按“不修改37条旧保留轨迹”的既定边界保留，并在 manifest 中显式列出。

## 边界

该冻结只证明数据身份、审核决定、动作路径、证据记录和哈希可复现；不证明模型训练效果。
未经重新生成新版本，不得静默修改本目录文件。
"""


def build_freeze(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"冻结目录已存在，拒绝覆盖：{output_dir}")
    lock = audit_round3_lock()
    approved_ids = set(lock["approved_ids"])
    approved_fingerprints = dict(lock["approved_fingerprints"])

    decisions = _read_jsonl(SOURCE_DECISIONS)
    if Counter(row["decision"] for row in decisions) != Counter({"approve": 38, "reject": 87}):
        raise ValueError("round3（第三轮审核）决定不是 38/87")
    salvage_decisions = _read_jsonl(SALVAGE_DECISIONS)
    if len(salvage_decisions) != 8 or any(row["decision"] != "reject" for row in salvage_decisions):
        raise ValueError("抢救二审结果不是 8 条全部拒绝")

    source_trajectory_rows = _read_jsonl(SOURCE_TRAJECTORIES)
    approved_source_rows = {
        row["candidate_id"]: row
        for row in source_trajectory_rows
        if row["candidate_id"] in approved_ids
    }
    source_trajectories = {
        row["candidate_id"]: CandidateTrajectory.model_validate(row)
        for row in approved_source_rows.values()
    }
    if set(source_trajectories) != approved_ids:
        raise ValueError("38 条批准轨迹身份不闭环")
    for case_id, trajectory in source_trajectories.items():
        if (
            trajectory.content_fingerprint != approved_fingerprints[case_id]
            or _original_fingerprint(approved_source_rows[case_id])
            != approved_fingerprints[case_id]
        ):
            raise ValueError(f"批准轨迹内容指纹漂移：{case_id}")

    old_raw, old_samples, new_samples = _approved_new_samples(
        approved_ids,
        approved_fingerprints,
    )
    all_samples = [*old_samples, *new_samples]
    if len(all_samples) != EXPECTED_TOTAL_SAMPLES:
        raise ValueError("冻结动作样本不是 163 条")
    if len({sample.sample_id for sample in all_samples}) != len(all_samples):
        raise ValueError("冻结动作样本 sample_id 不唯一")
    if len({sample.source_trace_id for sample in all_samples}) != EXPECTED_TOTAL_TRAJECTORIES:
        raise ValueError("冻结动作样本来源轨迹不是 75 条")
    if any(sample.split.value != "train" for sample in all_samples):
        raise ValueError("冻结数据包含非 train split")

    decisions_sha256 = _sha256_file(SOURCE_DECISIONS)
    index_rows = _trajectory_index(
        old_samples,
        new_samples,
        source_trajectories,
        decisions_sha256,
    )
    validation = _duplicate_and_leakage_report(
        index_rows,
        list(source_trajectories.values()),
    )
    provider_rows = _selected_provider_records(list(source_trajectories.values()))
    web_manifest = _selected_web_manifest(list(source_trajectories.values()))

    source_hashes = _source_hashes()
    created_at = datetime.now(UTC).isoformat(timespec="seconds")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sft-v2-freeze-", dir=output_dir.parent) as temp:
        work = Path(temp)
        # 旧 84 行保持源文件字节不变；新 79 行写成已审核正式训练状态。
        new_payload = b"".join(
            json.dumps(sample.model_dump(mode="json"), ensure_ascii=False, sort_keys=True).encode("utf-8")
            + b"\n"
            for sample in new_samples
        )
        (work / TRAIN_FILE).write_bytes(old_raw + new_payload)
        _write_jsonl(work / TRAJECTORY_INDEX_FILE, index_rows)
        _write_jsonl(
            work / NEW_TRAJECTORIES_FILE,
            [
                source_trajectories[case_id].model_dump(mode="json")
                for case_id in sorted(source_trajectories)
            ],
        )
        _write_jsonl(work / PROVIDER_FILE, provider_rows)
        _write_json(work / WEB_FILE, web_manifest)
        (work / SNAPSHOT_FILE).write_bytes(SOURCE_SNAPSHOT.read_bytes())

        manifest: dict[str, Any] = {
            "action_sample_count": EXPECTED_TOTAL_SAMPLES,
            "action_sample_count_new": EXPECTED_NEW_SAMPLES,
            "action_sample_count_old": EXPECTED_OLD_SAMPLES,
            "all_samples_approved_training_seed": True,
            "all_samples_reviewed": True,
            "created_at": created_at,
            "dataset_scope": "37 old retained + 38 round3 approved",
            "device_family_counts_new": dict(sorted(Counter(
                row.device_family for row in source_trajectories.values()
            ).items())),
            "excluded": {
                "round3_rejected_original_candidate_count": 87,
                "salvage_review_rejected_candidate_count": 8,
                "salvage_review_rejected_ids": sorted(row["candidate_id"] for row in salvage_decisions),
            },
            "formal_dataset_frozen": True,
            "freeze_status": "frozen_with_legacy_exceptions",
            "freeze_version": FREEZE_VERSION,
            "legacy_target": {
                "original_route_count": 17,
                "original_trajectory_target": 150,
                "status": "superseded_for_this_version_by_user_stop_expansion_decision",
            },
            "old_trajectory_count": EXPECTED_OLD_TRAJECTORIES,
            "provider_observation_count": len(provider_rows),
            "question_family_counts_new": dict(sorted(Counter(
                row.question_family for row in source_trajectories.values()
            ).items())),
            "round3_approved_trajectory_count": EXPECTED_NEW_TRAJECTORIES,
            "route_counts": _route_counts(index_rows),
            "source_file_sha256": source_hashes,
            "split_counts": {"train": EXPECTED_TOTAL_TRAJECTORIES},
            "training_performed": False,
            "trajectory_count": EXPECTED_TOTAL_TRAJECTORIES,
            "unique_query_count": len({_normalized_text(row["query"]) for row in index_rows}),
            "validation": validation,
            "web_evidence_case_count": len(web_manifest["candidate_facts"]),
        }
        (work / REPORT_FILE).write_text(_report_markdown(manifest), encoding="utf-8")
        frozen_files = [
            TRAIN_FILE,
            TRAJECTORY_INDEX_FILE,
            NEW_TRAJECTORIES_FILE,
            PROVIDER_FILE,
            WEB_FILE,
            SNAPSHOT_FILE,
            REPORT_FILE,
        ]
        manifest["files"] = {
            name: {
                "bytes": (work / name).stat().st_size,
                "sha256": _sha256_file(work / name),
            }
            for name in frozen_files
        }
        manifest["dataset_fingerprint"] = _sha256_bytes(_canonical_bytes({
            "files": manifest["files"],
            "freeze_version": FREEZE_VERSION,
            "trajectory_fingerprints": [
                row["content_fingerprint"] for row in index_rows
            ],
        }))
        _write_json(work / MANIFEST_FILE, manifest)
        work.replace(output_dir)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    result = build_freeze(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
