"""按三审 round3 结果保留 38 条通过项，并在原路线槽位重做 87 条失败候选。"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from app.rag.evaluation.action_providers import read_provider_observation_records
from app.rag.evaluation.case_schema import PlannerEvalCase
from app.rag.evaluation.sft_exporter import SftPlannerSample
from app.rag.query.contracts import QueryAction
from evaluation.stage8.build_environment_snapshot import write_environment_snapshot
from evaluation.stage9.sft_v2.build_sft_v2_candidates import (
    BATCH_ID,
    BATCH_VERSION,
    LOCAL_SOURCES,
    NEW_TRAJECTORY_COUNT,
    QUESTION_FAMILY_BY_DEVICE,
    WEB_SOURCES,
    CandidateDraftValidationError,
    CandidateSeed,
    CandidateTrajectory,
    SourceEvidence,
    _read_jsonl,
    _sha256_bytes,
    _stable_hash,
    assign_local_facts,
    assess_route_admission,
    bind_observed_source_evidence,
    build_allocations,
    build_cases,
    build_current_snapshot,
    build_outputs,
    capture_web_sources,
    draft_candidate_seeds,
    execute_trajectories,
    route_quota,
    validate_candidate_outputs,
    validate_generation,
    validate_pre_provider_profiles,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = PROJECT_ROOT / "evaluation/stage9/artifacts/sft_v2"
ROUND3_DIR = ARTIFACT_DIR / "independent_review_round3"
DEFAULT_OUTPUT_DIR = ARTIFACT_DIR / "repair_v2"
DEFAULT_CACHE_DIR = ARTIFACT_DIR / "repair_v2_dynamic_work_v3"
REPAIR_VERSION = "sft-v2-round3-repair-v2"

SOURCE_CASES = ARTIFACT_DIR / "sft_v2_new_candidate_cases.jsonl"
SOURCE_TRAJECTORIES = ARTIFACT_DIR / "sft_v2_new_candidate_trajectories.jsonl"
SOURCE_RECORDS = ARTIFACT_DIR / "sft_v2_provider_observations.jsonl"
SOURCE_WEB = ARTIFACT_DIR / "sft_v2_web_evidence_manifest.json"
SOURCE_POOL = ARTIFACT_DIR / "sft_v2_train_candidates.jsonl"
SOURCE_DECISIONS = ROUND3_DIR / "review_decisions.jsonl"
SOURCE_FIRST_REPAIR_DRAFTS = ARTIFACT_DIR / "repair_v2_work/sft_v2_question_drafts.jsonl"
SOURCE_FIRST_REPAIR_REPORT = ARTIFACT_DIR / "repair_v2_work/sft_v2_gate_failure_20260801.md"

FINAL_FILES = {
    "sft_v2_new_candidate_cases.jsonl",
    "sft_v2_new_candidate_trajectories.jsonl",
    "sft_v2_provider_observations.jsonl",
    "sft_v2_web_evidence_manifest.json",
    "sft_v2_environment_snapshot.json",
    "sft_v2_train_candidates.jsonl",
    "sft_v2_repair_manifest.json",
    "sft_v2_question_drafts.jsonl",
}

ASK_WEB_SOURCE_IDS = {
    "siemens-sinumerik-808-current",
    "rockwell-firmware-lifecycle-current",
    "abb-powertrain-lifecycle-current",
    "abb-legacy-servo-current",
}
REFUSE_WEB_SOURCE_IDS = {
    "siemens-sinumerik-808-current",
    "rockwell-firmware-lifecycle-current",
    "abb-powertrain-lifecycle-current",
    "osha-machine-guarding-current",
    "osha-1910-212-current",
    "nist-sp800-82r3-current",
    "nist-manufacturing-cybersecurity-current",
    "nist-ir8546-semiconductor-profile",
    "cisa-ics-advisories-2025-07-10",
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row.model_dump(mode="json") if hasattr(row, "model_dump") else row,
                ensure_ascii=False,
                sort_keys=True,
            ) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _original_fingerprint(row: dict[str, Any]) -> str:
    payload = {
        "case": row["case_contract"],
        "route": row["route"],
        "source_evidence": row["source_evidence"],
        "trace_steps": [{**step, "duration_ms": 0} for step in row["trace_steps"]],
        "generation_batch": row["generation_batch"],
    }
    return _stable_hash(payload)


def audit_round3_lock() -> dict[str, Any]:
    decisions = _read_jsonl(SOURCE_DECISIONS)
    trajectories = {row["candidate_id"]: row for row in _read_jsonl(SOURCE_TRAJECTORIES)}
    if len(decisions) != NEW_TRAJECTORY_COUNT or len(trajectories) != NEW_TRAJECTORY_COUNT:
        raise ValueError("三审决定或原候选不是 125 条")
    approved = {row["case_id"]: row for row in decisions if row["decision"] == "approve"}
    rejected = {row["case_id"]: row for row in decisions if row["decision"] == "reject"}
    if len(approved) != 38 or len(rejected) != 87 or set(approved) | set(rejected) != set(trajectories):
        raise ValueError("三审 38/87 身份集合不闭环")
    mismatches = []
    for case_id, decision in approved.items():
        trajectory = trajectories[case_id]
        if (
            trajectory["content_fingerprint"] != decision["content_fingerprint"]
            or _original_fingerprint(trajectory) != decision["content_fingerprint"]
        ):
            mismatches.append(case_id)
    if mismatches:
        raise ValueError(f"38 条通过项指纹漂移：{mismatches}")
    return {
        "approved_ids": sorted(approved),
        "rejected_ids": sorted(rejected),
        "approved_fingerprints": {
            case_id: approved[case_id]["content_fingerprint"] for case_id in sorted(approved)
        },
        "source_file_sha256": {
            path.name: _sha256_bytes(path.read_bytes())
            for path in (
                SOURCE_CASES, SOURCE_TRAJECTORIES, SOURCE_RECORDS, SOURCE_WEB,
                SOURCE_POOL, SOURCE_DECISIONS,
            )
        },
    }


def first_repair_partition(rejected_ids: set[str]) -> tuple[set[str], set[str]]:
    """从第一次真实执行报告锁定 44 条复用草稿和 43 条重新采样项。"""

    report = SOURCE_FIRST_REPAIR_REPORT.read_text(encoding="utf-8")
    failure_table = report.split("## 失败统计", 1)[1].split("## 停止原因", 1)[0]
    failed_ids = {
        f"sft-v2-new-{number}"
        for number in re.findall(r"\b\d{3}\b", failure_table)
    }
    reuse_ids = rejected_ids - failed_ids
    if len(failed_ids) != 43 or len(reuse_ids) != 44 or failed_ids | reuse_ids != rejected_ids:
        raise ValueError("第一次修复的 44/43 身份集合不闭环")

    drafts = {
        row["candidate_id"]: row for row in _read_jsonl(SOURCE_FIRST_REPAIR_DRAFTS)
    }
    if set(drafts) != rejected_ids or not reuse_ids.issubset(drafts):
        raise ValueError("第一次修复草稿无法完整恢复 44 条门禁通过项")
    return reuse_ids, failed_ids


def prepare_selective_draft_cache(
    cache_path: Path,
    *,
    reuse_ids: set[str],
) -> dict[str, dict[str, Any]]:
    """新缓存只继承 44 条草稿；43 条故意缺席以触发重新采样。"""

    source = {
        row["candidate_id"]: row for row in _read_jsonl(SOURCE_FIRST_REPAIR_DRAFTS)
    }
    promotion_path = cache_path.parent / "sft_v2_promoted_reuse_ids.json"
    promoted = set(
        (_json(promotion_path).get("candidate_ids") or [])
        if promotion_path.exists() else []
    )
    if cache_path.exists():
        cached = {row["candidate_id"]: row for row in _read_jsonl(cache_path)}
        for case_id in reuse_ids - promoted:
            for field in ("query", "trigger", "answer_points"):
                if cached[case_id].get(field) != source[case_id].get(field):
                    raise ValueError(f"44 条复用草稿发生改写：{case_id}.{field}")
        return cached
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    inherited = {case_id: source[case_id] for case_id in sorted(reuse_ids)}
    _write_jsonl(cache_path, inherited.values())
    return inherited


def _drop_cached_drafts(cache_path: Path, candidate_ids: set[str]) -> None:
    rows = [
        row for row in _read_jsonl(cache_path)
        if row["candidate_id"] not in candidate_ids
    ]
    _write_jsonl(cache_path, rows)


def draft_prevalidated_seeds(
    allocations: list[dict[str, Any]],
    *,
    local_facts: dict[str, dict[str, Any]],
    web_facts: dict[str, dict[str, Any]],
    drafts_path: Path,
    reuse_draft_ids: set[str],
    resample_ids: set[str],
    draft_batch_size: int,
) -> tuple[list[CandidateSeed], set[str], dict[str, int]]:
    """先闭环全部画像；失败项局部重采，44 条也不享受语义豁免。"""

    promotion_path = drafts_path.parent / "sft_v2_promoted_reuse_ids.json"
    promoted_ids = set(
        (_json(promotion_path).get("candidate_ids") or [])
        if promotion_path.exists() else []
    )
    immutable_ids = set(reuse_draft_ids) - promoted_ids
    attempts: Counter[str] = Counter()
    disabled_forced_ids = set(resample_ids)
    while True:
        failed_ids: set[str] = set()
        failure_reason = ""
        try:
            seeds = draft_candidate_seeds(
                allocations,
                local_facts=local_facts,
                web_facts=web_facts,
                batch_size=draft_batch_size,
                cache_path=drafts_path,
                immutable_draft_ids=immutable_ids,
                disabled_forced_draft_ids=disabled_forced_ids,
            )
        except CandidateDraftValidationError as exc:
            failed_ids = {exc.candidate_id}
            failure_reason = exc.reason
        except ValueError as exc:
            match = re.match(r"(sft-v2-new-\d{3})\b", str(exc))
            if match is None:
                raise
            failed_ids = {match.group(1)}
            failure_reason = str(exc)
        else:
            profile_failures = validate_pre_provider_profiles(seeds)
            if not profile_failures:
                return seeds, promoted_ids, dict(sorted(attempts.items()))
            failed_ids = set(profile_failures)
            failure_reason = json.dumps(profile_failures, ensure_ascii=False, sort_keys=True)

        for case_id in failed_ids:
            attempts[case_id] += 1
            if attempts[case_id] > 4:
                raise RuntimeError(
                    f"{case_id} 连续 4 次无法通过执行前画像门禁：{failure_reason}"
                )
            if case_id in immutable_ids:
                immutable_ids.remove(case_id)
                promoted_ids.add(case_id)
                _write_json(promotion_path, {
                    "promotion_reason": "pre_provider_semantic_or_profile_gate_failed",
                    "candidate_ids": sorted(promoted_ids),
                })
            disabled_forced_ids.add(case_id)
        _drop_cached_drafts(drafts_path, failed_ids)


def _probe_seed_by_category(seeds: list[CandidateSeed]) -> dict[str, CandidateSeed]:
    def pick(category: str, predicate) -> CandidateSeed:
        matched = [seed for seed in seeds if predicate(seed)]
        if not matched:
            raise RuntimeError(f"没有问题画像自然满足 {category} 探针条件的候选")
        return matched[0]

    return {
        "hyde": pick("HyDE", lambda seed: bool(
            seed.retrieval_subject_id
            and seed.answer_points
            and seed.question_profile.pre_search_terminal is None
            and seed.question_profile.terminology_gap
        )),
        "clarification": pick("检索后追问", lambda seed: bool(
            not seed.answer_points
            and seed.question_profile.pre_search_terminal is None
            and seed.question_profile.branch_selector
            and seed.question_profile.answer_changes_by_branch
            and (
                seed.retrieval_subject_id
                or any(item.source_type == "web" for item in seed.source_evidence)
            )
        )),
        "refusal": pick("检索后拒答", lambda seed: bool(
            seed.retrieval_subject_id
            and not seed.answer_points
            and seed.question_profile.pre_search_terminal is None
            and seed.question_profile.post_search_boundary
        )),
        "web": pick("网页答案", lambda seed: bool(
            seed.answer_points
            and seed.question_profile.pre_search_terminal is None
            and any(item.source_type == "web" for item in seed.source_evidence)
        )),
    }


def run_four_route_probes(
    seeds: list[CandidateSeed],
    *,
    snapshot: Any,
    cache_dir: Path,
) -> dict[str, Any]:
    """四类探针均按预先冻结的 Observation 门禁判定，不看采样标签放行。"""

    selected = _probe_seed_by_category(seeds)
    probe_seeds = list(selected.values())
    probe_cases = build_cases(probe_seeds)
    probe_index = 1
    records_path = cache_dir / f"sft_v2_four_route_probe_observations_{probe_index:02d}.jsonl"
    while records_path.exists():
        probe_index += 1
        records_path = cache_dir / f"sft_v2_four_route_probe_observations_{probe_index:02d}.jsonl"
    runs, records = execute_trajectories(
        probe_seeds,
        probe_cases,
        snapshot=snapshot,
        provider_records_path=records_path,
    )
    bound_seeds = bind_observed_source_evidence(probe_seeds, records)
    gates = validate_generation(
        bound_seeds, probe_cases, runs, records, raise_on_failure=False,
    )
    standards = {
        "hyde": ("hyde_target_improved",),
        "clarification": (
            "clarification_observation_grounded",
            "clarification_branch_profile_valid",
        ),
        "refusal": (
            "refusal_observation_grounded",
            "refusal_boundary_profile_valid",
        ),
        "web": (
            "web_evidence_bound",
            "web_fact_complete",
            "web_answer_coverage",
        ),
    }
    results: dict[str, Any] = {}
    for category, seed in selected.items():
        gate = gates[seed.candidate_id]
        required_checks = standards[category]
        passed = bool(
            gate["passed"]
            and all(gate["checks"].get(name, False) for name in required_checks)
        )
        results[category] = {
            "candidate_id": seed.candidate_id,
            "passed": passed,
            "required_checks": list(required_checks),
            "actual_route": gate["actual_route"],
            "failed_checks": gate["failed_checks"],
            "gate_details": gate["gate_details"],
        }
    report = {
        "probe_version": "sft-v2-four-route-probes-v1",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "all_passed": all(item["passed"] for item in results.values()),
        "results": results,
        "provider_records_file": records_path.name,
    }
    _write_json(cache_dir / "sft_v2_four_route_probe_report.json", report)
    return report


def _seed_from_preserved(row: CandidateTrajectory, case: PlannerEvalCase) -> CandidateSeed:
    web_query = ""
    for step in row.trace_steps:
        decision = step["decision"]
        if decision["action"] == QueryAction.WEB_SEARCH.value:
            web_query = decision["query"]
            break
    return CandidateSeed(
        candidate_id=row.candidate_id,
        sampling_target_route=[QueryAction(value) for value in row.route],
        reserve=row.reserve,
        device_family=row.device_family,
        question_family=row.question_family,
        missing_or_safety_trigger=row.missing_or_safety_trigger,
        source_evidence=[SourceEvidence.model_validate(item) for item in row.source_evidence],
        retrieval_subject_id=(case.expected_subject_ids[0] if case.expected_subject_ids else None),
        retrieval_subject_name=(case.expected_subject_names[0] if case.expected_subject_names else None),
        query=row.query,
        answer_points=list(case.expected_answer_points),
        web_search_query=web_query,
    )


def _rebalance_web_sources(
        allocations: list[dict[str, Any]],
        preserved: list[CandidateTrajectory],
) -> None:
    sources = {source.source_id: source for source in WEB_SOURCES}
    route_counts: dict[str, Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    for row in preserved:
        route = " -> ".join(row.route)
        for evidence in row.source_evidence:
            if evidence.source_type == "web":
                route_counts[route][evidence.source_id] += 1
                global_counts[evidence.source_id] += 1
    for item in allocations:
        if QueryAction.WEB_SEARCH not in item["route"]:
            continue
        terminal = item["route"][-1]
        allowed_ids = (
            ASK_WEB_SOURCE_IDS if terminal == QueryAction.ASK_CLARIFICATION
            else REFUSE_WEB_SOURCE_IDS if terminal == QueryAction.REFUSE
            else set(sources)
        )
        route = item["route_name"]
        ordered = sorted(
            (sources[source_id] for source_id in allowed_ids),
            key=lambda source: (
                route_counts[route][source.source_id],
                global_counts[source.source_id],
                source.source_id,
            ),
        )
        selected = ordered[0]
        item["web_source"] = selected
        item["question_family"] = QUESTION_FAMILY_BY_DEVICE[selected.device_family]
        if QueryAction.LOCAL_SEARCH in item["route"]:
            item["retrieval_subject"] = (
                LOCAL_SOURCES[1] if selected.publisher == "Siemens" else LOCAL_SOURCES[0]
            )
        route_counts[route][selected.source_id] += 1
        global_counts[selected.source_id] += 1


def _raw_pool_parts(approved_ids: set[str]) -> tuple[bytes, dict[str, list[bytes]]]:
    old = bytearray()
    approved: dict[str, list[bytes]] = defaultdict(list)
    for line in SOURCE_POOL.read_bytes().splitlines(keepends=True):
        row = json.loads(line)
        case_id = str(row["source_case_id"])
        if case_id.startswith("sft-v2-new-"):
            if case_id in approved_ids:
                approved[case_id].append(line)
        else:
            old.extend(line)
    if len({json.loads(line)["source_trace_id"] for line in old.splitlines()}) != 37:
        raise ValueError("旧 37 条轨迹前缀无法从候选池中唯一恢复")
    if set(approved) != approved_ids:
        raise ValueError("38 条通过项的 Action 样本不完整")
    return bytes(old), approved


def build_repair(
    *,
    output_dir: Path,
    cache_dir: Path,
    draft_batch_size: int,
) -> dict[str, Any]:
    lock = audit_round3_lock()
    if output_dir.exists():
        raise FileExistsError(f"repair_v2 目录已存在，拒绝覆盖：{output_dir}")
    if cache_dir == ARTIFACT_DIR / "repair_v2_work":
        raise ValueError("拒绝覆盖第一次修复的历史失败证据目录")
    cache_dir.mkdir(parents=True, exist_ok=True)
    approved_ids = set(lock["approved_ids"])
    rejected_ids = set(lock["rejected_ids"])
    reuse_draft_ids, resample_ids = first_repair_partition(rejected_ids)

    source_cases_raw = {row["case_id"]: row for row in _read_jsonl(SOURCE_CASES)}
    source_trajectories_raw = {
        row["candidate_id"]: row for row in _read_jsonl(SOURCE_TRAJECTORIES)
    }
    preserved_cases = [
        PlannerEvalCase.model_validate(source_cases_raw[case_id]) for case_id in sorted(approved_ids)
    ]
    preserved_trajectories = [
        CandidateTrajectory.model_validate(source_trajectories_raw[case_id])
        for case_id in sorted(approved_ids)
    ]
    preserved_seeds = [
        _seed_from_preserved(row, case)
        for row, case in zip(preserved_trajectories, preserved_cases, strict=True)
    ]

    allocations = [item for item in build_allocations() if item["candidate_id"] in rejected_ids]
    approved_queries = [row.query for row in preserved_trajectories]
    for item in allocations:
        item["forbidden_queries"] = approved_queries
    _rebalance_web_sources(allocations, preserved_trajectories)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".repair_v2-", dir=output_dir.parent) as temp:
        work = Path(temp)
        local_facts = assign_local_facts(allocations)
        web_captures, web_facts = capture_web_sources(allocations)
        drafts_path = cache_dir / "sft_v2_question_drafts.jsonl"
        prepare_selective_draft_cache(drafts_path, reuse_ids=reuse_draft_ids)
        repaired_seeds, promoted_reuse_ids, profile_resample_attempts = draft_prevalidated_seeds(
            allocations,
            local_facts=local_facts,
            web_facts=web_facts,
            drafts_path=drafts_path,
            reuse_draft_ids=reuse_draft_ids,
            resample_ids=resample_ids,
            draft_batch_size=draft_batch_size,
        )
        current_drafts = {
            row["candidate_id"]: row for row in _read_jsonl(drafts_path)
        }
        source_drafts = {
            row["candidate_id"]: row for row in _read_jsonl(SOURCE_FIRST_REPAIR_DRAFTS)
        }
        for case_id in reuse_draft_ids - promoted_reuse_ids:
            for field in ("query", "trigger", "answer_points"):
                if current_drafts[case_id].get(field) != source_drafts[case_id].get(field):
                    raise RuntimeError(f"44 条复用草稿发生改写：{case_id}.{field}")
        repaired_execution_cases = build_cases(repaired_seeds)
        execution_cases = sorted(
            [*preserved_cases, *repaired_execution_cases], key=lambda row: row.case_id
        )
        snapshot = build_current_snapshot(execution_cases).snapshot
        probe_report = run_four_route_probes(
            repaired_seeds,
            snapshot=snapshot,
            cache_dir=cache_dir,
        )
        if not probe_report["all_passed"]:
            failed_probe_ids = {
                category: result["candidate_id"]
                for category, result in probe_report["results"].items()
                if not result["passed"]
            }
            raise RuntimeError(
                "四类 Provider 探针未全部通过，禁止启动 87 条执行："
                + json.dumps(failed_probe_ids, ensure_ascii=False, sort_keys=True)
            )
        repaired_records_path = work / ".repaired_provider_observations.pending.jsonl"
        repaired_runs, repaired_records = execute_trajectories(
            repaired_seeds,
            repaired_execution_cases,
            snapshot=snapshot,
            provider_records_path=repaired_records_path,
        )
        # 真实 Provider（动作执行器）记录不能随临时目录一起消失。即使后续
        # Evaluator（评测器）门禁失败，也保留本轮 Observation（观察结果）供归因，
        # 但这些失败记录绝不进入候选池或盲审包。
        latest_records_path = cache_dir / "sft_v2_provider_observations.latest.jsonl"
        latest_records_path.write_bytes(repaired_records_path.read_bytes())
        repaired_seeds = bind_observed_source_evidence(repaired_seeds, repaired_records)
        repaired_gates = validate_generation(
            repaired_seeds,
            repaired_execution_cases,
            repaired_runs,
            repaired_records,
            raise_on_failure=False,
        )
        route_admission = assess_route_admission(
            repaired_seeds,
            repaired_runs,
            repaired_gates,
        )
        failed_gates = {
            case_id: gate for case_id, gate in repaired_gates.items() if not gate["passed"]
        }
        failed_by_check: dict[str, list[str]] = defaultdict(list)
        for case_id, gate in failed_gates.items():
            for check in gate["failed_checks"]:
                failed_by_check[check].append(case_id)
        gate_report = {
            "repair_version": REPAIR_VERSION,
            "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "candidate_count": len(repaired_seeds),
            "reused_draft_count": len(reuse_draft_ids),
            "resampled_draft_count": len(resample_ids),
            "reused_drafts_promoted_by_semantic_gate": sorted(promoted_reuse_ids),
            "profile_resample_attempts": profile_resample_attempts,
            "four_route_probe_report": probe_report,
            "passed_candidate_count": len(repaired_seeds) - len(failed_gates),
            "failed_candidate_count": len(failed_gates),
            "failed_by_check": {
                check: sorted(case_ids) for check, case_ids in sorted(failed_by_check.items())
            },
            "gates": repaired_gates,
            "route_admission": route_admission,
            "provider_records_file": latest_records_path.name,
            "eligible_for_blind_review": not failed_gates,
            "formal_dataset_frozen": False,
            "training_performed": False,
        }
        _write_json(cache_dir / "sft_v2_gate_report.latest.json", gate_report)
        if failed_gates:
            summary = ";".join(
                f"{check}={len(case_ids)}"
                for check, case_ids in sorted(failed_by_check.items())
            )
            raise RuntimeError(
                f"87 条修复候选仍有 {len(failed_gates)} 条未通过生成阶段门禁：{summary}"
            )
        if not route_admission["complete"]:
            raise RuntimeError(
                "真实路线尚未填满 87 条失败槽位，必须继续采样而不能改写路线："
                f"{route_admission['route_deficits']}"
            )
        actual_routes = {
            trajectory.case_id: list(trajectory.action_path) for trajectory in repaired_runs
        }
        repaired_cases = build_cases(repaired_seeds, routes_by_id=actual_routes)
        all_cases = sorted([*preserved_cases, *repaired_cases], key=lambda row: row.case_id)
        repaired_trajectories, repaired_samples = build_outputs(
            repaired_seeds,
            repaired_cases,
            repaired_runs,
            repaired_records,
            repaired_gates,
        )

        all_seeds = sorted([*preserved_seeds, *repaired_seeds], key=lambda row: row.candidate_id)
        all_trajectories = sorted(
            [*preserved_trajectories, *repaired_trajectories], key=lambda row: row.candidate_id
        )
        old_raw, approved_sample_lines = _raw_pool_parts(approved_ids)
        preserved_samples = [
            SftPlannerSample.model_validate(json.loads(line))
            for case_id in sorted(approved_ids) for line in approved_sample_lines[case_id]
        ]
        all_samples = [*preserved_samples, *repaired_samples]
        validation = validate_candidate_outputs(all_seeds, all_trajectories, all_samples)

        approved_record_ids = {
            record_id for row in preserved_trajectories for record_id in row.provider_record_ids
        }
        source_record_lines = {
            json.loads(line)["record_id"]: line
            for line in SOURCE_RECORDS.read_bytes().splitlines(keepends=True)
        }
        provider_path = work / "sft_v2_provider_observations.jsonl"
        provider_path.write_bytes(
            b"".join(source_record_lines[record_id] for record_id in sorted(approved_record_ids))
            + repaired_records_path.read_bytes()
        )

        _write_jsonl(work / "sft_v2_new_candidate_cases.jsonl", all_cases)
        _write_jsonl(work / "sft_v2_new_candidate_trajectories.jsonl", all_trajectories)
        (work / "sft_v2_question_drafts.jsonl").write_bytes(drafts_path.read_bytes())
        write_environment_snapshot(work / "sft_v2_environment_snapshot.json", snapshot)
        old_web = _json(SOURCE_WEB)
        approved_web_facts = {
            case_id: fact for case_id, fact in old_web["candidate_facts"].items()
            if case_id in approved_ids
        }
        _write_json(work / "sft_v2_web_evidence_manifest.json", {
            "manifest_version": REPAIR_VERSION,
            "generation_batch": BATCH_ID,
            "captured_sources": [*old_web["captured_sources"], *web_captures.values()],
            "candidate_facts": {**approved_web_facts, **web_facts},
            "preserved_round3_approved_count": len(approved_ids),
        })

        repaired_samples_by_case: dict[str, list[SftPlannerSample]] = defaultdict(list)
        for sample in repaired_samples:
            repaired_samples_by_case[sample.source_case_id].append(sample)
        pool_path = work / "sft_v2_train_candidates.jsonl"
        with pool_path.open("wb") as handle:
            handle.write(old_raw)
            for case_id in sorted(approved_ids | rejected_ids):
                if case_id in approved_ids:
                    for line in approved_sample_lines[case_id]:
                        handle.write(line)
                else:
                    for sample in sorted(
                        repaired_samples_by_case[case_id], key=lambda row: row.turn_index
                    ):
                        handle.write((json.dumps(
                            sample.model_dump(mode="json"), ensure_ascii=False, sort_keys=True,
                        ) + "\n").encode("utf-8"))

        repaired_fingerprints = {
            row.candidate_id: row.content_fingerprint for row in repaired_trajectories
        }
        unchanged = {
            row.candidate_id: row.content_fingerprint for row in preserved_trajectories
        } == lock["approved_fingerprints"]
        if not unchanged or any(
            repaired_fingerprints[case_id] == source_trajectories_raw[case_id]["content_fingerprint"]
            for case_id in rejected_ids
        ):
            raise RuntimeError("38 条通过项未保持原指纹，或 87 条失败项仍保留原指纹")

        manifest = {
            "repair_version": REPAIR_VERSION,
            "generation_batch": BATCH_ID,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "round3_only": True,
            "source_review_round": "round3",
            "approved_preserved_count": len(approved_ids),
            "rejected_regenerated_count": len(rejected_ids),
            "reused_draft_count": len(reuse_draft_ids),
            "resampled_draft_count": len(resample_ids),
            "reused_drafts_promoted_by_semantic_gate": sorted(promoted_reuse_ids),
            "four_route_probe_report": probe_report,
            "approved_fingerprints_unchanged": unchanged,
            "old_trajectory_count": 37,
            "new_candidate_count": len(all_trajectories),
            "candidate_pool_trajectory_count": 162,
            "route_counts": validation["route_counts"],
            "validation": validation,
            "source_file_sha256": lock["source_file_sha256"],
            "formal_dataset_frozen": False,
            "training_performed": False,
            "next_step": "9.3.22 independent review preparation only",
        }
        _write_json(work / "sft_v2_repair_manifest.json", manifest)
        if {path.name for path in work.iterdir() if not path.name.startswith(".")} != FINAL_FILES:
            raise RuntimeError("repair_v2 最终文件集合不完整")
        work.replace(output_dir)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnose-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--draft-batch-size", type=int, default=5)
    args = parser.parse_args(argv)
    result = audit_round3_lock() if args.diagnose_only else build_repair(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        draft_batch_size=args.draft_batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
