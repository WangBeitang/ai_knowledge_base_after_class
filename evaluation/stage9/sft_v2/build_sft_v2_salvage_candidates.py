"""从 round3 拒绝项中筛出真实可执行的 SFT V2 待审扩充候选。"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlsplit

from evaluation.stage8.build_environment_snapshot import write_environment_snapshot
from evaluation.stage9.sft_v2.build_sft_v2_candidates import (
    BATCH_ID,
    BATCH_VERSION,
    LOCAL_SOURCES,
    QUESTION_FAMILY_BY_DEVICE,
    CandidateQuestionProfile,
    CandidateSeed,
    CandidateTrajectory,
    SourceEvidence,
    _char_ngrams,
    _jaccard,
    _nontrain_queries_and_chunks,
    _normalized_text,
    _read_jsonl,
    _sha256_bytes,
    bind_observed_source_evidence,
    build_allocations,
    build_cases,
    build_current_snapshot,
    build_outputs,
    capture_web_sources,
    execute_trajectories,
    replay_recorded_trajectories,
    validate_generation,
    validate_pre_provider_profiles,
    assign_local_facts,
)
from evaluation.stage9.sft_v2.repair_sft_v2_candidates import (
    DEFAULT_CACHE_DIR as REPAIR_CACHE_DIR,
    SOURCE_POOL,
    _raw_pool_parts,
    _rebalance_web_sources,
    audit_round3_lock,
    first_repair_partition,
)
from app.rag.evaluation.action_providers import read_provider_observation_records
from app.rag.evaluation.sft_exporter import SftPlannerSample
from app.rag.query.contracts import QueryAction


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = PROJECT_ROOT / "evaluation/stage9/artifacts/sft_v2"
DEFAULT_OUTPUT_DIR = ARTIFACT_DIR / "salvage_pending_review_v1"
DEFAULT_WORK_DIR = ARTIFACT_DIR / "salvage_work_v1"
OVERRIDES_PATH = Path(__file__).with_name("salvage_v1_overrides.json")
SALVAGE_VERSION = "sft-v2-salvage-pending-review-v1"

EXTRA_RESCUE_IDS = {
    f"sft-v2-new-{number}"
    for number in (
        "027", "031", "033", "034", "063", "064", "065", "067", "068",
        "070", "071", "073", "074", "076", "077", "080", "081", "123",
    )
}

PLACEHOLDER_MARKERS = (
    "这种现场现象", "那个关键", "这个功能太少", "这套保护结构", "这种现象设置",
)
REALTIME_MARKERS = (
    "当前", "最新", "截至", "公告", "发布", "今天", "本周", "本月", "现行",
    "固件版本", "生命周期状态", "兼容性", "安全通告", "漏洞公告",
)
DIRECT_UNSAFE_MARKERS = (
    "绕过", "不使用上锁", "省略上锁", "未经设备验证", "未经验证", "立即执行",
    "未公开", "漏洞利用代码", "内部网络渗透", "不加锁", "跳过", "关闭保护",
    "唯一网络配置", "生产控制网", "写入命令", "直接上线", "来源无法核验",
)
DIRECT_MISSING_MARKERS = (
    "不知道", "不确定", "未确认", "无法确认", "无法读取", "铭牌模糊", "哪一项",
    "具体型号", "具体状态", "具体生命周期", "仅凭", "能确定", "需要确认",
)

# 这两条与已经三审通过的同来源问题语义高度重叠。仍执行真实路线并保存失败证据，
# 但不能进入待审池，避免把已知近义项再次交给独立审核。
KNOWN_SEMANTIC_DUPLICATES = {
    "sft-v2-new-033": "sft-v2-new-009",
    "sft-v2-new-071": "sft-v2-new-020",
    "sft-v2-new-097": "sft-v2-new-058",
    "sft-v2-new-122": "sft-v2-new-102",
    "sft-v2-new-125": "sft-v2-new-059",
}


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


def _load_overrides() -> dict[str, dict[str, Any]]:
    value = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != EXTRA_RESCUE_IDS:
        raise ValueError("18 条额外抢救覆盖项身份集合不闭环")
    return {str(key): dict(row) for key, row in value.items()}


def _selected_identity() -> tuple[dict[str, Any], set[str], set[str], set[str]]:
    lock = audit_round3_lock()
    rejected_ids = set(lock["rejected_ids"])
    reuse_ids, first_failed_ids = first_repair_partition(rejected_ids)
    if len(reuse_ids) != 44 or len(first_failed_ids) != 43:
        raise ValueError("第一次修复 44/43 身份集合漂移")
    if not EXTRA_RESCUE_IDS.issubset(first_failed_ids):
        raise ValueError("18 条额外抢救项不完全来自第一次门禁失败集合")
    selected_ids = reuse_ids | EXTRA_RESCUE_IDS
    if len(selected_ids) != 62:
        raise ValueError("待执行抢救集合不是 62 条")
    return lock, reuse_ids, first_failed_ids, selected_ids


def _build_authoring_seeds() -> tuple[
    dict[str, Any],
    set[str],
    set[str],
    list[CandidateSeed],
    list[dict[str, Any]],
    dict[str, Any],
]:
    lock, reuse_ids, first_failed_ids, selected_ids = _selected_identity()
    approved_ids = set(lock["approved_ids"])
    source_trajectories = {
        row["candidate_id"]: CandidateTrajectory.model_validate(row)
        for row in _read_jsonl(ARTIFACT_DIR / "sft_v2_new_candidate_trajectories.jsonl")
    }
    preserved = [source_trajectories[case_id] for case_id in sorted(approved_ids)]

    full_allocations = build_allocations()
    allocations = [
        item for item in full_allocations
        if item["candidate_id"] in set(lock["rejected_ids"])
    ]
    approved_queries = [row.query for row in preserved]
    for item in allocations:
        item["forbidden_queries"] = approved_queries
    # 使用与动态修复缓存相同的全量顺序完成来源分配，再筛 62 条，避免按子集重新洗牌。
    _rebalance_web_sources(allocations, preserved)
    # 44 条复用项继续使用 87 条修复批次的事实分配；18 条额外抢救项则显式复用
    # 初版 125 条来源分配中已经人工核验过的正式 chunk，避免因子集轮转把问题绑到
    # 同一文档里的另一段无关正文。
    local_facts = assign_local_facts(allocations)
    full_local_facts = assign_local_facts(full_allocations)
    for case_id in EXTRA_RESCUE_IDS:
        if case_id in full_local_facts:
            local_facts[case_id] = full_local_facts[case_id]
    selected_allocations = [
        item for item in allocations if item["candidate_id"] in selected_ids
    ]
    web_captures, web_facts = capture_web_sources(selected_allocations)

    drafts_path = REPAIR_CACHE_DIR / "sft_v2_question_drafts.jsonl"
    drafts = {row["candidate_id"]: row for row in _read_jsonl(drafts_path)}
    if not selected_ids.issubset(drafts):
        raise ValueError("动态修复缓存不能完整恢复 62 条当前草稿")
    overrides = _load_overrides()
    selected_drafts: dict[str, dict[str, Any]] = {}
    for case_id in sorted(selected_ids):
        row = dict(drafts[case_id])
        if case_id in overrides:
            row.update(overrides[case_id])
            row["salvage_override_version"] = SALVAGE_VERSION
        selected_drafts[case_id] = row

    seeds: list[CandidateSeed] = []
    for item in selected_allocations:
        case_id = item["candidate_id"]
        row = selected_drafts[case_id]
        local_source = item["local_source"]
        web_source = item["web_source"]
        evidences: list[SourceEvidence] = []
        if local_source is not None:
            fact = local_facts[case_id]
            evidences.append(SourceEvidence(
                source_type="local",
                source_id=(
                    f"{local_source.document_id}:{fact['chunk_id']}:"
                    f"{local_source.index_version}"
                ),
                publisher=(
                    "Rockwell Automation"
                    if local_source.document_id.startswith("doc_857") else
                    "Siemens"
                    if local_source.document_id.startswith("doc_98c") else
                    "document_owner"
                ),
                source_title=local_source.title,
                document_id=local_source.document_id,
                chunk_id=fact["chunk_id"],
                index_version=local_source.index_version,
                evidence_content_sha256=_sha256_bytes(fact["fact_text"].encode("utf-8")),
                fact_text=fact["fact_text"],
            ))
        if web_source is not None:
            fact = web_facts[case_id]
            evidences.append(SourceEvidence(
                source_type="web",
                source_id=web_source.source_id,
                publisher=web_source.publisher,
                source_title=web_source.source_title,
                url=web_source.url,
                captured_at=fact["captured_at"],
                response_sha256=fact["response_sha256"],
                evidence_content_sha256=fact["evidence_content_sha256"],
                fact_text=fact["fact_text"],
            ))
        source_for_family = local_source or web_source
        if source_for_family is None:
            raise ValueError(f"{case_id} 没有作者来源")
        query = re.sub(r"\s+", " ", str(row.get("query") or "")).strip()
        answer_points = [
            str(value).strip() for value in row.get("answer_points") or []
            if str(value).strip()
        ]
        seeds.append(CandidateSeed(
            candidate_id=case_id,
            sampling_target_route=list(item["route"]),
            reserve=bool(item["reserve"]),
            device_family=source_for_family.device_family,
            question_family=QUESTION_FAMILY_BY_DEVICE[source_for_family.device_family],
            missing_or_safety_trigger=str(row.get("trigger") or "").strip(),
            source_evidence=evidences,
            retrieval_subject_id=(
                item["retrieval_subject"].subject_id
                if item.get("retrieval_subject") else None
            ),
            retrieval_subject_name=(
                item["retrieval_subject"].subject_name
                if item.get("retrieval_subject") else None
            ),
            query=query,
            answer_points=answer_points,
            web_search_query=(
                f"site:{urlsplit(web_source.url).netloc} "
                f"{web_source.publisher} {web_source.source_title} {query}"
                if web_source is not None else ""
            ),
            question_profile=CandidateQuestionProfile.model_validate(
                row["question_profile"]
            ),
        ))
    return (
        lock,
        reuse_ids,
        first_failed_ids,
        seeds,
        [selected_drafts[case_id] for case_id in sorted(selected_drafts)],
        {"captured_sources": list(web_captures.values()), "candidate_facts": web_facts},
    )


def _preflight_failures(seeds: Sequence[CandidateSeed]) -> dict[str, list[str]]:
    failures: dict[str, list[str]] = defaultdict(list)
    for case_id, reasons in validate_pre_provider_profiles(seeds).items():
        failures[case_id].extend(reasons)
    for seed in seeds:
        if any(marker in seed.query for marker in PLACEHOLDER_MARKERS):
            failures[seed.candidate_id].append("query_contains_placeholder")
        facts = [evidence.fact_text for evidence in seed.source_evidence]
        for binding in seed.question_profile.claim_evidence_bindings:
            if not any(binding.evidence_span in fact for fact in facts):
                failures[seed.candidate_id].append("claim_span_not_in_authoring_evidence")
                break
        web_only_answer = bool(
            seed.answer_points
            and seed.retrieval_subject_id is None
            and any(item.source_type == "web" for item in seed.source_evidence)
            and not seed.question_profile.realtime_required
        )
        if web_only_answer:
            failures[seed.candidate_id].append("static_web_answer_not_allowed")
        if (
            seed.question_profile.realtime_required
            and not any(marker in seed.query for marker in REALTIME_MARKERS)
        ):
            failures[seed.candidate_id].append("realtime_profile_without_query_signal")
    return {key: sorted(set(value)) for key, value in sorted(failures.items())}


def _behavior_evidence(seed: CandidateSeed) -> SourceEvidence:
    fact = seed.query
    return SourceEvidence(
        source_type="behavior",
        source_id=f"user-query:{seed.candidate_id}",
        publisher="user_input",
        source_title="Pre-search user query boundary",
        evidence_content_sha256=_sha256_bytes(fact.encode("utf-8")),
        fact_text=fact,
    )


def _preserve_direct_behavior_evidence(
    seeds: Sequence[CandidateSeed],
    trajectories: Sequence[Any],
) -> list[CandidateSeed]:
    trajectories_by_id = {row.case_id: row for row in trajectories}
    result: list[CandidateSeed] = []
    for seed in seeds:
        route = list(trajectories_by_id[seed.candidate_id].action_path)
        if len(route) == 1 and route[0] in {
            QueryAction.ASK_CLARIFICATION, QueryAction.REFUSE,
        }:
            result.append(seed.model_copy(update={
                "source_evidence": [_behavior_evidence(seed)],
            }))
        else:
            result.append(seed)
    return result


def _base_queries(approved_ids: set[str]) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in _read_jsonl(SOURCE_POOL):
        case_id = str(row["source_case_id"])
        if case_id.startswith("sft-v2-new-") and case_id not in approved_ids:
            continue
        query = str((row.get("input_context") or {}).get("query") or "").strip()
        identity = (case_id, query)
        if query and identity not in seen:
            seen.add(identity)
            values.append(identity)
    return values


def _apply_policy_and_leakage_gates(
    seeds: Sequence[CandidateSeed],
    trajectories: Sequence[Any],
    gates: dict[str, dict[str, Any]],
    *,
    approved_ids: set[str],
) -> dict[str, dict[str, Any]]:
    seed_by_id = {seed.candidate_id: seed for seed in seeds}
    trajectory_by_id = {row.case_id: row for row in trajectories}
    base_queries = _base_queries(approved_ids)
    nontrain_queries, nontrain_chunks = _nontrain_queries_and_chunks()
    accepted_queries: list[tuple[str, str]] = []

    for case_id in sorted(seed_by_id):
        seed = seed_by_id[case_id]
        gate = gates[case_id]
        route = list(trajectory_by_id[case_id].action_path)
        checks = gate["checks"]
        details = gate["gate_details"]
        checks["web_realtime_policy"] = bool(
            QueryAction.WEB_SEARCH not in route
            or (
                seed.question_profile.realtime_required
                and any(marker in seed.query for marker in REALTIME_MARKERS)
            )
        )
        checks["query_placeholder_free"] = not any(
            marker in seed.query for marker in PLACEHOLDER_MARKERS
        )
        checks["source_evidence_present"] = bool(seed.source_evidence)
        if route == [QueryAction.REFUSE]:
            checks["direct_refusal_explicit"] = any(
                marker.lower() in seed.query.lower() for marker in DIRECT_UNSAFE_MARKERS
            )
        if route == [QueryAction.ASK_CLARIFICATION]:
            checks["direct_clarification_missing_field"] = any(
                marker in seed.query for marker in DIRECT_MISSING_MARKERS
            )
        if case_id in KNOWN_SEMANTIC_DUPLICATES:
            checks["known_semantic_duplicate_free"] = False
            details["known_semantic_duplicate"] = {
                "existing_case_id": KNOWN_SEMANTIC_DUPLICATES[case_id],
            }
        else:
            checks["known_semantic_duplicate_free"] = True

        duplicate: dict[str, Any] | None = None
        seed_grams = _char_ngrams(seed.query)
        for other_id, other_query in [*base_queries, *accepted_queries]:
            score = max(
                _jaccard(seed_grams, _char_ngrams(other_query)),
                SequenceMatcher(None, seed.query, other_query).ratio(),
            )
            if (
                _normalized_text(seed.query) == _normalized_text(other_query)
                or score >= 0.82
            ):
                duplicate = {
                    "other_case_id": other_id,
                    "score": round(score, 4),
                    "other_query": other_query,
                }
                break
        checks["exact_and_near_duplicate_free"] = duplicate is None
        if duplicate is not None:
            details["duplicate"] = duplicate

        split_leaks = []
        for nontrain_id, nontrain_query in nontrain_queries:
            score = _jaccard(seed_grams, _char_ngrams(nontrain_query))
            if (
                _normalized_text(seed.query) == _normalized_text(nontrain_query)
                or score >= 0.78
            ):
                split_leaks.append({
                    "nontrain_case_id": nontrain_id,
                    "score": round(score, 4),
                })
        chunk_leaks = sorted({
            str(evidence.chunk_id)
            for evidence in seed.source_evidence
            if evidence.source_type == "local"
            and str(evidence.chunk_id) in nontrain_chunks
        })
        checks["split_query_leak_free"] = not split_leaks
        checks["split_chunk_leak_free"] = not chunk_leaks
        if split_leaks or chunk_leaks:
            details["split_leakage"] = {
                "queries": split_leaks,
                "chunks": chunk_leaks,
            }

        failed_checks = sorted(name for name, passed in checks.items() if not passed)
        gate["failed_checks"] = failed_checks
        gate["passed"] = not failed_checks
        if gate["passed"]:
            accepted_queries.append((case_id, seed.query))
    return gates


def _filter_records(records: Sequence[Any], case_ids: set[str]) -> list[Any]:
    return [record for record in records if record.case_id in case_ids]


def _summary_counts(
    trajectories: Sequence[CandidateTrajectory],
) -> dict[str, dict[str, int]]:
    route_counts = Counter(" -> ".join(row.route) for row in trajectories)
    device_counts = Counter(row.device_family for row in trajectories)
    question_counts = Counter(row.question_family for row in trajectories)
    source_counts: Counter[str] = Counter()
    for row in trajectories:
        for evidence in row.source_evidence:
            source_counts[evidence.source_id] += 1
    return {
        "route_counts": dict(sorted(route_counts.items())),
        "device_family_counts": dict(sorted(device_counts.items())),
        "question_family_counts": dict(sorted(question_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
    }


def build_salvage_pool(
    *,
    output_dir: Path,
    work_dir: Path,
    replay_provider_records: Path | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"待审扩充目录已存在，拒绝覆盖：{output_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)
    (
        lock,
        reuse_ids,
        first_failed_ids,
        seeds,
        selected_drafts,
        web_evidence,
    ) = _build_authoring_seeds()
    preflight_failures = _preflight_failures(seeds)
    executable_seeds = [
        seed for seed in seeds if seed.candidate_id not in preflight_failures
    ]
    if not executable_seeds:
        raise RuntimeError("62 条抢救候选全部在 Provider 前结构门禁失败")

    execution_cases = build_cases(executable_seeds)
    snapshot = build_current_snapshot(execution_cases).snapshot
    provider_records_path = work_dir / "sft_v2_salvage_provider_observations.all.jsonl"
    if replay_provider_records is None:
        trajectories, records = execute_trajectories(
            executable_seeds,
            execution_cases,
            snapshot=snapshot,
            provider_records_path=provider_records_path,
        )
    else:
        trajectories, records = replay_recorded_trajectories(
            executable_seeds,
            execution_cases,
            snapshot=snapshot,
            provider_records_path=replay_provider_records,
        )
        if not provider_records_path.exists():
            provider_records_path.write_bytes(replay_provider_records.read_bytes())

    bound_seeds = bind_observed_source_evidence(executable_seeds, records)
    bound_seeds = _preserve_direct_behavior_evidence(bound_seeds, trajectories)
    provider_observation_case_ids = {record.case_id for record in records}
    gates = validate_generation(
        bound_seeds,
        execution_cases,
        trajectories,
        records,
        raise_on_failure=False,
    )
    gates = _apply_policy_and_leakage_gates(
        bound_seeds,
        trajectories,
        gates,
        approved_ids=set(lock["approved_ids"]),
    )
    passed_ids = {
        case_id for case_id, gate in gates.items() if gate["passed"]
    }
    passed_seeds = [
        seed for seed in bound_seeds if seed.candidate_id in passed_ids
    ]
    passed_trajectories_raw = [
        row for row in trajectories if row.case_id in passed_ids
    ]
    actual_routes = {
        row.case_id: list(row.action_path) for row in passed_trajectories_raw
    }
    final_cases = build_cases(passed_seeds, routes_by_id=actual_routes)
    trajectory_rows, samples = build_outputs(
        passed_seeds,
        final_cases,
        passed_trajectories_raw,
        records,
        gates,
    )

    if len({row.candidate_id for row in trajectory_rows}) != len(trajectory_rows):
        raise RuntimeError("待审抢救候选 candidate_id 不唯一")
    if len({row.content_fingerprint for row in trajectory_rows}) != len(trajectory_rows):
        raise RuntimeError("待审抢救候选 content_fingerprint 不唯一")
    source_original = {
        row["candidate_id"]: row["content_fingerprint"]
        for row in _read_jsonl(ARTIFACT_DIR / "sft_v2_new_candidate_trajectories.jsonl")
    }
    unchanged_rejected = [
        row.candidate_id for row in trajectory_rows
        if row.content_fingerprint == source_original[row.candidate_id]
    ]
    if unchanged_rejected:
        raise RuntimeError(f"抢救候选仍沿用三审拒绝指纹：{unchanged_rejected}")

    old_raw, approved_lines = _raw_pool_parts(set(lock["approved_ids"]))
    base_lines = [
        line
        for case_id in sorted(approved_lines)
        for line in approved_lines[case_id]
    ]
    base_bytes = old_raw + b"".join(base_lines)
    base_samples = [
        SftPlannerSample.model_validate(json.loads(line))
        for line in base_bytes.splitlines()
    ]
    if len({row.source_trace_id for row in base_samples}) != 75:
        raise RuntimeError("37 条旧数据与 38 条三审通过项的 75 条基础轨迹不闭环")
    combined_samples = [*base_samples, *samples]
    if len({row.sample_id for row in combined_samples}) != len(combined_samples):
        raise RuntimeError("待审扩充池 sample_id 不唯一")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".salvage-pending-", dir=output_dir.parent) as temp:
        work = Path(temp)
        _write_jsonl(work / "sft_v2_salvage_cases.jsonl", final_cases)
        _write_jsonl(work / "sft_v2_salvage_trajectories.jsonl", trajectory_rows)
        _write_jsonl(work / "sft_v2_salvage_action_samples.jsonl", samples)
        _write_jsonl(
            work / "sft_v2_salvage_provider_observations.accepted.jsonl",
            _filter_records(records, passed_ids),
        )
        _write_jsonl(work / "sft_v2_salvage_question_drafts.jsonl", selected_drafts)
        _write_jsonl(
            work / "sft_v2_train_candidates.pending_review.jsonl",
            combined_samples,
        )
        _write_json(work / "sft_v2_salvage_web_evidence_manifest.json", {
            "manifest_version": SALVAGE_VERSION,
            **web_evidence,
        })
        write_environment_snapshot(
            work / "sft_v2_salvage_environment_snapshot.json",
            snapshot,
        )
        failed_by_check: dict[str, list[str]] = defaultdict(list)
        for case_id, reasons in preflight_failures.items():
            for reason in reasons:
                failed_by_check[f"preflight:{reason}"].append(case_id)
        for case_id, gate in gates.items():
            for reason in gate["failed_checks"]:
                failed_by_check[reason].append(case_id)
        gate_report = {
            "gate_version": SALVAGE_VERSION,
            "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "selected_candidate_count": len(seeds),
            "dynamic_execution_candidate_count": len(executable_seeds),
            "provider_observation_candidate_count": len(provider_observation_case_ids),
            "direct_terminal_candidate_count": (
                len(executable_seeds) - len(provider_observation_case_ids)
            ),
            "preflight_failed_candidate_count": len(preflight_failures),
            "passed_candidate_count": len(passed_ids),
            "failed_candidate_count": len(seeds) - len(passed_ids),
            "preflight_failures": preflight_failures,
            "failed_by_check": {
                name: sorted(case_ids)
                for name, case_ids in sorted(failed_by_check.items())
            },
            "gates": gates,
            "eligible_for_blind_review": sorted(passed_ids),
            "formal_dataset_frozen": False,
            "training_performed": False,
        }
        _write_json(work / "sft_v2_salvage_gate_report.json", gate_report)

        counts = _summary_counts(trajectory_rows)
        manifest = {
            "manifest_version": SALVAGE_VERSION,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "source_review_round": "round3",
            "base_old_retained_trajectory_count": 37,
            "base_round3_approved_trajectory_count": 38,
            "base_reviewed_trajectory_count": 75,
            "base_action_sample_count": len(base_samples),
            "selected_salvage_candidate_count": len(seeds),
            "first_gate_passed_reuse_count": len(reuse_ids),
            "extra_rescue_from_first_failed_count": len(EXTRA_RESCUE_IDS),
            "first_failed_not_selected_count": len(first_failed_ids - EXTRA_RESCUE_IDS),
            "dynamic_execution_candidate_count": len(executable_seeds),
            "provider_observation_candidate_count": len(provider_observation_case_ids),
            "direct_terminal_candidate_count": (
                len(executable_seeds) - len(provider_observation_case_ids)
            ),
            "pending_review_salvage_trajectory_count": len(trajectory_rows),
            "pending_review_salvage_action_sample_count": len(samples),
            "pending_review_with_provider_observation_count": len(
                passed_ids & provider_observation_case_ids
            ),
            "pending_review_direct_terminal_count": len(
                passed_ids - provider_observation_case_ids
            ),
            "combined_candidate_pool_trajectory_count": 75 + len(trajectory_rows),
            "combined_candidate_pool_action_sample_count": len(combined_samples),
            "approved_fingerprints_unchanged": True,
            "source_file_sha256": lock["source_file_sha256"],
            "main_candidate_pool_sha256_unchanged": (
                _sha256_bytes(SOURCE_POOL.read_bytes())
                == lock["source_file_sha256"][SOURCE_POOL.name]
            ),
            **counts,
            "formal_dataset_frozen": False,
            "training_performed": False,
            "next_step": "independent blind review of salvage_pending_review_v1 only",
            "files": sorted(path.name for path in work.iterdir()),
        }
        _write_json(work / "sft_v2_salvage_manifest.json", manifest)
        work.replace(output_dir)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--replay-provider-records", type=Path)
    args = parser.parse_args(argv)
    result = build_salvage_pool(
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        replay_provider_records=args.replay_provider_records,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
