import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from app.rag.evaluation.case_schema import PlannerEvalCase
from evaluation.stage9.balanced_dev.export_blind_review_bundle import (
    BANNED_REVIEW_KEYS,
    validate_blind_review_bundle,
)
from evaluation.stage9.heldout_route_test.build_heldout_route_test import (
    CASE_SPECS,
    DEFAULT_BUILD_MANIFEST,
    DEFAULT_CASES_PATH,
    DEFAULT_FREEZE_MANIFEST,
    DEFAULT_MATRIX,
    DEFAULT_REWARD_PROFILE,
    DEFAULT_REVIEW_QUEUE,
    DEFAULT_SNAPSHOT,
    DEFAULT_SOURCE_IMPORT,
    DEFAULT_SPLIT_PATH,
    DEFAULT_WEB_EVIDENCE,
    GENERATED_PREFIX,
    HYDE_PROBES,
    ROUND1_DECISIONS,
    ROUND2_DECISIONS,
    _case_spec_fingerprint,
    build_heldout_route_test,
)
from evaluation.stage9.heldout_route_test.export_blind_review_bundle import (
    DEFAULT_OUTPUT_DIR,
    ROUND1_OUTPUT_DIR,
    export_heldout_blind_review_bundle,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _all_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def _temp_output_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "evidence_path": tmp_path / "artifacts/evidence.jsonl",
        "review_queue_path": tmp_path / "artifacts/review_queue.jsonl",
        "build_manifest_path": tmp_path / "artifacts/build_manifest.json",
        "freeze_manifest_path": tmp_path / "artifacts/freeze_manifest.json",
        "report_path": tmp_path / "reports/report.md",
    }


def test_checked_in_heldout_inventory_preserves_core_and_has_five_routes():
    rows = _read_jsonl(DEFAULT_CASES_PATH)
    test_cases = [
        PlannerEvalCase.model_validate(row) for row in rows if row["split"] == "test"
    ]
    core = [
        case for case in test_cases if not case.case_id.startswith(GENERATED_PREFIX)
    ]
    heldout = [
        case for case in test_cases if case.case_id.startswith(GENERATED_PREFIX)
    ]

    assert len(test_cases) == 60
    assert len(core) == 35
    assert len(heldout) == 25
    assert Counter(
        "web_required"
        if case.expected_behavior.should_call_web
        else "ask_clarification"
        if case.expected_behavior.should_ask_clarification
        else "safe_refuse"
        if case.expected_behavior.should_refuse
        else "hyde_fallback"
        if all(
            any(action.value == "hyde_search" for action in path)
            for path in case.acceptable_action_paths
        )
        else "local_answer"
        for case in heldout
    ) == {
        "local_answer": 5,
        "hyde_fallback": 5,
        "web_required": 5,
        "ask_clarification": 5,
        "safe_refuse": 5,
    }
    assert Counter(case.human_review_status.value for case in heldout) == {
        "reviewed": 25,
    }
    assert all(case.gold_origin.value == "heldout_gold" for case in heldout)
    assert len({case.leakage_group_id for case in heldout}) == 25

    split = json.loads(DEFAULT_SPLIT_PATH.read_text(encoding="utf-8"))
    assert len(split["core_answer_test_case_ids"]) == 35
    assert len(split["route_heldout_test_case_ids"]) == 25
    assert split["snapshot_id"] == "stage9-heldout-route-test-env-20260729-v3"
    assert split["logical_test_sets"]["route_heldout_test"][
        "allowed_for_model_selection"
    ] is False


def test_hyde_specs_only_keep_retrieval_probes_that_improve_target_rank():
    hyde_specs = [spec for spec in CASE_SPECS if spec.route_bucket == "hyde_fallback"]
    assert len(hyde_specs) == 5
    assert {spec.hyde_probe_id for spec in hyde_specs} == set(HYDE_PROBES)
    for probe in HYDE_PROBES.values():
        top_k = probe["top_k"]
        target = probe["target_chunk_index"]
        assert probe["original_target_rank"] is None
        assert target not in probe["original_top5_chunk_indices"]
        assert 1 <= probe["hypothetical_target_rank"] <= top_k
        assert (
            probe["hypothetical_top5_chunk_indices"][
                probe["hypothetical_target_rank"] - 1
            ]
            == target
        )
        assert len(probe["original_top5_chunk_indices"]) == top_k
        assert len(probe["hypothetical_top5_chunk_indices"]) == top_k
        assert probe["hypothetical_query"].strip()


def test_builder_is_idempotent_and_never_fakes_review_or_inference(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    split_path = tmp_path / "split.json"
    shutil.copy2(DEFAULT_CASES_PATH, cases_path)
    shutil.copy2(DEFAULT_SPLIT_PATH, split_path)
    paths = _temp_output_paths(tmp_path)
    missing_decisions = tmp_path / "missing_decisions.jsonl"

    first = build_heldout_route_test(
        cases_path=cases_path,
        split_path=split_path,
        snapshot_path=tmp_path / "missing_snapshot.json",
        decisions_path=missing_decisions,
        overwrite=True,
        **paths,
    )
    second = build_heldout_route_test(
        cases_path=cases_path,
        split_path=split_path,
        snapshot_path=tmp_path / "missing_snapshot.json",
        decisions_path=missing_decisions,
        overwrite=True,
        **paths,
    )

    assert first["route_heldout_test"] == second["route_heldout_test"]
    assert second["status"] == "candidate_complete_independent_review_pending"
    assert second["route_heldout_test"]["pending_count"] == 25
    assert second["core_answer_test"]["unchanged"] is True
    assert second["model_execution_performed"] is False
    assert second["heldout_inference_result_count"] == 0
    assert len(_read_jsonl(paths["review_queue_path"])) == 25


def test_explicit_independent_approvals_are_required_for_freeze_gate(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    split_path = tmp_path / "split.json"
    decisions_path = tmp_path / "decisions.jsonl"
    shutil.copy2(DEFAULT_CASES_PATH, cases_path)
    shutil.copy2(DEFAULT_SPLIT_PATH, split_path)
    paths = _temp_output_paths(tmp_path)
    decisions_path.write_text(
        "".join(
            json.dumps(
                {
                    "case_id": spec.case_id,
                    "case_fingerprint": _case_spec_fingerprint(spec),
                    "decision": "approved",
                    "reviewer_id": "independent-review-fixture",
                    "reviewer_role": "independent_agent",
                    "reviewed_at": "2026-07-28T12:30:00+00:00",
                    "evidence_check": "fixture passed",
                    "route_check": "fixture passed",
                    "leakage_check": "fixture passed",
                    "notes": "测试只验证门禁，不代表真实审核。",
                },
                ensure_ascii=False,
            )
            + "\n"
            for spec in CASE_SPECS
        ),
        encoding="utf-8",
    )

    result = build_heldout_route_test(
        cases_path=cases_path,
        split_path=split_path,
        snapshot_path=tmp_path / "missing_snapshot.json",
        decisions_path=decisions_path,
        overwrite=True,
        **paths,
    )

    assert result["status"] == "reviewed_freeze_complete"
    assert result["route_heldout_test"]["reviewed_count"] == 25
    assert result["route_heldout_test"]["pending_count"] == 0
    assert result["model_execution_performed"] is False


def test_checked_in_manifests_freeze_inputs_and_forbid_early_run():
    build = json.loads(DEFAULT_BUILD_MANIFEST.read_text(encoding="utf-8"))
    freeze = json.loads(DEFAULT_FREEZE_MANIFEST.read_text(encoding="utf-8"))

    assert build["core_answer_test"]["unchanged"] is True
    assert build["source_independence"]["source_document_overlap_count"] == 0
    assert build["cross_split_leakage_finding_count"] == 0
    assert build["model_execution_performed"] is False
    assert build["status"] == "reviewed_freeze_complete"
    assert build["route_heldout_test"]["reviewed_count"] == 25
    assert build["route_heldout_test"]["pending_count"] == 0
    assert build["route_heldout_test"]["rejected_count"] == 0
    assert DEFAULT_REVIEW_QUEUE.read_text(encoding="utf-8") == ""
    assert build["independent_review_decisions"] == [
        {
            "path": str(ROUND1_DECISIONS.relative_to(Path.cwd())),
            "sha256": hashlib.sha256(ROUND1_DECISIONS.read_bytes()).hexdigest(),
        },
        {
            "path": str(ROUND2_DECISIONS.relative_to(Path.cwd())),
            "sha256": hashlib.sha256(ROUND2_DECISIONS.read_bytes()).hexdigest(),
        },
    ]
    assert freeze["run_policy"] == "do_not_run_before_stage9_3_16_checkpoint_freeze"
    assert freeze["allowed_for_model_selection"] is False
    assert freeze["heldout_inference_result_count"] == 0
    assert freeze["inputs"] == {
        str(DEFAULT_SOURCE_IMPORT.relative_to(Path.cwd())): hashlib.sha256(
            DEFAULT_SOURCE_IMPORT.read_bytes()
        ).hexdigest(),
        str(DEFAULT_WEB_EVIDENCE.relative_to(Path.cwd())): hashlib.sha256(
            DEFAULT_WEB_EVIDENCE.read_bytes()
        ).hexdigest(),
        str(DEFAULT_MATRIX.relative_to(Path.cwd())): hashlib.sha256(
            DEFAULT_MATRIX.read_bytes()
        ).hexdigest(),
        str(DEFAULT_REWARD_PROFILE.relative_to(Path.cwd())): hashlib.sha256(
            DEFAULT_REWARD_PROFILE.read_bytes()
        ).hexdigest(),
        str(ROUND1_DECISIONS.relative_to(Path.cwd())): hashlib.sha256(
            ROUND1_DECISIONS.read_bytes()
        ).hexdigest(),
        str(ROUND2_DECISIONS.relative_to(Path.cwd())): hashlib.sha256(
            ROUND2_DECISIONS.read_bytes()
        ).hexdigest(),
        str(DEFAULT_SNAPSHOT.relative_to(Path.cwd())): hashlib.sha256(
            DEFAULT_SNAPSHOT.read_bytes()
        ).hexdigest(),
    }


def test_checked_in_round1_blind_review_bundle_remains_immutable_and_valid():
    result = validate_blind_review_bundle(ROUND1_OUTPUT_DIR)
    assert result["ok"] is True
    assert result["case_count"] == 25
    assert result["route_counts"] == {
        "local_answer": 5,
        "hyde_fallback": 5,
        "web_required": 5,
        "ask_clarification": 5,
        "safe_refuse": 5,
    }
    review_rows = _read_jsonl(ROUND1_OUTPUT_DIR / "review_cases.jsonl")
    assert not (set(_all_keys(review_rows)) & BANNED_REVIEW_KEYS)


def test_checked_in_round2_blind_review_bundle_contains_only_changed_cases():
    result = validate_blind_review_bundle(DEFAULT_OUTPUT_DIR)
    assert result["ok"] is True
    assert result["case_count"] == 5
    assert result["route_counts"] == {
        "hyde_fallback": 4,
        "ask_clarification": 1,
    }
    review_rows = _read_jsonl(DEFAULT_OUTPUT_DIR / "review_cases.jsonl")
    assert not (set(_all_keys(review_rows)) & BANNED_REVIEW_KEYS)
    assert {row["case_id"] for row in review_rows} == {
        "planner-test-heldout-hyde-display-joystick-poweroff",
        "planner-test-heldout-hyde-matebook-upgrade-screen-recovery",
        "planner-test-heldout-hyde-tablet-screen-reader-off",
        "planner-test-heldout-hyde-tablet-recording-transcript",
        "planner-test-heldout-ask-tablet-multiscreen-root-cause",
    }


def test_round2_blind_review_export_is_reproducible(tmp_path):
    output_dir = tmp_path / "blind_review_bundle"
    result = export_heldout_blind_review_bundle(
        output_dir=output_dir,
    )
    assert result["case_count"] == 5
    assert result["route_counts"] == {
        "hyde_fallback": 4,
        "ask_clarification": 1,
    }
    assert result["contamination_scan"] == "passed"
