import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from app.rag.evaluation.case_schema import PlannerEvalCase
from app.rag.query.contracts import QueryAction
from evaluation.stage9.balanced_dev.build_balanced_dev_cases import (
    CASE_SPECS,
    DEFAULT_CASES_PATH,
    DEFAULT_MATRIX,
    DEFAULT_RETIRED,
    DEFAULT_SOURCE_IMPORT,
    DEFAULT_SPLIT_PATH,
    DEFAULT_SUPERSEDED,
    DEFAULT_WEB_EVIDENCE,
    HYDE_PROBES,
    RETIRED_PENDING_DEV_IDS,
    SUPERSEDED_ROUND3_CASE_IDS,
    _case_spec_fingerprint,
    build_balanced_dev,
)


def _read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _temp_paths(tmp_path: Path):
    (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    return {
        "evidence_path": tmp_path / "artifacts/evidence.jsonl",
        "review_queue_path": tmp_path / "artifacts/review_queue.jsonl",
        "retired_path": tmp_path / "artifacts/retired.jsonl",
        "superseded_path": tmp_path / "artifacts/superseded.jsonl",
        "build_manifest_path": tmp_path / "artifacts/build_manifest.json",
        "report_path": tmp_path / "reports/report.md",
    }


def test_balanced_dev_artifacts_have_five_routes_and_real_chunk_identity():
    cases = [
        PlannerEvalCase.model_validate(row)
        for row in _read_jsonl(DEFAULT_CASES_PATH)
    ]
    dev = [case for case in cases if case.split.value == "dev"]
    assert len(dev) == 25

    route_counts = Counter(
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
        for case in dev
    )
    assert route_counts == {
        "local_answer": 5,
        "hyde_fallback": 5,
        "web_required": 5,
        "ask_clarification": 5,
        "safe_refuse": 5,
    }
    assert len({case.leakage_group_id for case in dev}) == 25

    source_import = json.loads(DEFAULT_SOURCE_IMPORT.read_text(encoding="utf-8"))
    frozen_chunks = {
        (
            str(chunk["document_id"]),
            str(chunk["chunk_id"]),
            int(chunk["index_version"]),
        )
        for document in source_import["documents"]
        for chunk in document["chunks"]
    }
    pending_case_ids = {
        row["case_id"]
        for row in _read_jsonl(
            Path("evaluation/stage9/artifacts/balanced_dev/second_review_queue.jsonl")
        )
    }
    balanced_cases = [
        case for case in dev if case.case_id.startswith("planner-dev-balanced-")
    ]
    # 9.3.18 的真实 Provider 探针证明两条 safe_refuse query 无法稳定召回安全证据。
    # 修订 query 后旧 fingerprint 审核必须失效，不能为了保持测试全绿而沿用旧批准。
    assert Counter(case.human_review_status.value for case in balanced_cases) == {
        "reviewed": 19,
        "pending": 2,
    }
    assert pending_case_ids == {
        "planner-dev-balanced-refuse-b5-force-pull-paper",
        "planner-dev-balanced-refuse-p5-touch-hot-surface",
    }
    assert {
        case.case_id
        for case in balanced_cases
        if case.human_review_status.value == "pending"
    } == pending_case_ids

    for case in dev:
        if not case.case_id.startswith("planner-dev-balanced-"):
            continue
        if case.expected_behavior.should_call_web:
            assert case.gold_origin.value == "heldout_gold"
            assert case.expected_chunks == []
            assert case.expected_web_evidence
            assert case.acceptable_action_paths == [
                [QueryAction.WEB_SEARCH, QueryAction.ANSWER]
            ]
        else:
            assert case.gold_origin.value == "production_chunk_gold"
            assert case.source_document_ids
            assert case.expected_chunks
            for chunk in case.expected_chunks:
                assert (
                    chunk.document_id,
                    str(chunk.chunk_id),
                    chunk.index_version,
                ) in frozen_chunks

    cases_by_id = {case.case_id: case for case in balanced_cases}
    for case_id in {
        "planner-dev-balanced-ask-printer-network-reset-model",
        "planner-dev-balanced-ask-id-copy-model",
    }:
        assert cases_by_id[case_id].expected_subject_ids == []
        assert cases_by_id[case_id].expected_subject_names == []


def test_hyde_cases_have_pre_result_retrieval_probe():
    hyde_specs = [spec for spec in CASE_SPECS if spec.route_bucket == "hyde_fallback"]
    assert len(hyde_specs) == 5
    assert {spec.hyde_probe_id for spec in hyde_specs} == set(HYDE_PROBES)
    for probe in HYDE_PROBES.values():
        assert probe["hypothetical_target_rank"] == 1
        assert probe["original_target_rank"] is None


def test_builder_is_idempotent_and_does_not_fake_second_review(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    split_path = tmp_path / "split.json"
    source_path = tmp_path / "source_import.json"
    web_evidence_path = tmp_path / "web_evidence.json"
    matrix_path = tmp_path / "matrix.json"
    decisions_path = tmp_path / "missing_decisions.jsonl"
    shutil.copy2(DEFAULT_CASES_PATH, cases_path)
    shutil.copy2(DEFAULT_SPLIT_PATH, split_path)
    shutil.copy2(DEFAULT_SOURCE_IMPORT, source_path)
    shutil.copy2(DEFAULT_WEB_EVIDENCE, web_evidence_path)
    shutil.copy2(DEFAULT_MATRIX, matrix_path)
    paths = _temp_paths(tmp_path)
    shutil.copy2(DEFAULT_RETIRED, paths["retired_path"])
    shutil.copy2(DEFAULT_SUPERSEDED, paths["superseded_path"])

    first = build_balanced_dev(
        cases_path=cases_path,
        split_path=split_path,
        source_import_path=source_path,
        web_evidence_path=web_evidence_path,
        matrix_path=matrix_path,
        decisions_path=decisions_path,
        overwrite=True,
        **paths,
    )
    second = build_balanced_dev(
        cases_path=cases_path,
        split_path=split_path,
        source_import_path=source_path,
        web_evidence_path=web_evidence_path,
        matrix_path=matrix_path,
        decisions_path=decisions_path,
        overwrite=True,
        **paths,
    )

    assert first["inventory"] == second["inventory"]
    assert second["new_candidate_count"] == 21
    assert second["retired_pending_dev_count"] == 3
    assert second["pending_second_review_count"] == 21
    assert second["inventory"]["review_gate_passed"] is False
    assert len(_read_jsonl(paths["review_queue_path"])) == 21
    assert {
        row["case_id"] for row in _read_jsonl(paths["retired_path"])
    } == set(RETIRED_PENDING_DEV_IDS)
    assert {
        row["case_id"] for row in _read_jsonl(paths["superseded_path"])
    } == SUPERSEDED_ROUND3_CASE_IDS


def test_only_explicit_independent_approvals_can_mark_new_cases_reviewed(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    split_path = tmp_path / "split.json"
    source_path = tmp_path / "source_import.json"
    web_evidence_path = tmp_path / "web_evidence.json"
    matrix_path = tmp_path / "matrix.json"
    decisions_path = tmp_path / "decisions.jsonl"
    shutil.copy2(DEFAULT_CASES_PATH, cases_path)
    shutil.copy2(DEFAULT_SPLIT_PATH, split_path)
    shutil.copy2(DEFAULT_SOURCE_IMPORT, source_path)
    shutil.copy2(DEFAULT_WEB_EVIDENCE, web_evidence_path)
    shutil.copy2(DEFAULT_MATRIX, matrix_path)
    paths = _temp_paths(tmp_path)
    shutil.copy2(DEFAULT_RETIRED, paths["retired_path"])
    shutil.copy2(DEFAULT_SUPERSEDED, paths["superseded_path"])

    decisions_path.write_text(
        "".join(
            json.dumps(
                {
                    "case_id": spec.case_id,
                    "case_fingerprint": _case_spec_fingerprint(spec),
                    "decision": "approved",
                    "reviewer_id": "independent-review-fixture",
                    "reviewer_role": "independent_agent",
                    "reviewed_at": "2026-07-28T06:00:00+00:00",
                    "evidence_check": "fixture passed",
                    "route_check": "fixture passed",
                    "leakage_check": "fixture passed",
                    "notes": "测试只验证状态门禁，不代表真实审核。",
                },
                ensure_ascii=False,
            )
            + "\n"
            for spec in CASE_SPECS
        ),
        encoding="utf-8",
    )
    result = build_balanced_dev(
        cases_path=cases_path,
        split_path=split_path,
        source_import_path=source_path,
        web_evidence_path=web_evidence_path,
        matrix_path=matrix_path,
        decisions_path=decisions_path,
        overwrite=True,
        **paths,
    )

    assert result["inventory"]["review_gate_passed"] is True
    assert result["pending_second_review_count"] == 0
    assert all(
        count == 5 for count in result["inventory"]["reviewed_counts"].values()
    )


def test_builder_rejects_stale_review_fingerprint(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    split_path = tmp_path / "split.json"
    source_path = tmp_path / "source_import.json"
    web_evidence_path = tmp_path / "web_evidence.json"
    matrix_path = tmp_path / "matrix.json"
    decisions_path = tmp_path / "decisions.jsonl"
    shutil.copy2(DEFAULT_CASES_PATH, cases_path)
    shutil.copy2(DEFAULT_SPLIT_PATH, split_path)
    shutil.copy2(DEFAULT_SOURCE_IMPORT, source_path)
    shutil.copy2(DEFAULT_WEB_EVIDENCE, web_evidence_path)
    shutil.copy2(DEFAULT_MATRIX, matrix_path)
    paths = _temp_paths(tmp_path)
    shutil.copy2(DEFAULT_RETIRED, paths["retired_path"])
    shutil.copy2(DEFAULT_SUPERSEDED, paths["superseded_path"])

    spec = CASE_SPECS[0]
    decisions_path.write_text(
        json.dumps(
            {
                "case_id": spec.case_id,
                "case_fingerprint": "0" * 64,
                "decision": "approved",
                "reviewer_id": "independent-review-fixture",
                "reviewer_role": "independent_agent",
                "reviewed_at": "2026-07-28T06:00:00+00:00",
                "evidence_check": "fixture passed",
                "route_check": "fixture passed",
                "leakage_check": "fixture passed",
                "notes": "故意使用失效 fingerprint。",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="case_fingerprint 已失效"):
        build_balanced_dev(
            cases_path=cases_path,
            split_path=split_path,
            source_import_path=source_path,
            web_evidence_path=web_evidence_path,
            matrix_path=matrix_path,
            decisions_path=decisions_path,
            overwrite=True,
            **paths,
        )
