import json
from pathlib import Path

import pytest

from app.rag.evaluation.case_schema import PlannerEvalCase
from evaluation.stage8_5.pipelines.common.paths import stage85_layout
from evaluation.stage8_5.pipelines.common.stage85_schema import read_jsonl
from evaluation.stage8_5.pipelines.curated_gold.build_source_grounded_gold import (
    GoldCaseAudit,
    GoldEvidenceChunk,
)
from evaluation.stage8_5.pipelines.curated_gold.import_gold_evidence import (
    GoldChunkRuntimeBinding,
    GoldImportManifest,
    SecondReviewDecision,
    build_prepared_documents,
    build_runtime_artifacts,
    reuse_frozen_snapshot_if_compatible,
    validate_second_review,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE85_DIR = PROJECT_ROOT / "evaluation/stage8_5"
LAYOUT = stage85_layout(STAGE85_DIR)


def test_stage85_second_review_and_runtime_binding_form_closed_snapshot_inputs():
    cases = read_jsonl(LAYOUT.curated_intermediate / "gold_cases_authoring.jsonl", PlannerEvalCase)
    audits = read_jsonl(LAYOUT.curated_review / "gold_case_audit.jsonl", GoldCaseAudit)
    evidence_chunks = read_jsonl(
        LAYOUT.curated_intermediate / "gold_evidence_chunks.jsonl",
        GoldEvidenceChunk,
    )
    decisions = read_jsonl(
        LAYOUT.curated_review / "gold_case_second_review.jsonl",
        SecondReviewDecision,
    )

    validate_second_review(cases, decisions)
    prepared_documents = build_prepared_documents(evidence_chunks)
    assert len(prepared_documents) == 2
    assert sum(len(document["chunks"]) for document in prepared_documents.values()) == 10

    bindings = []
    for chunk_id, evidence in enumerate(evidence_chunks, start=9001):
        bindings.append(GoldChunkRuntimeBinding(
            evidence_key=evidence.chunk_id,
            chunk_id=chunk_id,
            document_id=evidence.document_id,
            index_version=evidence.index_version,
            source_id=evidence.source_id,
            content_sha256="a" * 64,
        ))

    indexed_cases, verified_audits, runtime_bindings, split_manifest = build_runtime_artifacts(
        cases=cases,
        audits=audits,
        bindings=bindings,
        second_review_path=LAYOUT.curated_review / "gold_case_second_review.jsonl",
        snapshot_id="stage85-env-test-v1",
        split_created_at="2026-07-21T00:00:00+00:00",
    )

    assert len(indexed_cases) == 20
    assert len(runtime_bindings) == 20
    assert all(isinstance(case.expected_chunks[0].chunk_id, int) for case in indexed_cases)
    assert all(case.expected_chunks[0].chunk_id >= 9001 for case in indexed_cases)
    assert {audit.second_review_status for audit in verified_audits} == {"passed"}
    assert {audit.second_reviewer_type for audit in verified_audits} == {"independent_agent"}
    assert split_manifest.snapshot_id == "stage85-env-test-v1"
    assert split_manifest.created_at == "2026-07-21T00:00:00+00:00"
    assert len(split_manifest.train_case_ids) == 20
    assert split_manifest.dev_case_ids == []
    assert split_manifest.test_case_ids == []

    # 已物化 manifest 也必须显式保存记录数，不能让审计者只能通过重新读取文件推断。
    manifest = GoldImportManifest.model_validate(json.loads(
        (LAYOUT.curated_intermediate / "gold_import_manifest.json").read_text(encoding="utf-8")
    ))
    assert manifest.second_review_count == 20
    assert manifest.indexed_case_count == 20


def test_stage85_second_review_gate_rejects_any_case_needing_fix():
    case = _minimal_case()
    decision = SecondReviewDecision(
        case_id=case.case_id,
        decision="needs_fix",
        confidence="high",
        incorrect_fact_ids=["fact-1"],
    )

    with pytest.raises(ValueError, match="二审未通过"):
        validate_second_review([case], [decision])


def test_stage85_existing_snapshot_is_reused_only_when_inputs_match():
    snapshot_path = LAYOUT.curated_intermediate / "environment_snapshot_import_v1.json"
    snapshot_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    cases = read_jsonl(LAYOUT.curated_intermediate / "gold_cases_indexed.jsonl", PlannerEvalCase)

    result = reuse_frozen_snapshot_if_compatible(
        snapshot_path,
        snapshot_id=snapshot_payload["snapshot_id"],
        cases=cases,
        source_hashes=snapshot_payload["source_hashes"],
    )
    assert result is not None
    assert result.snapshot.created_at == snapshot_payload["created_at"]
    assert result.summary["case_reference_check"] == "passed"

    with pytest.raises(RuntimeError, match="输入文件 hash 已变化"):
        reuse_frozen_snapshot_if_compatible(
            snapshot_path,
            snapshot_id=snapshot_payload["snapshot_id"],
            cases=cases,
            source_hashes={"changed.jsonl": "b" * 64},
        )


def _minimal_case() -> PlannerEvalCase:
    return PlannerEvalCase.model_validate({
        "case_id": "gold-test-1",
        "case_group": "core",
        "split": "train",
        "leakage_group_id": "gold-test-group",
        "query": "测试问题",
        "dataset_ids": ["dataset_default_equipment_ops"],
        "owner_user_id": "eval_demo_user",
        "source_document_ids": ["doc-test"],
        "source_index_versions": {"doc-test": 1},
        "expected_chunks": [{
            "document_id": "doc-test",
            "chunk_id": "chunk-test",
            "index_version": 1,
            "answer_point_ids": ["ap-1"],
        }],
        "expected_answer_points": ["测试答案"],
        "expected_behavior": {
            "should_answer": True,
            "should_refuse": False,
            "should_ask_clarification": False,
            "should_call_web": False,
        },
        "acceptable_action_paths": [["local_search", "answer"]],
        "label_source": "api_assisted",
        "human_review_status": "reviewed",
    })
