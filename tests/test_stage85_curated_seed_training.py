from pathlib import Path

import pytest

from app.rag.evaluation.case_schema import GoldOrigin, PlannerEvalCase
from evaluation.stage8_5.pipelines.common.paths import stage85_layout
from evaluation.stage8_5.pipelines.common.stage85_schema import read_jsonl
from evaluation.stage8_5.pipelines.curated_gold.build_source_grounded_gold import GoldCaseAudit
from evaluation.stage8_5.pipelines.sft_seed.prepare_curated_seed_training import (
    prepare_curated_seed_cases,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE85_DIR = PROJECT_ROOT / "evaluation/stage8_5"
LAYOUT = stage85_layout(STAGE85_DIR)


def test_stage85_curated_seed_cases_are_training_ready_and_train_only():
    cases = read_jsonl(LAYOUT.curated_intermediate / "gold_cases_indexed.jsonl", PlannerEvalCase)
    audits = read_jsonl(LAYOUT.curated_review / "gold_case_audit.jsonl", GoldCaseAudit)

    prepared, manifest = prepare_curated_seed_cases(
        indexed_cases=cases,
        audits=audits,
        snapshot_id="stage85-env-test-v2",
        created_at="2026-07-21T00:00:00+00:00",
    )

    assert len(prepared) == 20
    assert {case.gold_origin for case in prepared} == {GoldOrigin.CURATED_SEED_GOLD}
    assert all(case.expected_subject_ids for case in prepared)
    assert all(len(case.expected_subject_names) == 1 for case in prepared)
    assert all(case.expected_identifiers == {} for case in prepared)
    assert all("second_agent_review=passed" in case.notes for case in prepared)
    assert all("training_status=approved" in case.notes for case in prepared)
    assert len(manifest.train_case_ids) == 20
    assert manifest.dev_case_ids == []
    assert manifest.test_case_ids == []


def test_stage85_curated_seed_requires_passed_second_review():
    cases = read_jsonl(LAYOUT.curated_intermediate / "gold_cases_indexed.jsonl", PlannerEvalCase)
    audits = read_jsonl(LAYOUT.curated_review / "gold_case_audit.jsonl", GoldCaseAudit)
    audits[0] = audits[0].model_copy(update={"second_review_status": "pending"})

    with pytest.raises(ValueError, match="独立二审未通过"):
        prepare_curated_seed_cases(
            indexed_cases=cases,
            audits=audits,
            snapshot_id="stage85-env-test-v2",
        )
