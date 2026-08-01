import hashlib
import json
from pathlib import Path

import pytest

from app.rag.evaluation.grpo_case_exporter import (
    GrpoCaseExportManifest,
    GrpoTrainingCase,
    load_grpo_training_cases,
)
from evaluation.stage9.grpo.export_sft_v2_grpo_cases import build_grpo_case_dataset


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_grpo_case_export_reuses_all_frozen_75_cases(tmp_path: Path) -> None:
    output = tmp_path / "grpo-cases"
    manifest = build_grpo_case_dataset(output)
    cases = load_grpo_training_cases(output / "grpo_train_cases.jsonl")

    assert manifest.case_count == 75
    assert manifest.unique_query_count == 75
    assert len(cases) == 75
    assert len({row.case_contract.case_id for row in cases}) == 75
    assert len({row.record_fingerprint for row in cases}) == 75
    assert set(manifest.origin_counts) == {"old_retained_reviewed", "round3_approved"}
    assert manifest.origin_counts == {"old_retained_reviewed": 37, "round3_approved": 38}


def test_grpo_case_export_contains_only_reviewed_train_contracts(tmp_path: Path) -> None:
    output = tmp_path / "grpo-cases"
    manifest = build_grpo_case_dataset(output)
    cases = load_grpo_training_cases(output / "grpo_train_cases.jsonl")

    assert manifest.all_cases_train_only is True
    assert manifest.all_cases_reviewed is True
    assert manifest.all_reference_routes_accepted is True
    assert manifest.rollout_generated is False
    assert manifest.training_performed is False
    for row in cases:
        assert row.case_contract.split.value == "train"
        assert row.case_contract.human_review_status.value == "reviewed"
        assert row.reference_trajectory.route in row.case_contract.acceptable_action_paths


def test_grpo_case_export_manifest_hashes_outputs(tmp_path: Path) -> None:
    output = tmp_path / "grpo-cases"
    build_grpo_case_dataset(output)
    manifest = json.loads((output / "grpo_case_manifest.json").read_text(encoding="utf-8"))
    parsed_manifest = GrpoCaseExportManifest.model_validate(manifest)

    for name, expected in manifest["output_file_sha256"].items():
        assert _sha256(output / name) == expected
    assert manifest["source_dataset_fingerprint"] == (
        "92e8146962d6fddc44a8fc1c001ff5306bc01cb6d2c3c7f0866db8d15aeae0e6"
    )
    assert manifest["rollout_source"] == "training_time_policy_sampling"
    assert parsed_manifest.case_count == 75


def test_grpo_case_schema_rejects_pending_case(tmp_path: Path) -> None:
    output = tmp_path / "grpo-cases"
    build_grpo_case_dataset(output)
    raw = json.loads((output / "grpo_train_cases.jsonl").read_text(encoding="utf-8").splitlines()[0])
    raw["case_contract"]["human_review_status"] = "pending"

    with pytest.raises(ValueError, match="必须已经人工审核"):
        GrpoTrainingCase.model_validate(raw)
