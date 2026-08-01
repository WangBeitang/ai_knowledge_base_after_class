import json
from pathlib import Path

from evaluation.stage9.model_planner.sft_dataset import load_sft_samples
from evaluation.stage9.sft_v2.freeze_sft_v2_reviewed_dataset import build_freeze


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_freeze_builds_only_reviewed_75(tmp_path: Path) -> None:
    output = tmp_path / "frozen"
    manifest = build_freeze(output)

    samples = load_sft_samples(output / "sft_v2_train.jsonl")
    trajectories = _read_jsonl(output / "sft_v2_trajectory_index.jsonl")
    assert manifest["freeze_status"] == "frozen_with_legacy_exceptions"
    assert manifest["trajectory_count"] == 75
    assert manifest["action_sample_count"] == 163
    assert len(samples) == 163
    assert len(trajectories) == 75
    assert len({row["source_trace_id"] for row in trajectories}) == 75
    assert {sample.review_status for sample in samples} == {"reviewed"}
    assert {sample.artifact_status.value for sample in samples} == {"approved_training_seed"}


def test_freeze_excludes_all_rejected_candidate_versions(tmp_path: Path) -> None:
    output = tmp_path / "frozen"
    manifest = build_freeze(output)
    samples = load_sft_samples(output / "sft_v2_train.jsonl")
    case_ids = {sample.source_case_id for sample in samples}

    assert manifest["excluded"]["round3_rejected_original_candidate_count"] == 87
    assert manifest["excluded"]["salvage_review_rejected_candidate_count"] == 8
    assert not case_ids.intersection(manifest["excluded"]["salvage_review_rejected_ids"])
    assert manifest["validation"]["split_query_leak_count"] == 0
    assert manifest["validation"]["split_chunk_leak_count"] == 0
    assert manifest["validation"]["new_or_cross_generation_near_duplicate_count"] == 0


def test_freeze_keeps_approved_new_fingerprints_and_provider_records(tmp_path: Path) -> None:
    output = tmp_path / "frozen"
    manifest = build_freeze(output)
    trajectories = _read_jsonl(output / "sft_v2_approved_new_trajectories.jsonl")
    provider_rows = _read_jsonl(output / "sft_v2_provider_observations.jsonl")

    required_ids = {record_id for row in trajectories for record_id in row["provider_record_ids"]}
    assert len(trajectories) == 38
    assert len({row["content_fingerprint"] for row in trajectories}) == 38
    assert required_ids == {row["record_id"] for row in provider_rows}
    assert len(provider_rows) == manifest["provider_observation_count"] == 41
