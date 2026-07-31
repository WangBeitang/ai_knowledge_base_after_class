import hashlib
import json
from pathlib import Path

import pytest

from evaluation.stage9.reward_validation.regress_reward_v1_1_replay import (
    DEFAULT_OUTPUT,
    DEFAULT_REPORT,
    RewardV11ReplayRegression,
    render_report,
    run_regression,
    write_outputs,
)
from evaluation.stage9.reward_validation.validate_reward_v1_1_balanced_dev import (
    DEFAULT_PROFILE,
    DEFAULT_REWARD_IMPLEMENTATION,
    EXPECTED_PROFILE_SHA256,
    EXPECTED_REWARD_IMPLEMENTATION_SHA256,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reward_v1_1_replay_regression_passes_without_mutation():
    profile_before = _sha256(DEFAULT_PROFILE)
    reward_before = _sha256(DEFAULT_REWARD_IMPLEMENTATION)

    output = run_regression()

    assert output.summary.decision == "pass_keep_v1_1"
    assert output.summary.case_count == 25
    assert output.source_trajectory_count == 231
    assert output.scored_trajectory_count == 179
    assert output.skipped_missing_observation_count == 52
    assert output.summary.trajectory_count == 179
    assert output.summary.inversion_count == 0
    assert output.summary.minimum_case_margin == pytest.approx(0.07)
    assert output.summary.minimum_route_margin == pytest.approx(0.1076923077)
    assert output.critical_anti_pattern_counts == {}
    assert output.action_provider == "replay_action_provider"
    assert output.model_execution_performed is False
    assert output.heldout_inference_result_count == 0
    assert all(
        case.correct_trajectory_count >= 1
        and case.incorrect_trajectory_count >= 1
        for case in output.cases
    )
    assert all(
        path.reason == "missing_non_required_replay_observation"
        for path in output.skipped_paths
    )
    assert _sha256(DEFAULT_PROFILE) == profile_before == EXPECTED_PROFILE_SHA256
    assert (
        _sha256(DEFAULT_REWARD_IMPLEMENTATION)
        == reward_before
        == EXPECTED_REWARD_IMPLEMENTATION_SHA256
    )


def test_checked_in_replay_regression_and_report_are_reproducible():
    checked_in = RewardV11ReplayRegression.model_validate_json(
        DEFAULT_OUTPUT.read_text(encoding="utf-8")
    )
    regenerated = run_regression()

    assert regenerated == checked_in
    assert render_report(regenerated) == DEFAULT_REPORT.read_text(encoding="utf-8")


def test_regression_rejects_modified_frozen_replay_contract(tmp_path: Path):
    output = run_regression()
    contract_path = Path(output.inputs["replay_contract_9_3_18"].path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["records_sha256"] = "0" * 64
    modified = tmp_path / "modified_contract.json"
    modified.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Replay contract 与当前"):
        run_regression(replay_contract_path=modified)


def test_regression_outputs_refuse_silent_overwrite(tmp_path: Path):
    output = run_regression()
    output_path = tmp_path / "regression.json"
    report_path = tmp_path / "report.md"
    write_outputs(output, output_path=output_path, report_path=report_path)

    with pytest.raises(FileExistsError):
        write_outputs(output, output_path=output_path, report_path=report_path)
