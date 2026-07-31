import hashlib
import json
from pathlib import Path

import pytest

from evaluation.stage9.reward_validation.validate_reward_v1_1_balanced_dev import (
    DEFAULT_OUTPUT,
    DEFAULT_PROFILE,
    DEFAULT_REPORT,
    DEFAULT_REWARD_IMPLEMENTATION,
    EXPECTED_PROFILE_SHA256,
    EXPECTED_REWARD_IMPLEMENTATION_SHA256,
    RewardV11BalancedDevValidation,
    render_report,
    run_validation,
    write_outputs,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_historical_validation_remains_readable_and_reward_immutable():
    output = RewardV11BalancedDevValidation.model_validate_json(
        DEFAULT_OUTPUT.read_text(encoding="utf-8")
    )

    assert output.summary.decision == "pass_keep_v1_1"
    assert output.summary.case_count == 25
    assert output.summary.trajectory_count == 231
    assert output.summary.route_bucket_count == 5
    assert output.summary.inversion_count == 0
    assert output.summary.minimum_case_margin == pytest.approx(0.07)
    assert output.summary.minimum_route_margin == pytest.approx(0.1076923077)
    assert output.summary.answer_placeholder_case_count == 15
    assert len(output.balanced_dev_case_ids) == 25
    assert output.non_dev_case_count_ignored == 71
    assert {bucket.route_bucket.value: bucket.case_count for bucket in output.buckets} == {
        "local_answer": 5,
        "hyde_fallback": 5,
        "web_required": 5,
        "ask_clarification": 5,
        "safe_refuse": 5,
    }
    assert all(case.attribution == "no_issue" for case in output.cases)
    assert output.profile_mutation_performed is False
    assert output.reward_mutation_performed is False
    assert output.model_execution_performed is False
    assert output.heldout_inference_result_count == 0
    assert _sha256(DEFAULT_PROFILE) == EXPECTED_PROFILE_SHA256
    assert (
        _sha256(DEFAULT_REWARD_IMPLEMENTATION)
        == EXPECTED_REWARD_IMPLEMENTATION_SHA256
    )
    assert render_report(output) == DEFAULT_REPORT.read_text(encoding="utf-8")


def test_historical_validation_rejects_current_case_registry_drift():
    with pytest.raises(
        ValueError,
        match="EnvironmentSnapshot 未绑定当前 planner_cases SHA256",
    ):
        run_validation()


def test_validation_rejects_silently_modified_v1_1_profile(tmp_path: Path):
    profile = json.loads(DEFAULT_PROFILE.read_text(encoding="utf-8"))
    profile["weights"]["behavior"] = 0.34
    modified_profile = tmp_path / "modified_reward_v1_1.json"
    modified_profile.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="profile SHA256 已变化"):
        run_validation(profile_path=modified_profile)


def test_validation_outputs_refuse_silent_overwrite(tmp_path: Path):
    output = RewardV11BalancedDevValidation.model_validate_json(
        DEFAULT_OUTPUT.read_text(encoding="utf-8")
    )
    output_path = tmp_path / "validation.json"
    report_path = tmp_path / "report.md"
    write_outputs(output, output_path=output_path, report_path=report_path)

    with pytest.raises(FileExistsError):
        write_outputs(output, output_path=output_path, report_path=report_path)
