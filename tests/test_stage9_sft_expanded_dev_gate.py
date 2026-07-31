import hashlib
import json
from pathlib import Path

import pytest

from app.rag.evaluation.baseline_runner import BaselineEvalOutput
from app.rag.evaluation.case_schema import load_planner_cases
from app.rag.query.contracts import PlannerDecision, PlannerReasonCode, QueryAction
from evaluation.stage9.admission.run_sft_expanded_dev_gate import (
    ADMISSION_VERSION,
    AdmissionDecision,
    DEFAULT_CASES,
    DEFAULT_SNAPSHOT,
    load_admission_contract,
    load_and_build_admission,
    main,
    render_report,
    write_admission_outputs,
    _case_admission,
)
from evaluation.stage9.admission.run_sft_v1_corrected_replay_eval import (
    AttributionCategory,
    build_corrected_replay_evaluation,
    load_corrected_replay_preflight,
    main as corrected_replay_main,
)
from evaluation.stage9.providers.record_expanded_dev_observations import (
    DEFAULT_RECORDS,
)
from evaluation.stage9.model_planner.eval_model_planner import (
    run_model_planner_eval,
)
from evaluation.stage9.providers.validate_expanded_dev_replay import (
    DEFAULT_CONTRACT_OUTPUT,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_expanded_dev_gate_allows_a_complete_five_route_candidate(tmp_path):
    checkpoint_dir = _write_checkpoint(tmp_path)
    eval_path = _write_eval(tmp_path, checkpoint_dir=checkpoint_dir)
    legacy_inputs = _write_legacy_contract_inputs(tmp_path)

    admission = load_and_build_admission(
        eval_output_path=eval_path,
        checkpoint_dir=checkpoint_dir,
        **legacy_inputs,
    )

    assert admission.admission_version == ADMISSION_VERSION
    assert admission.summary.decision == AdmissionDecision.ALLOW_STAGE9_4
    assert admission.summary.eligible_for_stage9_4 is True
    assert admission.summary.case_count == 25
    assert admission.summary.route_macro_accuracy == 1.0
    assert admission.summary.execution_failure_count == 0
    assert admission.summary.forbidden_action_count == 0
    assert admission.summary.safe_refuse_dangerous_false_release_count == 0
    assert admission.summary.actual_terminal_action_counts == {
        "answer": 15,
        "ask_clarification": 5,
        "refuse": 5,
    }
    assert admission.summary.terminal_confusion_matrix["answer"]["answer"] == 15
    assert admission.summary.failure_category_counts == {}
    assert len(admission.buckets) == 5
    assert all(bucket.case_count == 5 for bucket in admission.buckets)
    assert admission.heldout_inference_result_count == 0
    assert admission.retrieval_quality_verified is False
    assert admission.answer_quality_interpretable_as_model_quality is False
    assert admission.inputs["adapter_weights"].sha256
    assert admission.inputs["tokenizer_json"].sha256
    assert admission.balanced_dev_canonical_sha256 == (
        "3cd44738a185747578389a9484e2f6ed521644005578012b76b94d62a008c8cc"
    )

    report = render_report(admission)
    assert "允许进入 9.4" in report
    assert "heldout 推理结果数：`0`" in report
    assert "不证明真实 Milvus/Web 召回质量" in report


def test_expanded_dev_gate_blocks_a_dangerous_false_release(tmp_path):
    checkpoint_dir = _write_checkpoint(tmp_path)
    eval_path = _write_eval(
        tmp_path,
        checkpoint_dir=checkpoint_dir,
        unsafe_false_release=True,
    )
    legacy_inputs = _write_legacy_contract_inputs(tmp_path)

    admission = load_and_build_admission(
        eval_output_path=eval_path,
        checkpoint_dir=checkpoint_dir,
        **legacy_inputs,
    )

    assert admission.summary.decision == AdmissionDecision.TRAIN_SFT_V2
    assert admission.summary.eligible_for_stage9_4 is False
    assert admission.summary.safe_refuse_dangerous_false_release_count == 1
    safety_gate = next(
        gate
        for gate in admission.gate_checks
        if gate.name == "safe_refuse_dangerous_false_release_count"
    )
    assert safety_gate.passed is False
    failed = [
        case for case in admission.cases if case.safe_refuse_false_release
    ]
    assert len(failed) == 1
    assert "safe_refuse_false_release" in failed[0].failure_categories


def test_expanded_dev_gate_rejects_debug_checkpoint_before_model_load(tmp_path):
    checkpoint_dir = _write_checkpoint(tmp_path, training_backend="debug_memorized")
    legacy_inputs = _write_legacy_contract_inputs(tmp_path)

    with pytest.raises(ValueError, match="debug_memorized"):
        load_admission_contract(
            checkpoint_dir=checkpoint_dir,
            **legacy_inputs,
        )


def test_expanded_dev_preflight_does_not_execute_model_or_write_outputs(
    tmp_path,
    capsys,
):
    checkpoint_dir = _write_checkpoint(tmp_path)
    eval_output = tmp_path / "should_not_exist/eval.json"
    decision_output = tmp_path / "should_not_exist/decision.json"
    report_output = tmp_path / "should_not_exist/report.md"
    legacy_inputs = _write_legacy_contract_inputs(tmp_path)

    exit_code = main(
        [
            "--checkpoint",
            str(checkpoint_dir),
            "--eval-output",
            str(eval_output),
            "--decision-output",
            str(decision_output),
            "--report",
            str(report_output),
            "--snapshot",
            str(legacy_inputs["snapshot_path"]),
            "--reward-validation",
            str(legacy_inputs["reward_validation_path"]),
            "--preflight-only",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["preflight_only"] is True
    assert payload["model_execution_performed"] is False
    assert payload["heldout_inference_result_count"] == 0
    assert payload["case_count"] == 25
    assert not eval_output.exists()
    assert not decision_output.exists()
    assert not report_output.exists()


def test_expanded_dev_gate_rejects_non_dev_result_and_silent_overwrite(tmp_path):
    checkpoint_dir = _write_checkpoint(tmp_path)
    eval_path = _write_eval(tmp_path, checkpoint_dir=checkpoint_dir)
    legacy_inputs = _write_legacy_contract_inputs(tmp_path)
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    payload["results"][0]["case_id"] = "planner-test-core-001"
    eval_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="非 balanced dev"):
        load_and_build_admission(
            eval_output_path=eval_path,
            checkpoint_dir=checkpoint_dir,
            **legacy_inputs,
        )

    clean_eval = _write_eval(
        tmp_path,
        checkpoint_dir=checkpoint_dir,
        filename="clean_eval.json",
    )
    admission = load_and_build_admission(
        eval_output_path=clean_eval,
        checkpoint_dir=checkpoint_dir,
        **legacy_inputs,
    )
    decision_path = tmp_path / "outputs/decision.json"
    report_path = tmp_path / "outputs/report.md"
    write_admission_outputs(
        admission,
        decision_output_path=decision_path,
        report_path=report_path,
    )
    with pytest.raises(FileExistsError, match="拒绝静默覆盖"):
        write_admission_outputs(
            admission,
            decision_output_path=decision_path,
            report_path=report_path,
        )


def test_corrected_replay_preflight_binds_old_eval_and_writes_nothing(
    tmp_path,
    capsys,
):
    checkpoint_dir = _write_checkpoint(tmp_path)
    old_eval = _write_eval(tmp_path, checkpoint_dir=checkpoint_dir)
    old_decision = _write_legacy_decision(
        tmp_path,
        eval_path=old_eval,
        checkpoint_dir=checkpoint_dir,
    )
    corrected_eval = tmp_path / "outputs/corrected_eval.json"
    comparison = tmp_path / "outputs/comparison.json"
    report = tmp_path / "outputs/report.md"

    exit_code = corrected_replay_main(
        [
            "--checkpoint",
            str(checkpoint_dir),
            "--old-eval",
            str(old_eval),
            "--old-decision",
            str(old_decision),
            "--corrected-eval-output",
            str(corrected_eval),
            "--comparison-output",
            str(comparison),
            "--report",
            str(report),
            "--preflight-only",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["preflight_only"] is True
    assert payload["model_execution_performed"] is False
    assert payload["heldout_inference_result_count"] == 0
    assert payload["case_count"] == 25
    assert payload["provider_records_sha256"] == (
        "3513e5e550dbe182ce55b9d7c3e461b280a9bc2ad64ed51aa5cc2747d0a1e1e7"
    )
    assert not corrected_eval.exists()
    assert not comparison.exists()
    assert not report.exists()


def test_corrected_replay_comparison_marks_old_false_negative_as_corrected(
    tmp_path,
):
    checkpoint_dir = _write_checkpoint(tmp_path)
    old_eval = _write_eval(
        tmp_path,
        checkpoint_dir=checkpoint_dir,
        unsafe_false_release=True,
        filename="old_eval.json",
    )
    old_decision = _write_legacy_decision(
        tmp_path,
        eval_path=old_eval,
        checkpoint_dir=checkpoint_dir,
    )
    new_eval_path = _write_eval(
        tmp_path,
        checkpoint_dir=checkpoint_dir,
        filename="new_eval.json",
    )
    _rewrite_as_replay_eval(new_eval_path)
    preflight = load_corrected_replay_preflight(
        checkpoint_dir=checkpoint_dir,
        old_eval_path=old_eval,
        old_decision_path=old_decision,
    )
    new_eval = BaselineEvalOutput.model_validate_json(
        new_eval_path.read_text(encoding="utf-8")
    )

    comparison = build_corrected_replay_evaluation(
        preflight=preflight,
        new_eval=new_eval,
        old_eval_path=old_eval,
        corrected_eval_path=tmp_path / "final/corrected_eval.json",
        corrected_eval_content_path=new_eval_path,
        provider_records_path=DEFAULT_RECORDS,
        replay_contract_path=DEFAULT_CONTRACT_OUTPUT,
    )

    assert comparison.summary.eligible_for_stage9_4 is False
    assert comparison.heldout_inference_result_count == 0
    assert comparison.action_provider == "replay_action_provider"
    assert comparison.summary.old_route_correct_count == 24
    assert comparison.summary.new_route_correct_count == 25
    assert comparison.summary.route_correct_delta == 1
    assert len(
        comparison.summary.provider_false_negative_corrected_case_ids
    ) == 1
    corrected_case = next(
        case for case in comparison.cases
        if case.attribution
        == AttributionCategory.PROVIDER_FALSE_NEGATIVE_CORRECTED
    )
    assert corrected_case.route_bucket.value == "safe_refuse"
    assert corrected_case.old_observation.full_observation_available is False


def test_corrected_replay_preflight_rejects_frozen_record_hash_drift(tmp_path):
    checkpoint_dir = _write_checkpoint(tmp_path)
    old_eval = _write_eval(tmp_path, checkpoint_dir=checkpoint_dir)
    old_decision = _write_legacy_decision(
        tmp_path,
        eval_path=old_eval,
        checkpoint_dir=checkpoint_dir,
    )
    replay_contract = json.loads(
        DEFAULT_CONTRACT_OUTPUT.read_text(encoding="utf-8")
    )
    replay_contract["records_sha256"] = "0" * 64
    bad_contract = tmp_path / "bad_replay_contract.json"
    bad_contract.write_text(
        json.dumps(replay_contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="输入已偏离 9.3.19"):
        load_corrected_replay_preflight(
            checkpoint_dir=checkpoint_dir,
            old_eval_path=old_eval,
            old_decision_path=old_decision,
            replay_contract_path=bad_contract,
        )


def test_corrected_replay_eval_persists_structured_trace_evidence(
    tmp_path,
    monkeypatch,
):
    case = next(
        case
        for case in load_planner_cases(DEFAULT_CASES)
        if case.case_id == "planner-dev-balanced-local-rs12-10a-current"
    )

    class FakePlanner:
        policy_version = "test-corrected-replay"

        def plan(self, context):
            return PlannerDecision(
                action=(
                    QueryAction.LOCAL_SEARCH
                    if context.planner_step == 0
                    else QueryAction.ANSWER
                ),
                query=case.query,
                reason_code=(
                        PlannerReasonCode.INITIAL_LOCAL_SEARCH
                        if context.planner_step == 0
                        else PlannerReasonCode.LOCAL_EVIDENCE_SUFFICIENT
                ),
            )

    monkeypatch.setattr(
        "evaluation.stage9.model_planner.eval_model_planner."
        "ModelPlanner.from_checkpoint",
        lambda _path: FakePlanner(),
    )

    output = run_model_planner_eval(
        checkpoint_dir=tmp_path / "unused-checkpoint",
        cases=[case],
        snapshot_path=DEFAULT_SNAPSHOT,
        split="dev",
        provider_name="replay",
        provider_records_path=DEFAULT_RECORDS,
        include_trace_evidence=True,
    )

    trace_steps = output.results[0].usage["trace_steps"]
    assert len(trace_steps) == 2
    assert trace_steps[0]["decision"]["action"] == "local_search"
    assert trace_steps[0]["decision"]["query"] == case.query
    assert trace_steps[0]["output_observation"]["status"] == "success"
    assert trace_steps[0]["output_observation"]["candidate_count"] == 5
    assert trace_steps[1]["decision"]["action"] == "answer"
    assert trace_steps[1]["output_observation"] is None


def _write_checkpoint(
    tmp_path: Path,
    *,
    training_backend: str = "transformers_causal_lm",
) -> Path:
    run_id = f"candidate-{training_backend}"
    checkpoint_dir = tmp_path / run_id
    adapter_dir = checkpoint_dir / "model/adapter"
    tokenizer_dir = checkpoint_dir / "tokenizer"
    adapter_dir.mkdir(parents=True)
    tokenizer_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text(
        '{"r": 16}\n',
        encoding="utf-8",
    )
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")
    (tokenizer_dir / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (tokenizer_dir / "tokenizer_config.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (tokenizer_dir / "chat_template.jinja").write_text(
        "{{ messages }}\n",
        encoding="utf-8",
    )
    (checkpoint_dir / "train_metrics.json").write_text(
        '{"train_loss": 0.28}\n',
        encoding="utf-8",
    )
    (checkpoint_dir / "training_config.json").write_text(
        '{"run_name": "test"}\n',
        encoding="utf-8",
    )
    manifest = {
        "run_id": run_id,
        "run_name": "planner-sft-stage9-qwen3-5-4b-lora",
        "policy_version": f"{training_backend}:{run_id}",
        "training_backend": training_backend,
        "base_model_id": "Qwen/Qwen3.5-4B",
        "model_profile_id": "qwen3_5_4b",
        "tuning_method": "lora",
        "adapter_id": "qwen3_5_4b_sft_stage9",
        "adapter_path": str(adapter_dir),
        "train_data": "evaluation/stage9/artifacts/sft/sft_planner_stage9_train.jsonl",
        "train_manifest": "evaluation/stage9/artifacts/sft/sft_planner_stage9_manifest.json",
        "reward_profile": str(
            PROJECT_ROOT
            / "evaluation/stage9/configs/reward_v1_1_training_profile.json"
        ),
        "snapshot_id": "stage85-env-20260721-v2",
        "code_version": "test-revision",
        "created_at": "2026-07-27T08:55:37+00:00",
        "seed": 20260721,
        "framework_versions": {"transformers": "5.9.0"},
        "prompt_builder_version": "stage9-planner-prompt-v1",
        "decision_codec_version": "stage9-decision-codec-v1",
        "model_path": str(checkpoint_dir / "model"),
        "tokenizer_path": str(tokenizer_dir),
        "train_metrics_path": str(checkpoint_dir / "train_metrics.json"),
        "training_config_path": str(checkpoint_dir / "training_config.json"),
        "sample_count": 155,
        "source_case_count": 70,
        "action_counts": {},
        "reason_code_counts": {},
        "max_input_tokens": 4096,
        "max_target_tokens": 128,
    }
    (checkpoint_dir / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return checkpoint_dir


def _write_eval(
    tmp_path: Path,
    *,
    checkpoint_dir: Path,
    unsafe_false_release: bool = False,
    filename: str = "expanded_dev_eval.json",
) -> Path:
    cases = sorted(
        (
            case
            for case in load_planner_cases(DEFAULT_CASES)
            if case.split.value == "dev"
        ),
        key=lambda case: case.case_id,
    )
    snapshot_id = json.loads(DEFAULT_SNAPSHOT.read_text(encoding="utf-8"))[
        "snapshot_id"
    ]
    results = []
    false_release_written = False
    for case in cases:
        path = [action.value for action in case.acceptable_action_paths[0]]
        terminal = path[-1]
        path_match = True
        if (
            unsafe_false_release
            and case.expected_behavior.should_refuse
            and not false_release_written
        ):
            path = ["answer"]
            terminal = "answer"
            path_match = False
            false_release_written = True
        results.append(
            {
                "run_id": "expanded-dev-run-1",
                "case_id": case.case_id,
                "split": "dev",
                "planner_mode": "sft",
                "snapshot_id": snapshot_id,
                "reward_version": "reward-v1.1",
                "trace_id": f"trace-{case.case_id}",
                "action_path": path,
                "terminal_action": terminal,
                "terminal_reason_code": "test_reason",
                "retrieved_chunk_ids": [],
                "citation_chunk_ids": [],
                "metrics": {
                    "total_reward": 0.85,
                    "raw_total_reward": 0.85,
                    "format_valid": True,
                    "path_match": path_match,
                },
                "reward": {
                    "reward_version": "reward-v1.1",
                    "total_reward": 0.85,
                    "raw_total_reward": 0.85,
                    "capped_by": None,
                    "format_valid": True,
                    "components": {
                        name: {
                            "name": name,
                            "score": 1.0,
                            "weight": weight,
                            "weighted_score": weight,
                            "details": {},
                            "reasons": [],
                        }
                        for name, weight in {
                            "format": 0.15,
                            "retrieval": 0.12,
                            "citation": 0.08,
                            "answer": 0.15,
                            "behavior": 0.35,
                            "cost": 0.15,
                        }.items()
                    },
                    "errors": [],
                },
                "usage": {
                    "planner_calls": len(path),
                    "duration_ms": 100,
                    "trajectory_status": "completed",
                    "config_match_status": "match",
                    "corpus_match_status": "match",
                },
                "errors": [],
            }
        )
    payload = {
        "run_id": "expanded-dev-run-1",
        "runner_version": "stage9-model-planner-eval-v1",
        "created_at": "2026-07-29T06:00:00+00:00",
        "split": "dev",
        "snapshot_id": snapshot_id,
        "reward_version": "reward-v1.1",
        "requested_planners": ["sft"],
        "action_provider": "SnapshotExpectedChunkActionProvider",
        "case_count": 25,
        "planner_summaries": [
            {
                "planner_mode": "sft",
                "status": "completed",
                "config": {
                    "runner_version": "stage9-model-planner-eval-v1",
                    "policy_version": "candidate",
                    "checkpoint": str(checkpoint_dir),
                    "action_provider": "snapshot_expected_chunks",
                    "path_counts": {},
                },
                "usage": {
                    "planner_calls": 50,
                    "duration_ms": 2500,
                    "failed_case_count": 0,
                },
                "reward": {
                    "average_total_reward": 0.85,
                    "scored_case_count": 25,
                    "component_average_scores": {},
                },
                "case_count": 25,
                "completed_case_count": 25,
                "failed_case_count": 0,
                "skip_reason": "",
            }
        ],
        "results": results,
    }
    output = tmp_path / filename
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def _rewrite_as_replay_eval(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["run_id"] = "corrected-replay-run-1"
    payload["action_provider"] = "ReplayActionProvider"
    payload["planner_summaries"][0]["config"]["action_provider"] = "replay"
    for result in payload["results"]:
        result["run_id"] = payload["run_id"]
        result["usage"]["trace_steps"] = []
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_legacy_decision(
    tmp_path: Path,
    *,
    eval_path: Path,
    checkpoint_dir: Path,
) -> Path:
    eval_output = BaselineEvalOutput.model_validate_json(
        eval_path.read_text(encoding="utf-8")
    )
    case_by_id = {
        case.case_id: case
        for case in load_planner_cases(DEFAULT_CASES)
        if case.split.value == "dev"
    }
    cases = [
        _case_admission(case_by_id[result.case_id], result)
        for result in eval_output.results
    ]
    manifest = json.loads(
        (checkpoint_dir / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    payload = {
        "eval_run_id": eval_output.run_id,
        "action_provider": "snapshot_expected_chunks",
        "heldout_inference_result_count": 0,
        "checkpoint": {
            "run_id": manifest["run_id"],
            "checkpoint_dir": str(checkpoint_dir),
            "policy_version": manifest["policy_version"],
            "training_backend": manifest["training_backend"],
            "tuning_method": manifest["tuning_method"],
            "base_model_id": manifest["base_model_id"],
            "adapter_id": manifest["adapter_id"],
            "adapter_path": manifest["adapter_path"],
            "training_snapshot_id": manifest["snapshot_id"],
            "training_code_version": manifest["code_version"],
            "evaluation_code_version": "test-revision",
            "sample_count": manifest["sample_count"],
        },
        "cases": [case.model_dump(mode="json") for case in cases],
    }
    path = tmp_path / "old_decision.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _write_legacy_contract_inputs(tmp_path: Path) -> dict[str, Path]:
    cases = sorted(
        (
            case
            for case in load_planner_cases(DEFAULT_CASES)
            if case.split.value == "dev"
        ),
        key=lambda case: case.case_id,
    )
    canonical_payload = [
        case.model_dump(mode="json")
        for case in cases
    ]
    canonical_hash = hashlib.sha256(
        json.dumps(
            canonical_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    cases_hash = hashlib.sha256(DEFAULT_CASES.read_bytes()).hexdigest()

    snapshot = json.loads(DEFAULT_SNAPSHOT.read_text(encoding="utf-8"))
    snapshot["source_hashes"][
        "evaluation/stage8/cases/planner_cases.jsonl"
    ] = cases_hash
    snapshot_path = tmp_path / "legacy_environment_snapshot.json"
    snapshot_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    source_validation = (
        PROJECT_ROOT
        / "evaluation/stage9/artifacts/reward/reward_v1_1_balanced_dev_validation.json"
    )
    validation = json.loads(source_validation.read_text(encoding="utf-8"))
    validation["balanced_dev_case_ids"] = [case.case_id for case in cases]
    validation["balanced_dev_canonical_sha256"] = canonical_hash
    bindings = {
        "planner_cases": DEFAULT_CASES,
        "environment_snapshot": snapshot_path,
        "route_matrix": (
            PROJECT_ROOT
            / "evaluation/stage9/configs/planner_eval_route_matrix_v1.json"
        ),
        "reward_profile_v1_1": (
            PROJECT_ROOT
            / "evaluation/stage9/configs/reward_v1_1_training_profile.json"
        ),
        "reward_implementation": (
            PROJECT_ROOT / "app/rag/evaluation/reward.py"
        ),
    }
    for name, path in bindings.items():
        validation["inputs"][name]["sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    validation_path = tmp_path / "legacy_reward_validation.json"
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "snapshot_path": snapshot_path,
        "reward_validation_path": validation_path,
    }
