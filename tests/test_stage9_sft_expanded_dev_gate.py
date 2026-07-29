import json
from pathlib import Path

import pytest

from app.rag.evaluation.case_schema import load_planner_cases
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
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_expanded_dev_gate_allows_a_complete_five_route_candidate(tmp_path):
    checkpoint_dir = _write_checkpoint(tmp_path)
    eval_path = _write_eval(tmp_path, checkpoint_dir=checkpoint_dir)

    admission = load_and_build_admission(
        eval_output_path=eval_path,
        checkpoint_dir=checkpoint_dir,
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
        "6058ff8f17570ceea62163b3504f660163a1ecd53457ea7a927b16999423f5d1"
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

    admission = load_and_build_admission(
        eval_output_path=eval_path,
        checkpoint_dir=checkpoint_dir,
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

    with pytest.raises(ValueError, match="debug_memorized"):
        load_admission_contract(checkpoint_dir=checkpoint_dir)


def test_expanded_dev_preflight_does_not_execute_model_or_write_outputs(
    tmp_path,
    capsys,
):
    checkpoint_dir = _write_checkpoint(tmp_path)
    eval_output = tmp_path / "should_not_exist/eval.json"
    decision_output = tmp_path / "should_not_exist/decision.json"
    report_output = tmp_path / "should_not_exist/report.md"

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
        )

    clean_eval = _write_eval(
        tmp_path,
        checkpoint_dir=checkpoint_dir,
        filename="clean_eval.json",
    )
    admission = load_and_build_admission(
        eval_output_path=clean_eval,
        checkpoint_dir=checkpoint_dir,
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
