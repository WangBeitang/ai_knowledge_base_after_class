import json
from pathlib import Path

import pytest

from evaluation.stage9.model_planner.analyze_sft_dev_results import (
    ANALYSIS_VERSION,
    CaseAnalysisStatus,
    FailureAttribution,
    analyze_sft_dev_results,
    render_markdown_report,
    write_analysis_outputs,
)


def test_analyze_sft_dev_results_separates_model_and_evaluator_failures(tmp_path):
    eval_path, cases_path, log_path = _write_fixture(tmp_path)

    analysis = analyze_sft_dev_results(
        eval_path=eval_path,
        cases_path=cases_path,
        dev_log_path=log_path,
        eval_logical_path="archive/sft_eval_dev.json",
        cases_logical_path="archive/planner_cases.jsonl",
        dev_log_logical_path="archive/dev_eval.log",
        source_archive_name="sft-v1.tar.gz",
        source_archive_sha256="a" * 64,
    )

    assert analysis.analysis_version == ANALYSIS_VERSION
    assert analysis.case_count == 3
    assert analysis.summary["path_match_count"] == 2
    assert analysis.summary["model_route_failure_case_count"] == 1
    assert analysis.summary["hyde_actual_case_count"] == 0
    assert analysis.summary["real_retrieval_quality_verified_case_count"] == 0
    assert analysis.terminal_confusion_matrix["answer"]["answer"] == 1
    assert analysis.terminal_confusion_matrix["refuse"]["ask_clarification"] == 1
    assert analysis.dataset_attributions == [FailureAttribution.INSUFFICIENT_COVERAGE]

    by_id = {case.case_id: case for case in analysis.cases}
    answer = by_id["dev-answer"]
    assert answer.analysis_status == CaseAnalysisStatus.EVALUATOR_LIMITED
    assert answer.primary_attribution == FailureAttribution.EVALUATOR_LIMITATION
    assert answer.attributions == [
        FailureAttribution.EVALUATOR_LIMITATION,
        FailureAttribution.PROVIDER_LIMITATION,
    ]
    assert answer.answer_score_interpretable_as_planner_quality is False

    realtime = by_id["dev-realtime"]
    assert realtime.analysis_status == CaseAnalysisStatus.MODEL_ROUTE_FAILURE
    assert realtime.primary_attribution == FailureAttribution.MODEL_ERROR
    assert realtime.web_behavior_match is False
    assert realtime.label_review_required is True

    clarification = by_id["dev-clarify"]
    assert clarification.analysis_status == CaseAnalysisStatus.PASSED_WITH_COST_OBSERVATION
    assert clarification.primary_attribution is None
    assert clarification.wall_duration_ms == 3200


def test_analysis_report_preserves_evidence_boundaries_and_markdown_table(tmp_path):
    eval_path, cases_path, log_path = _write_fixture(tmp_path)
    analysis = analyze_sft_dev_results(
        eval_path=eval_path,
        cases_path=cases_path,
        dev_log_path=log_path,
    )

    report = render_markdown_report(analysis)

    assert "不能把 answer=0.2857 表述为 Qwen3.5-4B 回答能力差" in report
    assert "不代表真实 Milvus/Web 质量" in report
    assert "`ask_clarification`<br>`local_search -> ask_clarification`" in report
    assert "进入 9.3.12" in report


def test_analysis_output_refuses_silent_overwrite(tmp_path):
    eval_path, cases_path, log_path = _write_fixture(tmp_path)
    analysis = analyze_sft_dev_results(
        eval_path=eval_path,
        cases_path=cases_path,
        dev_log_path=log_path,
    )
    output_json = tmp_path / "out/analysis.json"
    output_report = tmp_path / "out/report.md"

    write_analysis_outputs(
        analysis=analysis,
        output_json=output_json,
        output_report=output_report,
        overwrite=False,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["analysis_version"] == ANALYSIS_VERSION
    assert output_report.read_text(encoding="utf-8").startswith("# 阶段 9 SFT v1")
    with pytest.raises(FileExistsError):
        write_analysis_outputs(
            analysis=analysis,
            output_json=output_json,
            output_report=output_report,
            overwrite=False,
        )


def test_analysis_rejects_inconsistent_path_match(tmp_path):
    eval_path, cases_path, log_path = _write_fixture(tmp_path)
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    payload["results"][0]["metrics"]["path_match"] = False
    eval_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="path_match"):
        analyze_sft_dev_results(
            eval_path=eval_path,
            cases_path=cases_path,
            dev_log_path=log_path,
        )


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    cases = [
        _case(
            case_id="dev-answer",
            case_group="core",
            terminal="answer",
            paths=[["local_search", "answer"], ["local_search", "hyde_search", "answer"]],
            reviewed=True,
        ),
        _case(
            case_id="dev-realtime",
            case_group="realtime",
            terminal="refuse",
            paths=[["web_search", "refuse"]],
            reviewed=False,
            should_call_web=True,
        ),
        _case(
            case_id="dev-clarify",
            case_group="clarification",
            terminal="ask_clarification",
            paths=[["ask_clarification"], ["local_search", "ask_clarification"]],
            reviewed=False,
        ),
    ]
    results = [
        _result(
            case_id="dev-answer",
            actual_path=["local_search", "answer"],
            terminal="answer",
            expected_terminal="answer",
            path_match=True,
            should_call_web=False,
            answer_score=0.0,
            cost_score=1.0,
            extra_steps=0,
        ),
        _result(
            case_id="dev-realtime",
            actual_path=["local_search", "ask_clarification"],
            terminal="ask_clarification",
            expected_terminal="refuse",
            path_match=False,
            should_call_web=True,
            answer_score=0.0,
            cost_score=1.0,
            extra_steps=0,
        ),
        _result(
            case_id="dev-clarify",
            actual_path=["local_search", "ask_clarification"],
            terminal="ask_clarification",
            expected_terminal="ask_clarification",
            path_match=True,
            should_call_web=False,
            answer_score=1.0,
            cost_score=0.92,
            extra_steps=1,
        ),
    ]
    eval_payload = {
        "run_id": "dev-run-1",
        "runner_version": "stage9-model-planner-eval-v1",
        "created_at": "2026-07-27T00:00:00+00:00",
        "split": "dev",
        "snapshot_id": "snapshot-1",
        "reward_version": "reward-v1.1",
        "requested_planners": ["sft"],
        "action_provider": "SnapshotExpectedChunkActionProvider",
        "case_count": 3,
        "planner_summaries": [{
            "planner_mode": "sft",
            "status": "completed",
            "config": {
                "checkpoint": "evaluation/checkpoints/checkpoint-1",
                "action_provider": "snapshot_expected_chunks",
            },
            "reward": {
                "average_total_reward": 0.79,
                "component_average_scores": {
                    "answer": 0.2857142857142857,
                    "behavior": 0.7,
                    "citation": 1.0,
                    "cost": 0.97,
                    "format": 1.0,
                    "retrieval": 1.0,
                },
            },
            "completed_case_count": 3,
            "failed_case_count": 0,
        }],
        "results": results,
    }

    eval_path = tmp_path / "sft_eval_dev.json"
    cases_path = tmp_path / "planner_cases.jsonl"
    log_path = tmp_path / "dev_eval.log"
    eval_path.write_text(json.dumps(eval_payload, ensure_ascii=False), encoding="utf-8")
    cases_path.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n",
        encoding="utf-8",
    )
    log_path.write_text(
        "\n".join([
            "[dev_eval] case=1/3 case_id=dev-answer status=completed duration_ms=2500 action_path=local_search -> answer",
            "[dev_eval] case=2/3 case_id=dev-realtime status=completed duration_ms=5000 action_path=local_search -> ask_clarification",
            "[dev_eval] case=3/3 case_id=dev-clarify status=completed duration_ms=3200 action_path=local_search -> ask_clarification",
        ]) + "\n",
        encoding="utf-8",
    )
    return eval_path, cases_path, log_path


def _case(
        *,
        case_id: str,
        case_group: str,
        terminal: str,
        paths: list[list[str]],
        reviewed: bool,
        should_call_web: bool = False,
) -> dict:
    flags = {
        "should_answer": terminal == "answer",
        "should_refuse": terminal == "refuse",
        "should_ask_clarification": terminal == "ask_clarification",
        "should_call_web": should_call_web,
    }
    return {
        "case_id": case_id,
        "case_group": case_group,
        "split": "dev",
        "query": f"query for {case_id}",
        "acceptable_action_paths": paths,
        "expected_behavior": flags,
        "label_source": "manual" if reviewed else "synthetic",
        "human_review_status": "reviewed" if reviewed else "pending",
    }


def _result(
        *,
        case_id: str,
        actual_path: list[str],
        terminal: str,
        expected_terminal: str,
        path_match: bool,
        should_call_web: bool,
        answer_score: float,
        cost_score: float,
        extra_steps: int,
) -> dict:
    used_web = "web_search" in actual_path
    scores = {
        "format": 1.0,
        "retrieval": 1.0,
        "citation": 1.0,
        "answer": answer_score,
        "behavior": 1.0 if path_match else 0.1,
        "cost": cost_score,
    }
    components = {
        name: {
            "score": score,
            "details": {},
        }
        for name, score in scores.items()
    }
    components["behavior"]["details"] = {
        "expected_terminal": expected_terminal,
        "actual_terminal": terminal,
        "path_match": path_match,
        "should_call_web": should_call_web,
        "used_web": used_web,
    }
    components["cost"]["details"] = {"extra_steps": extra_steps}
    return {
        "run_id": "dev-run-1",
        "case_id": case_id,
        "action_path": actual_path,
        "terminal_action": terminal,
        "terminal_reason_code": "test_reason",
        "retrieved_chunk_ids": [1] if terminal == "answer" else [],
        "citation_chunk_ids": [1] if terminal == "answer" else [],
        "metrics": {
            "total_reward": sum(scores.values()) / len(scores),
            "path_match": path_match,
        },
        "reward": {"components": components},
        "usage": {"duration_ms": 1},
    }
