import json
from pathlib import Path
from typing import Any

from app.rag.evaluation.baseline_runner import BaselineEvalOutput
from app.rag.evaluation.case_schema import PlannerEvalCase, PlannerEvalResult
from app.rag.evaluation.sft_exporter import (
    SftExportConfig,
    export_sft_samples,
    export_sft_samples_from_files,
)


def test_stage8_sft_export_splits_valid_rule_trajectory_into_action_samples():
    case = _case()
    result = _result(case_id=case.case_id)
    output = export_sft_samples(
        eval_output=_eval_output([result]),
        cases=[case],
        config=SftExportConfig(reward_threshold=0.80),
    )

    assert len(output.samples) == 2
    assert output.samples[0].target_decision["action"] == "local_search"
    assert output.samples[1].target_decision["action"] == "answer"
    assert output.samples[1].input_context["latest_observation"]["retrieved_chunk_ids"] == ["12345"]
    assert output.samples[0].label_source == "rule"
    assert output.samples[0].review_status == "reviewed"
    assert output.manifest.sample_count == 2
    assert output.manifest.source_counts == {"rule": 2}
    assert output.manifest.source_ratios == {"rule": 1.0}
    assert not _contains_key(output.samples[0].model_dump(mode="python"), "content")
    assert not _contains_key(output.samples[0].model_dump(mode="python"), "answer_prompt")


def test_stage8_sft_export_rejects_test_demo_and_format_invalid_results():
    test_case = _case(case_id="sft-test-answer", split="test")
    demo_case = _case(case_id="sft-demo-answer", split="demo_regression", case_group="demo")
    invalid_case = _case(case_id="sft-train-invalid")
    results = [
        _result(case_id=test_case.case_id, split="test"),
        _result(case_id=demo_case.case_id, split="demo_regression"),
        _result(case_id=invalid_case.case_id, format_valid=False),
    ]

    output = export_sft_samples(
        eval_output=_eval_output(results),
        cases=[test_case, demo_case, invalid_case],
    )

    assert output.samples == []
    assert output.manifest.filter_counts["split_not_allowed"] == 2
    assert output.manifest.filter_counts["format_invalid"] == 1


def test_stage8_sft_export_rejects_low_reward_api_teacher_even_when_path_matches():
    case = _case()
    result = _result(
        case_id=case.case_id,
        planner_mode="api",
        total_reward=0.40,
        path_match=True,
    )

    output = export_sft_samples(
        eval_output=_eval_output([result], requested_planners=["api"]),
        cases=[case],
        config=SftExportConfig(reward_threshold=0.80),
    )

    assert output.samples == []
    assert output.manifest.filter_counts == {"api_reward_below_threshold": 1}


def test_stage8_sft_export_from_files_writes_jsonl_and_manifest(tmp_path: Path):
    case = _case()
    result = _result(case_id=case.case_id)
    eval_output = _eval_output([result])
    eval_path = tmp_path / "planner_eval_train.json"
    cases_path = tmp_path / "planner_cases.jsonl"
    output_path = tmp_path / "sft_planner_train.jsonl"
    manifest_path = tmp_path / "sft_manifest.json"

    eval_path.write_text(
        json.dumps(eval_output.to_json_dict(), ensure_ascii=False),
        encoding="utf-8",
    )
    cases_path.write_text(
        json.dumps(case.model_dump(mode="json"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    export = export_sft_samples_from_files(
        eval_result_path=eval_path,
        cases_path=cases_path,
        output_path=output_path,
        manifest_path=manifest_path,
    )

    jsonl_lines = output_path.read_text(encoding="utf-8").splitlines()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(jsonl_lines) == len(export.samples) == 2
    assert json.loads(jsonl_lines[0])["source_case_id"] == case.case_id
    assert manifest["sample_count"] == 2
    assert manifest["excluded_payloads"] == [
        "full_chunk_content",
        "answer_prompt",
        "private_chain_of_thought",
        "model_reasoning_text",
    ]


def _eval_output(
        results: list[PlannerEvalResult],
        *,
        requested_planners: list[str] | None = None,
) -> BaselineEvalOutput:
    return BaselineEvalOutput(
        run_id="stage8_eval_sft_test",
        created_at="2026-07-19T00:00:00+00:00",
        split=results[0].split,
        snapshot_id="stage8-env-sft-test",
        reward_version="reward-v1",
        requested_planners=requested_planners or ["rule"],
        action_provider="pytest",
        case_count=len({result.case_id for result in results}),
        planner_summaries=[],
        results=results,
    )


def _case(
        *,
        case_id: str = "sft-train-answer",
        split: str = "train",
        case_group: str = "core",
) -> PlannerEvalCase:
    return PlannerEvalCase(
        case_id=case_id,
        case_group=case_group,
        split=split,
        leakage_group_id=f"{case_id}-group",
        query="HAK180 的 E020 是什么故障？",
        dataset_ids=["dataset_default_equipment_ops"],
        owner_user_id="eval_demo_user",
        tenant_id="tenant_default",
        privacy_scope="public_demo",
        source_document_ids=["doc_hak180_manual"],
        source_index_versions={"doc_hak180_manual": 3},
        expected_subject_ids=["subject_hak180"],
        expected_subject_names=["HAK 180 烫金机"],
        expected_chunks=[
            {
                "document_id": "doc_hak180_manual",
                "chunk_id": 12345,
                "index_version": 3,
                "relevance": "required",
                "answer_point_ids": ["alarm_meaning"],
            }
        ],
        expected_answer_points=["说明 E020 的故障含义"],
        expected_behavior={
            "should_answer": True,
            "should_refuse": False,
            "should_ask_clarification": False,
            "should_call_web": False,
            "forbidden_actions": ["web_search"],
        },
        acceptable_action_paths=[["local_search", "answer"]],
        expected_identifiers={"equipment_model": ["HAK 180"], "alarm_code": ["E020"]},
        label_source="manual",
        human_review_status="reviewed",
    )


def _result(
        *,
        case_id: str,
        split: str = "train",
        planner_mode: str = "rule",
        total_reward: float = 0.95,
        format_valid: bool = True,
        path_match: bool = True,
) -> PlannerEvalResult:
    return PlannerEvalResult(
        run_id="stage8_eval_sft_test",
        case_id=case_id,
        split=split,
        planner_mode=planner_mode,
        snapshot_id="stage8-env-sft-test",
        reward_version="reward-v1",
        trace_id=f"trace_{planner_mode}_{case_id}",
        action_path=["local_search", "answer"],
        terminal_action="answer",
        terminal_reason_code="local_evidence_sufficient",
        retrieved_chunk_ids=[12345],
        citation_chunk_ids=[12345],
        metrics={
            "total_reward": total_reward,
            "raw_total_reward": total_reward,
            "format_valid": format_valid,
            "path_match": path_match,
            "recall_at_k": 1.0,
            "citation_hit_rate": 1.0,
        },
        reward=_reward(total_reward=total_reward, format_valid=format_valid, path_match=path_match),
        usage={"planner_calls": 2, "total_tokens": 0},
        errors=[],
    )


def _reward(*, total_reward: float, format_valid: bool, path_match: bool) -> dict[str, Any]:
    return {
        "reward_version": "reward-v1",
        "total_reward": total_reward,
        "raw_total_reward": total_reward,
        "capped_by": None if format_valid else "invalid_format",
        "format_valid": format_valid,
        "components": {
            "format": {"score": 1.0 if format_valid else 0.0, "details": {}, "reasons": []},
            "retrieval": {"score": 1.0, "details": {"recall_at_k": 1.0}, "reasons": []},
            "citation": {
                "score": 1.0,
                "details": {"citation_hit_rate": 1.0, "invalid_citation_count": 0},
                "reasons": [],
            },
            "answer": {"score": 1.0, "details": {"answer_point_coverage": 1.0}, "reasons": []},
            "behavior": {"score": 1.0, "details": {"path_match": path_match}, "reasons": []},
            "cost": {"score": 1.0, "details": {}, "reasons": []},
        },
        "errors": [],
    }


def _contains_key(payload: Any, key: str) -> bool:
    if isinstance(payload, dict):
        return key in payload or any(_contains_key(value, key) for value in payload.values())
    if isinstance(payload, list):
        return any(_contains_key(value, key) for value in payload)
    return False
