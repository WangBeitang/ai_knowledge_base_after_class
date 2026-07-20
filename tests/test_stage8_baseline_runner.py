import json
from pathlib import Path

import pytest

from app.rag.evaluation.baseline_runner import (
    parse_planner_modes,
    run_baseline_evaluation,
    run_baseline_evaluation_from_files,
)
from app.rag.evaluation.case_schema import EnvironmentSnapshot, PlannerEvalCase, PlannerMode
from app.rag.query.config import RETRIEVAL_CONFIG_VERSION
from app.rag.query.contracts import QueryAction


def test_stage8_baseline_runner_runs_rule_and_skips_unavailable_planners():
    output = run_baseline_evaluation(
        cases=[_answer_case()],
        snapshot=_snapshot(),
        split="dev",
        planners="rule,api,local_base",
        run_id="pytest_stage8_baseline",
    )

    summaries = {summary.planner_mode: summary for summary in output.planner_summaries}
    assert summaries[PlannerMode.RULE].status == "completed"
    assert summaries[PlannerMode.API].status == "skipped"
    assert "API Planner" in summaries[PlannerMode.API].skip_reason
    assert summaries[PlannerMode.LOCAL_BASE].status == "skipped"
    assert "本地零样本" in summaries[PlannerMode.LOCAL_BASE].skip_reason

    assert output.action_provider == "snapshot_expected_chunks"
    assert len(output.results) == 1
    result = output.results[0]
    assert result.planner_mode is PlannerMode.RULE
    assert result.action_path == [QueryAction.LOCAL_SEARCH, QueryAction.ANSWER]
    assert result.reward["reward_version"] == "reward-v1.1"
    assert result.metrics["recall_at_k"] == 1.0
    assert result.usage["planner_calls"] == 2

    json.dumps(output.to_json_dict(), ensure_ascii=False)


def test_stage8_baseline_runner_from_files_writes_json(tmp_path: Path):
    cases_path = tmp_path / "planner_cases.jsonl"
    snapshot_path = tmp_path / "environment_snapshot.json"
    output_path = tmp_path / "planner_eval_dev.json"

    cases_path.write_text(
        json.dumps(_answer_case().model_dump(mode="json"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    snapshot_path.write_text(
        _snapshot().model_dump_json(),
        encoding="utf-8",
    )

    output = run_baseline_evaluation_from_files(
        cases_path=cases_path,
        snapshot_path=snapshot_path,
        split="dev",
        planners="rule",
        output_path=output_path,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert output.run_id == payload["run_id"]
    assert payload["results"][0]["case_id"] == "baseline-dev-hak180-e020"
    assert payload["planner_summaries"][0]["reward"]["scored_case_count"] == 1


def test_stage8_baseline_runner_rejects_unknown_planner():
    with pytest.raises(ValueError, match="阶段 8.6 只支持"):
        parse_planner_modes("rule,grpo")


def _snapshot() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        snapshot_id="stage8-baseline-test-v1",
        created_at="2026-07-19T00:00:00+00:00",
        created_by="pytest",
        dataset_ids=["dataset_default_equipment_ops"],
        test_user_ids=["eval_demo_user"],
        documents=[
            {
                "document_id": "doc_hak180_manual",
                "dataset_id": "dataset_default_equipment_ops",
                "index_version": 3,
                "visibility": "public",
                "chunk_count": 1,
            }
        ],
        enabled_chunks={"doc_hak180_manual": [12345]},
        disabled_chunks=[],
        retrieval_config_version=RETRIEVAL_CONFIG_VERSION,
        retrieval_config_snapshot={
            "retrieval_mode": "dense_learned_sparse_bm25",
            "per_channel_topk": 5,
            "fusion_topk": 5,
            "rerank_min_topk": 2,
            "rerank_max_topk": 5,
            "rrf_k": 60,
            "evidence_threshold": 0.75,
            "web_fallback_enabled": True,
        },
        policy_version="rule-v1",
        planner_registry=[
            {
                "planner_mode": "rule",
                "enabled_online": True,
                "enabled_for_eval": True,
                "unavailable_reason": "",
            },
            {
                "planner_mode": "api",
                "enabled_online": False,
                "enabled_for_eval": False,
                "unavailable_reason": "API Planner provider 未配置",
            },
            {
                "planner_mode": "local_base",
                "enabled_online": False,
                "enabled_for_eval": False,
                "unavailable_reason": "本地零样本 Planner 模型未配置",
            },
        ],
    )


def _answer_case() -> PlannerEvalCase:
    return PlannerEvalCase(
        case_id="baseline-dev-hak180-e020",
        case_group="core",
        split="dev",
        leakage_group_id="baseline-hak180-e020",
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
