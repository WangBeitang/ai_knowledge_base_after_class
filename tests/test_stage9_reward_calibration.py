import json
from pathlib import Path

from app.rag.evaluation.case_schema import EnvironmentSnapshot, PlannerEvalCase
from app.rag.query.config import RETRIEVAL_CONFIG_VERSION
from app.rag.query.contracts import QueryAction
from evaluation.stage9.reward_calibration.action_path_suite import (
    ACTION_PATH_SUITE_VERSION,
    build_action_path_suite,
)
from evaluation.stage9.reward_calibration.generate_reward_calibration_report import (
    build_markdown_report,
)
from evaluation.stage9.reward_calibration.run_reward_calibration import (
    RewardCalibrationOutput,
    load_reward_calibration_output,
    run_reward_calibration,
    write_reward_calibration_output,
    write_reward_training_profile,
)


def test_stage9_action_path_suite_adds_fallback_paths_for_hyde_case():
    paths = build_action_path_suite(_answer_case())

    assert ACTION_PATH_SUITE_VERSION == "stage9-action-path-suite-v1"
    assert len(paths) >= 10
    assert [QueryAction.LOCAL_SEARCH, QueryAction.ANSWER] in [path.action_path for path in paths]
    assert [QueryAction.LOCAL_SEARCH, QueryAction.HYDE_SEARCH, QueryAction.REFUSE] in [
        path.action_path
        for path in paths
    ]


def test_stage9_reward_calibration_runs_multi_paths_and_freezes_profile(tmp_path: Path):
    cases = [_answer_case(), _clarification_case()]
    output = run_reward_calibration(
        cases=cases,
        snapshot=_snapshot(),
        split="dev",
        case_path="planner_cases.jsonl",
        snapshot_path="environment_snapshot.json",
        run_id="pytest_stage9_reward_calibration",
    )

    assert output.case_count == 2
    assert output.path_count >= 14
    assert output.summary.min_paths_per_case >= 7
    assert output.summary.freeze_decision == "frozen"
    assert output.training_profile.decision == "frozen"
    assert output.training_profile.reward_version == "reward-v1.1"
    assert output.training_profile.weights["behavior"] > output.training_profile.weights["retrieval"]
    assert all(result.reward["components"] for result in output.results)

    answer_results = [result for result in output.results if result.case_id == "stage9-cal-answer"]
    best_answer = min(answer_results, key=lambda result: result.route_rank)
    assert best_answer.path_id == "local_answer"

    output_path = tmp_path / "reward_v1_1_baseline_dev.json"
    profile_path = tmp_path / "reward_v1_1_training_profile.json"
    write_reward_calibration_output(output, output_path)
    write_reward_training_profile(output.training_profile, profile_path)

    loaded = load_reward_calibration_output(output_path)
    profile_payload = json.loads(profile_path.read_text(encoding="utf-8"))
    assert RewardCalibrationOutput.model_validate_json(output_path.read_text(encoding="utf-8"))
    assert loaded.run_id == output.run_id
    assert profile_payload["action_path_suite_version"] == ACTION_PATH_SUITE_VERSION


def test_stage9_reward_calibration_report_contains_core_tables():
    output = run_reward_calibration(
        cases=[_answer_case()],
        snapshot=_snapshot(),
        split="dev",
        case_path="planner_cases.jsonl",
        snapshot_path="environment_snapshot.json",
        run_id="pytest_stage9_reward_report",
    )

    report = build_markdown_report(output)

    assert "阶段 9 Reward v1.1 dev 多轨迹校准报告" in report
    assert "Reward 分项统计" in report
    assert "各 case 最优路线" in report
    assert "local_answer" in report


def _snapshot() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        snapshot_id="stage9-reward-calibration-test-v1",
        created_at="2026-07-21T00:00:00+00:00",
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
    )


def _answer_case() -> PlannerEvalCase:
    return PlannerEvalCase(
        case_id="stage9-cal-answer",
        case_group="core",
        split="dev",
        leakage_group_id="stage9-cal-answer",
        query="HAK180 的 E020 是什么故障？",
        dataset_ids=["dataset_default_equipment_ops"],
        owner_user_id="eval_demo_user",
        tenant_id="tenant_default",
        privacy_scope="public_demo",
        source_document_ids=["doc_hak180_manual"],
        source_index_versions={"doc_hak180_manual": 3},
        expected_subject_ids=["subject_hak180"],
        expected_subject_names=["HAK180 E020 证据 12345"],
        expected_chunks=[
            {
                "document_id": "doc_hak180_manual",
                "chunk_id": 12345,
                "index_version": 3,
                "relevance": "required",
                "answer_point_ids": ["offline_answer"],
            }
        ],
        expected_answer_points=["离线评测基于 HAK180 E020 证据 12345 形成 answer 终态"],
        expected_behavior={
            "should_answer": True,
            "should_refuse": False,
            "should_ask_clarification": False,
            "should_call_web": False,
            "forbidden_actions": ["web_search"],
        },
        acceptable_action_paths=[["local_search", "answer"], ["local_search", "hyde_search", "answer"]],
        expected_identifiers={"alarm_code": ["E020"]},
        label_source="manual",
        human_review_status="reviewed",
    )


def _clarification_case() -> PlannerEvalCase:
    return PlannerEvalCase(
        case_id="stage9-cal-ask",
        case_group="clarification",
        split="dev",
        leakage_group_id="stage9-cal-ask",
        query="这个报警应该怎么处理？",
        dataset_ids=["dataset_default_equipment_ops"],
        owner_user_id="eval_demo_user",
        tenant_id="tenant_default",
        privacy_scope="public_demo",
        expected_chunks=[],
        expected_answer_points=[],
        expected_behavior={
            "should_answer": False,
            "should_refuse": False,
            "should_ask_clarification": True,
            "should_call_web": False,
            "forbidden_actions": ["web_search", "hyde_search"],
        },
        acceptable_action_paths=[["ask_clarification"], ["local_search", "ask_clarification"]],
        expected_identifiers={},
        label_source="manual",
        human_review_status="reviewed",
    )
