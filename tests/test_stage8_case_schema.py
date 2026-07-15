from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.rag.evaluation.case_schema import (
    CaseSplit,
    EnvironmentSnapshot,
    PlannerEvalCase,
    PlannerEvalResult,
    SplitManifest,
    load_planner_cases,
    validate_case_collection,
    validate_cases_for_sft_export,
)
from app.rag.query.contracts import QueryAction


def _valid_case_payload(**overrides):
    payload = {
        "case_id": "dev-alarm-e020-001",
        "case_group": "core",
        "split": "dev",
        "leakage_group_id": "hak180-e020",
        "query": "HAK180 设备的 E020 是什么故障？",
        "query_variants": ["HAK-180 报 E020 怎么处理？"],
        "dataset_ids": ["dataset_default_equipment_ops"],
        "owner_user_id": "eval_demo_user",
        "tenant_id": "tenant_default",
        "privacy_scope": "public_demo",
        "source_document_ids": ["doc_hak180_manual"],
        "source_index_versions": {"doc_hak180_manual": 3},
        "expected_subject_ids": ["subject_hak180"],
        "expected_subject_names": ["HAK 180 烫金机"],
        "expected_chunks": [
            {
                "document_id": "doc_hak180_manual",
                "chunk_id": 12345,
                "index_version": 3,
                "relevance": "required",
                "answer_point_ids": ["alarm_meaning", "first_check"],
            }
        ],
        "expected_answer_points": [
            "说明 E020 的故障含义",
            "给出首轮排查顺序",
            "提醒必要的停机或安全确认",
        ],
        "expected_behavior": {
            "should_answer": True,
            "should_refuse": False,
            "should_ask_clarification": False,
            "should_call_web": False,
            "web_required_reason": "",
            "forbidden_actions": ["web_search"],
        },
        "acceptable_action_paths": [
            ["local_search", "answer"],
            ["local_search", "hyde_search", "answer"],
        ],
        "expected_identifiers": {
            "equipment_model": ["HAK 180"],
            "alarm_code": ["E020"],
        },
        "label_source": "manual",
        "human_review_status": "reviewed",
        "notes": "",
    }
    payload.update(overrides)
    return payload


def test_stage8_planner_case_schema_accepts_valid_sample():
    case = PlannerEvalCase(**_valid_case_payload())

    assert case.split is CaseSplit.DEV
    assert case.expected_chunks[0].chunk_id == 12345
    assert case.acceptable_action_paths[0][-1] is QueryAction.ANSWER
    assert case.model_dump(mode="json")["expected_behavior"]["forbidden_actions"] == ["web_search"]


def test_stage8_planner_case_rejects_answer_sample_without_expected_chunks():
    payload = _valid_case_payload(expected_chunks=[])

    with pytest.raises(ValidationError, match="expected_chunks"):
        PlannerEvalCase(**payload)


def test_stage8_case_collection_rejects_duplicate_case_id():
    first = PlannerEvalCase(**_valid_case_payload())
    second_payload = _valid_case_payload(leakage_group_id="hak180-e021")
    second = PlannerEvalCase(**second_payload)

    with pytest.raises(ValueError, match="case_id 不能重复"):
        validate_case_collection([first, second])


def test_stage8_case_collection_rejects_same_leakage_group_cross_split():
    first = PlannerEvalCase(**_valid_case_payload(case_id="train-alarm-e020-001", split="train"))
    second = PlannerEvalCase(**_valid_case_payload(case_id="test-alarm-e020-001", split="test"))

    with pytest.raises(ValueError, match="leakage_group_id 不能跨 split"):
        validate_case_collection([first, second])


def test_stage8_sft_export_rejects_test_and_demo_cases():
    train_case = PlannerEvalCase(**_valid_case_payload(case_id="train-alarm-e020-001", split="train"))
    dev_case = PlannerEvalCase(**_valid_case_payload(case_id="dev-alarm-e020-001", split="dev"))
    validate_cases_for_sft_export([train_case, dev_case])

    test_case = PlannerEvalCase(**_valid_case_payload(case_id="test-alarm-e020-001", split="test"))
    demo_case = PlannerEvalCase(**_valid_case_payload(
        case_id="demo-alarm-e020-001",
        split="demo_regression",
        case_group="demo",
    ))

    with pytest.raises(ValueError, match="test/demo 样本不能导出训练"):
        validate_cases_for_sft_export([test_case, demo_case])


def test_stage8_environment_snapshot_and_result_schema_accept_structured_payloads():
    snapshot = EnvironmentSnapshot(
        snapshot_id="stage8-env-test-v1",
        created_at="2026-07-15T00:00:00+00:00",
        created_by="pytest",
        dataset_ids=["dataset_default_equipment_ops"],
        test_user_ids=["eval_demo_user"],
        documents=[
            {
                "document_id": "doc_hak180_manual",
                "dataset_id": "dataset_default_equipment_ops",
                "index_version": 3,
                "visibility": "public",
                "chunk_count": 10,
            }
        ],
        enabled_chunks={"doc_hak180_manual": [12345]},
        disabled_chunks=[],
        retrieval_config_version="retrieval-stage5-final-v1",
        retrieval_config_snapshot={"retrieval_mode": "dense_learned_sparse_bm25"},
        policy_version="rule-v1",
    )
    result = PlannerEvalResult(
        run_id="run_stage8_test",
        case_id="dev-alarm-e020-001",
        split="dev",
        planner_mode="rule",
        snapshot_id=snapshot.snapshot_id,
        reward_version="reward-v1",
        action_path=["local_search", "answer"],
        terminal_action="answer",
        retrieved_chunk_ids=[12345],
        citation_chunk_ids=[12345],
        metrics={"recall_at_k": 1.0, "citation_hit": True},
        reward={"total_reward": 1.0},
    )

    assert snapshot.documents[0].index_version == 3
    assert result.action_path == [QueryAction.LOCAL_SEARCH, QueryAction.ANSWER]


def test_stage8_split_manifest_rejects_case_id_in_multiple_splits():
    with pytest.raises(ValidationError, match="不能同时属于"):
        SplitManifest(
            manifest_id="manifest_test",
            created_at="2026-07-15T00:00:00+00:00",
            train_case_ids=["case_1"],
            test_case_ids=["case_1"],
        )


def test_stage8_template_jsonl_can_be_loaded():
    template_path = Path("evaluation/stage8/cases/planner_cases.template.jsonl")

    cases = load_planner_cases(template_path)

    assert cases[0].case_id == "template-dev-alarm-e020"
    assert cases[0].human_review_status.value == "pending"


def test_stage8_task82_demo_and_planner_case_files_are_usable():
    demo_cases = load_planner_cases("evaluation/stage8/cases/demo_regression_cases.jsonl")
    planner_cases = load_planner_cases("evaluation/stage8/cases/planner_cases.jsonl")

    assert 10 <= len(demo_cases) <= 20
    assert len(planner_cases) > len(demo_cases) * 3
    assert {case.split.value for case in demo_cases} == {"demo_regression"}
    assert all(case.human_review_status.value == "reviewed" for case in demo_cases)

    planner_groups = Counter(case.case_group.value for case in planner_cases)
    for expected_group in {"core", "colloquial", "clarification", "refusal", "realtime", "private_doc"}:
        assert planner_groups[expected_group] >= 1

    core_answer_cases = [
        case for case in planner_cases
        if case.case_group.value == "core" and case.expected_behavior.should_answer
    ]
    assert core_answer_cases
    assert all(case.expected_chunks for case in core_answer_cases)
    assert all(case.expected_answer_points for case in core_answer_cases)


def test_stage8_task82_split_manifest_matches_case_files():
    manifest = SplitManifest.model_validate_json(
        Path("evaluation/stage8/cases/split_manifest.json").read_text(encoding="utf-8")
    )
    demo_cases = load_planner_cases("evaluation/stage8/cases/demo_regression_cases.jsonl")
    planner_cases = load_planner_cases("evaluation/stage8/cases/planner_cases.jsonl")

    case_ids_by_split = {
        "train": {case.case_id for case in planner_cases if case.split.value == "train"},
        "dev": {case.case_id for case in planner_cases if case.split.value == "dev"},
        "test": {case.case_id for case in planner_cases if case.split.value == "test"},
        "demo_regression": {case.case_id for case in demo_cases},
    }

    assert set(manifest.train_case_ids) == case_ids_by_split["train"]
    assert set(manifest.dev_case_ids) == case_ids_by_split["dev"]
    assert set(manifest.test_case_ids) == case_ids_by_split["test"]
    assert set(manifest.demo_regression_case_ids) == case_ids_by_split["demo_regression"]

    all_cases = planner_cases + demo_cases
    assert set(manifest.leakage_group_to_split) == {case.leakage_group_id for case in all_cases}
    assert all(
        manifest.leakage_group_to_split[case.leakage_group_id] == case.split
        for case in all_cases
    )
