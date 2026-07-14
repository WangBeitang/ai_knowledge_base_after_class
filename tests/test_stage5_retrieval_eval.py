import json
from pathlib import Path

import pytest

from evaluation.stage5 import run_retrieval_eval as eval_module
from evaluation.stage5.run_retrieval_eval import load_cases, metrics_for_ranking, summarize_results


def test_ranking_metrics_use_binary_relevance_and_stable_rank_order():
    metrics = metrics_for_ranking(["chunk-a", "chunk-c"], ["chunk-x", "chunk-a", "chunk-c"])

    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr"] == 0.5
    assert 0 < metrics["ndcg"] < 1


def test_formal_eval_rejects_too_small_case_set(tmp_path):
    case_path = tmp_path / "cases.jsonl"
    case_path.write_text(json.dumps({
        "query_id": "case-1",
        "split": "dev",
        "query": "E020 是什么故障？",
        "dataset_id": "dataset-a",
        "owner_user_id": "user-a",
        "expected_subject_id": "subject-a",
        "expected_chunk_ids": ["chunk-a"],
        "expected_identifiers": {"alarm_code": ["E020"]},
        "expected_identifier_resolution_status": "fallback_exact_match",
        "should_answer": True,
        "should_use_web": False,
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="60 条评测样本"):
        load_cases(case_path)
    assert len(load_cases(case_path, allow_small_set=True)) == 1


def test_summary_keeps_unmeasured_resource_cost_as_none():
    result = {
        "query_id": "case-1",
        "split": "dev",
        "case_group": "core",
        "paired_query_id": None,
        "mode": "dense_learned_sparse",
        "recall_at_k": 1.0,
        "mrr": 0.5,
        "ndcg": 0.7,
        "citation_hit": 1,
        "identifier_expectation_match": True,
        "latency_ms": 12.0,
        "first_relevant_rank": 2,
    }

    summary = summarize_results([result])["core"]["dense_learned_sparse"]

    assert summary["citation_hit_rate"] == 1.0
    assert summary["index_size_bytes"] is None
    assert summary["ingestion_duration_ms"] is None


def test_load_cases_requires_colloquial_pair_with_same_labels(tmp_path):
    case_path = tmp_path / "cases.jsonl"
    base = {
        "split": "held_out",
        "query": "怎么处理？",
        "dataset_id": "dataset-a",
        "owner_user_id": "user-a",
        "expected_subject_id": "subject-a",
        "expected_chunk_ids": ["chunk-a"],
        "expected_identifiers": {},
        "expected_identifier_resolution_status": "not_applicable",
        "should_answer": True,
        "should_use_web": False,
    }
    rows = [
        {**base, "query_id": "core-case"},
        {
            **base,
            "query_id": "held-colloquial-case",
            "expected_chunk_ids": ["chunk-b"],
            "paired_query_id": "core-case",
        },
    ]
    case_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="相同 subject/chunk"):
        load_cases(case_path, allow_small_set=True)


def test_summary_separates_core_and_colloquial_and_reports_pair_delta():
    common = {
        "split": "held_out",
        "mode": "dense_bm25",
        "recall_at_k": 1.0,
        "ndcg": 1.0,
        "citation_hit": 1,
        "identifier_expectation_match": True,
        "latency_ms": 10.0,
    }
    results = [
        {
            **common,
            "query_id": "core-case",
            "case_group": "core",
            "paired_query_id": None,
            "mrr": 0.5,
            "first_relevant_rank": 2,
        },
        {
            **common,
            "query_id": "held-colloquial-case",
            "case_group": "colloquial",
            "paired_query_id": "core-case",
            "mrr": 1.0,
            "first_relevant_rank": 1,
        },
    ]

    summary = summarize_results(results)

    assert summary["core"]["dense_bm25"]["case_count"] == 1
    assert summary["colloquial"]["dense_bm25"]["case_count"] == 1
    assert summary["colloquial_pairs"][0]["mrr_delta"] == 0.5


def test_run_warms_each_mode_before_recording_results(monkeypatch):
    calls = []

    def fake_evaluate_case(case, mode):
        calls.append(mode.value)
        return {
            "query_id": case.query_id,
            "split": "dev",
            "case_group": "core",
            "paired_query_id": None,
            "mode": mode.value,
            "recall_at_k": 1.0,
            "mrr": 1.0,
            "ndcg": 1.0,
            "first_relevant_rank": 1,
            "citation_hit": 1,
            "identifier_expectation_match": True,
            "latency_ms": 1.0,
        }

    monkeypatch.setattr(eval_module, "evaluate_case", fake_evaluate_case)
    case = type("Case", (), {"query_id": "case-1"})()

    report = eval_module.run([case])

    expected_modes = [mode.value for mode in eval_module.EVALUATION_MODES]
    assert calls == [*expected_modes, *expected_modes]
    assert report["summary"]["core"]["dense_bm25"]["case_count"] == 1


def test_project_formal_case_set_has_expected_core_and_colloquial_counts():
    case_path = Path(__file__).parents[1] / "evaluation/stage5/retrieval_cases.jsonl"

    cases = load_cases(case_path)

    assert len(cases) == 60
    assert sum(case.is_colloquial for case in cases) == 10
    assert sum(not case.is_colloquial for case in cases) == 50
    assert sum(case.split == "dev" for case in cases) == 20
    assert sum(case.split == "held_out" for case in cases) == 40
