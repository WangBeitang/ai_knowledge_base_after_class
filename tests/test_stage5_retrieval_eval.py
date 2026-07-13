import json

import pytest

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
        "expected_identifier_resolution_status": "exact_match",
        "should_answer": True,
        "should_use_web": False,
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="24～40"):
        load_cases(case_path)
    assert len(load_cases(case_path, allow_small_set=True)) == 1


def test_summary_keeps_unmeasured_resource_cost_as_none():
    result = {
        "query_id": "case-1",
        "split": "dev",
        "mode": "dense_learned_sparse",
        "recall_at_k": 1.0,
        "mrr": 0.5,
        "ndcg": 0.7,
        "citation_hit": 1,
        "latency_ms": 12.0,
    }

    summary = summarize_results([result])["dense_learned_sparse"]

    assert summary["citation_hit_rate"] == 1.0
    assert summary["index_size_bytes"] is None
    assert summary["ingestion_duration_ms"] is None

