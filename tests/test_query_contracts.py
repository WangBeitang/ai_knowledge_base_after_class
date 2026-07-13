import pytest

from app.rag.query.rerank_service import check_params as check_rerank_params
from app.rag.query.rrf_service import check_params as check_rrf_params
from app.rag.query.rrf_service import fuse_by_rrf


def _local_candidate(chunk_id="chunk-1"):
    return {
        "document_id": "doc-1",
        "chunk_id": chunk_id,
        "dataset_id": "dataset-1",
        "index_version": 1,
        "chunk_index": 0,
        "title": "测试文档",
        "source_title": "测试手册",
        "content": "测试正文",
        "source_type": "local",
        "retrieval_channels": ["dense", "learned_sparse", "original"],
        "retrieval_rank": 1,
        "retrieval_score": 0.5,
        "rerank_score": None,
        "url": None,
    }


def test_rrf_allows_single_retrieval_action():
    state = {
        "embedding_chunks": [_local_candidate()],
        "hyde_embedding_chunks": [],
        "web_search_docs": [],
    }

    embedding_chunks, hyde_chunks, web_chunks = check_rrf_params(state)
    result = fuse_by_rrf(state)

    assert embedding_chunks
    assert hyde_chunks == []
    assert web_chunks == []
    assert result["rrf_chunks"][0]["chunk_id"] == "chunk-1"


def test_rrf_keeps_empty_retrieval_result_for_planner_observation():
    state = {
        "embedding_chunks": [],
        "hyde_embedding_chunks": [],
        "web_search_docs": [],
    }

    assert check_rrf_params(state) == ([], [], [])
    assert fuse_by_rrf(state)["rrf_chunks"] == []


def test_rerank_reads_only_cross_action_rrf_candidates():
    candidates, rewritten_query = check_rerank_params({
        "rrf_chunks": [_local_candidate()],
        # Web 已经在 RRF 中，rerank 不再读取这个旧字段并重复追加。
        "web_search_docs": [],
        "rewritten_query": "HAK 180 烫金机怎么操作？",
    })

    assert candidates[0].chunk_id == "chunk-1"
    assert rewritten_query


def test_rerank_rejects_no_candidate_docs():
    with pytest.raises(ValueError):
        check_rerank_params({
            "rrf_chunks": [],
            "rewritten_query": "HAK 180 烫金机怎么操作？",
        })
