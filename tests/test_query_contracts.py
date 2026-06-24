import pytest

from app.rag.query.rerank_service import check_params as check_rerank_params
from app.rag.query.rrf_service import check_params as check_rrf_params
from app.rag.query.rrf_service import fuse_by_rrf


def test_rrf_allows_single_retrieval_path():
    state = {
        "embedding_chunks": [
            {"chunk_id": "chunk-1", "content": "a"},
        ],
        "hyde_embedding_chunks": [],
    }

    embedding_chunks, hyde_embedding_chunks = check_rrf_params(state)
    result = fuse_by_rrf(state)

    assert embedding_chunks
    assert hyde_embedding_chunks == []
    assert result["rrf_chunks"][0]["chunk_id"] == "chunk-1"


def test_rrf_rejects_empty_retrieval_results():
    with pytest.raises(ValueError):
        check_rrf_params({"embedding_chunks": [], "hyde_embedding_chunks": []})


def test_rerank_allows_empty_web_search_docs():
    rrf_chunks, web_search_docs, rewritten_query = check_rerank_params(
        {
            "rrf_chunks": [{"chunk_id": "chunk-1", "content": "a"}],
            "web_search_docs": [],
            "rewritten_query": "HAK 180 烫金机怎么操作？",
        }
    )

    assert rrf_chunks
    assert web_search_docs == []
    assert rewritten_query


def test_rerank_rejects_no_candidate_docs():
    with pytest.raises(ValueError):
        check_rerank_params(
            {
                "rrf_chunks": [],
                "web_search_docs": [],
                "rewritten_query": "HAK 180 烫金机怎么操作？",
            }
        )
