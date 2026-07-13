import pytest

from app.rag.query.citation_service import build_citations
from app.rag.query.contracts import EvidenceSourceType, RetrievalCandidate, RetrievalChannel


def _local(chunk_id="chunk-1", score=0.9):
    return RetrievalCandidate(
        document_id="doc-1",
        chunk_id=chunk_id,
        dataset_id="dataset-a",
        index_version=1,
        chunk_index=0,
        title="本地手册",
        source_title="HAK 180 手册.pdf",
        content="本地证据",
        source_type=EvidenceSourceType.LOCAL,
        retrieval_channels=[RetrievalChannel.ORIGINAL],
        retrieval_rank=1,
        retrieval_score=0.1,
        rerank_score=score,
    ).model_dump(mode="json")


def _web(url="https://example.com/guide#section", score=0.8):
    return RetrievalCandidate(
        title="官网公告",
        content="联网证据",
        source_type=EvidenceSourceType.WEB,
        retrieval_channels=[RetrievalChannel.WEB],
        retrieval_rank=2,
        retrieval_score=0.08,
        rerank_score=score,
        url=url,
    ).model_dump(mode="json")


def test_citations_keep_final_order_and_deduplicate_local_and_web_identity():
    citations = build_citations([
        _local(),
        _local(),
        _web(),
        _web("https://EXAMPLE.com/guide"),
    ])

    assert [citation.source_type.value for citation in citations] == ["local", "web"]
    assert citations[0].document_id == "doc-1"
    assert citations[0].chunk_id == "chunk-1"
    assert citations[1].document_id is None
    assert citations[1].source == "https://example.com/guide#section"


def test_citation_rejects_candidate_that_has_not_completed_rerank():
    candidate = _local()
    candidate["rerank_score"] = None

    with pytest.raises(ValueError, match="生成 Citation 前候选必须已经完成 rerank"):
        build_citations([candidate])

