from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.rag.query import answer_service, rerank_service, web_search_service
from app.rag.query.config import channels_for_retrieval_mode
from app.rag.query.contracts import RetrievalCandidate, RetrievalChannel, RetrievalMode
from app.rag.query.rrf_service import (
    canonicalize_web_url,
    fuse_ranked_candidate_lists,
)
from app.shared.clients.milvus_utils import (
    create_hybrid_search_requests,
    hybrid_search,
)


def _local_candidate(
        chunk_id,
        *,
        channels,
        rank=1,
        content=None,
        retrieval_score=0.5,
):
    return RetrievalCandidate(
        document_id=f"doc-{chunk_id}",
        chunk_id=chunk_id,
        dataset_id="dataset_ops",
        index_version=3,
        chunk_index=rank - 1,
        title=f"本地候选 {chunk_id}",
        source_title="HAK 180 操作手册",
        subject_id="subject_hak_180",
        standard_subject_name="HAK 180 烫金机",
        content=content or f"本地正文 {chunk_id}",
        equipment_model="HAK 180",
        source_type="local",
        retrieval_channels=channels,
        retrieval_rank=rank,
        retrieval_score=retrieval_score,
        rerank_score=None,
        url=None,
    ).model_dump(mode="json")


def _web_candidate(url, *, rank=1, title="厂家公告", content="最新维修通知"):
    return RetrievalCandidate(
        document_id=None,
        chunk_id=None,
        dataset_id=None,
        index_version=None,
        chunk_index=None,
        title=title,
        source_title=title,
        content=content,
        source_type="web",
        retrieval_channels=["web"],
        retrieval_rank=rank,
        retrieval_score=0.0,
        rerank_score=None,
        url=url,
    ).model_dump(mode="json")


@pytest.mark.parametrize(
    ("mode", "expected_fields"),
    [
        (RetrievalMode.DENSE_LEARNED_SPARSE, ["dense_vector", "sparse_vector"]),
        (RetrievalMode.DENSE_BM25, ["dense_vector", "bm25_sparse_vector"]),
        (
            RetrievalMode.DENSE_LEARNED_SPARSE_BM25,
            ["dense_vector", "sparse_vector", "bm25_sparse_vector"],
        ),
    ],
)
def test_retrieval_modes_create_correct_ann_requests_with_same_filter(mode, expected_fields):
    expr = 'dataset_id in ["dataset_ops"] AND enabled == true'
    requests = create_hybrid_search_requests(
        dense_vector=[0.1, 0.2],
        sparse_vector={1: 0.3},
        expr=expr,
        retrieval_mode=mode.value,
        query_text="HAK 180 E020",
    )

    assert [request.anns_field for request in requests] == expected_fields
    assert all(request.expr == expr for request in requests)
    if mode in {RetrievalMode.DENSE_BM25, RetrievalMode.DENSE_LEARNED_SPARSE_BM25}:
        assert requests[-1].data == ["HAK 180 E020"]
        assert requests[-1].param["metric_type"] == "BM25"


def test_bm25_mode_requires_original_query_text():
    with pytest.raises(ValueError, match="必须提供非空 query_text"):
        create_hybrid_search_requests(
            dense_vector=[0.1],
            sparse_vector={1: 0.2},
            retrieval_mode=RetrievalMode.DENSE_BM25.value,
        )


def test_retrieval_mode_channels_are_stable_and_explicit():
    assert channels_for_retrieval_mode(RetrievalMode.DENSE_LEARNED_SPARSE) == [
        RetrievalChannel.DENSE,
        RetrievalChannel.LEARNED_SPARSE,
    ]
    assert channels_for_retrieval_mode(RetrievalMode.DENSE_BM25) == [
        RetrievalChannel.DENSE,
        RetrievalChannel.BM25,
    ]
    assert channels_for_retrieval_mode(RetrievalMode.DENSE_LEARNED_SPARSE_BM25) == [
        RetrievalChannel.DENSE,
        RetrievalChannel.LEARNED_SPARSE,
        RetrievalChannel.BM25,
    ]


def test_chunk_search_can_select_rrf_without_changing_weighted_subject_search():
    captured_rankers = []

    class FakeClient:
        def hybrid_search(self, **kwargs):
            captured_rankers.append(kwargs["ranker"].__class__.__name__)
            return [[]]

    client = FakeClient()
    hybrid_search(
        client=client,
        collection_name="subject_aliases",
        reqs=["dense", "sparse"],
        ranker_weights=(0.9, 0.1),
        ranker_type="weighted",
    )
    hybrid_search(
        client=client,
        collection_name="chunks",
        reqs=["dense", "sparse", "bm25"],
        ranker_type="rrf",
        rrf_k=60,
    )

    assert captured_rankers == ["WeightedRanker", "RRFRanker"]


def test_retrieval_candidate_enforces_local_and_web_identity_boundaries():
    local = RetrievalCandidate.model_validate(_local_candidate("chunk-a", channels=["original", "dense"]))
    web = RetrievalCandidate.model_validate(_web_candidate("https://example.com/notice"))
    assert local.document_id == "doc-chunk-a"
    assert web.document_id is None

    invalid_web = _web_candidate("https://example.com/notice")
    invalid_web["chunk_id"] = "fake-web-chunk"
    with pytest.raises(ValidationError, match="不能伪造"):
        RetrievalCandidate.model_validate(invalid_web)

    invalid_local = _local_candidate("chunk-a", channels=["original", "dense"])
    invalid_local["document_id"] = None
    with pytest.raises(ValidationError, match="本地候选必须包含"):
        RetrievalCandidate.model_validate(invalid_local)


def test_outer_rrf_fuses_original_hyde_and_web_once_and_keeps_provenance():
    original = [
        _local_candidate("chunk-a", channels=["dense", "original"], rank=1),
        _local_candidate("chunk-b", channels=["dense", "original"], rank=2),
    ]
    hyde = [
        _local_candidate("chunk-b", channels=["learned_sparse", "hyde"], rank=1),
        _local_candidate("chunk-c", channels=["dense", "hyde"], rank=2),
    ]
    web = [_web_candidate("https://example.com/latest", rank=1)]

    fused = fuse_ranked_candidate_lists([original, hyde, web], limit=10, k=60)

    assert [candidate["chunk_id"] for candidate in fused[:1]] == ["chunk-b"]
    chunk_b = fused[0]
    assert chunk_b["retrieval_channels"] == ["dense", "learned_sparse", "original", "hyde"]
    assert chunk_b["document_id"] == "doc-chunk-b"
    assert any(candidate["source_type"] == "web" for candidate in fused)
    assert all(candidate["rerank_score"] is None for candidate in fused)


def test_outer_rrf_result_does_not_depend_on_action_list_arrival_order():
    original = [_local_candidate("chunk-a", channels=["dense", "original"])]
    hyde = [_local_candidate("chunk-a", channels=["learned_sparse", "hyde"])]
    web = [_web_candidate("https://example.com/latest")]

    forward = fuse_ranked_candidate_lists([original, hyde, web], limit=10)
    reversed_order = fuse_ranked_candidate_lists([web, hyde, original], limit=10)

    assert forward == reversed_order


@pytest.mark.parametrize(
    "candidate_lists",
    [
        [[_local_candidate("chunk-a", channels=["original", "dense"])]],
        [
            [_local_candidate("chunk-a", channels=["original", "dense"])],
            [_local_candidate("chunk-b", channels=["hyde", "dense"])],
        ],
        [
            [_local_candidate("chunk-a", channels=["original", "dense"])],
            [_web_candidate("https://example.com/latest")],
        ],
        [[_web_candidate("https://example.com/latest")]],
    ],
)
def test_outer_rrf_supports_every_executed_action_combination(candidate_lists):
    assert fuse_ranked_candidate_lists(candidate_lists)


def test_web_candidates_deduplicate_by_canonical_url_without_fake_local_id():
    first = [_web_candidate("HTTPS://Example.com/notice/#section")]
    second = [_web_candidate("https://example.com/notice")]

    fused = fuse_ranked_candidate_lists([first, second], limit=10)

    assert len(fused) == 1
    assert canonicalize_web_url(first[0]["url"]) == canonicalize_web_url(second[0]["url"])
    assert fused[0]["document_id"] is None
    assert fused[0]["chunk_id"] is None


def test_unified_rerank_preserves_local_and_web_metadata(monkeypatch):
    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(str(text))

    class FakeReranker:
        tokenizer = FakeTokenizer()

        def compute_score(self, qa_pairs, normalize=True):
            assert normalize is True
            return [0.2, 0.9]

    fake_provider = SimpleNamespace(reranker_model=lambda: FakeReranker())
    monkeypatch.setattr(rerank_service, "llm_provider", fake_provider)

    local = _local_candidate("chunk-a", channels=["dense", "original"])
    web = _web_candidate("https://example.com/latest")
    result = rerank_service.rerank_documents({
        "rrf_chunks": [local, web],
        "rewritten_query": "HAK 180 最新维修通知",
        "retrieval_observation": None,
    })["reranked_docs"]

    assert result[0]["source_type"] == "web"
    assert result[0]["document_id"] is None
    assert result[0]["url"] == "https://example.com/latest"
    assert result[1]["source_type"] == "local"
    assert result[1]["document_id"] == "doc-chunk-a"
    assert result[1]["chunk_id"] == "chunk-a"
    assert result[1]["equipment_model"] == "HAK 180"
    assert result[1]["retrieval_score"] == local["retrieval_score"]
    assert result[1]["rerank_score"] == 0.2


def test_web_only_answer_does_not_require_fake_local_subject():
    web = _web_candidate("https://example.com/latest")
    web["rerank_score"] = 0.9
    reranked_docs, subject_names, query, history = answer_service.check_params({
        "reranked_docs": [web],
        "standard_subject_names": [],
        "rewritten_query": "今天有没有新的厂家通知？",
        "history": [],
    })

    assert reranked_docs[0]["source_type"] == "web"
    assert subject_names == []
    assert query == "今天有没有新的厂家通知？"
    assert history == []


def test_web_search_formats_pages_as_web_retrieval_candidates(monkeypatch):
    async def fake_web_search(query):
        return SimpleNamespace(content=[SimpleNamespace(text='''{
            "pages": [
                {"title": "厂家通知", "url": "https://example.com/a", "snippet": "通知正文"},
                {"title": "无链接结果", "snippet": "应被忽略"}
            ]
        }''')])

    monkeypatch.setattr(web_search_service, "web_search_func", fake_web_search)
    result = web_search_service.search_by_web({"rewritten_query": "最新通知"})["web_search_docs"]

    assert len(result) == 1
    assert result[0]["source_type"] == "web"
    assert result[0]["retrieval_channels"] == ["web"]
    assert result[0]["document_id"] is None
    assert result[0]["url"] == "https://example.com/a"
