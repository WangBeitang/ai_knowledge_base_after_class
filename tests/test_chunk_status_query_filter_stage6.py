import pytest

from app.process.query.agent.state import create_query_default_state
from app.rag.query import embedding_search_service, hyde_search_service
from app.rag.query.chunk_retrieval_utils import (
    build_chunk_retrieval_filter,
    build_disabled_chunk_exclusion_filter,
)
from app.rag.query.chunk_status_filter_service import get_disabled_chunk_ids_for_query


class FakeLLMProvider:
    def __init__(self):
        self.documents = []

    def embed_documents(self, documents):
        self.documents.append(documents)
        return {"dense": [[0.1]], "sparse": [{1: 0.2}]}


class ExcludingFakeMilvusGateway:
    chunk_collection_name = "chunks"

    def __init__(self):
        self.exprs = []

    def create_requests(self, dense_vector, sparse_vector, expr=None, **kwargs):
        self.exprs.append(expr)
        return [expr]

    def hybrid_search(self, *, reqs, **kwargs):
        expr = reqs[0]
        if "chunk_id not in [1001]" in expr:
            return [[]]
        return [[_milvus_item(1001)]]


class FakeStatusRepository:
    def __init__(self):
        self.calls = []

    def list_disabled_chunk_ids(self, **kwargs):
        self.calls.append(kwargs)
        return [1001]


def _milvus_item(chunk_id):
    return {
        "id": chunk_id,
        "distance": 0.91,
        "entity": {
            "chunk_id": chunk_id,
            "dataset_id": "dataset_ops",
            "document_id": "doc-1001",
            "owner_user_id": "user_a",
            "tenant_id": "tenant_default",
            "visibility": "private",
            "index_version": 1,
            "chunk_index": 0,
            "enabled": True,
            "subject_id": "subject_hak_180",
            "standard_subject_name": "HAK 180 烫金机",
            "content": "E020 报警处理步骤。",
            "title": "E020 报警",
        },
    }


def _query_state(**overrides):
    state = create_query_default_state(
        session_id="session-stage6-query-filter",
        original_query="HAK 180 的 E020 怎么处理？",
        rewritten_query="HAK 180 的 E020 怎么处理？",
        owner_user_id="user_a",
        tenant_id="tenant_default",
        dataset_ids=["dataset_ops"],
        subject_ids=["subject_hak_180"],
        query_identifiers={},
    )
    state.update(overrides)
    return state


def test_disabled_chunk_exclusion_filter_uses_blacklist_not_second_enabled_whitelist():
    assert build_disabled_chunk_exclusion_filter(None) == ""
    assert build_disabled_chunk_exclusion_filter([]) == ""
    assert build_disabled_chunk_exclusion_filter([1001, "1002", 'manual"3', 1001]) == (
        'chunk_id not in [1001,1002,"manual\\"3"]'
    )

    with pytest.raises(ValueError, match="不能直接传入单个字符串"):
        build_disabled_chunk_exclusion_filter("1001")
    with pytest.raises(ValueError, match="不能包含空 chunk_id"):
        build_disabled_chunk_exclusion_filter(["  "])


def test_retrieval_filter_keeps_milvus_enabled_and_appends_manual_disabled_exclusion():
    expr = build_chunk_retrieval_filter(
        dataset_ids=["dataset_ops"],
        subject_ids=["subject_hak_180"],
        owner_user_id="user_a",
        tenant_id="tenant_default",
        query_identifiers={"alarm_code": ["E020"]},
        disabled_chunk_ids=[1001],
    )

    assert "enabled == true" in expr
    assert 'alarm_code in ["E020"]' in expr
    assert expr.endswith("AND chunk_id not in [1001]")


def test_disabled_chunk_ids_loader_only_reads_mongo_when_query_filter_is_enabled():
    repository = FakeStatusRepository()

    assert get_disabled_chunk_ids_for_query(
        _query_state(chunk_status_filter_enabled=False),
        status_repository=repository,
    ) == []
    assert repository.calls == []

    disabled_chunk_ids = get_disabled_chunk_ids_for_query(
        _query_state(chunk_status_filter_enabled=True),
        status_repository=repository,
    )

    assert disabled_chunk_ids == [1001]
    assert repository.calls == [{
        "dataset_ids": ["dataset_ops"],
        "owner_user_id": "user_a",
        "tenant_id": "tenant_default",
    }]


def test_embedding_search_excludes_mongo_disabled_chunk_from_actual_query(monkeypatch):
    fake_gateway = ExcludingFakeMilvusGateway()
    monkeypatch.setattr(embedding_search_service, "milvus_gateway", fake_gateway)
    monkeypatch.setattr(embedding_search_service, "llm_provider", FakeLLMProvider())
    monkeypatch.setattr(
        embedding_search_service,
        "get_disabled_chunk_ids_for_query",
        lambda state: [1001],
    )

    result = embedding_search_service.search_by_embedding(
        _query_state(chunk_status_filter_enabled=True)
    )

    assert result["embedding_chunks"] == []
    assert result["disabled_chunk_ids"] == [1001]
    assert fake_gateway.exprs == [
        'dataset_id in ["dataset_ops"] '
        'AND enabled == true '
        'AND (visibility == "public" '
        'OR (visibility == "shared" AND tenant_id == "tenant_default") '
        'OR owner_user_id == "user_a") '
        'AND subject_id in ["subject_hak_180"] '
        'AND chunk_id not in [1001]'
    ]


def test_hyde_search_excludes_mongo_disabled_chunk_from_actual_query(monkeypatch):
    fake_gateway = ExcludingFakeMilvusGateway()
    generated_queries = []
    monkeypatch.setattr(hyde_search_service, "milvus_gateway", fake_gateway)
    monkeypatch.setattr(hyde_search_service, "llm_provider", FakeLLMProvider())
    monkeypatch.setattr(
        hyde_search_service,
        "generate_hyde_answer",
        lambda query: generated_queries.append(query) or "检查 E020 报警。",
    )
    monkeypatch.setattr(
        hyde_search_service,
        "get_disabled_chunk_ids_for_query",
        lambda state: [1001],
    )

    result = hyde_search_service.search_by_hyde(
        _query_state(
            chunk_status_filter_enabled=True,
            standard_subject_names=["HAK 180 烫金机"],
        )
    )

    assert result["hyde_embedding_chunks"] == []
    assert result["disabled_chunk_ids"] == [1001]
    assert generated_queries == [
        "已确认设备主体：HAK 180 烫金机。\n用户问题：HAK 180 的 E020 怎么处理？"
    ]
    assert len(fake_gateway.exprs) == 1
    assert "enabled == true" in fake_gateway.exprs[0]
    assert "chunk_id not in [1001]" in fake_gateway.exprs[0]
