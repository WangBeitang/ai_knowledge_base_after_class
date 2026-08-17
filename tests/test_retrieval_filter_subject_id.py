import copy

import pytest

from app.process.query.agent.state import create_query_default_state
from app.rag.query import embedding_search_service
from app.rag.query import hyde_search_service
from app.rag.query.chunk_retrieval_utils import (
    build_chunk_access_filter,
    build_chunk_retrieval_filter,
    build_structured_identifier_filter,
)
from app.shared.clients.milvus_utils import create_hybrid_search_requests


class FakeLLMProvider:
    def __init__(self):
        self.documents = []

    def embed_documents(self, documents):
        self.documents.append(documents)
        return {"dense": [[0.1]], "sparse": [{1: 0.2}]}


class FakeMilvusGateway:
    chunk_collection_name = "chunks"

    def __init__(self):
        self.exprs = []

    def create_requests(self, dense_vector, sparse_vector, expr=None, limit=5, **kwargs):
        self.exprs.append(expr)
        return [f"req:{expr}"]

    def hybrid_search(self, **kwargs):
        return [
            [
                {
                    "id": 1001,
                    "distance": 0.86,
                    "entity": {
                    "chunk_id": 1001,
                    "dataset_id": "dataset_ops",
                    "document_id": "doc-1001",
                    "index_version": 1,
                    "chunk_index": 0,
                    "enabled": True,
                        "subject_id": "subject_hak_180",
                        "standard_subject_name": "HAK 180 烫金机",
                        "content": "新数据按 subject_id 召回。",
                        "title": "标准主题手册",
                    },
                }
            ]
        ]


def _expected_access_filter(user_id="user_a", tenant_id="tenant_default"):
    return (
        'dataset_id in ["dataset_ops"] '
        'AND enabled == true '
        'AND (visibility == "public" '
        f'OR (visibility == "shared" AND tenant_id == "{tenant_id}") '
        f'OR owner_user_id == "{user_id}")'
    )


def _query_state(**overrides):
    state = create_query_default_state(
        session_id="session-filter-stage5",
        rewritten_query="HAK 180 的 E021 报警怎么处理？",
        owner_user_id="user_a",
        tenant_id="tenant_default",
        dataset_ids=["dataset_ops"],
        subject_ids=["subject_hak_180"],
        query_identifiers={},
    )
    state.update(overrides)
    return state


@pytest.mark.parametrize("user_id", ["user_a", "user_b"])
def test_access_filter_keeps_public_shared_and_current_owner_in_one_parenthesized_group(user_id):
    expr = build_chunk_access_filter(
        dataset_ids=["dataset_ops"],
        owner_user_id=user_id,
        tenant_id="tenant_default",
    )

    # user_a 与 user_b 生成相同的 public/shared 规则，私有分支只绑定当前用户本人。
    assert expr == _expected_access_filter(user_id=user_id)
    assert expr.count("AND (") == 1
    assert expr.endswith(f'OR owner_user_id == "{user_id}")')


def test_retrieval_filter_combines_access_subject_and_identifiers_in_stable_order():
    expr = build_chunk_retrieval_filter(
        dataset_ids=[" dataset_ops ", "dataset_ops"],
        subject_ids=["subject_hak_180", " subject_hak_180 "],
        owner_user_id=" user_a ",
        tenant_id=" tenant_default ",
        # 故意使用与 schema 不同的字典顺序，验证最终表达式仍按固定字段顺序输出。
        query_identifiers={
            "alarm_code": ["E021", " E021 "],
            "equipment_model": ["HAK 180"],
        },
    )

    assert expr == (
        _expected_access_filter()
        + ' AND subject_id in ["subject_hak_180"]'
        + ' AND equipment_model in ["HAK 180"]'
        + ' AND alarm_code in ["E021"]'
    )


def test_filter_escapes_every_dynamic_string_literal():
    expr = build_chunk_retrieval_filter(
        dataset_ids=['data"set\\ops'],
        subject_ids=['subject\nHAK"180'],
        owner_user_id='user"a',
        tenant_id="tenant\\default",
        query_identifiers={"alarm_code": ['E"021\\A']},
    )

    assert 'dataset_id in ["data\\"set\\\\ops"]' in expr
    assert 'subject_id in ["subject HAK\\"180"]' in expr
    assert 'owner_user_id == "user\\"a"' in expr
    assert 'tenant_id == "tenant\\\\default"' in expr
    assert 'alarm_code in ["E\\"021\\\\A"]' in expr


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dataset_ids": []}, "dataset_ids 不能为空，禁止退化为全库检索"),
        ({"dataset_ids": ["", "  "]}, "dataset_ids 不能为空，禁止退化为全库检索"),
        ({"subject_ids": []}, "subject_ids 不能为空，禁止退化为全库检索"),
        ({"subject_ids": ["\n"]}, "subject_ids 不能为空，禁止退化为全库检索"),
        ({"owner_user_id": ""}, "owner_user_id 不能为空"),
        ({"tenant_id": "  "}, "tenant_id 不能为空"),
    ],
)
def test_filter_rejects_missing_scope_instead_of_falling_back_to_full_collection(overrides, message):
    kwargs = {
        "dataset_ids": ["dataset_ops"],
        "subject_ids": ["subject_hak_180"],
        "owner_user_id": "user_a",
        "tenant_id": "tenant_default",
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        build_chunk_retrieval_filter(**kwargs)


@pytest.mark.parametrize("field_name", ["dataset_ids", "subject_ids"])
def test_filter_rejects_single_string_where_a_list_is_required(field_name):
    kwargs = {
        "dataset_ids": ["dataset_ops"],
        "subject_ids": ["subject_hak_180"],
        "owner_user_id": "user_a",
        "tenant_id": "tenant_default",
    }
    kwargs[field_name] = "mistaken_single_string"

    with pytest.raises(ValueError, match=f"{field_name} 必须是字符串列表"):
        build_chunk_retrieval_filter(**kwargs)


def test_structured_identifier_filter_is_optional_but_rejects_invalid_claims():
    assert build_structured_identifier_filter() == ""
    assert build_structured_identifier_filter({}) == ""

    with pytest.raises(ValueError, match="不支持的字段：sop_number"):
        build_structured_identifier_filter({"sop_number": ["SOP-001"]})

    with pytest.raises(ValueError, match="query_identifiers.alarm_code 不能为空"):
        build_structured_identifier_filter({"alarm_code": ["", "  "]})


def test_embedding_and_hyde_reuse_exactly_the_same_filter_expr(monkeypatch):
    fake_gateway = FakeMilvusGateway()
    fake_llm_provider = FakeLLMProvider()
    monkeypatch.setattr(embedding_search_service, "llm_provider", fake_llm_provider)
    monkeypatch.setattr(embedding_search_service, "milvus_gateway", fake_gateway)
    monkeypatch.setattr(hyde_search_service, "llm_provider", fake_llm_provider)
    monkeypatch.setattr(hyde_search_service, "milvus_gateway", fake_gateway)
    monkeypatch.setattr(
        hyde_search_service,
        "generate_hyde_answer",
        lambda query: ("开机前检查急停按钮。", {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "answer_usage_available": True}),
    )

    state = _query_state(query_identifiers={"alarm_code": ["E021"]})
    embedding_result = embedding_search_service.search_by_embedding(copy.deepcopy(state))
    hyde_result = hyde_search_service.search_by_hyde(copy.deepcopy(state))

    expected_expr = (
        _expected_access_filter()
        + ' AND subject_id in ["subject_hak_180"]'
        + ' AND alarm_code in ["E021"]'
    )
    assert fake_gateway.exprs == [expected_expr, expected_expr]
    assert embedding_result["embedding_chunks"][0]["source_type"] == "local"
    assert "original" in embedding_result["embedding_chunks"][0]["retrieval_channels"]
    assert hyde_result["hyde_embedding_chunks"][0]["source_type"] == "local"
    assert "hyde" in hyde_result["hyde_embedding_chunks"][0]["retrieval_channels"]


def test_dense_and_learned_sparse_requests_receive_the_same_filter_expr():
    # 当前 hybrid search 内含 dense（稠密向量）和 learned sparse（学习式稀疏向量）
    # 两个 AnnSearchRequest；二者必须携带完全相同的权限表达式。
    filter_expr = build_chunk_retrieval_filter(
        dataset_ids=["dataset_ops"],
        subject_ids=["subject_hak_180"],
        owner_user_id="user_a",
        tenant_id="tenant_default",
    )

    dense_request, learned_sparse_request = create_hybrid_search_requests(
        dense_vector=[0.1, 0.2],
        sparse_vector={1: 0.3},
        expr=filter_expr,
    )

    assert dense_request.expr == filter_expr
    assert learned_sparse_request.expr == filter_expr


@pytest.mark.parametrize("missing_field", ["dataset_ids", "subject_ids", "owner_user_id", "tenant_id"])
def test_embedding_search_fails_before_embedding_or_milvus_when_scope_is_missing(monkeypatch, missing_field):
    fake_gateway = FakeMilvusGateway()
    fake_llm_provider = FakeLLMProvider()
    monkeypatch.setattr(embedding_search_service, "llm_provider", fake_llm_provider)
    monkeypatch.setattr(embedding_search_service, "milvus_gateway", fake_gateway)

    empty_value = [] if missing_field.endswith("_ids") else ""
    state = _query_state(**{missing_field: empty_value})

    with pytest.raises(ValueError):
        embedding_search_service.search_by_embedding(state)

    assert fake_llm_provider.documents == []
    assert fake_gateway.exprs == []


def test_embedding_query_preserves_formatted_result_metadata(monkeypatch):
    fake_gateway = FakeMilvusGateway()
    monkeypatch.setattr(embedding_search_service, "llm_provider", FakeLLMProvider())
    monkeypatch.setattr(embedding_search_service, "milvus_gateway", fake_gateway)

    result = embedding_search_service.query_chunk_by_milvus(
        rewritten_query="HAK 180 怎么开机？",
        filter_expr="expected-filter",
    )

    assert fake_gateway.exprs == ["expected-filter"]
    assert result[0]["chunk_id"] == 1001
    assert result[0]["source_type"] == "local"
    assert result[0]["document_id"] == "doc-1001"
    assert result[0]["enabled"] is True
