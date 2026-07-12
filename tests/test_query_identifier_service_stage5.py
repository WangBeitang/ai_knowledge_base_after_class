import copy

import pytest

from app.process.query.agent.state import create_query_default_state
from app.rag.query import answer_service, embedding_search_service, rerank_service, rrf_service
from app.rag.query.chunk_retrieval_utils import build_chunk_retrieval_filter
from app.rag.query.contracts import (
    IdentifierResolutionStatus,
    ObservationStatus,
    PlannerReasonCode,
    QueryAction,
    RetrievalObservation,
)
from app.rag.query.query_identifier_service import (
    append_identifiers_to_query,
    extract_query_identifiers,
    rank_identifier_suggestions,
)


class FakeLLMProvider:
    """只提供 embedding；若测试意外进入答案 LLM，会由具体用例显式令其失败。"""

    def __init__(self):
        self.documents = []

    def embed_documents(self, documents):
        self.documents.append(documents)
        return {"dense": [[0.1]], "sparse": [{1: 0.2}]}


class TwoStageFakeGateway:
    chunk_collection_name = "chunks"

    def __init__(self, *, exact_items=None, broad_items=None, dictionary_records=None):
        self.exact_items = exact_items or []
        self.broad_items = broad_items or []
        self.dictionary_records = dictionary_records or []
        self.search_exprs = []
        self.dictionary_exprs = []

    def create_requests(self, dense_vector, sparse_vector, expr=None, limit=5):
        self.search_exprs.append(expr)
        return [expr]

    def hybrid_search(self, *, reqs, **kwargs):
        expr = reqs[0]
        items = self.exact_items if "alarm_code in" in expr else self.broad_items
        return [items]

    def query_entities(self, *, filter_expr, **kwargs):
        self.dictionary_exprs.append(filter_expr)
        return copy.deepcopy(self.dictionary_records)


def _milvus_item(
        chunk_id,
        content,
        *,
        equipment_model="HAK 180",
        alarm_code="",
        title="HAK 180 手册",
):
    return {
        "id": chunk_id,
        "distance": 0.9,
        "entity": {
            "chunk_id": chunk_id,
            "dataset_id": "dataset_ops",
            "document_id": "doc-1",
            "owner_user_id": "user_a",
            "tenant_id": "tenant_default",
            "visibility": "private",
            "enabled": True,
            "subject_id": "subject_hak_180",
            "standard_subject_name": "HAK 180 烫金机",
            "content": content,
            "title": title,
            "equipment_model": equipment_model,
            "alarm_code": alarm_code,
        },
    }


def _query_state(query="HAK 180 的 E020 怎么处理？"):
    return create_query_default_state(
        session_id="session-stage5-identifiers",
        original_query=query,
        rewritten_query=query,
        owner_user_id="user_a",
        tenant_id="tenant_default",
        dataset_ids=["dataset_ops"],
        subject_ids=["subject_hak_180"],
        standard_subject_names=["HAK 180 烫金机"],
        query_identifiers=extract_query_identifiers(query),
    )


@pytest.mark.parametrize("variant", ["HAK180", "HAK-180", "HAK 180", "hak_180"])
def test_equipment_model_variants_share_one_canonical_form(variant):
    assert extract_query_identifiers(f"{variant} 怎么开机？") == {
        "equipment_model": ["HAK 180"]
    }


def test_alarm_codes_are_extracted_independently_and_ambiguous_words_are_ignored():
    assert extract_query_identifiers("E021 和 E020 分别是什么故障？") == {
        "alarm_code": ["E021", "E020"]
    }
    assert extract_query_identifiers("这是 SOP 操作流程，版本 20，需要检查设备。") == {}


def test_sop_and_explicit_part_number_are_lexical_identifiers():
    identifiers = extract_query_identifiers("查看 SOP-001，零件编号 AB-123")
    assert identifiers == {"sop_code": ["SOP-001"], "part_number": ["AB-123"]}

    enhanced = append_identifiers_to_query("请查看操作步骤", identifiers)
    assert "SOP 编号 SOP-001" in enhanced
    assert "零件编号 AB-123" in enhanced

    # lexical-only 字段可以进入 State/检索文本，但不会生成不存在的 Milvus schema 条件。
    expr = build_chunk_retrieval_filter(
        dataset_ids=["dataset_ops"],
        subject_ids=["subject_hak_180"],
        owner_user_id="user_a",
        tenant_id="tenant_default",
        query_identifiers=identifiers,
    )
    assert "sop_code in" not in expr
    assert "part_number in" not in expr


def test_empty_identifier_snapshot_is_not_reparsed_from_rewritten_query(monkeypatch):
    gateway = TwoStageFakeGateway(
        broad_items=[_milvus_item(10, "HAK 180 通用说明。", alarm_code="")],
    )
    monkeypatch.setattr(embedding_search_service, "milvus_gateway", gateway)
    monkeypatch.setattr(embedding_search_service, "llm_provider", FakeLLMProvider())
    state = _query_state("这个问题怎么处理？")
    state["rewritten_query"] = "HAK 180 烫金机这个问题怎么处理？"
    state["query_identifiers"] = {}

    result = embedding_search_service.search_by_embedding(state)

    assert result["query_identifiers"] == {}
    assert (
        result["retrieval_observation"].identifier_resolution_status
        == IdentifierResolutionStatus.NOT_APPLICABLE
    )
    assert len(gateway.search_exprs) == 1
    assert "equipment_model in" not in gateway.search_exprs[0]


def test_explicit_part_number_matches_same_code_in_content_without_repeated_label(monkeypatch):
    gateway = TwoStageFakeGateway(
        broad_items=[_milvus_item(11, "更换备件 AB123 后重新开机。", alarm_code="")],
    )
    monkeypatch.setattr(embedding_search_service, "milvus_gateway", gateway)
    monkeypatch.setattr(embedding_search_service, "llm_provider", FakeLLMProvider())

    result = embedding_search_service.search_by_embedding(
        _query_state("零件编号 AB-123 应该怎么更换？")
    )
    observation = result["retrieval_observation"]

    assert observation.identifier_resolution_status == IdentifierResolutionStatus.EXACT_MATCH
    assert observation.requested_identifiers == {"part_number": ["AB-123"]}
    assert observation.matched_identifiers == {"part_number": ["AB-123"]}
    assert observation.used_structured_filter is False
    assert observation.filter_fallback is False


def test_structured_exact_match_uses_one_search_and_keeps_user_identifier(monkeypatch):
    gateway = TwoStageFakeGateway(
        exact_items=[_milvus_item(1, "E020 表示温度异常。", alarm_code="E020")],
    )
    llm = FakeLLMProvider()
    monkeypatch.setattr(embedding_search_service, "milvus_gateway", gateway)
    monkeypatch.setattr(embedding_search_service, "llm_provider", llm)

    result = embedding_search_service.search_by_embedding(_query_state("E020 怎么处理？"))
    observation = result["retrieval_observation"]

    assert observation.identifier_resolution_status == IdentifierResolutionStatus.EXACT_MATCH
    assert observation.requested_identifiers == {"alarm_code": ["E020"]}
    assert observation.matched_identifiers == {"alarm_code": ["E020"]}
    assert observation.used_structured_filter is True
    assert observation.filter_fallback is False
    assert len(gateway.search_exprs) == 1
    assert gateway.dictionary_exprs == []
    assert len(llm.documents) == 1


def test_metadata_miss_but_same_code_in_content_is_fallback_exact_match(monkeypatch):
    gateway = TwoStageFakeGateway(
        exact_items=[],
        broad_items=[_milvus_item(2, "屏幕显示 E020 时请先停机。", alarm_code="")],
    )
    monkeypatch.setattr(embedding_search_service, "milvus_gateway", gateway)
    monkeypatch.setattr(embedding_search_service, "llm_provider", FakeLLMProvider())

    result = embedding_search_service.search_by_embedding(_query_state("E020 怎么处理？"))
    observation = result["retrieval_observation"]

    assert observation.identifier_resolution_status == IdentifierResolutionStatus.FALLBACK_EXACT_MATCH
    assert observation.matched_identifiers == {"alarm_code": ["E020"]}
    assert observation.used_structured_filter is True
    assert observation.filter_fallback is True
    assert len(result["embedding_chunks"]) == 1
    assert len(gateway.search_exprs) == 2
    # 两段 expr 的权限、dataset、subject 和 enabled 完全相同，第二段只少 alarm_code。
    assert 'dataset_id in ["dataset_ops"]' in gateway.search_exprs[1]
    assert 'subject_id in ["subject_hak_180"]' in gateway.search_exprs[1]
    assert "enabled == true" in gateway.search_exprs[1]
    assert "alarm_code in" not in gateway.search_exprs[1]


def test_different_near_code_requires_confirmation_and_never_becomes_citation(monkeypatch):
    gateway = TwoStageFakeGateway(
        exact_items=[],
        broad_items=[_milvus_item(3, "E021 表示传感器异常。", alarm_code="E021")],
        dictionary_records=[{"chunk_id": 3, "equipment_model": "HAK 180", "alarm_code": "E021"}],
    )
    monkeypatch.setattr(embedding_search_service, "milvus_gateway", gateway)
    monkeypatch.setattr(embedding_search_service, "llm_provider", FakeLLMProvider())

    result = embedding_search_service.search_by_embedding(_query_state("E020 怎么处理？"))
    observation = result["retrieval_observation"]

    assert observation.identifier_resolution_status == IdentifierResolutionStatus.SUGGESTION_REQUIRED
    assert observation.requested_identifiers == {"alarm_code": ["E020"]}
    assert observation.suggested_identifiers == {"alarm_code": ["E021"]}
    assert observation.matched_identifiers == {}
    assert observation.citation_count == 0
    assert "E020" in observation.clarification_question
    assert "E021" in observation.clarification_question
    assert result["query_identifiers"] == {"alarm_code": ["E020"]}


def test_cross_device_candidate_is_not_silently_suggested():
    requested = {"equipment_model": ["HAK 180"], "alarm_code": ["E020"]}
    unauthorized_or_cross_device_records = [
        {"equipment_model": "HAK 200", "alarm_code": "E021"},
        {"equipment_model": "HAK 180", "alarm_code": "E099"},
    ]
    assert rank_identifier_suggestions(requested, unauthorized_or_cross_device_records) == {}

    compatible_record = [{"equipment_model": "HAK 180", "alarm_code": "E021"}]
    assert rank_identifier_suggestions(requested, compatible_record) == {
        "alarm_code": ["E021"]
    }


def test_no_same_code_or_reliable_candidate_returns_not_found(monkeypatch):
    gateway = TwoStageFakeGateway(
        exact_items=[],
        broad_items=[_milvus_item(4, "通用维护说明，不包含报警码。", alarm_code="")],
        dictionary_records=[],
    )
    monkeypatch.setattr(embedding_search_service, "milvus_gateway", gateway)
    monkeypatch.setattr(embedding_search_service, "llm_provider", FakeLLMProvider())

    result = embedding_search_service.search_by_embedding(_query_state("E020 怎么处理？"))
    observation = result["retrieval_observation"]

    assert observation.identifier_resolution_status == IdentifierResolutionStatus.NOT_FOUND
    assert observation.matched_identifiers == {}
    assert observation.suggested_identifiers == {}
    assert "核对设备屏幕" in observation.clarification_question


def test_answer_guard_delivers_question_without_calling_answer_llm(monkeypatch):
    observation = RetrievalObservation(
        action=QueryAction.LOCAL_SEARCH,
        status=ObservationStatus.SUCCESS,
        candidate_count=1,
        requested_identifiers={"alarm_code": ["E020"]},
        identifier_resolution_status=IdentifierResolutionStatus.SUGGESTION_REQUIRED,
        suggested_identifiers={"alarm_code": ["E021"]},
        clarification_question="没有找到 E020，是否要查询 E021？",
        used_structured_filter=True,
        filter_fallback=True,
    )
    state = _query_state("E020 怎么处理？")
    state["retrieval_observation"] = observation
    state["reranked_docs"] = [{"title": "E021", "text": "不应进入答案", "type": "milvus"}]

    monkeypatch.setattr(
        answer_service.llm_provider,
        "chat",
        lambda: pytest.fail("编号未确认时不应调用答案 LLM"),
    )
    monkeypatch.setattr(answer_service, "try_return_existing_answer", lambda current: bool(current.get("answer")))
    monkeypatch.setattr(answer_service, "save_assistant_message", lambda current: None)

    result = answer_service.generate_answer(state)
    assert result["answer"] == "没有找到 E020，是否要查询 E021？"
    assert result["citations"] == []
    assert result["image_urls"] == []
    assert result["terminal_reason_code"] == PlannerReasonCode.IDENTIFIER_CONFIRMATION_REQUIRED


def test_terminal_identifier_observation_skips_empty_rrf_and_rerank(monkeypatch):
    observation = RetrievalObservation(
        action=QueryAction.LOCAL_SEARCH,
        status=ObservationStatus.EMPTY,
        requested_identifiers={"alarm_code": ["E020"]},
        identifier_resolution_status=IdentifierResolutionStatus.NOT_FOUND,
        clarification_question="请核对 E020。",
        used_structured_filter=True,
        filter_fallback=True,
    )
    state = _query_state("E020 怎么处理？")
    state["retrieval_observation"] = observation
    state["embedding_chunks"] = []
    state["hyde_embedding_chunks"] = []
    state["web_search_docs"] = [{"title": "网页", "snippet": "不应参与编号纠错"}]

    assert rrf_service.fuse_by_rrf(state)["rrf_chunks"] == []
    monkeypatch.setattr(
        rerank_service,
        "reranker_score",
        lambda pairs: pytest.fail("编号未确认时不应调用 reranker"),
    )
    assert rerank_service.rerank_documents(state)["reranked_docs"] == []
