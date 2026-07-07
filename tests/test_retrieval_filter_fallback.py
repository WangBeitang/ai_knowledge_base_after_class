from app.rag.query import embedding_search_service
from app.rag.query import hyde_search_service


class FakeLLMProvider:
    def embed_documents(self, documents):
        return {"dense": [[0.1]], "sparse": [{1: 0.2}]}


class FakeMilvusGateway:
    chunk_collection_name = "chunks"

    def __init__(self):
        self.exprs = []

    def create_requests(self, dense_vector, sparse_vector, expr=None, limit=5):
        self.exprs.append(expr)
        return [f"req:{expr}"]

    def hybrid_search(self, **kwargs):
        expr = self.exprs[-1]
        if expr.startswith("subject_id"):
            return [[]]
        return [
            [
                {
                    "id": 1001,
                    "distance": 0.86,
                    "entity": {
                        "chunk_id": 1001,
                        "subject_name": "HAK 180 烫金机",
                        "content": "旧数据只有 subject_name，也应该能召回。",
                        "title": "旧手册",
                    },
                }
            ]
        ]


def test_embedding_search_fallbacks_to_subject_name_when_subject_id_has_no_hits(monkeypatch):
    fake_gateway = FakeMilvusGateway()
    monkeypatch.setattr(embedding_search_service, "llm_provider", FakeLLMProvider())
    monkeypatch.setattr(embedding_search_service, "milvus_gateway", fake_gateway)

    result = embedding_search_service.query_chunk_by_milvus(
        subject_ids=["subject_hak_180"],
        subject_names=["HAK 180 烫金机"],
        rewritten_query="HAK 180 怎么开机？",
    )

    assert fake_gateway.exprs == [
        'subject_id in ["subject_hak_180"]',
        'subject_name in ["HAK 180 烫金机"]',
    ]
    assert result[0]["chunk_id"] == 1001
    assert result[0]["type"] == "milvus"


def test_hyde_search_fallbacks_to_subject_name_when_subject_id_has_no_hits(monkeypatch):
    fake_gateway = FakeMilvusGateway()
    monkeypatch.setattr(hyde_search_service, "llm_provider", FakeLLMProvider())
    monkeypatch.setattr(hyde_search_service, "milvus_gateway", fake_gateway)

    result = hyde_search_service.query_chunk_by_milvus(
        subject_ids=["subject_hak_180"],
        subject_names=["HAK 180 烫金机"],
        rewritten_query="HAK 180 怎么开机？",
        hyde_answer="开机前检查急停按钮。",
    )

    assert fake_gateway.exprs == [
        'subject_id in ["subject_hak_180"]',
        'subject_name in ["HAK 180 烫金机"]',
    ]
    assert result[0]["chunk_id"] == 1001
    assert result[0]["type"] == "hyde"
