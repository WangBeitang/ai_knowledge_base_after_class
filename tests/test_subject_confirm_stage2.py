from app.rag.query import subject_name_confirm_service as service


def test_classify_subject_aliases_confirms_standard_subject(monkeypatch):
    monkeypatch.setattr(service, "SUBJECT_NAME_CONFIRM_THRESHOLD", 0.8)
    monkeypatch.setattr(service, "SUBJECT_NAME_CANDIDATE_THRESHOLD", 0.5)

    confirmed_records, candidate_names = service.classify_subject_aliases(
        {
            "HAK180": [
                {
                    "alias": "HAK180",
                    "subject_id": "subject_hak_180",
                    "standard_subject_name": "HAK 180 烫金机",
                    "score": 0.92,
                }
            ]
        }
    )

    assert confirmed_records == [
        {
            "subject_id": "subject_hak_180",
            "standard_subject_name": "HAK 180 烫金机",
        }
    ]
    assert candidate_names == []


def test_classify_subject_aliases_returns_candidates(monkeypatch):
    monkeypatch.setattr(service, "SUBJECT_NAME_CONFIRM_THRESHOLD", 0.8)
    monkeypatch.setattr(service, "SUBJECT_NAME_CANDIDATE_THRESHOLD", 0.5)
    monkeypatch.setattr(service, "SUBJECT_NAME_OPTIONS_TOPK", 2)

    confirmed_records, candidate_names = service.classify_subject_aliases(
        {
            "180": [
                {
                    "alias": "HAK180",
                    "subject_id": "subject_hak_180",
                    "standard_subject_name": "HAK 180 烫金机",
                    "score": 0.7,
                },
                {
                    "alias": "HAK180A",
                    "subject_id": "subject_hak_180_a",
                    "standard_subject_name": "HAK 180A 烫金机",
                    "score": 0.65,
                },
            ]
        }
    )

    assert confirmed_records == []
    assert candidate_names == ["HAK 180 烫金机", "HAK 180A 烫金机"]


def test_confirm_subject_name_writes_subject_ids_and_standard_names(monkeypatch):
    monkeypatch.setattr(service, "params_check", lambda state: ("HAK180怎么开机？", "session-1"))
    monkeypatch.setattr(service, "load_history", lambda session_id, user_id: [])
    monkeypatch.setattr(
        service,
        "query_rewrite_and_subject_name_recognition",
        lambda original_query, history_text: ("HAK180 怎么开机？", ["HAK180"]),
    )
    monkeypatch.setattr(
        service,
        "search_subject_alias_in_milvus",
        lambda subject_mentions: {
            "HAK180": [
                {
                    "alias": "HAK180",
                    "subject_id": "subject_hak_180",
                    "standard_subject_name": "HAK 180 烫金机",
                    "score": 0.95,
                }
            ]
        },
    )
    monkeypatch.setattr(
        service,
        "classify_subject_aliases",
        lambda search_result_dict: (
            [
                {
                    "subject_id": "subject_hak_180",
                    "standard_subject_name": "HAK 180 烫金机",
                }
            ],
            [],
        ),
    )

    state = {
        "session_id": "session-1",
        "original_query": "HAK180怎么开机？",
        "is_stream": False,
    }

    result = service.confirm_subject_name(state)

    assert result["subject_ids"] == ["subject_hak_180"]
    assert result["standard_subject_names"] == ["HAK 180 烫金机"]
    assert result["rewritten_query"] == "HAK180 怎么开机？"


def test_search_subject_alias_in_milvus_returns_alias_subject_mapping(monkeypatch):
    class FakeLLMProvider:
        def embed_documents(self, documents):
            return {"dense": [[0.1]], "sparse": [{1: 0.2}]}

    class FakeGateway:
        subject_alias_collection = "subject_aliases"

        def __init__(self):
            self.created_expr = None

        def create_requests(self, dense_vector, sparse_vector, limit=5, expr=None):
            self.created_expr = expr
            return ["req"]

        def hybrid_search(self, **kwargs):
            assert kwargs["collection_name"] == "subject_aliases"
            assert kwargs["output_fields"] == ["alias", "alias_type", "subject_id", "standard_subject_name"]
            return [
                [
                    {
                        "distance": 0.91,
                        "entity": {
                            "alias": "HAK180",
                            "alias_type": "file_title",
                            "subject_id": "subject_hak_180",
                            "standard_subject_name": "HAK 180 烫金机",
                        },
                    }
                ]
            ]

    fake_gateway = FakeGateway()
    monkeypatch.setattr(service, "llm_provider", FakeLLMProvider())
    monkeypatch.setattr(service, "milvus_gateway", fake_gateway)

    result = service.search_subject_alias_in_milvus(["HAK180"])

    assert fake_gateway.created_expr is None
    assert result == {
        "HAK180": [
            {
                "alias": "HAK180",
                "alias_type": "file_title",
                "subject_id": "subject_hak_180",
                "standard_subject_name": "HAK 180 烫金机",
                "score": 0.91,
            }
        ]
    }
