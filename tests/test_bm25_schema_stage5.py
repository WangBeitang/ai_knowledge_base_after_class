import pytest

from app.rag.import_ import embedding_service, index_service
from app.rag.import_.lexical_text_service import (
    LEXICAL_ANALYZER_PARAMS,
    LEXICAL_TEXT_MAX_LENGTH,
    build_chunk_lexical_text,
    build_identifier_variants,
    validate_lexical_analyzer,
)


def test_identifier_variants_preserve_same_code_but_do_not_correct_different_code():
    variants = build_identifier_variants("HAK-180 报警 E021，对比 E020 与 SOP 2024-01")

    assert ["HAK180", "HAK-180", "HAK 180"] == variants[:3]
    assert {"E021", "E-021", "E 021"}.issubset(variants)
    assert {"E020", "E-020", "E 020"}.issubset(variants)
    assert variants.index("E021") < variants.index("E020")
    # 两个报警码必须各自保留，不能因为只差一位就合并或相互纠正。
    assert variants.count("E021") == 1
    assert variants.count("E020") == 1


def test_build_chunk_lexical_text_uses_fixed_order_and_appends_variants():
    lexical_text = build_chunk_lexical_text(
        {
            "standard_subject_name": "HAK 180 烫金机",
            "equipment_model": "HAK-180",
            "alarm_code": "E021",
            "part_name": "温度传感器",
            "sop_type": "故障排查",
            "safety_level": "断电后操作",
            "maintenance_stage": "故障定位",
            "source_title": "HAK180 安全手册",
            "title": "E021 报警说明",
            "parent_title": "报警处理",
            "content": "检查温度传感器接线。",
        }
    )

    lines = lexical_text.splitlines()
    assert lines[:11] == [
        "HAK 180 烫金机",
        "HAK-180",
        "E021",
        "温度传感器",
        "故障排查",
        "断电后操作",
        "故障定位",
        "HAK180 安全手册",
        "E021 报警说明",
        "报警处理",
        "检查温度传感器接线。",
    ]
    assert "HAK180" in lines[-1]
    assert "HAK-180" in lines[-1]
    assert "HAK 180" in lines[-1]
    assert "E021" in lines[-1]
    assert "E-021" in lines[-1]
    assert "E 021" in lines[-1]


def test_build_chunk_lexical_text_truncates_copy_but_keeps_identifier_variants():
    original_content = "正文" * LEXICAL_TEXT_MAX_LENGTH + " HAK180 E021"
    chunk = {
        "equipment_model": "HAK180",
        "alarm_code": "E021",
        "content": original_content,
    }

    lexical_text = build_chunk_lexical_text(chunk)

    assert len(lexical_text) <= LEXICAL_TEXT_MAX_LENGTH
    assert "HAK-180" in lexical_text
    assert "E-021" in lexical_text
    assert chunk["content"] == original_content


def test_validate_lexical_analyzer_uses_real_schema_params_and_accepts_required_tokens():
    class FakeClient:
        def __init__(self):
            self.texts = None
            self.analyzer_params = None

        def run_analyzer(self, texts, analyzer_params):
            self.texts = texts
            self.analyzer_params = analyzer_params
            return [["hak180", "报警", "e021", "温度", "传感器", "故障"]]

    client = FakeClient()

    result = validate_lexical_analyzer(client)

    assert result[0][:3] == ["hak180", "报警", "e021"]
    assert client.analyzer_params == LEXICAL_ANALYZER_PARAMS
    assert "HAK180" in client.texts[0]
    assert "E021" in client.texts[0]


@pytest.mark.parametrize(
    "tokens",
    [
        [],
        ["报警", "e021"],
        ["hak180", "e021"],
    ],
)
def test_validate_lexical_analyzer_rejects_missing_identifier_or_chinese_token(tokens):
    class FakeClient:
        def run_analyzer(self, texts, analyzer_params):
            return [tokens]

    with pytest.raises(RuntimeError, match="Analyzer"):
        validate_lexical_analyzer(FakeClient())


def test_generate_embeddings_writes_only_new_learned_sparse_field(monkeypatch):
    monkeypatch.setattr(
        embedding_service.llm_provider,
        "embed_documents",
        lambda texts: {"dense": [[0.1, 0.2]], "sparse": [{3: 0.8}]},
    )
    chunks = [{"standard_subject_name": "HAK 180", "content": "报警 E021"}]

    result = embedding_service.generate_embeddings(chunks)

    assert result[0]["dense_vector"] == [0.1, 0.2]
    assert result[0]["learned_sparse_vector"] == {3: 0.8}
    assert "sparse_vector" not in result[0]
    assert "bm25_sparse_vector" not in result[0]


def test_existing_old_chunk_collection_requires_explicit_rebuild(monkeypatch):
    class FakeClient:
        def has_collection(self, collection_name):
            return True

        def describe_collection(self, collection_name):
            return {
                "fields": [
                    {"name": "dense_vector"},
                    {"name": "sparse_vector"},
                ]
            }

    class FakeGateway:
        client = FakeClient()
        chunk_collection_name = "chunks"

    monkeypatch.setattr(index_service, "milvus_gateway", FakeGateway())

    with pytest.raises(RuntimeError, match="删除该 collection 后重新导入"):
        index_service.prepare_chunks_collection({})


def test_existing_collection_with_fields_but_without_bm25_function_requires_rebuild(monkeypatch):
    class FakeClient:
        def has_collection(self, collection_name):
            return True

        def describe_collection(self, collection_name):
            return {
                "fields": [
                    {"name": name} for name in index_service.CHUNK_SCHEMA_REQUIRED_FIELDS
                ],
                "functions": [],
            }

    class FakeGateway:
        client = FakeClient()
        chunk_collection_name = "chunks"

    monkeypatch.setattr(index_service, "milvus_gateway", FakeGateway())

    with pytest.raises(RuntimeError, match="Function：chunk_lexical_text_bm25"):
        index_service.prepare_chunks_collection({})


@pytest.mark.parametrize("forbidden_field", ["sparse_vector", "bm25_sparse_vector"])
def test_insert_chunks_rejects_old_or_function_output_vector(monkeypatch, forbidden_field):
    class FakeClient:
        def insert(self, **kwargs):
            raise AssertionError("非法字段必须在调用 Milvus 前被拒绝")

    class FakeGateway:
        client = FakeClient()
        chunk_collection_name = "chunks"

    monkeypatch.setattr(index_service, "milvus_gateway", FakeGateway())
    chunk = {"content": "测试", forbidden_field: {1: 0.5}}

    with pytest.raises(ValueError, match=forbidden_field):
        index_service.insert_chunks([chunk])
