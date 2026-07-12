from app.rag.import_ import index_service
from app.rag.query.chunk_retrieval_utils import (
    CHUNK_OUTPUT_FIELDS,
    format_chunk_search_item,
)


def test_prepare_chunks_collection_includes_stage2_subject_fields(monkeypatch):
    class FakeSchema:
        def __init__(self):
            self.field_names = []
            self.field_descriptions = {}

        def add_field(self, *, field_name, **kwargs):
            self.field_names.append(field_name)
            self.field_descriptions[field_name] = kwargs.get("description", "")

    class FakeIndexParams:
        def add_index(self, **kwargs):
            pass

    class FakeClient:
        def __init__(self):
            self.schema = FakeSchema()
            self.created_collection = None
            self.loaded_collection = None

        def has_collection(self, collection_name):
            return False

        def create_schema(self, **kwargs):
            return self.schema

        def prepare_index_params(self):
            return FakeIndexParams()

        def create_collection(self, **kwargs):
            self.created_collection = kwargs["collection_name"]

        def load_collection(self, collection_name):
            self.loaded_collection = collection_name

    class FakeGateway:
        def __init__(self):
            self.client = FakeClient()
            self.chunk_collection_name = "chunks"

    fake_gateway = FakeGateway()
    monkeypatch.setattr(index_service, "milvus_gateway", fake_gateway)

    index_service.prepare_chunks_collection({})

    assert fake_gateway.client.created_collection == "chunks"
    assert fake_gateway.client.loaded_collection == "chunks"
    assert {
        "subject_id",
        "standard_subject_name",
        "equipment_model",
        "alarm_code",
        "part_name",
        "sop_type",
        "safety_level",
        "maintenance_stage",
    }.issubset(set(fake_gateway.client.schema.field_names))
    assert "subject_name" not in fake_gateway.client.schema.field_names
    assert "subject_name" not in CHUNK_OUTPUT_FIELDS
    assert "document_id" in CHUNK_OUTPUT_FIELDS
    assert "owner_user_id" in CHUNK_OUTPUT_FIELDS
    assert "chunk_index" in CHUNK_OUTPUT_FIELDS
    assert "标准主题稳定业务 ID" in fake_gateway.client.schema.field_descriptions["subject_id"]
    assert "设备型号" in fake_gateway.client.schema.field_descriptions["equipment_model"]
    assert "报警码或故障码" in fake_gateway.client.schema.field_descriptions["alarm_code"]
    assert "部件名称" in fake_gateway.client.schema.field_descriptions["part_name"]
    assert "SOP 类型" in fake_gateway.client.schema.field_descriptions["sop_type"]
    assert "安全等级或风险级别" in fake_gateway.client.schema.field_descriptions["safety_level"]
    assert "维护阶段" in fake_gateway.client.schema.field_descriptions["maintenance_stage"]


def test_normalize_chunk_subject_fields_backfills_stage2_fields():
    chunks = [
        {
            "content": "开机前检查急停按钮。",
        }
    ]

    result = index_service.normalize_chunk_subject_fields(chunks)

    assert result[0]["standard_subject_name"] == ""
    assert result[0]["subject_id"] == ""
    assert result[0]["equipment_model"] == ""
    assert result[0]["maintenance_stage"] == ""


def test_format_chunk_search_item_preserves_stage2_fields():
    item = {
        "id": 1,
        "distance": 0.87,
            "entity": {
                "dataset_id": "dataset_default_equipment_ops",
                "document_id": "doc_1",
                "owner_user_id": "user_a",
                "tenant_id": "tenant_default",
                "visibility": "private",
                "index_version": 2,
                "chunk_index": 0,
                "enabled": True,
                "source_title": "HAK180说明书",
                "subject_id": "subject_hak_180",
                "standard_subject_name": "HAK 180 烫金机",
                "equipment_model": "HAK 180",
            "alarm_code": "E101",
            "content": "报警 E101 表示温度异常。",
            "title": "报警说明",
        },
    }

    result = format_chunk_search_item(
        item,
        retrieval_channels=["dense", "learned_sparse", "original"],
        retrieval_rank=1,
    )

    assert result["chunk_id"] == 1
    assert result["dataset_id"] == "dataset_default_equipment_ops"
    assert result["document_id"] == "doc_1"
    assert result["index_version"] == 2
    assert result["chunk_index"] == 0
    assert result["source_title"] == "HAK180说明书"
    assert result["subject_id"] == "subject_hak_180"
    assert result["standard_subject_name"] == "HAK 180 烫金机"
    assert result["equipment_model"] == "HAK 180"
    assert result["alarm_code"] == "E101"
    assert result["source_type"] == "local"
    assert result["retrieval_channels"] == ["dense", "learned_sparse", "original"]
    assert result["retrieval_rank"] == 1
    assert result["retrieval_score"] == 0.87
