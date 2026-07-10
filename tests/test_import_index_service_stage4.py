from app.rag.import_ import index_service


class FakeSchema:
    def __init__(self):
        self.field_names = []
        self.field_types = {}
        self.field_descriptions = {}

    def add_field(self, *, field_name, datatype, **kwargs):
        self.field_names.append(field_name)
        self.field_types[field_name] = datatype
        self.field_descriptions[field_name] = kwargs.get("description", "")


class FakeIndexParams:
    def add_index(self, **kwargs):
        pass


def test_prepare_chunks_collection_includes_stage4_document_fields(monkeypatch):
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

    field_names = set(fake_gateway.client.schema.field_names)
    assert {
        "dataset_id",
        "document_id",
        "owner_user_id",
        "tenant_id",
        "visibility",
        "index_version",
        "chunk_index",
        "enabled",
        "source_title",
    }.issubset(field_names)
    assert fake_gateway.client.schema.field_types["index_version"] == index_service.DataType.INT64
    assert fake_gateway.client.schema.field_types["chunk_index"] == index_service.DataType.INT64
    assert fake_gateway.client.schema.field_types["enabled"] == index_service.DataType.BOOL
    assert "不再作为导入幂等删除条件" in fake_gateway.client.schema.field_descriptions["file_title"]


def test_normalize_chunk_document_fields_backfills_stage4_metadata():
    chunks = [
        {"content": "开机前检查急停按钮。"},
        {"content": "启动设备。", "file_title": "chunk旧标题"},
    ]
    state = {
        "dataset_id": "dataset_default_equipment_ops",
        "document_id": "doc_1",
        "owner_user_id": "user_a",
        "tenant_id": "",
        "visibility": "",
        "index_version": 3,
    }

    result = index_service.normalize_chunk_document_fields(chunks, state, "HAK180说明书")

    assert result[0]["dataset_id"] == "dataset_default_equipment_ops"
    assert result[0]["document_id"] == "doc_1"
    assert result[0]["owner_user_id"] == "user_a"
    assert result[0]["tenant_id"] == "tenant_default"
    assert result[0]["visibility"] == "private"
    assert result[0]["index_version"] == 3
    assert result[0]["chunk_index"] == 0
    assert result[1]["chunk_index"] == 1
    assert result[0]["enabled"] is True
    assert result[0]["source_title"] == "HAK180说明书"
    assert result[0]["file_title"] == "HAK180说明书"
    assert result[1]["file_title"] == "chunk旧标题"


def test_index_chunks_deletes_by_document_id_and_inserts_stage4_metadata(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.deleted_filter = ""
            self.inserted_data = []

        def has_collection(self, collection_name):
            return True

        def delete(self, *, collection_name, filter):
            self.deleted_filter = filter

        def insert(self, *, collection_name, data):
            self.inserted_data = data
            return {"insert_count": len(data), "ids": [101, 102]}

    class FakeGateway:
        def __init__(self):
            self.client = FakeClient()
            self.chunk_collection_name = "chunks"

    fake_gateway = FakeGateway()
    monkeypatch.setattr(index_service, "milvus_gateway", fake_gateway)

    state = {
        "dataset_id": "dataset_default_equipment_ops",
        "document_id": "doc_a",
        "owner_user_id": "user_a",
        "tenant_id": "tenant_default",
        "visibility": "private",
        "index_version": 2,
        "file_title": "同名说明书",
        "chunks": [
            {"content": "第一段", "title": "标题1", "dense_vector": [0.1], "sparse_vector": {1: 0.5}},
            {"content": "第二段", "title": "标题2", "dense_vector": [0.2], "sparse_vector": {2: 0.4}},
        ],
    }

    result = index_service.index_chunks(state)

    assert fake_gateway.client.deleted_filter == "document_id=='doc_a'"
    assert "file_title" not in fake_gateway.client.deleted_filter
    assert fake_gateway.client.inserted_data[0]["document_id"] == "doc_a"
    assert fake_gateway.client.inserted_data[0]["owner_user_id"] == "user_a"
    assert fake_gateway.client.inserted_data[0]["chunk_index"] == 0
    assert fake_gateway.client.inserted_data[1]["chunk_index"] == 1
    assert fake_gateway.client.inserted_data[0]["enabled"] is True
    assert fake_gateway.client.inserted_data[0]["source_title"] == "同名说明书"
    assert result["chunks"][0]["chunk_id"] == 101
    assert result["chunks"][1]["chunk_id"] == 102
