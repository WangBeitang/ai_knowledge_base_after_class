import pytest

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


def test_index_chunks_replaces_only_current_document(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.rows = [
                {"chunk_id": 1, "document_id": "doc_a", "file_title": "同名说明书", "content": "旧内容1"},
                {"chunk_id": 2, "document_id": "doc_a", "file_title": "同名说明书", "content": "旧内容2"},
                {"chunk_id": 3, "document_id": "doc_b", "file_title": "同名说明书", "content": "其他文档内容"},
            ]
            self.next_id = 100

        def has_collection(self, collection_name):
            return True

        def delete(self, *, collection_name, filter):
            document_id = filter.split("==", 1)[1].strip("'\"")
            self.rows = [row for row in self.rows if row.get("document_id") != document_id]

        def insert(self, *, collection_name, data):
            ids = []
            for chunk in data:
                self.next_id += 1
                ids.append(self.next_id)
                self.rows.append({**chunk, "chunk_id": self.next_id})
            return {"insert_count": len(data), "ids": ids}

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
        "index_version": 2,
        "file_title": "同名说明书",
        "chunks": [
            {"content": "新内容1", "dense_vector": [0.1], "sparse_vector": {1: 0.5}},
            {"content": "新内容2", "dense_vector": [0.2], "sparse_vector": {2: 0.4}},
        ],
    }

    index_service.index_chunks(state)

    document_a_rows = [row for row in fake_gateway.client.rows if row["document_id"] == "doc_a"]
    document_b_rows = [row for row in fake_gateway.client.rows if row["document_id"] == "doc_b"]
    assert [row["content"] for row in document_a_rows] == ["新内容1", "新内容2"]
    assert [row["chunk_index"] for row in document_a_rows] == [0, 1]
    assert [row["content"] for row in document_b_rows] == ["其他文档内容"]


def test_index_chunks_does_not_restore_old_chunks_when_insert_fails(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.rows = [
                {"document_id": "doc_a", "content": "已过期内容"},
                {"document_id": "doc_b", "content": "其他文档内容"},
            ]

        def has_collection(self, collection_name):
            return True

        def delete(self, *, collection_name, filter):
            document_id = filter.split("==", 1)[1].strip("'\"")
            self.rows = [row for row in self.rows if row.get("document_id") != document_id]

        def insert(self, *, collection_name, data):
            raise RuntimeError("Milvus insert failed")

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
        "index_version": 2,
        "file_title": "HAK180说明书",
        "chunks": [
            {"content": "新内容", "dense_vector": [0.1], "sparse_vector": {1: 0.5}},
        ],
    }

    with pytest.raises(RuntimeError, match="Milvus insert failed"):
        index_service.index_chunks(state)

    assert [row["document_id"] for row in fake_gateway.client.rows] == ["doc_b"]


def test_remove_old_chunks_skips_missing_collection(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.delete_called = False

        def has_collection(self, collection_name):
            return False

        def delete(self, **kwargs):
            self.delete_called = True

    class FakeGateway:
        def __init__(self):
            self.client = FakeClient()
            self.chunk_collection_name = "chunks"

    fake_gateway = FakeGateway()
    monkeypatch.setattr(index_service, "milvus_gateway", fake_gateway)

    index_service.remove_old_chunks("doc_failed_before_index")

    assert fake_gateway.client.delete_called is False
