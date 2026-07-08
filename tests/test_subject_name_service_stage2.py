from app.rag.import_ import subject_name_service as service


class FakeSchema:
    def __init__(self, **kwargs):
        self.description = kwargs.get("description", "")
        self.field_names = []
        self.field_descriptions = {}

    def add_field(self, *, field_name, **kwargs):
        self.field_names.append(field_name)
        self.field_descriptions[field_name] = kwargs.get("description", "")


class FakeIndexParams:
    def add_index(self, **kwargs):
        pass


class FakeCollectionClient:
    def __init__(self):
        self.schemas = []
        self.created_collections = []

    def has_collection(self, collection_name):
        return False

    def create_schema(self, **kwargs):
        schema = FakeSchema(**kwargs)
        self.schemas.append(schema)
        return schema

    def prepare_index_params(self):
        return FakeIndexParams()

    def create_collection(self, **kwargs):
        self.created_collections.append(kwargs["collection_name"])


def test_build_subject_id_is_stable_and_aliases_are_deduped():
    assert service.build_subject_id(" HAK   180 烫金机 ") == service.build_subject_id("hak 180 烫金机")

    aliases = service.build_subject_aliases(
        standard_subject_name="HAK 180 烫金机",
        file_title="HAK180 操作手册",
        llm_subject_name=" HAK 180 烫金机 ",
    )

    assert aliases == ["HAK 180 烫金机", "HAK 180", "HAK180", "HAK-180", "HAK180 操作手册"]


def test_build_model_code_aliases_extracts_common_model_spellings():
    assert service.build_model_code_aliases("HAK 180 烫金机操作手册") == ["HAK 180", "HAK180", "HAK-180"]


def test_prepare_standard_subject_collection_adds_field_descriptions(monkeypatch):
    class FakeGateway:
        def __init__(self):
            self.client = FakeCollectionClient()
            self.standard_subject_collection = "standard_subjects"

    fake_gateway = FakeGateway()
    monkeypatch.setattr(service, "milvus_gateway", fake_gateway)

    service.prepare_standard_subject_collection({})

    schema = fake_gateway.client.schemas[0]
    assert fake_gateway.client.created_collections == ["standard_subjects"]
    assert "标准主题主表" in schema.description
    assert "标准主题稳定业务 ID" in schema.field_descriptions["subject_id"]
    assert "标准主题名称" in schema.field_descriptions["standard_subject_name"]
    assert "设备型号" in schema.field_descriptions["equipment_model"]
    assert "报警码或故障码" in schema.field_descriptions["alarm_code"]
    assert "部件名称" in schema.field_descriptions["part_name"]
    assert "SOP 类型" in schema.field_descriptions["sop_type"]
    assert "安全等级或风险级别" in schema.field_descriptions["safety_level"]
    assert "维护阶段" in schema.field_descriptions["maintenance_stage"]


def test_prepare_subject_alias_collection_adds_field_descriptions(monkeypatch):
    class FakeGateway:
        def __init__(self):
            self.client = FakeCollectionClient()
            self.subject_alias_collection = "subject_aliases"

    fake_gateway = FakeGateway()
    monkeypatch.setattr(service, "milvus_gateway", fake_gateway)

    service.prepare_subject_alias_collection({})

    schema = fake_gateway.client.schemas[0]
    assert fake_gateway.client.created_collections == ["subject_aliases"]
    assert "标准主题别名索引" in schema.description
    assert "主体别名文本" in schema.field_descriptions["alias"]
    assert "别名来源类型" in schema.field_descriptions["alias_type"]
    assert "标准主题 ID" in schema.field_descriptions["subject_id"]


def test_recognize_and_index_subject_name_backfills_standard_subject(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        service,
        "recognize_standard_subject_name",
        lambda context, file_title: ("HAK 180 烫金机", "HAK180"),
    )
    monkeypatch.setattr(service, "generate_embeddings", lambda text: ([0.1], {1: 0.2}))
    monkeypatch.setattr(service, "prepare_standard_subject_collection", lambda state: None)
    monkeypatch.setattr(service, "prepare_subject_alias_collection", lambda state: None)
    monkeypatch.setattr(service, "insert_standard_subject", lambda **kwargs: captured.update(standard=kwargs))
    monkeypatch.setattr(
        service,
        "insert_subject_aliases",
        lambda *args: captured.update(alias_args=args),
    )

    state = {
        "file_title": "HAK180 操作手册",
        "equipment_model": "HAK 180",
        "chunks": [
            {
                "title": "开机流程",
                "content": "HAK 180 烫金机开机前需要检查急停按钮。",
            }
        ],
    }

    result = service.recognize_and_index_subject_name(state)

    assert result["standard_subject_name"] == "HAK 180 烫金机"
    assert result["subject_id"] == service.build_subject_id("HAK 180 烫金机")
    assert result["subject_aliases"] == [
        "HAK 180 烫金机",
        "HAK 180",
        "HAK180",
        "HAK-180",
        "HAK180 操作手册",
    ]

    chunk = result["chunks"][0]
    assert chunk["subject_id"] == result["subject_id"]
    assert chunk["standard_subject_name"] == "HAK 180 烫金机"
    assert chunk["equipment_model"] == "HAK 180"

    assert captured["standard"]["subject_id"] == result["subject_id"]
    assert captured["standard"]["standard_subject_name"] == "HAK 180 烫金机"
    assert captured["standard"]["subject_aliases"] == result["subject_aliases"]
    assert captured["alias_args"] == (
        result["subject_id"],
        "HAK 180 烫金机",
        result["subject_aliases"],
        "HAK180 操作手册",
        "HAK180",
    )


def test_insert_subject_aliases_writes_one_row_per_alias(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.delete_calls = []
            self.insert_calls = []

        def delete(self, **kwargs):
            self.delete_calls.append(kwargs)

        def insert(self, **kwargs):
            self.insert_calls.append(kwargs)

    class FakeGateway:
        def __init__(self):
            self.client = FakeClient()
            self.subject_alias_collection = "subject_aliases"

    fake_gateway = FakeGateway()
    monkeypatch.setattr(service, "milvus_gateway", fake_gateway)
    monkeypatch.setattr(
        service,
        "generate_batch_embeddings",
        lambda aliases: [
            {"dense_vector": [index], "sparse_vector": {index: 1.0}}
            for index, _ in enumerate(aliases)
        ],
    )

    service.insert_subject_aliases(
        subject_id="subject_1",
        standard_subject_name="HAK 180 烫金机",
        subject_aliases=["HAK 180 烫金机", "HAK180 操作手册", "HAK180"],
        file_title="HAK180 操作手册",
        llm_subject_name="HAK180",
    )

    assert fake_gateway.client.delete_calls == [
        {
            "collection_name": "subject_aliases",
            "filter": 'subject_id=="subject_1" and file_title=="HAK180 操作手册"',
        }
    ]
    insert_call = fake_gateway.client.insert_calls[0]
    assert insert_call["collection_name"] == "subject_aliases"
    assert [row["alias"] for row in insert_call["data"]] == [
        "HAK 180 烫金机",
        "HAK180 操作手册",
        "HAK180",
    ]
    assert [row["alias_type"] for row in insert_call["data"]] == ["standard", "file_title", "llm"]
