import pytest

from app.infra.persistence import chunk_status_repository as repo_module


class FakeInsertResult:
    inserted_id = "fake-id"


class FakeUpdateResult:
    def __init__(self, matched_count=1, modified_count=1):
        self.matched_count = matched_count
        self.modified_count = modified_count


class FakeCursor:
    def __init__(self, items):
        self.items = [dict(item) for item in items]

    def sort(self, field_or_fields, direction=None):
        fields = field_or_fields
        if isinstance(field_or_fields, str):
            fields = [(field_or_fields, direction)]
        for field_name, sort_direction in reversed(fields):
            reverse = sort_direction < 0
            self.items.sort(
                key=lambda item: (item.get(field_name) is None, str(item.get(field_name, ""))),
                reverse=reverse,
            )
        return self

    def limit(self, limit):
        self.items = self.items[:limit]
        return self

    def __iter__(self):
        return iter(self.items)


class FakeCollection:
    def __init__(self):
        self.items = []
        self.indexes = []

    def create_index(self, index_fields, unique=False):
        self.indexes.append((tuple(index_fields), unique))

    @classmethod
    def _matches(cls, document, query):
        for field_name, value in query.items():
            if field_name == "$or":
                if not any(cls._matches(document, branch) for branch in value):
                    return False
                continue

            document_value = document.get(field_name)
            if isinstance(value, dict):
                if "$in" in value and document_value not in value["$in"]:
                    return False
            elif document_value != value:
                return False
        return True

    @staticmethod
    def _insert_base_document(query):
        document = {}
        for field_name, value in query.items():
            if field_name.startswith("$") or isinstance(value, dict):
                continue
            document[field_name] = value
        return document

    def insert_one(self, document):
        self.items.append(dict(document))
        return FakeInsertResult()

    def update_one(self, query, update, upsert=False):
        for document in self.items:
            if self._matches(document, query):
                if "$set" in update:
                    document.update(update["$set"])
                return FakeUpdateResult()

        if not upsert:
            return FakeUpdateResult(matched_count=0, modified_count=0)
        document = self._insert_base_document(query)
        if "$setOnInsert" in update:
            document.update(update["$setOnInsert"])
        if "$set" in update:
            document.update(update["$set"])
        self.items.append(document)
        return FakeUpdateResult()

    def find_one(self, query):
        for document in self.items:
            if self._matches(document, query):
                return dict(document)
        return None

    def find(self, query):
        return FakeCursor(document for document in self.items if self._matches(document, query))


class FakeDatabase:
    def __init__(self):
        self.collections = {}

    def __getitem__(self, collection_name):
        self.collections.setdefault(collection_name, FakeCollection())
        return self.collections[collection_name]


class FakeMongoClient:
    def __init__(self):
        self.databases = {}

    def __getitem__(self, db_name):
        self.databases.setdefault(db_name, FakeDatabase())
        return self.databases[db_name]


def build_repository():
    return repo_module.ChunkStatusRepository(
        client=FakeMongoClient(),
        mongo_url="mongodb://fake",
        db_name="test_db",
    )


def status_event(**overrides):
    event = {
        "event_id": "event_1",
        "document_id": "doc_1",
        "chunk_id": 101,
        "dataset_id": "dataset_ops",
        "owner_user_id": "user_a",
        "tenant_id": "tenant_default",
        "visibility": "private",
        "index_version": 3,
        "chunk_index": 5,
        "operator_user_id": "user_a",
        "operation": "disable",
        "previous_enabled": True,
        "enabled": False,
        "reason_type": "header_footer",
        "reason_detail": "重复页脚",
        "source": "manual",
        "human_confirmed": True,
        "created_at": "2026-07-15T08:00:00+00:00",
    }
    event.update(overrides)
    return event


def status_override(**overrides):
    override = {
        "document_id": "doc_1",
        "chunk_id": 101,
        "index_version": 3,
        "dataset_id": "dataset_ops",
        "owner_user_id": "user_a",
        "tenant_id": "tenant_default",
        "visibility": "private",
        "manual_status": repo_module.MANUAL_STATUS_DISABLED,
        "latest_event_id": "event_1",
    }
    override.update(overrides)
    return override


def test_module_import_does_not_initialize_repository_singleton():
    assert repo_module._chunk_status_repository is None


def test_ensure_indexes_creates_event_and_override_indexes():
    repository = build_repository()

    event_indexes = repository.events.indexes
    override_indexes = repository.overrides.indexes

    assert ((("event_id", repo_module.ASCENDING),), True) in event_indexes
    assert (
        (
            ("document_id", repo_module.ASCENDING),
            ("chunk_id", repo_module.ASCENDING),
            ("index_version", repo_module.ASCENDING),
            ("created_at", repo_module.DESCENDING),
        ),
        False,
    ) in event_indexes
    assert (
        (
            ("document_id", repo_module.ASCENDING),
            ("chunk_id", repo_module.ASCENDING),
            ("index_version", repo_module.ASCENDING),
        ),
        True,
    ) in override_indexes
    assert ((("dataset_id", repo_module.ASCENDING), ("manual_status", repo_module.ASCENDING)), False) in override_indexes


def test_record_event_inserts_audit_event_without_mongo_id():
    repository = build_repository()

    event = repository.record_event({**status_event(), "_id": "mongo_internal"})

    assert event["event_id"] == "event_1"
    assert event["document_id"] == "doc_1"
    assert event["chunk_id"] == 101
    assert "_id" not in event
    assert repository.events.items[0]["reason_type"] == "header_footer"


def test_list_events_filters_identity_and_returns_newest_first():
    repository = build_repository()
    repository.record_event(status_event(event_id="event_old", created_at="2026-07-15T08:00:00+00:00"))
    repository.record_event(status_event(event_id="event_new", created_at="2026-07-15T09:00:00+00:00"))
    repository.record_event(status_event(event_id="event_other_chunk", chunk_id=202, created_at="2026-07-15T10:00:00+00:00"))

    events = repository.list_events(
        document_id="doc_1",
        chunk_id=101,
        index_version=3,
        limit=10,
    )

    assert [event["event_id"] for event in events] == ["event_new", "event_old"]


def test_set_override_upserts_current_manual_status_and_adds_updated_at():
    repository = build_repository()

    first = repository.set_override(status_override())
    second = repository.set_override(status_override(manual_status=repo_module.MANUAL_STATUS_ENABLED, latest_event_id="event_2"))

    assert len(repository.overrides.items) == 1
    assert first["manual_status"] == repo_module.MANUAL_STATUS_DISABLED
    assert second["manual_status"] == repo_module.MANUAL_STATUS_ENABLED
    assert second["latest_event_id"] == "event_2"
    assert second["updated_at"]


def test_set_override_accepts_early_enabled_field_but_stores_manual_status():
    repository = build_repository()

    override = repository.set_override(status_override(enabled=False, manual_status=None))

    assert override["manual_status"] == repo_module.MANUAL_STATUS_DISABLED


def test_get_overrides_filters_by_document_chunks_and_index_version():
    repository = build_repository()
    repository.set_override(status_override(chunk_id=101, latest_event_id="event_101"))
    repository.set_override(status_override(chunk_id=202, latest_event_id="event_202"))
    repository.set_override(status_override(document_id="doc_2", chunk_id=303, latest_event_id="event_303"))

    overrides = repository.get_overrides(
        document_id="doc_1",
        chunk_ids=[202, 404],
        index_version=3,
    )

    assert [override["chunk_id"] for override in overrides] == [202]
    assert overrides[0]["latest_event_id"] == "event_202"


def test_list_disabled_chunk_ids_keeps_scope_and_visibility_boundaries():
    repository = build_repository()
    repository.set_override(status_override(chunk_id=101, visibility="private", owner_user_id="user_a"))
    repository.set_override(status_override(chunk_id=202, visibility="private", owner_user_id="user_b"))
    repository.set_override(status_override(chunk_id=303, visibility="shared", tenant_id="tenant_default", owner_user_id="user_b"))
    repository.set_override(status_override(chunk_id=404, visibility="shared", tenant_id="tenant_other", owner_user_id="user_b"))
    repository.set_override(status_override(chunk_id=505, visibility="public", owner_user_id="user_b"))
    repository.set_override(status_override(chunk_id=606, dataset_id="dataset_other", owner_user_id="user_a"))
    repository.set_override(status_override(chunk_id=707, manual_status=repo_module.MANUAL_STATUS_ENABLED, owner_user_id="user_a"))

    disabled_chunk_ids = repository.list_disabled_chunk_ids(
        dataset_ids=["dataset_ops"],
        owner_user_id="user_a",
        tenant_id="tenant_default",
        document_id="doc_1",
        index_version=3,
    )

    assert disabled_chunk_ids == [101, 303, 505]


@pytest.mark.parametrize(
    ("method_name", "kwargs", "message"),
    [
        (
            "list_events",
            {"document_id": " ", "chunk_id": 101, "index_version": 3},
            "document_id 不能为空",
        ),
        (
            "list_events",
            {"document_id": "doc_1", "chunk_id": "", "index_version": 3},
            "chunk_id 不能为空",
        ),
        (
            "list_events",
            {"document_id": "doc_1", "chunk_id": 101, "index_version": -1},
            "index_version 必须是大于等于 0 的整数",
        ),
        (
            "list_disabled_chunk_ids",
            {"dataset_ids": [], "owner_user_id": "user_a", "tenant_id": "tenant_default"},
            "dataset_ids 不能为空，禁止退化为全库查询",
        ),
        (
            "list_disabled_chunk_ids",
            {"dataset_ids": "dataset_ops", "owner_user_id": "user_a", "tenant_id": "tenant_default"},
            "dataset_ids 必须是字符串列表",
        ),
    ],
)
def test_repository_rejects_invalid_scope(method_name, kwargs, message):
    repository = build_repository()

    with pytest.raises(ValueError, match=message):
        getattr(repository, method_name)(**kwargs)
