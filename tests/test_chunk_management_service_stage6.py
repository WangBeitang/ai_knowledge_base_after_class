import pytest

from app.infra.persistence.chunk_status_repository import (
    MANUAL_STATUS_DISABLED,
    MANUAL_STATUS_ENABLED,
    MANUAL_STATUS_NONE,
)
from app.rag.import_.chunk_management_service import (
    ChunkManagementService,
    ChunkPermissionError,
    ChunkStateError,
    ChunkVersionConflictError,
)


def document(**overrides):
    data = {
        "document_id": "doc_1",
        "dataset_id": "dataset_ops",
        "owner_user_id": "user_a",
        "tenant_id": "tenant_default",
        "visibility": "private",
        "index_version": 2,
        "chunk_count": 3,
        "status": "completed",
    }
    data.update(overrides)
    return data


def chunk(**overrides):
    data = {
        "chunk_id": 1001,
        "document_id": "doc_1",
        "dataset_id": "dataset_ops",
        "owner_user_id": "user_a",
        "tenant_id": "tenant_default",
        "visibility": "private",
        "index_version": 2,
        "chunk_index": 0,
        "enabled": True,
        "source_title": "HAK180说明书",
        "content": "报警 E021 表示主电机过载，需要检查负载和驱动器。",
        "title": "E021 报警处理",
        "parent_title": "报警排查",
        "subject_id": "subject_hak_180",
        "standard_subject_name": "HAK 180 烫金机",
        "equipment_model": "HAK 180",
        "alarm_code": "E021",
        "part_name": "",
        "sop_type": "",
        "safety_level": "",
        "maintenance_stage": "",
    }
    data.update(overrides)
    return data


def event(**overrides):
    data = {
        "event_id": "event_1",
        "document_id": "doc_1",
        "chunk_id": 1002,
        "dataset_id": "dataset_ops",
        "owner_user_id": "user_a",
        "tenant_id": "tenant_default",
        "visibility": "private",
        "index_version": 2,
        "chunk_index": 1,
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
    data.update(overrides)
    return data


class FakeMetadataRepository:
    def __init__(self, documents):
        self.documents = documents

    def get_visible_document(self, *, document_id, owner_user_id, tenant_id):
        return self.documents.get(document_id, {})


class FakeStatusRepository:
    def __init__(self, *, overrides=None, events=None):
        self.overrides = list(overrides or [])
        self.events = list(events or [])
        self.recorded_events = []
        self.set_overrides = []

    def get_overrides(self, *, document_id, chunk_ids=None, index_version=None):
        chunk_id_set = set(chunk_ids) if chunk_ids is not None else None
        result = []
        for override in self.overrides:
            if override.get("document_id") != document_id:
                continue
            if index_version is not None and override.get("index_version") != index_version:
                continue
            if chunk_id_set is not None and override.get("chunk_id") not in chunk_id_set:
                continue
            result.append(dict(override))
        return result

    def list_events(self, *, document_id, chunk_id, index_version, limit=20):
        result = [
            dict(item)
            for item in self.events
            if item.get("document_id") == document_id
            and item.get("chunk_id") == chunk_id
            and item.get("index_version") == index_version
        ]
        result.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return result[:limit]

    def record_event(self, event):
        stored = dict(event)
        self.events.append(stored)
        self.recorded_events.append(stored)
        return stored

    def set_override(self, override):
        stored = dict(override)
        self.set_overrides.append(stored)
        self.overrides = [
            item
            for item in self.overrides
            if not (
                item.get("document_id") == stored.get("document_id")
                and item.get("chunk_id") == stored.get("chunk_id")
                and item.get("index_version") == stored.get("index_version")
            )
        ]
        self.overrides.append(stored)
        return stored


class FakeVectorGateway:
    chunk_collection_name = "chunks"

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.calls = []

    def query_entities(self, *, collection_name, filter_expr, output_fields, limit):
        self.calls.append({
            "collection_name": collection_name,
            "filter_expr": filter_expr,
            "output_fields": list(output_fields),
            "limit": limit,
        })
        result = []
        for item in self.chunks:
            if f'document_id == "{item["document_id"]}"' not in filter_expr:
                continue
            if f"index_version == {item['index_version']}" not in filter_expr:
                continue
            if "chunk_id ==" in filter_expr and f"chunk_id == {item['chunk_id']}" not in filter_expr:
                continue
            if "enabled == true" in filter_expr and not item.get("enabled"):
                continue
            if "enabled == false" in filter_expr and item.get("enabled"):
                continue
            result.append(dict(item))
        return result[:limit]


def build_service(*, metadata=None, status=None, vector=None):
    return ChunkManagementService(
        metadata_repository=metadata or FakeMetadataRepository({"doc_1": document()}),
        status_repository=status or FakeStatusRepository(),
        vector_gateway=vector or FakeVectorGateway([chunk()]),
    )


def test_list_document_chunks_merges_manual_override_and_latest_event():
    status = FakeStatusRepository(
        overrides=[{
            "document_id": "doc_1",
            "chunk_id": 1002,
            "index_version": 2,
            "manual_status": MANUAL_STATUS_DISABLED,
        }],
        events=[event()],
    )
    vector = FakeVectorGateway([
        chunk(chunk_id=1002, chunk_index=1, content="第 23 页 / 共 120 页"),
        chunk(chunk_id=1001, chunk_index=0, content="有效正文"),
        chunk(chunk_id=1003, index_version=1, chunk_index=0, content="旧版本正文"),
    ])
    service = build_service(status=status, vector=vector)

    result = service.list_document_chunks(
        document_id="doc_1",
        user_id="user_a",
        enabled=None,
        limit=10,
    )

    assert [item["chunk_id"] for item in result["items"]] == [1001, 1002]
    assert "content" not in result["items"][0]
    assert result["items"][0]["manual_status"] == MANUAL_STATUS_NONE
    assert result["items"][0]["effective_enabled"] is True
    assert result["items"][1]["manual_status"] == MANUAL_STATUS_DISABLED
    assert result["items"][1]["effective_enabled"] is False
    assert result["items"][1]["latest_event"]["reason_type"] == "header_footer"
    assert "enabled ==" not in vector.calls[0]["filter_expr"]
    assert 'document_id == "doc_1"' in vector.calls[0]["filter_expr"]
    assert "index_version == 2" in vector.calls[0]["filter_expr"]


def test_list_document_chunks_filters_by_effective_enabled_after_route_b_override():
    status = FakeStatusRepository(overrides=[{
        "document_id": "doc_1",
        "chunk_id": 1002,
        "index_version": 2,
        "manual_status": MANUAL_STATUS_DISABLED,
    }])
    vector = FakeVectorGateway([
        chunk(chunk_id=1001, chunk_index=0, enabled=True),
        chunk(chunk_id=1002, chunk_index=1, enabled=True),
        chunk(chunk_id=1003, chunk_index=2, enabled=False),
    ])
    service = build_service(status=status, vector=vector)

    result = service.list_document_chunks(
        document_id="doc_1",
        user_id="user_a",
        enabled=False,
        limit=10,
    )

    assert [item["chunk_id"] for item in result["items"]] == [1002, 1003]
    assert result["items"][0]["manual_status"] == MANUAL_STATUS_DISABLED
    assert result["items"][1]["manual_status"] == MANUAL_STATUS_NONE
    assert result["items"][1]["enabled"] is False
    assert "enabled == false" not in vector.calls[0]["filter_expr"]


def test_get_chunk_detail_returns_content_and_current_override_state():
    status = FakeStatusRepository(overrides=[{
        "document_id": "doc_1",
        "chunk_id": 1001,
        "index_version": 2,
        "manual_status": MANUAL_STATUS_ENABLED,
    }])
    service = build_service(status=status, vector=FakeVectorGateway([chunk(content="完整正文")]))

    detail = service.get_chunk_detail(
        document_id="doc_1",
        chunk_id="1001",
        user_id="user_a",
    )

    assert detail["chunk_id"] == 1001
    assert detail["content"] == "完整正文"
    assert detail["manual_status"] == MANUAL_STATUS_ENABLED
    assert detail["effective_enabled"] is True


def test_disable_chunk_records_event_and_updates_override():
    status = FakeStatusRepository()
    service = build_service(status=status, vector=FakeVectorGateway([chunk()]))

    response = service.change_chunk_enabled(
        document_id="doc_1",
        chunk_id="1001",
        user_id="user_a",
        enabled=False,
        expected_index_version=2,
        reason_type="garbled_text",
        reason_detail="OCR 乱码",
        trace_id="trace_1",
    )

    assert response["changed"] is True
    assert response["manual_status"] == MANUAL_STATUS_DISABLED
    assert response["effective_enabled"] is False
    assert len(status.recorded_events) == 1
    assert status.recorded_events[0]["operation"] == "disable"
    assert status.recorded_events[0]["previous_enabled"] is True
    assert status.recorded_events[0]["enabled"] is False
    assert status.recorded_events[0]["trace_id"] == "trace_1"
    assert status.set_overrides[0]["manual_status"] == MANUAL_STATUS_DISABLED
    assert status.set_overrides[0]["latest_event_id"] == response["latest_event"]["event_id"]


def test_same_target_state_is_noop_and_does_not_write_event():
    status = FakeStatusRepository(
        overrides=[{
            "document_id": "doc_1",
            "chunk_id": 1001,
            "index_version": 2,
            "manual_status": MANUAL_STATUS_DISABLED,
        }],
        events=[event(chunk_id=1001)],
    )
    service = build_service(status=status, vector=FakeVectorGateway([chunk()]))

    response = service.change_chunk_enabled(
        document_id="doc_1",
        chunk_id=1001,
        user_id="user_a",
        enabled=False,
        expected_index_version=2,
        reason_type="header_footer",
    )

    assert response["changed"] is False
    assert response["latest_event"]["event_id"] == "event_1"
    assert status.recorded_events == []
    assert status.set_overrides == []


def test_change_rejects_stale_expected_index_version():
    service = build_service(vector=FakeVectorGateway([chunk()]))

    with pytest.raises(ChunkVersionConflictError, match="expected_index_version=1"):
        service.change_chunk_enabled(
            document_id="doc_1",
            chunk_id=1001,
            user_id="user_a",
            enabled=False,
            expected_index_version=1,
            reason_type="header_footer",
        )


def test_public_document_can_be_read_but_non_owner_cannot_change_state():
    metadata = FakeMetadataRepository({
        "doc_1": document(owner_user_id="user_b", visibility="public"),
    })
    vector = FakeVectorGateway([chunk(owner_user_id="user_b", visibility="public")])
    service = build_service(metadata=metadata, vector=vector)

    result = service.list_document_chunks(document_id="doc_1", user_id="user_a")

    assert result["items"][0]["chunk_id"] == 1001
    with pytest.raises(ChunkPermissionError, match="只能查看该 chunk"):
        service.change_chunk_enabled(
            document_id="doc_1",
            chunk_id=1001,
            user_id="user_a",
            enabled=False,
            expected_index_version=2,
            reason_type="header_footer",
        )


def test_route_b_cannot_enable_milvus_disabled_chunk():
    status = FakeStatusRepository(overrides=[{
        "document_id": "doc_1",
        "chunk_id": 1001,
        "index_version": 2,
        "manual_status": MANUAL_STATUS_DISABLED,
    }])
    service = build_service(status=status, vector=FakeVectorGateway([chunk(enabled=False)]))

    with pytest.raises(ChunkStateError, match="Milvus enabled=false"):
        service.change_chunk_enabled(
            document_id="doc_1",
            chunk_id=1001,
            user_id="user_a",
            enabled=True,
            expected_index_version=2,
            reason_type="manual_restore",
        )


def test_list_chunk_events_checks_current_chunk_before_returning_history():
    status = FakeStatusRepository(events=[event(chunk_id=1001)])
    service = build_service(status=status, vector=FakeVectorGateway([chunk()]))

    result = service.list_chunk_events(
        document_id="doc_1",
        chunk_id="1001",
        user_id="user_a",
        limit=5,
    )

    assert result["chunk_id"] == 1001
    assert result["index_version"] == 2
    assert [item["event_id"] for item in result["items"]] == ["event_1"]
