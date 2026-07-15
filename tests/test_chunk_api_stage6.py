from fastapi.testclient import TestClient

from app.api.http import import_server
from app.rag.import_.chunk_management_service import (
    ChunkNotFoundError,
    ChunkPermissionError,
    ChunkStateError,
    ChunkVersionConflictError,
)


def api_event(**overrides):
    data = {
        "event_id": "event_1",
        "document_id": "doc_1",
        "chunk_id": 1001,
        "dataset_id": "dataset_ops",
        "owner_user_id": "user_a",
        "tenant_id": "tenant_default",
        "visibility": "private",
        "index_version": 2,
        "chunk_index": 0,
        "operator_user_id": "user_a",
        "operation": "disable",
        "previous_enabled": True,
        "enabled": False,
        "reason_type": "garbled_text",
        "reason_detail": "OCR 乱码",
        "source": "manual",
        "human_confirmed": True,
        "created_at": "2026-07-15T08:00:00+00:00",
    }
    data.update(overrides)
    return data


def api_list_item(**overrides):
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
        "manual_status": "none",
        "effective_enabled": True,
        "title": "E021 报警处理",
        "parent_title": "报警排查",
        "source_title": "HAK180说明书",
        "content_preview": "报警 E021 表示主电机过载",
        "content_length": 15,
        "subject_id": "subject_hak_180",
        "standard_subject_name": "HAK 180 烫金机",
        "equipment_model": "HAK 180",
        "alarm_code": "E021",
        "part_name": "",
        "sop_type": "",
        "safety_level": "",
        "maintenance_stage": "",
        "latest_event": None,
    }
    data.update(overrides)
    return data


class FakeChunkManagementService:
    def __init__(self, *, error=None):
        self.error = error
        self.calls = []

    def _maybe_raise(self):
        if self.error:
            raise self.error

    def list_document_chunks(self, **kwargs):
        self.calls.append(("list_document_chunks", kwargs))
        self._maybe_raise()
        return {
            "code": 200,
            "items": [
                api_list_item(),
                api_list_item(
                    chunk_id=1002,
                    chunk_index=1,
                    manual_status="disabled",
                    effective_enabled=False,
                    latest_event=api_event(chunk_id=1002),
                ),
            ],
        }

    def get_chunk_detail(self, **kwargs):
        self.calls.append(("get_chunk_detail", kwargs))
        self._maybe_raise()
        return {
            **api_list_item(),
            "content": "报警 E021 表示主电机过载，需要检查负载和驱动器。",
        }

    def change_chunk_enabled(self, **kwargs):
        self.calls.append(("change_chunk_enabled", kwargs))
        self._maybe_raise()
        return {
            "code": 200,
            "message": "chunk 已禁用",
            "changed": True,
            "document_id": "doc_1",
            "chunk_id": 1001,
            "index_version": 2,
            "enabled": True,
            "manual_status": "disabled",
            "effective_enabled": False,
            "latest_event": api_event(),
        }

    def list_chunk_events(self, **kwargs):
        self.calls.append(("list_chunk_events", kwargs))
        self._maybe_raise()
        return {
            "code": 200,
            "document_id": "doc_1",
            "chunk_id": 1001,
            "index_version": 2,
            "items": [api_event()],
        }


def test_list_document_chunks_endpoint_passes_user_and_enabled_filter(monkeypatch):
    service = FakeChunkManagementService()
    monkeypatch.setattr(import_server, "get_chunk_management_service", lambda: service)

    response = TestClient(import_server.app).get(
        "/documents/doc_1/chunks?enabled=false&limit=50",
        headers={"X-User-Id": "user_a"},
    )

    assert response.status_code == 200
    data = response.json()
    assert [item["chunk_id"] for item in data["items"]] == [1001, 1002]
    assert data["items"][0]["content_preview"] == "报警 E021 表示主电机过载"
    assert "content" not in data["items"][0]
    assert data["items"][1]["manual_status"] == "disabled"
    assert data["items"][1]["latest_event"]["reason_type"] == "garbled_text"
    assert service.calls[0] == (
        "list_document_chunks",
        {
            "document_id": "doc_1",
            "user_id": "user_a",
            "tenant_id": "tenant_default",
            "enabled": False,
            "limit": 50,
        },
    )


def test_chunk_detail_endpoint_returns_full_content(monkeypatch):
    service = FakeChunkManagementService()
    monkeypatch.setattr(import_server, "get_chunk_management_service", lambda: service)

    response = TestClient(import_server.app).get(
        "/documents/doc_1/chunks/1001",
        headers={"X-User-Id": "user_a"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["chunk_id"] == 1001
    assert data["content"] == "报警 E021 表示主电机过载，需要检查负载和驱动器。"
    assert service.calls[0][1]["chunk_id"] == "1001"


def test_change_chunk_enabled_endpoint_passes_request_contract(monkeypatch):
    service = FakeChunkManagementService()
    monkeypatch.setattr(import_server, "get_chunk_management_service", lambda: service)

    response = TestClient(import_server.app).patch(
        "/documents/doc_1/chunks/1001/enabled",
        headers={"X-User-Id": "user_a"},
        json={
            "enabled": False,
            "expected_index_version": 2,
            "reason_type": "garbled_text",
            "reason_detail": "OCR 乱码",
            "trace_id": "trace_1",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["changed"] is True
    assert data["manual_status"] == "disabled"
    assert data["latest_event"]["trace_id"] is None
    assert service.calls[0][1] == {
        "document_id": "doc_1",
        "chunk_id": "1001",
        "user_id": "user_a",
        "tenant_id": "tenant_default",
        "enabled": False,
        "expected_index_version": 2,
        "reason_type": "garbled_text",
        "reason_detail": "OCR 乱码",
        "trace_id": "trace_1",
    }


def test_list_chunk_events_endpoint_returns_event_history(monkeypatch):
    service = FakeChunkManagementService()
    monkeypatch.setattr(import_server, "get_chunk_management_service", lambda: service)

    response = TestClient(import_server.app).get(
        "/documents/doc_1/chunks/1001/events?limit=10",
        headers={"X-User-Id": "user_a"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == "doc_1"
    assert data["chunk_id"] == 1001
    assert data["items"][0]["operation"] == "disable"
    assert service.calls[0][1]["limit"] == 10


def test_chunk_endpoint_requires_user_id_header(monkeypatch):
    service = FakeChunkManagementService()
    monkeypatch.setattr(import_server, "get_chunk_management_service", lambda: service)

    response = TestClient(import_server.app).get("/documents/doc_1/chunks")

    assert response.status_code == 400
    assert response.json()["detail"] == "缺少 X-User-Id 请求头"
    assert service.calls == []


def test_change_chunk_enabled_rejects_invalid_reason_contract(monkeypatch):
    service = FakeChunkManagementService()
    monkeypatch.setattr(import_server, "get_chunk_management_service", lambda: service)

    response = TestClient(import_server.app).patch(
        "/documents/doc_1/chunks/1001/enabled",
        headers={"X-User-Id": "user_a"},
        json={
            "enabled": False,
            "expected_index_version": 2,
            "reason_type": "other",
            "reason_detail": "",
        },
    )

    assert response.status_code == 422
    assert service.calls == []


def test_chunk_service_errors_map_to_http_status(monkeypatch):
    cases = [
        (ChunkNotFoundError("chunk_id=1001 不存在"), 404),
        (ChunkPermissionError("阶段 6 第一版只允许 document owner 启停 chunk"), 403),
        (ChunkVersionConflictError("expected_index_version=1 与当前 index_version=2 不一致"), 409),
        (ChunkStateError("Milvus enabled=false 的 chunk 不能通过路线 B 人工恢复"), 409),
    ]

    for error, expected_status in cases:
        service = FakeChunkManagementService(error=error)
        monkeypatch.setattr(import_server, "get_chunk_management_service", lambda service=service: service)

        response = TestClient(import_server.app).get(
            "/documents/doc_1/chunks/1001",
            headers={"X-User-Id": "user_a"},
        )

        assert response.status_code == expected_status
        assert response.json()["detail"] == str(error)
