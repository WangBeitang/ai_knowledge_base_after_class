from pathlib import Path

from fastapi.testclient import TestClient

from app.api.http import import_server


client = TestClient(import_server.app)


def test_delete_document_requires_user_header():
    response = client.delete("/documents/doc_1")

    assert response.status_code == 400
    assert response.json()["detail"] == "缺少 X-User-Id 请求头"


def test_delete_document_returns_404_for_other_owner(monkeypatch):
    monkeypatch.setattr(
        import_server,
        "delete_document_service",
        lambda document_id, owner_user_id: (_ for _ in ()).throw(
            import_server.DocumentNotFoundError(f"document_id={document_id} 不存在")
        ),
    )

    response = client.delete("/documents/doc_user_b", headers={"X-User-Id": "user_a"})

    assert response.status_code == 404


def test_delete_document_returns_deleted_metadata(monkeypatch):
    monkeypatch.setattr(
        import_server,
        "delete_document_service",
        lambda document_id, owner_user_id: {
            "document_id": document_id,
            "status": "deleted",
            "deleted_at": "2026-07-10T00:00:00+00:00",
        },
    )

    response = client.delete("/documents/doc_1", headers={"X-User-Id": "user_a"})

    assert response.status_code == 200
    assert response.json() == {
        "code": 200,
        "message": "文档删除成功",
        "document_id": "doc_1",
        "status": "deleted",
        "deleted_at": "2026-07-10T00:00:00+00:00",
    }


def test_delete_document_returns_409_for_processing_document(monkeypatch):
    monkeypatch.setattr(
        import_server,
        "delete_document_service",
        lambda document_id, owner_user_id: (_ for _ in ()).throw(
            import_server.DocumentStateError("document当前正在处理")
        ),
    )

    response = client.delete("/documents/doc_1", headers={"X-User-Id": "user_a"})

    assert response.status_code == 409


def test_rebuild_document_schedules_existing_import_graph(monkeypatch, tmp_path):
    source_file = tmp_path / "manual.pdf"
    source_file.write_bytes(b"pdf")
    local_dir = tmp_path / "rebuild"
    local_dir.mkdir()
    registered_tasks = []
    invoke_calls = []

    monkeypatch.setattr(
        import_server,
        "prepare_document_rebuild",
        lambda document_id, owner_user_id: {
            "task_id": "task_rebuild",
            "document_id": document_id,
            "dataset_id": "dataset_default_equipment_ops",
            "owner_user_id": owner_user_id,
            "tenant_id": "tenant_custom",
            "visibility": "shared",
            "index_version": 3,
            "source_file_path": Path(source_file),
            "local_dir": Path(local_dir),
        },
    )
    monkeypatch.setattr(
        import_server,
        "register_persistent_task",
        lambda *args: registered_tasks.append(args),
    )
    monkeypatch.setattr(import_server, "invoke_graph", lambda **kwargs: invoke_calls.append(kwargs))

    response = client.post("/documents/doc_1/rebuild", headers={"X-User-Id": "user_a"})

    assert response.status_code == 200
    assert response.json() == {
        "code": 200,
        "message": "重建索引任务已创建",
        "task_id": "task_rebuild",
        "document_id": "doc_1",
        "dataset_id": "dataset_default_equipment_ops",
        "index_version": 3,
    }
    assert registered_tasks == [
        ("task_rebuild", "doc_1", "dataset_default_equipment_ops", "user_a")
    ]
    assert invoke_calls[0]["document_id"] == "doc_1"
    assert invoke_calls[0]["tenant_id"] == "tenant_custom"
    assert invoke_calls[0]["visibility"] == "shared"
    assert invoke_calls[0]["local_file_path_obj"] == source_file


def test_rebuild_document_requires_user_header():
    response = client.post("/documents/doc_1/rebuild")

    assert response.status_code == 400


def test_rebuild_document_returns_404_for_other_owner(monkeypatch):
    monkeypatch.setattr(
        import_server,
        "prepare_document_rebuild",
        lambda document_id, owner_user_id: (_ for _ in ()).throw(
            import_server.DocumentNotFoundError(f"document_id={document_id} 不存在")
        ),
    )

    response = client.post("/documents/doc_user_b/rebuild", headers={"X-User-Id": "user_a"})

    assert response.status_code == 404


def test_rebuild_document_returns_409_for_invalid_state(monkeypatch):
    monkeypatch.setattr(
        import_server,
        "prepare_document_rebuild",
        lambda document_id, owner_user_id: (_ for _ in ()).throw(
            import_server.DocumentStateError("document已删除")
        ),
    )

    response = client.post("/documents/doc_1/rebuild", headers={"X-User-Id": "user_a"})

    assert response.status_code == 409
