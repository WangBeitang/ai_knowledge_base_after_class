from pathlib import Path

import pytest

from app.rag.import_ import document_lifecycle_service as lifecycle_service


class FakeRepository:
    def __init__(self, document):
        self.document = dict(document)
        self.calls = []

    def get_document(self, document_id, owner_user_id):
        if self.document.get("document_id") != document_id:
            return {}
        if self.document.get("owner_user_id") != owner_user_id:
            return {}
        return dict(self.document)

    def mark_document_deleted(self, *, document_id, owner_user_id):
        self.calls.append(("mark_deleted", document_id, owner_user_id))
        self.document.update({"status": "deleted", "deleted_at": "2026-07-10T00:00:00+00:00"})
        return dict(self.document)

    def create_rebuild_task_metadata(self, *, document_id, task_id, owner_user_id, local_dir):
        self.calls.append(("create_rebuild", document_id, task_id, owner_user_id, local_dir))
        self.document.update(
            {
                "latest_task_id": task_id,
                "index_version": int(self.document.get("index_version", 1)) + 1,
                "local_dir": local_dir,
                "status": "processing",
                "parse_status": "pending",
                "index_status": "pending",
            }
        )
        return dict(self.document), {"task_id": task_id, "task_type": "rebuild_index"}


class FakeMinioGateway:
    def __init__(self, calls):
        self.calls = calls

    def delete_image_prefix(self, image_prefix):
        self.calls.append(("delete_images", image_prefix))
        return 1


def _completed_document(file_path="/tmp/manual.pdf"):
    return {
        "document_id": "doc_1",
        "dataset_id": "dataset_default_equipment_ops",
        "owner_user_id": "user_a",
        "tenant_id": "tenant_custom",
        "visibility": "shared",
        "status": "completed",
        "index_version": 2,
        "file_path": file_path,
        "image_prefix": "kb-images/doc_1",
    }


def test_delete_document_cleans_external_resources_before_soft_delete(monkeypatch):
    calls = []
    repo = FakeRepository(_completed_document())
    original_mark_deleted = repo.mark_document_deleted

    def record_mark_deleted(**kwargs):
        calls.append(("mark_deleted", kwargs["document_id"], kwargs["owner_user_id"]))
        return original_mark_deleted(**kwargs)

    monkeypatch.setattr(repo, "mark_document_deleted", record_mark_deleted)
    monkeypatch.setattr(lifecycle_service, "get_import_metadata_repository", lambda: repo)
    monkeypatch.setattr(lifecycle_service, "remove_old_chunks", lambda document_id: calls.append(("delete_chunks", document_id)))
    monkeypatch.setattr(lifecycle_service, "minio_gateway", FakeMinioGateway(calls))

    result = lifecycle_service.delete_document("doc_1", "user_a")

    assert result["status"] == "deleted"
    assert calls == [
        ("delete_chunks", "doc_1"),
        ("delete_images", "kb-images/doc_1"),
        ("mark_deleted", "doc_1", "user_a"),
    ]


def test_delete_document_does_not_touch_resources_for_other_owner(monkeypatch):
    calls = []
    repo = FakeRepository(_completed_document())
    monkeypatch.setattr(lifecycle_service, "get_import_metadata_repository", lambda: repo)
    monkeypatch.setattr(lifecycle_service, "remove_old_chunks", lambda document_id: calls.append(document_id))
    monkeypatch.setattr(lifecycle_service, "minio_gateway", FakeMinioGateway(calls))

    with pytest.raises(lifecycle_service.DocumentNotFoundError):
        lifecycle_service.delete_document("doc_1", "user_b")

    assert calls == []
    assert repo.calls == []


def test_delete_document_does_not_soft_delete_when_image_cleanup_fails(monkeypatch):
    class FailingMinioGateway:
        def delete_image_prefix(self, image_prefix):
            raise RuntimeError("minio unavailable")

    repo = FakeRepository(_completed_document())
    monkeypatch.setattr(lifecycle_service, "get_import_metadata_repository", lambda: repo)
    monkeypatch.setattr(lifecycle_service, "remove_old_chunks", lambda document_id: None)
    monkeypatch.setattr(lifecycle_service, "minio_gateway", FailingMinioGateway())

    with pytest.raises(RuntimeError, match="minio unavailable"):
        lifecycle_service.delete_document("doc_1", "user_a")

    assert repo.document["status"] == "completed"
    assert repo.calls == []


@pytest.mark.parametrize("status", ["uploaded", "processing"])
def test_delete_document_rejects_non_terminal_document(monkeypatch, status):
    calls = []
    document = {**_completed_document(), "status": status}
    repo = FakeRepository(document)
    monkeypatch.setattr(lifecycle_service, "get_import_metadata_repository", lambda: repo)
    monkeypatch.setattr(lifecycle_service, "remove_old_chunks", lambda document_id: calls.append(document_id))
    monkeypatch.setattr(lifecycle_service, "minio_gateway", FakeMinioGateway(calls))

    with pytest.raises(lifecycle_service.DocumentStateError, match="当前正在处理"):
        lifecycle_service.delete_document("doc_1", "user_a")

    assert calls == []
    assert repo.calls == []


def test_prepare_rebuild_checks_source_file_before_creating_task(monkeypatch, tmp_path):
    missing_file = tmp_path / "missing.pdf"
    repo = FakeRepository(_completed_document(str(missing_file)))
    monkeypatch.setattr(lifecycle_service, "get_import_metadata_repository", lambda: repo)
    monkeypatch.setattr(lifecycle_service, "PROJECT_ROOT", tmp_path)

    with pytest.raises(lifecycle_service.DocumentStateError, match="原始文件不存在"):
        lifecycle_service.prepare_document_rebuild("doc_1", "user_a")

    assert repo.calls == []


def test_prepare_rebuild_reuses_document_and_preserves_access_metadata(monkeypatch, tmp_path):
    source_file = tmp_path / "manual.pdf"
    source_file.write_bytes(b"pdf")
    repo = FakeRepository(_completed_document(str(source_file)))
    monkeypatch.setattr(lifecycle_service, "get_import_metadata_repository", lambda: repo)
    monkeypatch.setattr(lifecycle_service, "PROJECT_ROOT", tmp_path)

    result = lifecycle_service.prepare_document_rebuild("doc_1", "user_a")

    assert result["document_id"] == "doc_1"
    assert result["task_id"] != ""
    assert result["index_version"] == 3
    assert result["tenant_id"] == "tenant_custom"
    assert result["visibility"] == "shared"
    assert result["source_file_path"] == Path(source_file)
    assert result["local_dir"].is_dir()
    assert repo.calls[0][0] == "create_rebuild"
    assert repo.calls[0][1] == "doc_1"
    assert repo.calls[0][2] == result["task_id"]


@pytest.mark.parametrize("status", ["deleted", "processing"])
def test_prepare_rebuild_rejects_invalid_document_state(monkeypatch, tmp_path, status):
    source_file = tmp_path / "manual.pdf"
    source_file.write_bytes(b"pdf")
    repo = FakeRepository({**_completed_document(str(source_file)), "status": status})
    monkeypatch.setattr(lifecycle_service, "get_import_metadata_repository", lambda: repo)

    with pytest.raises(lifecycle_service.DocumentStateError):
        lifecycle_service.prepare_document_rebuild("doc_1", "user_a")

    assert repo.calls == []
