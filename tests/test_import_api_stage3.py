from fastapi.testclient import TestClient

from app.api.http import import_server
from app.api.schema.import_schema import TaskStatusSchema, UploadSchema
from app.shared.utils import task_utils


class FakeImportMetadataRepository:
    def __init__(self, *, tasks=None, documents=None, should_raise=False):
        self.tasks = tasks or {}
        self.documents = documents or {}
        self.should_raise = should_raise

    def get_task(self, task_id):
        if self.should_raise:
            raise RuntimeError("mongo unavailable")
        return self.tasks.get(task_id, {})

    def get_document(self, document_id):
        return self.documents.get(document_id, {})


def test_import_schemas_include_stage3_fields():
    upload = UploadSchema(
        message="上传成功，正在处理中...",
        task_ids=["task_1"],
        document_ids=["doc_1"],
        dataset_id="dataset_default_equipment_ops",
    )
    status = TaskStatusSchema(
        task_id="task_1",
        status="failed",
        document_id="doc_1",
        dataset_id="dataset_default_equipment_ops",
        failed_node="node_import_milvus",
        error_message="Milvus 入库失败",
        created_at="2026-07-08T08:00:00+00:00",
        updated_at="2026-07-08T08:01:00+00:00",
    )

    assert upload.document_ids == ["doc_1"]
    assert upload.dataset_id == "dataset_default_equipment_ops"
    assert status.document_id == "doc_1"
    assert status.dataset_id == "dataset_default_equipment_ops"
    assert status.failed_node == "node_import_milvus"
    assert status.error_message == "Milvus 入库失败"
    assert status.created_at
    assert status.updated_at


def test_status_prefers_mongo_task_record(monkeypatch):
    fake_repo = FakeImportMetadataRepository(
        tasks={
            "task_1": {
                "task_id": "task_1",
                "document_id": "doc_1",
                "dataset_id": "dataset_default_equipment_ops",
                "status": "failed",
                "done_nodes": ["upload_file", "node_entry"],
                "running_nodes": ["node_import_milvus"],
                "failed_node": "node_import_milvus",
                "error_message": "Milvus 入库失败",
                "created_at": "2026-07-08T08:00:00+00:00",
                "updated_at": "2026-07-08T08:01:00+00:00",
            }
        }
    )
    monkeypatch.setattr(import_server, "get_import_metadata_repository", lambda: fake_repo)

    response = TestClient(import_server.app).get("/status/task_1")

    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == "task_1"
    assert data["document_id"] == "doc_1"
    assert data["dataset_id"] == "dataset_default_equipment_ops"
    assert data["status"] == "failed"
    assert data["done_list"] == ["开始上传文件", "检查文件"]
    assert data["running_list"] == ["导入向量库"]
    assert data["failed_node"] == "node_import_milvus"
    assert data["error_message"] == "Milvus 入库失败"
    assert data["created_at"] == "2026-07-08T08:00:00+00:00"
    assert data["updated_at"] == "2026-07-08T08:01:00+00:00"


def test_status_falls_back_to_memory_when_mongo_unavailable(monkeypatch):
    task_id = "task_memory_fallback"
    task_utils.clear_task(task_id)
    task_utils.add_running_task(task_id, "node_entry")
    task_utils.add_done_task(task_id, "node_entry")
    task_utils.add_running_task(task_id, "node_pdf_to_md")
    task_utils.update_task_status(task_id, task_utils.TASK_STATUS_PROCESSING)
    fake_repo = FakeImportMetadataRepository(should_raise=True)
    monkeypatch.setattr(import_server, "get_import_metadata_repository", lambda: fake_repo)

    response = TestClient(import_server.app).get(f"/status/{task_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == task_id
    assert data["status"] == task_utils.TASK_STATUS_PROCESSING
    assert data["done_list"] == ["检查文件"]
    assert data["running_list"] == ["PDF转Markdown"]
    assert data["document_id"] == ""
    assert data["dataset_id"] == ""

    task_utils.clear_task(task_id)


def test_document_status_returns_mongo_document(monkeypatch):
    fake_repo = FakeImportMetadataRepository(
        documents={
            "doc_1": {
                "document_id": "doc_1",
                "dataset_id": "dataset_default_equipment_ops",
                "latest_task_id": "task_1",
                "file_name": "HAK180说明书.pdf",
                "file_path": "/tmp/HAK180说明书.pdf",
                "local_dir": "/tmp/task_1",
                "status": "completed",
                "parse_status": "completed",
                "index_status": "completed",
                "chunk_count": 8,
                "subject_id": "subject_hak_180",
                "standard_subject_name": "HAK 180 烫金机",
                "md_path": "/tmp/HAK180说明书.md",
                "failed_node": "",
                "error_message": "",
                "created_at": "2026-07-08T08:00:00+00:00",
                "updated_at": "2026-07-08T08:01:00+00:00",
            }
        }
    )
    monkeypatch.setattr(import_server, "get_import_metadata_repository", lambda: fake_repo)

    response = TestClient(import_server.app).get("/documents/doc_1")

    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == "doc_1"
    assert data["dataset_id"] == "dataset_default_equipment_ops"
    assert data["latest_task_id"] == "task_1"
    assert data["index_status"] == "completed"
    assert data["chunk_count"] == 8
    assert data["subject_id"] == "subject_hak_180"
    assert data["standard_subject_name"] == "HAK 180 烫金机"
    assert data["created_at"] == "2026-07-08T08:00:00+00:00"
    assert data["updated_at"] == "2026-07-08T08:01:00+00:00"


def test_document_status_returns_404_for_missing_document(monkeypatch):
    fake_repo = FakeImportMetadataRepository()
    monkeypatch.setattr(import_server, "get_import_metadata_repository", lambda: fake_repo)

    response = TestClient(import_server.app).get("/documents/doc_missing")

    assert response.status_code == 404
    assert "document_id=doc_missing 不存在" in response.json()["detail"]
