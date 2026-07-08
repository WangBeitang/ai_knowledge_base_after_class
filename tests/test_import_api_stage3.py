from fastapi.testclient import TestClient

from app.api.http import import_server
from app.api.schema.import_schema import TaskStatusSchema, UploadSchema
from app.shared.utils import task_utils


class FakeImportMetadataRepository:
    def __init__(self, *, tasks=None, documents=None, should_raise=False):
        self.tasks = tasks or {}
        self.documents = documents or {}
        self.should_raise = should_raise
        self.create_calls = []

    def get_task(self, task_id, owner_user_id):
        if self.should_raise:
            raise RuntimeError("mongo unavailable")
        task = self.tasks.get(task_id, {})
        if task and task.get("owner_user_id") != owner_user_id:
            return {}
        return task

    def get_document(self, document_id, owner_user_id):
        document = self.documents.get(document_id, {})
        if document and document.get("owner_user_id") != owner_user_id:
            return {}
        return document

    def list_documents(self, *, owner_user_id, dataset_id, status=None, keyword=None, limit=20):
        documents = []
        for document in self.documents.values():
            if document.get("owner_user_id") != owner_user_id:
                continue
            if document.get("dataset_id") != dataset_id:
                continue
            if status and document.get("status") != status:
                continue
            if keyword and keyword.lower() not in document.get("file_name", "").lower():
                continue
            documents.append(document)
        return documents[:limit]

    def list_tasks(self, *, owner_user_id, document_id=None, dataset_id=None, status=None, limit=20):
        tasks = []
        for task in self.tasks.values():
            if task.get("owner_user_id") != owner_user_id:
                continue
            if document_id and task.get("document_id") != document_id:
                continue
            if dataset_id and task.get("dataset_id") != dataset_id:
                continue
            if status and task.get("status") != status:
                continue
            tasks.append(task)
        return tasks[:limit]

    def create_import_metadata(self, **kwargs):
        self.create_calls.append(kwargs)
        document = {
            "document_id": kwargs["document_id"],
            "dataset_id": kwargs["dataset_id"],
            "latest_task_id": kwargs["task_id"],
            "owner_user_id": kwargs["owner_user_id"],
        }
        task = {
            "task_id": kwargs["task_id"],
            "document_id": kwargs["document_id"],
            "dataset_id": kwargs["dataset_id"],
            "owner_user_id": kwargs["owner_user_id"],
        }
        self.documents[kwargs["document_id"]] = document
        self.tasks[kwargs["task_id"]] = task
        return document, task


class FailingImportApp:
    def __init__(self, task_id, failed_node, error_message):
        self.task_id = task_id
        self.failed_node = failed_node
        self.error_message = error_message
        self.last_state = None

    def invoke(self, state):
        self.last_state = state
        task_utils.add_running_task(self.task_id, self.failed_node)
        raise RuntimeError(self.error_message)


def test_import_schemas_include_stage3_fields():
    upload = UploadSchema(
        message="上传成功，正在处理中...",
        task_ids=["task_1"],
        document_ids=["doc_1"],
        dataset_id="dataset_default_equipment_ops",
        owner_user_id="user_a",
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
    assert upload.owner_user_id == "user_a"
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
                "owner_user_id": "user_a",
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

    response = TestClient(import_server.app).get("/status/task_1", headers={"X-User-Id": "user_a"})

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

    response = TestClient(import_server.app).get(f"/status/{task_id}", headers={"X-User-Id": "user_a"})

    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == task_id
    assert data["status"] == task_utils.TASK_STATUS_PROCESSING
    assert data["done_list"] == ["检查文件"]
    assert data["running_list"] == ["PDF转Markdown"]
    assert data["document_id"] == ""
    assert data["dataset_id"] == ""

    task_utils.clear_task(task_id)


def test_status_does_not_fallback_to_memory_when_owner_mismatch(monkeypatch):
    task_id = "task_owner_mismatch"
    task_utils.clear_task(task_id)
    task_utils.add_running_task(task_id, "node_import_milvus")
    task_utils.update_task_status(task_id, task_utils.TASK_STATUS_PROCESSING)
    fake_repo = FakeImportMetadataRepository(
        tasks={
            task_id: {
                "task_id": task_id,
                "document_id": "doc_user_b",
                "dataset_id": "dataset_default_equipment_ops",
                "owner_user_id": "user_b",
                "status": "processing",
                "done_nodes": [],
                "running_nodes": ["node_import_milvus"],
            }
        }
    )
    monkeypatch.setattr(import_server, "get_import_metadata_repository", lambda: fake_repo)

    response = TestClient(import_server.app).get(f"/status/{task_id}", headers={"X-User-Id": "user_a"})

    assert response.status_code == 404
    assert "task_id=task_owner_mismatch 不存在" in response.json()["detail"]

    task_utils.clear_task(task_id)


def test_document_status_returns_mongo_document(monkeypatch):
    fake_repo = FakeImportMetadataRepository(
        documents={
            "doc_1": {
                "document_id": "doc_1",
                "dataset_id": "dataset_default_equipment_ops",
                "owner_user_id": "user_a",
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

    response = TestClient(import_server.app).get("/documents/doc_1", headers={"X-User-Id": "user_a"})

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


def test_list_documents_filters_by_owner_user_id(monkeypatch):
    fake_repo = FakeImportMetadataRepository(
        documents={
            "doc_user_a": {
                "document_id": "doc_user_a",
                "dataset_id": "dataset_default_equipment_ops",
                "owner_user_id": "user_a",
                "latest_task_id": "task_user_a",
                "file_name": "HAK180说明书.pdf",
                "status": "completed",
                "parse_status": "completed",
                "index_status": "completed",
                "chunk_count": 8,
            },
            "doc_user_b": {
                "document_id": "doc_user_b",
                "dataset_id": "dataset_default_equipment_ops",
                "owner_user_id": "user_b",
                "latest_task_id": "task_user_b",
                "file_name": "HAK180维修手册.pdf",
                "status": "failed",
                "parse_status": "completed",
                "index_status": "failed",
                "chunk_count": 0,
            },
        }
    )
    monkeypatch.setattr(import_server, "get_import_metadata_repository", lambda: fake_repo)

    response = TestClient(import_server.app).get("/documents", headers={"X-User-Id": "user_a"})

    assert response.status_code == 200
    data = response.json()
    assert [item["document_id"] for item in data["items"]] == ["doc_user_a"]


def test_document_status_returns_404_for_other_user_document(monkeypatch):
    fake_repo = FakeImportMetadataRepository(
        documents={
            "doc_user_b": {
                "document_id": "doc_user_b",
                "dataset_id": "dataset_default_equipment_ops",
                "owner_user_id": "user_b",
                "latest_task_id": "task_user_b",
                "file_name": "HAK180维修手册.pdf",
                "status": "failed",
            }
        }
    )
    monkeypatch.setattr(import_server, "get_import_metadata_repository", lambda: fake_repo)

    response = TestClient(import_server.app).get("/documents/doc_user_b", headers={"X-User-Id": "user_a"})

    assert response.status_code == 404
    assert "document_id=doc_user_b 不存在" in response.json()["detail"]


def test_document_tasks_filters_by_owner_user_id(monkeypatch):
    fake_repo = FakeImportMetadataRepository(
        tasks={
            "task_user_a": {
                "task_id": "task_user_a",
                "document_id": "doc_shared_name",
                "dataset_id": "dataset_default_equipment_ops",
                "owner_user_id": "user_a",
                "status": "completed",
                "done_nodes": ["upload_file"],
                "running_nodes": [],
            },
            "task_user_b": {
                "task_id": "task_user_b",
                "document_id": "doc_shared_name",
                "dataset_id": "dataset_default_equipment_ops",
                "owner_user_id": "user_b",
                "status": "failed",
                "done_nodes": [],
                "running_nodes": ["node_import_milvus"],
            },
        }
    )
    monkeypatch.setattr(import_server, "get_import_metadata_repository", lambda: fake_repo)

    response = TestClient(import_server.app).get(
        "/documents/doc_shared_name/tasks",
        headers={"X-User-Id": "user_a"},
    )

    assert response.status_code == 200
    data = response.json()
    assert [item["task_id"] for item in data["items"]] == ["task_user_a"]


def test_document_status_returns_404_for_missing_document(monkeypatch):
    fake_repo = FakeImportMetadataRepository()
    monkeypatch.setattr(import_server, "get_import_metadata_repository", lambda: fake_repo)

    response = TestClient(import_server.app).get("/documents/doc_missing", headers={"X-User-Id": "user_a"})

    assert response.status_code == 404
    assert "document_id=doc_missing 不存在" in response.json()["detail"]


def test_upload_requires_user_id_header():
    response = TestClient(import_server.app).post(
        "/upload",
        files={"files": ("demo.md", b"# demo", "text/markdown")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "缺少 X-User-Id 请求头"


def test_upload_writes_owner_user_id_and_returns_it(monkeypatch, tmp_path):
    fake_repo = FakeImportMetadataRepository()
    background_calls = []
    monkeypatch.setattr(import_server, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(import_server, "get_import_metadata_repository", lambda: fake_repo)
    monkeypatch.setattr(import_server, "register_persistent_task", lambda *args: None)
    monkeypatch.setattr(import_server, "add_running_task", lambda *args: None)
    monkeypatch.setattr(import_server, "add_done_task", lambda *args: None)
    monkeypatch.setattr(
        import_server,
        "invoke_graph",
        lambda **kwargs: background_calls.append(kwargs),
    )

    response = TestClient(import_server.app).post(
        "/upload",
        headers={"X-User-Id": "user_a"},
        files={"files": ("demo.md", b"# demo", "text/markdown")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["owner_user_id"] == "user_a"
    assert fake_repo.create_calls[0]["owner_user_id"] == "user_a"
    assert background_calls[0]["owner_user_id"] == "user_a"


def test_invoke_graph_marks_failed_task_with_current_running_node(monkeypatch, tmp_path):
    task_id = "task_invoke_failed"
    task_utils.clear_task(task_id)
    failure_calls = []
    import_app = FailingImportApp(
        task_id=task_id,
        failed_node="node_import_milvus",
        error_message="Milvus 入库失败",
    )
    monkeypatch.setattr(
        import_server,
        "kb_import_app",
        import_app,
    )
    monkeypatch.setattr(
        import_server,
        "safe_mark_import_failed",
        lambda task_id, failed_node, error_message: failure_calls.append(
            {
                "task_id": task_id,
                "failed_node": failed_node,
                "error_message": error_message,
            }
        ),
    )

    import_server.invoke_graph(
        task_id=task_id,
        dataset_id="dataset_default_equipment_ops",
        document_id="doc_1",
        owner_user_id="user_a",
        local_file_path_obj=tmp_path / "demo.pdf",
        local_dir_path_obj=tmp_path,
    )

    assert task_utils.get_task_status(task_id) == task_utils.TASK_STATUS_FAILED
    assert failure_calls == [
        {
            "task_id": task_id,
            "failed_node": "node_import_milvus",
            "error_message": "Milvus 入库失败",
        }
    ]
    assert import_app.last_state["owner_user_id"] == "user_a"
    assert import_app.last_state["tenant_id"] == "tenant_default"
    assert import_app.last_state["visibility"] == "private"

    task_utils.clear_task(task_id)
