import pytest

from app.infra.persistence import import_metadata_repository as repo_module


class FakeUpdateResult:
    def __init__(self, matched_count=1, modified_count=1):
        self.matched_count = matched_count
        self.modified_count = modified_count


class FakeInsertResult:
    inserted_id = "fake-id"


class FakeCursor:
    def __init__(self, items):
        self.items = list(items)

    def sort(self, field_name, direction):
        reverse = direction < 0
        self.items.sort(key=lambda item: item.get(field_name, ""), reverse=reverse)
        return self

    def limit(self, limit):
        self.items = self.items[:limit]
        return self

    def __iter__(self):
        return iter(self.items)


class FakeCollection:
    def __init__(self, key_field):
        self.key_field = key_field
        self.items = {}
        self.indexes = []

    def create_index(self, index_fields, unique=False):
        self.indexes.append((index_fields, unique))

    @staticmethod
    def _matches(document, query):
        for field_name, value in query.items():
            document_value = document.get(field_name)
            if isinstance(value, dict) and "$regex" in value:
                if value["$regex"].lower() not in str(document_value).lower():
                    return False
            elif isinstance(value, dict) and "$ne" in value:
                if document_value == value["$ne"]:
                    return False
            elif isinstance(value, dict) and "$in" in value:
                if document_value not in value["$in"]:
                    return False
            elif document_value != value:
                return False
        return True

    def update_one(self, query, update, upsert=False):
        # 真实 Mongo 会拒绝同一路径同时出现在多个更新操作符中；测试桩也保持这个约束，
        # 避免 ensure_default_dataset 之类代码只在连接真实数据库时才暴露 path conflict。
        set_on_insert_fields = set(update.get("$setOnInsert", {}))
        set_fields = set(update.get("$set", {}))
        if set_on_insert_fields.intersection(set_fields):
            raise ValueError("Mongo update path conflict")
        matched_key = next(
            (key for key, document in self.items.items() if self._matches(document, query)),
            None,
        )
        if matched_key is None:
            if not upsert:
                return FakeUpdateResult(matched_count=0, modified_count=0)
            key = query[self.key_field]
            self.items[key] = {self.key_field: key}
            matched_key = key

        if "$setOnInsert" in update:
            for field_name, value in update["$setOnInsert"].items():
                self.items[matched_key].setdefault(field_name, value)
        if "$set" in update:
            self.items[matched_key].update(update["$set"])
        return FakeUpdateResult()

    def insert_one(self, document):
        self.items[document[self.key_field]] = dict(document)
        return FakeInsertResult()

    def find_one(self, query):
        for document in self.items.values():
            if self._matches(document, query):
                return dict(document)
        return None

    def find(self, query):
        result = []
        for document in self.items.values():
            if self._matches(document, query):
                result.append(dict(document))
        return FakeCursor(result)


def build_fake_repository():
    repo = object.__new__(repo_module.ImportMetadataRepository)
    repo.datasets = FakeCollection("dataset_id")
    repo.documents = FakeCollection("document_id")
    repo.tasks = FakeCollection("task_id")
    return repo


def test_ensure_indexes_includes_owner_user_id_indexes():
    repo = build_fake_repository()

    repo._ensure_indexes()

    assert ((("owner_user_id", repo_module.ASCENDING), ("dataset_id", repo_module.ASCENDING), ("updated_at", repo_module.DESCENDING)), False) in [
        (tuple(index_fields), unique) for index_fields, unique in repo.documents.indexes
    ]
    assert ((("owner_user_id", repo_module.ASCENDING), ("document_id", repo_module.ASCENDING)), False) in [
        (tuple(index_fields), unique) for index_fields, unique in repo.documents.indexes
    ]
    assert ((("owner_user_id", repo_module.ASCENDING), ("status", repo_module.ASCENDING)), False) in [
        (tuple(index_fields), unique) for index_fields, unique in repo.documents.indexes
    ]
    assert ((("owner_user_id", repo_module.ASCENDING), ("task_id", repo_module.ASCENDING)), False) in [
        (tuple(index_fields), unique) for index_fields, unique in repo.tasks.indexes
    ]
    assert ((("owner_user_id", repo_module.ASCENDING), ("document_id", repo_module.ASCENDING), ("created_at", repo_module.DESCENDING)), False) in [
        (tuple(index_fields), unique) for index_fields, unique in repo.tasks.indexes
    ]
    assert ((("status", repo_module.ASCENDING),), False) in [
        (tuple(index_fields), unique) for index_fields, unique in repo.tasks.indexes
    ]


def test_create_import_metadata_creates_default_dataset_document_and_task():
    repo = build_fake_repository()

    document, task = repo.create_import_metadata(
        dataset_id=repo_module.DEFAULT_DATASET_ID,
        document_id="doc_1",
        task_id="task_1",
        owner_user_id="user_a",
        file_name="HAK180说明书.pdf",
        file_path="/tmp/HAK180说明书.pdf",
        local_dir="/tmp/task_1",
    )

    dataset = repo.get_dataset(repo_module.DEFAULT_DATASET_ID)

    assert dataset["name"] == repo_module.DEFAULT_DATASET_NAME
    assert document["document_id"] == "doc_1"
    assert document["owner_user_id"] == "user_a"
    assert document["tenant_id"] == repo_module.DEFAULT_TENANT_ID
    assert document["visibility"] == repo_module.DEFAULT_VISIBILITY
    assert document["latest_task_id"] == "task_1"
    assert document["index_version"] == repo_module.DEFAULT_INDEX_VERSION
    assert document["deleted_at"] == ""
    assert document["image_prefix"] == ""
    assert document["parse_result_zip_path"] == ""
    assert document["parse_result_dir"] == ""
    assert document["status"] == repo_module.STATUS_UPLOADED
    assert document["parse_status"] == repo_module.STATUS_PENDING
    assert document["index_status"] == repo_module.STATUS_PENDING
    assert document["error_code"] == ""
    assert task["task_id"] == "task_1"
    assert task["document_id"] == "doc_1"
    assert task["owner_user_id"] == "user_a"
    assert task["tenant_id"] == repo_module.DEFAULT_TENANT_ID
    assert task["task_type"] == repo_module.TASK_TYPE_IMPORT
    assert task["error_code"] == ""


def test_create_rebuild_task_metadata_reuses_document_and_increments_index_version():
    repo = build_fake_repository()
    repo.create_import_metadata(
        dataset_id=repo_module.DEFAULT_DATASET_ID,
        document_id="doc_1",
        task_id="task_import",
        owner_user_id="user_a",
        file_name="HAK180说明书.pdf",
        file_path="/tmp/HAK180说明书.pdf",
        local_dir="/tmp/task_import",
    )

    document, task = repo.create_rebuild_task_metadata(
        document_id="doc_1",
        task_id="task_rebuild",
        owner_user_id="user_a",
        local_dir="/tmp/task_rebuild",
    )

    stored_document = repo.get_document("doc_1", "user_a")
    stored_task = repo.get_task("task_rebuild", "user_a")

    assert document["document_id"] == "doc_1"
    assert document["latest_task_id"] == "task_rebuild"
    assert document["index_version"] == repo_module.DEFAULT_INDEX_VERSION + 1
    assert document["status"] == repo_module.STATUS_PROCESSING
    assert document["parse_status"] == repo_module.STATUS_PENDING
    assert document["error_code"] == ""
    assert document["local_dir"] == "/tmp/task_rebuild"
    assert task["task_type"] == repo_module.TASK_TYPE_REBUILD_INDEX
    assert task["status"] == repo_module.STATUS_PENDING
    assert task["error_code"] == ""
    assert stored_document["latest_task_id"] == "task_rebuild"
    assert stored_document["index_version"] == repo_module.DEFAULT_INDEX_VERSION + 1
    assert stored_task["task_type"] == repo_module.TASK_TYPE_REBUILD_INDEX


def test_create_rebuild_task_metadata_rejects_missing_or_deleted_document():
    repo = build_fake_repository()

    with pytest.raises(ValueError, match="document_id=doc_missing 不存在"):
        repo.create_rebuild_task_metadata(
            document_id="doc_missing",
            task_id="task_rebuild",
            owner_user_id="user_a",
            local_dir="/tmp/task_rebuild",
        )

    repo.create_import_metadata(
        dataset_id=repo_module.DEFAULT_DATASET_ID,
        document_id="doc_deleted",
        task_id="task_import",
        owner_user_id="user_a",
        file_name="HAK180说明书.pdf",
        file_path="/tmp/HAK180说明书.pdf",
        local_dir="/tmp/task_import",
    )
    repo.update_document("doc_deleted", status=repo_module.STATUS_DELETED)

    with pytest.raises(ValueError, match="document_id=doc_deleted 已删除"):
        repo.create_rebuild_task_metadata(
            document_id="doc_deleted",
            task_id="task_rebuild_deleted",
            owner_user_id="user_a",
            local_dir="/tmp/task_rebuild_deleted",
        )


def test_create_import_metadata_uses_existing_custom_dataset_without_creating_default():
    repo = build_fake_repository()
    repo.datasets.insert_one(
        {
            "dataset_id": "dataset_after_sales",
            "name": "售后维修知识库",
        }
    )

    document, task = repo.create_import_metadata(
        dataset_id="dataset_after_sales",
        document_id="doc_1",
        task_id="task_1",
        owner_user_id="user_a",
        file_name="HAK180维修手册.pdf",
        file_path="/tmp/HAK180维修手册.pdf",
        local_dir="/tmp/task_1",
    )

    assert document["dataset_id"] == "dataset_after_sales"
    assert task["dataset_id"] == "dataset_after_sales"
    assert repo.get_dataset(repo_module.DEFAULT_DATASET_ID) == {}


def test_create_import_metadata_rejects_unknown_custom_dataset():
    repo = build_fake_repository()

    try:
        repo.create_import_metadata(
            dataset_id="dataset_missing",
            document_id="doc_1",
            task_id="task_1",
            owner_user_id="user_a",
            file_name="HAK180维修手册.pdf",
            file_path="/tmp/HAK180维修手册.pdf",
            local_dir="/tmp/task_1",
        )
    except ValueError as e:
        assert "dataset_id=dataset_missing 不存在" in str(e)
    else:
        raise AssertionError("未知 dataset_id 应该拒绝导入")


@pytest.mark.parametrize(
    "failed_node",
    ["upload_file", "node_entry", "node_pdf_to_md", "node_md_img"],
)
def test_mark_import_failed_records_parse_stage_for_parse_nodes(failed_node):
    repo = build_fake_repository()
    repo.create_import_metadata(
        dataset_id=repo_module.DEFAULT_DATASET_ID,
        document_id="doc_1",
        task_id="task_1",
        owner_user_id="user_a",
        file_name="HAK180说明书.pdf",
        file_path="/tmp/HAK180说明书.pdf",
        local_dir="/tmp/task_1",
    )

    repo.mark_import_failed("task_1", failed_node, "解析阶段失败")

    task = repo.get_task("task_1", "user_a")
    document = repo.get_document("doc_1", "user_a")

    assert task["status"] == repo_module.STATUS_FAILED
    assert task["running_nodes"] == []
    assert task["failed_node"] == failed_node
    assert task["error_message"] == "解析阶段失败"
    assert document["status"] == repo_module.STATUS_FAILED
    assert document["parse_status"] == repo_module.STATUS_FAILED
    assert document["index_status"] == repo_module.STATUS_PENDING


@pytest.mark.parametrize(
    "failed_node",
    ["node_subject_name_recognition", "node_bge_embedding", "node_import_milvus"],
)
def test_mark_import_failed_records_index_stage_for_index_nodes(failed_node):
    repo = build_fake_repository()
    repo.create_import_metadata(
        dataset_id=repo_module.DEFAULT_DATASET_ID,
        document_id="doc_1",
        task_id="task_1",
        owner_user_id="user_a",
        file_name="HAK180说明书.pdf",
        file_path="/tmp/HAK180说明书.pdf",
        local_dir="/tmp/task_1",
    )
    repo.update_document(
        "doc_1",
        parse_status=repo_module.STATUS_COMPLETED,
        index_status=repo_module.STATUS_PROCESSING,
    )

    repo.mark_import_failed("task_1", failed_node, "索引阶段失败")

    task = repo.get_task("task_1", "user_a")
    document = repo.get_document("doc_1", "user_a")

    assert task["status"] == repo_module.STATUS_FAILED
    assert task["running_nodes"] == []
    assert task["failed_node"] == failed_node
    assert task["error_message"] == "索引阶段失败"
    assert document["status"] == repo_module.STATUS_FAILED
    assert document["parse_status"] == repo_module.STATUS_COMPLETED
    assert document["index_status"] == repo_module.STATUS_FAILED


def test_mark_import_completed_updates_task_and_document():
    repo = build_fake_repository()
    repo.create_import_metadata(
        dataset_id=repo_module.DEFAULT_DATASET_ID,
        document_id="doc_1",
        task_id="task_1",
        owner_user_id="user_a",
        file_name="HAK180说明书.pdf",
        file_path="/tmp/HAK180说明书.pdf",
        local_dir="/tmp/task_1",
    )
    repo.update_task_nodes("task_1", ["node_import_milvus"], ["node_entry"])

    repo.mark_import_completed("task_1")

    task = repo.get_task("task_1", "user_a")
    document = repo.get_document("doc_1", "user_a")

    assert task["status"] == repo_module.STATUS_COMPLETED
    assert task["running_nodes"] == []
    assert document["status"] == repo_module.STATUS_COMPLETED
    assert document["parse_status"] == repo_module.STATUS_COMPLETED
    assert document["index_status"] == repo_module.STATUS_COMPLETED


def test_list_documents_and_tasks_use_document_as_history_entry():
    repo = build_fake_repository()
    repo.create_import_metadata(
        dataset_id=repo_module.DEFAULT_DATASET_ID,
        document_id="doc_1",
        task_id="task_1",
        owner_user_id="user_a",
        file_name="HAK180说明书.pdf",
        file_path="/tmp/HAK180说明书.pdf",
        local_dir="/tmp/task_1",
    )

    documents = repo.list_documents(owner_user_id="user_a", keyword="HAK180")
    tasks = repo.list_tasks(document_id="doc_1", owner_user_id="user_a")

    assert [document["document_id"] for document in documents] == ["doc_1"]
    assert [task["task_id"] for task in tasks] == ["task_1"]


def test_mark_document_deleted_keeps_history_and_list_hides_it_by_default():
    repo = build_fake_repository()
    repo.create_import_metadata(
        dataset_id=repo_module.DEFAULT_DATASET_ID,
        document_id="doc_1",
        task_id="task_1",
        owner_user_id="user_a",
        file_name="HAK180说明书.pdf",
        file_path="/tmp/HAK180说明书.pdf",
        local_dir="/tmp/task_1",
    )
    repo.update_document(
        "doc_1",
        subject_id="subject_hak_180",
        status=repo_module.STATUS_COMPLETED,
    )

    deleted_document = repo.mark_document_deleted(
        document_id="doc_1",
        owner_user_id="user_a",
    )

    assert deleted_document["status"] == repo_module.STATUS_DELETED
    assert deleted_document["deleted_at"]
    assert deleted_document["latest_task_id"] == "task_1"
    assert deleted_document["file_path"] == "/tmp/HAK180说明书.pdf"
    assert deleted_document["subject_id"] == "subject_hak_180"
    assert repo.list_documents(owner_user_id="user_a") == []
    assert [
        item["document_id"]
        for item in repo.list_documents(owner_user_id="user_a", status=repo_module.STATUS_DELETED)
    ] == ["doc_1"]


def test_mark_document_deleted_rejects_other_owner():
    repo = build_fake_repository()
    repo.create_import_metadata(
        dataset_id=repo_module.DEFAULT_DATASET_ID,
        document_id="doc_1",
        task_id="task_1",
        owner_user_id="user_a",
        file_name="HAK180说明书.pdf",
        file_path="/tmp/HAK180说明书.pdf",
        local_dir="/tmp/task_1",
    )

    with pytest.raises(ValueError, match="document_id=doc_1 不存在"):
        repo.mark_document_deleted(document_id="doc_1", owner_user_id="user_b")


def test_repository_queries_filter_by_owner_user_id():
    repo = build_fake_repository()
    repo.create_import_metadata(
        dataset_id=repo_module.DEFAULT_DATASET_ID,
        document_id="doc_user_a",
        task_id="task_user_a",
        owner_user_id="user_a",
        file_name="HAK180说明书.pdf",
        file_path="/tmp/HAK180说明书.pdf",
        local_dir="/tmp/task_user_a",
    )
    repo.create_import_metadata(
        dataset_id=repo_module.DEFAULT_DATASET_ID,
        document_id="doc_user_b",
        task_id="task_user_b",
        owner_user_id="user_b",
        file_name="HAK180维修手册.pdf",
        file_path="/tmp/HAK180维修手册.pdf",
        local_dir="/tmp/task_user_b",
    )

    assert repo.get_document("doc_user_a", "user_a")["document_id"] == "doc_user_a"
    assert repo.get_document("doc_user_a", "user_b") == {}
    assert repo.get_task("task_user_a", "user_a")["task_id"] == "task_user_a"
    assert repo.get_task("task_user_a", "user_b") == {}

    documents = repo.list_documents(owner_user_id="user_a", keyword="HAK180")
    tasks = repo.list_tasks(document_id="doc_user_a", owner_user_id="user_a")

    assert [document["document_id"] for document in documents] == ["doc_user_a"]
    assert [task["task_id"] for task in tasks] == ["task_user_a"]


@pytest.mark.parametrize(
    ("running_node", "expected_parse_status", "expected_index_status"),
    [
        ("node_pdf_to_md", repo_module.STATUS_FAILED, repo_module.STATUS_PENDING),
        ("node_bge_embedding", repo_module.STATUS_COMPLETED, repo_module.STATUS_FAILED),
        ("", repo_module.STATUS_PENDING, repo_module.STATUS_PENDING),
    ],
)
def test_reconcile_interrupted_tasks_marks_task_and_latest_document_failed(
        running_node,
        expected_parse_status,
        expected_index_status,
):
    repo = build_fake_repository()
    repo.create_import_metadata(
        dataset_id=repo_module.DEFAULT_DATASET_ID,
        document_id="doc_1",
        task_id="task_1",
        owner_user_id="user_a",
        file_name="HAK180说明书.pdf",
        file_path="/tmp/HAK180说明书.pdf",
        local_dir="/tmp/task_1",
    )
    if running_node == "node_bge_embedding":
        repo.update_document("doc_1", parse_status=repo_module.STATUS_COMPLETED)
    repo.tasks.items["task_1"].update({
        "status": repo_module.STATUS_PROCESSING,
        "running_nodes": [running_node] if running_node else [],
        "done_nodes": ["node_entry"],
    })
    repo.documents.items["doc_1"]["status"] = repo_module.STATUS_PROCESSING

    summary = repo.reconcile_interrupted_tasks()
    task = repo.get_task("task_1", "user_a")
    document = repo.get_document("doc_1", "user_a")

    assert summary == {
        "examined_task_count": 1,
        "failed_task_count": 1,
        "completed_task_count": 0,
        "failed_document_count": 1,
    }
    assert task["status"] == repo_module.STATUS_FAILED
    assert task["running_nodes"] == []
    assert task["done_nodes"] == ["node_entry"]
    assert task["failed_node"] == running_node
    assert task["error_code"] == repo_module.ERROR_CODE_IMPORT_SERVICE_RESTARTED
    assert task["error_message"] == repo_module.IMPORT_SERVICE_RESTARTED_MESSAGE
    assert task["failed_at"]
    assert document["status"] == repo_module.STATUS_FAILED
    assert document["failed_node"] == running_node
    assert document["error_code"] == repo_module.ERROR_CODE_IMPORT_SERVICE_RESTARTED
    assert document["parse_status"] == expected_parse_status
    assert document["index_status"] == expected_index_status

    # 第二次启动不应重复修改已经收口的终态记录。
    assert repo.reconcile_interrupted_tasks() == {
        "examined_task_count": 0,
        "failed_task_count": 0,
        "completed_task_count": 0,
        "failed_document_count": 0,
    }


def test_reconcile_interrupted_old_task_does_not_overwrite_newer_document():
    repo = build_fake_repository()
    repo.create_import_metadata(
        dataset_id=repo_module.DEFAULT_DATASET_ID,
        document_id="doc_1",
        task_id="task_old",
        owner_user_id="user_a",
        file_name="HAK180说明书.pdf",
        file_path="/tmp/HAK180说明书.pdf",
        local_dir="/tmp/task_old",
    )
    repo.tasks.items["task_old"].update({
        "status": repo_module.STATUS_PROCESSING,
        "running_nodes": ["node_import_milvus"],
    })
    repo.documents.items["doc_1"].update({
        "latest_task_id": "task_new",
        "status": repo_module.STATUS_COMPLETED,
        "parse_status": repo_module.STATUS_COMPLETED,
        "index_status": repo_module.STATUS_COMPLETED,
    })

    summary = repo.reconcile_interrupted_tasks()

    assert summary["failed_task_count"] == 1
    assert summary["failed_document_count"] == 0
    assert repo.get_task("task_old", "user_a")["status"] == repo_module.STATUS_FAILED
    assert repo.get_document("doc_1", "user_a")["status"] == repo_module.STATUS_COMPLETED


def test_reconcile_repairs_task_when_latest_document_was_already_committed():
    repo = build_fake_repository()
    repo.create_import_metadata(
        dataset_id=repo_module.DEFAULT_DATASET_ID,
        document_id="doc_1",
        task_id="task_1",
        owner_user_id="user_a",
        file_name="HAK180说明书.pdf",
        file_path="/tmp/HAK180说明书.pdf",
        local_dir="/tmp/task_1",
    )
    repo.tasks.items["task_1"].update({
        "status": repo_module.STATUS_PROCESSING,
        "running_nodes": ["node_import_milvus"],
    })
    repo.documents.items["doc_1"].update({
        "status": repo_module.STATUS_COMPLETED,
        "parse_status": repo_module.STATUS_COMPLETED,
        "index_status": repo_module.STATUS_COMPLETED,
    })

    summary = repo.reconcile_interrupted_tasks()
    task = repo.get_task("task_1", "user_a")

    assert summary["completed_task_count"] == 1
    assert summary["failed_task_count"] == 0
    assert task["status"] == repo_module.STATUS_COMPLETED
    assert task["running_nodes"] == []
    assert task["error_code"] == ""
    assert task["completed_at"]


def test_reconcile_leaves_existing_terminal_tasks_unchanged():
    repo = build_fake_repository()
    repo.create_import_metadata(
        dataset_id=repo_module.DEFAULT_DATASET_ID,
        document_id="doc_1",
        task_id="task_1",
        owner_user_id="user_a",
        file_name="HAK180说明书.pdf",
        file_path="/tmp/HAK180说明书.pdf",
        local_dir="/tmp/task_1",
    )
    repo.mark_import_completed("task_1")
    before_task = repo.get_task("task_1", "user_a")
    before_document = repo.get_document("doc_1", "user_a")

    summary = repo.reconcile_interrupted_tasks()

    assert summary["examined_task_count"] == 0
    assert repo.get_task("task_1", "user_a") == before_task
    assert repo.get_document("doc_1", "user_a") == before_document


def test_safe_reconcile_does_not_block_startup_when_mongo_is_unavailable(monkeypatch):
    def raise_mongo_error():
        raise RuntimeError("mongo unavailable")

    monkeypatch.setattr(repo_module, "get_import_metadata_repository", raise_mongo_error)

    assert repo_module.safe_reconcile_interrupted_tasks() == {
        "examined_task_count": 0,
        "failed_task_count": 0,
        "completed_task_count": 0,
        "failed_document_count": 0,
    }
