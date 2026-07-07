from app.infra.persistence import import_metadata_repository as repo_module


class FakeUpdateResult:
    matched_count = 1
    modified_count = 1


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

    def update_one(self, query, update, upsert=False):
        key = query[self.key_field]
        if key not in self.items:
            if not upsert:
                return FakeUpdateResult()
            self.items[key] = {self.key_field: key}

        if "$setOnInsert" in update:
            for field_name, value in update["$setOnInsert"].items():
                self.items[key].setdefault(field_name, value)
        if "$set" in update:
            self.items[key].update(update["$set"])
        return FakeUpdateResult()

    def insert_one(self, document):
        self.items[document[self.key_field]] = dict(document)
        return FakeInsertResult()

    def find_one(self, query):
        for document in self.items.values():
            if all(document.get(field_name) == value for field_name, value in query.items()):
                return dict(document)
        return None

    def find(self, query):
        result = []
        for document in self.items.values():
            is_match = True
            for field_name, value in query.items():
                document_value = document.get(field_name)
                if isinstance(value, dict) and "$regex" in value:
                    if value["$regex"].lower() not in str(document_value).lower():
                        is_match = False
                        break
                elif document_value != value:
                    is_match = False
                    break
            if is_match:
                result.append(dict(document))
        return FakeCursor(result)


def build_fake_repository():
    repo = object.__new__(repo_module.ImportMetadataRepository)
    repo.datasets = FakeCollection("dataset_id")
    repo.documents = FakeCollection("document_id")
    repo.tasks = FakeCollection("task_id")
    return repo


def test_create_import_metadata_creates_default_dataset_document_and_task():
    repo = build_fake_repository()

    document, task = repo.create_import_metadata(
        dataset_id=repo_module.DEFAULT_DATASET_ID,
        document_id="doc_1",
        task_id="task_1",
        file_name="HAK180说明书.pdf",
        file_path="/tmp/HAK180说明书.pdf",
        local_dir="/tmp/task_1",
    )

    dataset = repo.get_dataset(repo_module.DEFAULT_DATASET_ID)

    assert dataset["name"] == repo_module.DEFAULT_DATASET_NAME
    assert document["document_id"] == "doc_1"
    assert document["latest_task_id"] == "task_1"
    assert document["status"] == repo_module.STATUS_UPLOADED
    assert document["parse_status"] == repo_module.STATUS_PENDING
    assert document["index_status"] == repo_module.STATUS_PENDING
    assert task["task_id"] == "task_1"
    assert task["document_id"] == "doc_1"
    assert task["task_type"] == repo_module.TASK_TYPE_IMPORT


def test_mark_import_failed_records_failed_node_and_document_stage():
    repo = build_fake_repository()
    repo.create_import_metadata(
        dataset_id=repo_module.DEFAULT_DATASET_ID,
        document_id="doc_1",
        task_id="task_1",
        file_name="HAK180说明书.pdf",
        file_path="/tmp/HAK180说明书.pdf",
        local_dir="/tmp/task_1",
    )

    repo.mark_import_failed("task_1", "node_pdf_to_md", "MinerU 解析失败")

    task = repo.get_task("task_1")
    document = repo.get_document("doc_1")

    assert task["status"] == repo_module.STATUS_FAILED
    assert task["failed_node"] == "node_pdf_to_md"
    assert task["error_message"] == "MinerU 解析失败"
    assert document["status"] == repo_module.STATUS_FAILED
    assert document["parse_status"] == repo_module.STATUS_FAILED
    assert document["index_status"] == repo_module.STATUS_PENDING


def test_mark_import_completed_updates_task_and_document():
    repo = build_fake_repository()
    repo.create_import_metadata(
        dataset_id=repo_module.DEFAULT_DATASET_ID,
        document_id="doc_1",
        task_id="task_1",
        file_name="HAK180说明书.pdf",
        file_path="/tmp/HAK180说明书.pdf",
        local_dir="/tmp/task_1",
    )
    repo.update_task_nodes("task_1", ["node_import_milvus"], ["node_entry"])

    repo.mark_import_completed("task_1")

    task = repo.get_task("task_1")
    document = repo.get_document("doc_1")

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
        file_name="HAK180说明书.pdf",
        file_path="/tmp/HAK180说明书.pdf",
        local_dir="/tmp/task_1",
    )

    documents = repo.list_documents(keyword="HAK180")
    tasks = repo.list_tasks(document_id="doc_1")

    assert [document["document_id"] for document in documents] == ["doc_1"]
    assert [task["task_id"] for task in tasks] == ["task_1"]
