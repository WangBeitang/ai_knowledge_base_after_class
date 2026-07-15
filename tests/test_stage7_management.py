import pytest

from app.api.schema.dataset_schema import DatasetCreateRequest, DatasetMemberRole, DatasetVisibility
from app.rag.management.access_control_service import AccessControlService, PermissionDeniedError, ResourceNotFoundError
from app.rag.management.conversation_management_service import ConversationManagementService
from app.rag.management.planner_management_service import get_planner_status
from app.rag.management.trace_feedback_service import TraceFeedbackService


class FakeMetadataRepository:
    def __init__(self):
        self.datasets = {
            "dataset_private": {
                "dataset_id": "dataset_private",
                "owner_user_id": "owner",
                "visibility": "private",
                "status": "active",
            },
            "dataset_public": {
                "dataset_id": "dataset_public",
                "owner_user_id": "owner",
                "visibility": "public",
                "status": "active",
            },
        }
        self.members = {
            ("dataset_private", "editor"): {
                "dataset_id": "dataset_private",
                "user_id": "editor",
                "role": "editor",
            }
        }
        self.documents = {
            "doc_shared": {
                "document_id": "doc_shared",
                "dataset_id": "dataset_private",
                "owner_user_id": "owner",
                "visibility": "shared",
                "status": "completed",
            }
        }

    def get_dataset(self, dataset_id):
        return dict(self.datasets.get(dataset_id) or {})

    def get_dataset_member(self, *, dataset_id, user_id):
        return dict(self.members.get((dataset_id, user_id)) or {})

    def get_visible_document(self, *, document_id, owner_user_id, tenant_id):
        document = self.documents.get(document_id)
        if not document:
            return {}
        if document["owner_user_id"] == owner_user_id or document["visibility"] in {"public", "shared"}:
            return dict(document)
        return {}


def test_dataset_schema_rejects_invalid_dataset_id():
    with pytest.raises(ValueError, match="dataset_id 只能包含"):
        DatasetCreateRequest(dataset_id="bad/id", name="测试知识库")

    request = DatasetCreateRequest(name="测试知识库", visibility=DatasetVisibility.PRIVATE)
    assert request.visibility is DatasetVisibility.PRIVATE
    assert DatasetMemberRole.EDITOR.value == "editor"


def test_access_control_distinguishes_public_read_and_editor_write():
    service = AccessControlService(metadata_repository=FakeMetadataRepository())

    _dataset, role = service.require_dataset_read(dataset_id="dataset_public", user_id="guest")
    assert role == "viewer"
    with pytest.raises(PermissionDeniedError):
        service.require_dataset_write(dataset_id="dataset_public", user_id="guest")

    _dataset, role = service.require_dataset_write(dataset_id="dataset_private", user_id="editor")
    assert role == "editor"

    with pytest.raises(ResourceNotFoundError):
        service.require_dataset_read(dataset_id="dataset_private", user_id="guest")


def test_access_control_allows_dataset_editor_to_operate_shared_document():
    service = AccessControlService(metadata_repository=FakeMetadataRepository())

    document, role = service.require_document_write(document_id="doc_shared", user_id="editor")

    assert document["document_id"] == "doc_shared"
    assert role == "editor"


class FakeConversationRepository:
    def __init__(self):
        self.cleared = []

    def list_conversations(self, user_id, limit=50):
        return [{"session_id": f"session-{user_id}", "message_count": 2}]

    def list_recent(self, session_id, limit=50, user_id=None):
        return [{"_id": "msg_1", "user_id": user_id, "session_id": session_id, "role": "user", "text": "问题"}]

    def clear_session(self, session_id, user_id=None, hidden_at=None):
        self.cleared.append((user_id, session_id, bool(hidden_at)))
        return 1


def test_conversation_service_passes_user_id_to_repository():
    repository = FakeConversationRepository()
    service = ConversationManagementService(repository=repository)

    assert service.list_conversations(user_id="user_a")["items"][0]["session_id"] == "session-user_a"
    detail = service.get_conversation(user_id="user_a", session_id="same_session")
    assert detail["items"][0]["user_id"] == "user_a"
    result = service.hide_conversation(user_id="user_a", session_id="same_session")

    assert result["deleted_count"] == 1
    assert result["hidden_at"]
    assert repository.cleared == [("user_a", "same_session", True)]


def test_planner_status_exposes_rule_only_online_mode():
    status = get_planner_status()

    assert status["online_mode"] == "rule"
    assert status["policy_version"] == "rule-v1"
    assert any(item["planner_mode"] == "grpo" and not item["enabled_online"] for item in status["registered_planners"])


class FakeTraceRepository:
    def get_trace(self, trace_id, owner_user_id=None):
        if owner_user_id != "user_a":
            return {}
        return {
            "trace_id": trace_id,
            "session_id": "session_1",
            "owner_user_id": owner_user_id,
            "dataset_ids": ["dataset_ops"],
            "original_query": "E021 怎么处理",
            "status": "completed",
        }

    def list_traces(self, **kwargs):
        return [self.get_trace("trace_1", owner_user_id=kwargs["owner_user_id"])]


class FakeFeedbackRepository:
    def __init__(self):
        self.created = []

    def create_feedback(self, feedback):
        self.created.append(feedback)
        return dict(feedback)

    def list_feedbacks(self, *, trace_id, limit=50):
        return list(self.created)


def test_trace_feedback_binds_feedback_to_visible_trace():
    feedback_repo = FakeFeedbackRepository()
    service = TraceFeedbackService(
        trace_repository=FakeTraceRepository(),
        feedback_repository=feedback_repo,
    )

    result = service.create_feedback(
        trace_id="trace_1",
        user_id="user_a",
        payload={
            "expected_subject_ids": ["subject_1"],
            "expected_actions": ["local_search", "answer"],
            "expected_chunks": [],
            "expected_answer_points": ["说明报警含义"],
            "expected_behavior": {"should_call_web": False},
            "rating": 5,
            "notes": "",
        },
    )

    assert result["trace_id"] == "trace_1"
    assert result["operator_user_id"] == "user_a"
    assert result["dataset_ids"] == ["dataset_ops"]
    assert service.list_feedbacks(trace_id="trace_1", user_id="user_a")["items"][0]["rating"] == 5
