"""阶段 7 Conversation 管理服务。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.infra.persistence.history_repository import history_repository


class ConversationManagementService:
    """当前用户会话列表、详情和隐藏。"""

    def __init__(self, *, repository=None) -> None:
        self.repository = repository or history_repository

    def list_conversations(self, *, user_id: str, limit: int = 50) -> dict[str, Any]:
        return {"code": 200, "items": self.repository.list_conversations(user_id, limit=limit)}

    def get_conversation(self, *, user_id: str, session_id: str, limit: int = 50) -> dict[str, Any]:
        messages = self.repository.list_recent(session_id, limit=limit, user_id=user_id)
        messages.reverse()
        return {"code": 200, "session_id": session_id, "items": messages}

    def hide_conversation(self, *, user_id: str, session_id: str) -> dict[str, Any]:
        hidden_at = datetime.now(timezone.utc).isoformat()
        deleted_count = self.repository.clear_session(session_id, user_id=user_id, hidden_at=hidden_at)
        return {
            "code": 200,
            "session_id": session_id,
            "deleted_count": deleted_count,
            "hidden_at": hidden_at,
        }


_conversation_management_service: ConversationManagementService | None = None


def get_conversation_management_service() -> ConversationManagementService:
    global _conversation_management_service
    if _conversation_management_service is None:
        _conversation_management_service = ConversationManagementService()
    return _conversation_management_service
