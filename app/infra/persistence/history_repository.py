from typing import Any

from app.shared.clients.mongo_history_utils import (
    clear_history,
    get_recent_messages,
    list_user_conversations,
    save_chat_message,
    update_message_standard_subject_names
)

class HistoryRepository:
    def list_recent(self, session_id: str, limit: int = 10, user_id: str | None = None) -> list[dict]:
        return get_recent_messages(session_id, limit=limit, user_id=user_id)

    def save_message(
        self,
        *,
        user_id: str,
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        standard_subject_names: list[str] | None = None,
        image_urls: list[str] | None = None,
        message_id: str | None = None,
        citations: list[dict[str, Any]] | None = None,
        trace_id: str = "",
        terminal_reason_code: str = "",
    ) -> str:
        return save_chat_message(
            user_id=user_id,
            session_id=session_id,
            role=role,
            text=text,
            rewritten_query=rewritten_query,
            standard_subject_names=standard_subject_names,
            image_urls=image_urls,
            message_id=message_id,
            citations=citations,
            trace_id=trace_id,
            terminal_reason_code=terminal_reason_code,
        )

    def clear_session(self, session_id: str, user_id: str | None = None, hidden_at: str | None = None) -> int:
        return clear_history(session_id, user_id=user_id, hidden_at=hidden_at)

    def list_conversations(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        return list_user_conversations(user_id, limit=limit)

    def update_standard_subject_names(self, ids: list[str], standard_subject_names: list[str]) -> int:
        return update_message_standard_subject_names(ids, standard_subject_names)


history_repository = HistoryRepository()
