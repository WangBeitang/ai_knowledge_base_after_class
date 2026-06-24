from app.shared.clients.mongo_history_utils import (
    clear_history,
    get_recent_messages,
    save_chat_message,
    update_message_subject_names
)

class HistoryRepository:
    def list_recent(self, session_id: str, limit: int = 10) -> list[dict]:
        return get_recent_messages(session_id, limit=limit)

    def save_message(
        self,
        *,
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        subject_names: list[str] | None = None,
        image_urls: list[str] | None = None,
        message_id: str | None = None,
    ) -> str:
        return save_chat_message(
            session_id=session_id,
            role=role,
            text=text,
            rewritten_query=rewritten_query,
            subject_names=subject_names,
            image_urls=image_urls,
            message_id=message_id,
        )

    def clear_session(self, session_id: str) -> int:
        return clear_history(session_id)

    def update_subject_names(self, ids: list[str], subject_names: list[str]) -> int:
        return update_message_subject_names(ids, subject_names)


history_repository = HistoryRepository()
