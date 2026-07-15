"""阶段 7 Conversation 管理 API 契约。"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.rag.query.contracts import Citation, PlannerReasonCode


class ConversationSchemaModel(BaseModel):
    """Conversation API schema 公共基类。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


class ConversationListItemSchema(ConversationSchemaModel):
    """当前用户会话列表单项。"""

    session_id: str = Field(min_length=1)
    title: str = ""
    latest_message_preview: str = ""
    message_count: int = Field(default=0, ge=0)
    latest_trace_id: str = ""
    updated_at: Any = None


class ConversationListSchema(ConversationSchemaModel):
    """当前用户会话列表响应。"""

    code: int = 200
    items: list[ConversationListItemSchema] = Field(default_factory=list)


class ConversationMessageSchema(ConversationSchemaModel):
    """当前用户某个会话内的一条消息。"""

    id: str
    user_id: str = ""
    session_id: str
    role: str
    text: str
    rewritten_query: str = ""
    standard_subject_names: list[str] = Field(default_factory=list)
    image_urls: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    trace_id: str = ""
    terminal_reason_code: PlannerReasonCode | None = None
    ts: Any = None


class ConversationDetailSchema(ConversationSchemaModel):
    """当前用户会话详情响应。"""

    code: int = 200
    session_id: str = Field(min_length=1)
    items: list[ConversationMessageSchema] = Field(default_factory=list)


class ConversationDeleteSchema(ConversationSchemaModel):
    """隐藏当前用户会话响应。"""

    code: int = 200
    session_id: str = Field(min_length=1)
    deleted_count: int = Field(default=0, ge=0)
    hidden_at: str = ""


__all__ = [
    "ConversationDeleteSchema",
    "ConversationDetailSchema",
    "ConversationListItemSchema",
    "ConversationListSchema",
    "ConversationMessageSchema",
]
