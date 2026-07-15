"""阶段 7 Trace 人工反馈 API 契约。"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TraceFeedbackSchemaModel(BaseModel):
    """Trace Feedback schema 公共基类。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


class ExpectedChunkSchema(TraceFeedbackSchemaModel):
    """人工期望引用的 chunk 版本身份。"""

    document_id: str = Field(min_length=1)
    chunk_id: int | str
    index_version: int = Field(ge=0)


class ExpectedBehaviorSchema(TraceFeedbackSchemaModel):
    """人工期望的 Planner 行为。"""

    should_ask_clarification: bool | None = None
    should_refuse: bool | None = None
    should_call_web: bool | None = None


class TraceFeedbackCreateRequest(TraceFeedbackSchemaModel):
    """写入 Trace 人工反馈。Feedback 是阶段 8 数据来源，不等于最终训练样本。"""

    expected_subject_ids: list[str] = Field(default_factory=list)
    expected_actions: list[str] = Field(default_factory=list)
    expected_chunks: list[ExpectedChunkSchema] = Field(default_factory=list)
    expected_answer_points: list[str] = Field(default_factory=list)
    expected_behavior: ExpectedBehaviorSchema = Field(default_factory=ExpectedBehaviorSchema)
    rating: int | None = Field(default=None, ge=1, le=5)
    notes: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def validate_has_signal(self) -> Self:
        if (
            not self.expected_subject_ids
            and not self.expected_actions
            and not self.expected_chunks
            and not self.expected_answer_points
            and self.rating is None
            and not self.notes
            and self.expected_behavior.should_ask_clarification is None
            and self.expected_behavior.should_refuse is None
            and self.expected_behavior.should_call_web is None
        ):
            raise ValueError("反馈至少需要提供一个有效标注字段")
        return self


class TraceFeedbackSchema(TraceFeedbackSchemaModel):
    """Trace 人工反馈响应。"""

    code: int = 200
    feedback_id: str
    trace_id: str
    operator_user_id: str
    dataset_ids: list[str] = Field(default_factory=list)
    expected_subject_ids: list[str] = Field(default_factory=list)
    expected_actions: list[str] = Field(default_factory=list)
    expected_chunks: list[ExpectedChunkSchema] = Field(default_factory=list)
    expected_answer_points: list[str] = Field(default_factory=list)
    expected_behavior: ExpectedBehaviorSchema = Field(default_factory=ExpectedBehaviorSchema)
    rating: int | None = None
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""


class TraceFeedbackListSchema(TraceFeedbackSchemaModel):
    """Trace 人工反馈列表响应。"""

    code: int = 200
    trace_id: str
    items: list[TraceFeedbackSchema] = Field(default_factory=list)


__all__ = [
    "ExpectedBehaviorSchema",
    "ExpectedChunkSchema",
    "TraceFeedbackCreateRequest",
    "TraceFeedbackListSchema",
    "TraceFeedbackSchema",
]
