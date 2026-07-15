"""阶段 7 Retrieval Test API 契约。"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.rag.query.config import RETRIEVAL_DEFAULT_MODE
from app.rag.query.contracts import Citation
from app.shared.config.knowledge_base_config import DEFAULT_DATASET_ID


class RetrievalTestSchemaModel(BaseModel):
    """Retrieval Test schema 公共基类。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


class RetrievalTestRunRequest(RetrievalTestSchemaModel):
    """运行一次检索测试。Retrieval Test 不写聊天历史，但必须写 Trace。"""

    query: str = Field(min_length=1)
    test_user_id: str | None = None
    dataset_ids: list[str] = Field(default_factory=lambda: [DEFAULT_DATASET_ID])
    retrieval_mode: str = Field(default_factory=lambda: RETRIEVAL_DEFAULT_MODE.value)
    planner_mode: str = "rule"
    web_fallback_enabled: bool | None = None
    generate_answer: bool = True

    @field_validator("dataset_ids")
    @classmethod
    def validate_dataset_ids(cls, dataset_ids: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for dataset_id in dataset_ids:
            value = str(dataset_id or "").strip()
            if value and value not in seen:
                normalized.append(value)
                seen.add(value)
        if not normalized:
            raise ValueError("dataset_ids 至少包含一个非空知识库 ID")
        return normalized


class TraceSummarySchema(RetrievalTestSchemaModel):
    """Trace 摘要响应。只返回结构化结果，不返回模型私有思维链。"""

    trace_id: str
    session_id: str = ""
    owner_user_id: str = ""
    dataset_ids: list[str] = Field(default_factory=list)
    original_query: str = ""
    status: str = ""
    planner_type: str = ""
    policy_version: str = ""
    retrieval_config_version: str = ""
    planner_steps: list[dict[str, Any]] = Field(default_factory=list)
    channel_hits: list[dict[str, Any]] = Field(default_factory=list)
    final_citations: list[Citation] = Field(default_factory=list)
    total_duration_ms: int = 0
    terminal_action: str | None = None
    terminal_reason_code: str | None = None
    execution_source: str = "chat"


class RetrievalTestRunResponse(RetrievalTestSchemaModel):
    """检索测试运行响应。"""

    code: int = 200
    trace_id: str
    session_id: str
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    trace: TraceSummarySchema | None = None


class RetrievalReplayRequest(RetrievalTestSchemaModel):
    """基于 trace_id 重放检索测试。"""

    planner_mode: str = "rule"
    strict_config_match: bool = True


class RetrievalReplayResponse(RetrievalTestSchemaModel):
    """Trace 重放响应。"""

    code: int = 200
    original_trace_id: str
    replay_trace_id: str = ""
    config_match_status: str = "unknown"
    corpus_match_status: str = "unknown"
    message: str = ""


__all__ = [
    "RetrievalReplayRequest",
    "RetrievalReplayResponse",
    "RetrievalTestRunRequest",
    "RetrievalTestRunResponse",
    "TraceSummarySchema",
]
