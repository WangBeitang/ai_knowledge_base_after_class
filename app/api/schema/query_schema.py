from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.rag.query.contracts import Citation, PlannerReasonCode
from app.shared.config.knowledge_base_config import DEFAULT_DATASET_ID


# 第一版查询入口只允许一次选择有限数量的知识库。这个限制作用于去空、去重后的
# dataset 集合，既保留调用方的原始顺序，也防止后续 Milvus 过滤表达式无限膨胀。
MAX_QUERY_DATASET_COUNT = 10


class QueryRequestParam(BaseModel):
    query: str
    session_id: str
    is_stream: bool = False
    # 省略字段时查询默认设备运维知识库；显式传 [] 则视为调用错误，而不是偷偷扩大或
    # 改写查询范围。default_factory 还能保证不同请求不会共享同一个可变 list。
    dataset_ids: list[str] = Field(default_factory=lambda: [DEFAULT_DATASET_ID])

    @field_validator("dataset_ids")
    @classmethod
    def normalize_dataset_ids(cls, dataset_ids: list[str]) -> list[str]:
        """
        规范化调用方指定的知识库范围。

        处理顺序为：去除首尾空白 -> 丢弃空字符串 -> 保序去重 -> 校验唯一值数量。
        这里不对显式空列表应用默认值，因为调用方已经表达了“没有目标知识库”；返回
        422 能更早暴露请求构造错误，也避免后续权限过滤意外退回默认知识库。
        """
        normalized_dataset_ids: list[str] = []
        seen_dataset_ids: set[str] = set()

        for dataset_id in dataset_ids:
            normalized_dataset_id = dataset_id.strip()
            if not normalized_dataset_id or normalized_dataset_id in seen_dataset_ids:
                continue
            normalized_dataset_ids.append(normalized_dataset_id)
            seen_dataset_ids.add(normalized_dataset_id)

        if not normalized_dataset_ids:
            raise ValueError("dataset_ids 至少包含一个非空知识库 ID")
        if len(normalized_dataset_ids) > MAX_QUERY_DATASET_COUNT:
            raise ValueError(f"dataset_ids 最多包含 {MAX_QUERY_DATASET_COUNT} 个不同的知识库 ID")

        return normalized_dataset_ids


class QueryStreamResponse(BaseModel):
    message: str
    session_id: str


class QueryNotStreamResponse(BaseModel):
    message: str
    session_id: str
    answer: str
    done_list: list
    image_urls: list
    # trace_id 唯一标识这一次查询执行，便于日志和 Mongo Retrieval Trace 对照。
    trace_id: str
    # citations 只包含真正进入答案上下文的最终证据；追问和拒答为空列表。
    citations: list[Citation]
    # terminal_reason_code 说明为什么最终回答、追问或拒答，不包含自由文本思维链。
    terminal_reason_code: PlannerReasonCode


class QueryTaskStatusResponse(BaseModel):
    """流式后台任务的轮询快照；FINAL SSE 与该结构共享最终交付字段。"""

    session_id: str
    status: str
    done_list: list[str]
    running_list: list[str]
    answer: str = ""
    error: str = ""
    image_urls: list[str] = Field(default_factory=list)
    trace_id: str = ""
    citations: list[Citation] = Field(default_factory=list)
    terminal_reason_code: PlannerReasonCode | None = None


class ClearHistoryResponse(BaseModel):
    message: str
    deleted_count: int


class HistoryItem(BaseModel):
    id: str
    session_id: str
    role: str
    text: str
    rewritten_query: str
    standard_subject_names: list[str]
    image_urls: list[str]
    citations: list[Citation] = Field(default_factory=list)
    trace_id: str = ""
    terminal_reason_code: PlannerReasonCode | None = None
    ts: Any


class HistoryResponse(BaseModel):
    session_id: str
    items: list[HistoryItem]
