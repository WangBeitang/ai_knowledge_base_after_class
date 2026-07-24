"""阶段 7 Planner 管理 API 契约。"""

from pydantic import BaseModel, ConfigDict, Field


class PlannerSchemaModel(BaseModel):
    """Planner API schema 公共基类。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


class PlannerRegistryItemSchema(PlannerSchemaModel):
    """一个已注册 Planner 实现的状态。"""

    planner_mode: str
    planner_type: str = ""
    enabled_online: bool = False
    enabled_for_eval: bool = False
    unavailable_reason: str = ""
    policy_version: str = ""
    provider: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    prompt_version: str | None = None
    endpoint: str | None = None


class PlannerStatusSchema(PlannerSchemaModel):
    """当前 Planner 配置摘要。"""

    code: int = 200
    online_mode: str = "rule"
    planner_type: str = "rule"
    policy_version: str
    provider: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    prompt_version: str | None = None
    endpoint: str | None = None
    current_unavailable_reason: str = ""
    retrieval_config_version: str
    max_steps: int = Field(ge=1)
    web_fallback_enabled: bool
    rule_config: dict[str, object] = Field(default_factory=dict)
    registered_planners: list[PlannerRegistryItemSchema] = Field(default_factory=list)


__all__ = [
    "PlannerRegistryItemSchema",
    "PlannerStatusSchema",
]
