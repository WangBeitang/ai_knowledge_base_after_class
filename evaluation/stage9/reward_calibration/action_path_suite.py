"""阶段 9 Reward v1.1 多轨迹校准路线套件。

Action path suite 的中文含义是“动作路线套件”。它不是训练数据本身，而是在同一个
dev case 上强制执行多条 Planner 路线，用来验证 Reward 是否能区分合理与不合理决策。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.rag.evaluation.case_schema import PlannerEvalCase
from app.rag.query.contracts import QueryAction


ACTION_PATH_SUITE_VERSION = "stage9-action-path-suite-v1"


class ActionPathSuiteModel(BaseModel):
    """多轨迹校准 schema 公共基类；拒绝未知字段，避免校准产物字段漂移。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


class CalibrationPathSpec(ActionPathSuiteModel):
    """
    一条待执行的 Planner Action 路线。

    path_id 是报告和 JSON 结果中的稳定短名；action_path 是真正交给
    OfflineRagEnvironment.run_action_path 的动作序列；route_family 用于按路线类型汇总。
    """

    path_id: str = Field(min_length=1, description="路线稳定 ID，例如 local_answer。")
    action_path: list[QueryAction] = Field(min_length=1, description="待执行的 Action 序列。")
    route_family: str = Field(min_length=1, description="路线类别，例如 answer、fallback、refuse。")
    purpose: str = Field(min_length=1, description="中文说明：这条路线用于验证哪类 Reward 行为。")


BASE_PATH_SPECS: tuple[CalibrationPathSpec, ...] = (
    CalibrationPathSpec(
        path_id="local_answer",
        action_path=[QueryAction.LOCAL_SEARCH, QueryAction.ANSWER],
        route_family="answer",
        purpose="验证本地检索后直接回答是否得到合理高分。",
    ),
    CalibrationPathSpec(
        path_id="local_hyde_answer",
        action_path=[QueryAction.LOCAL_SEARCH, QueryAction.HYDE_SEARCH, QueryAction.ANSWER],
        route_family="hyde_fallback",
        purpose="验证 HyDE fallback 必要和不必要时的 Reward 差异。",
    ),
    CalibrationPathSpec(
        path_id="local_ask",
        action_path=[QueryAction.LOCAL_SEARCH, QueryAction.ASK_CLARIFICATION],
        route_family="ask_clarification",
        purpose="验证检索后追问是否只在歧义场景拿高分。",
    ),
    CalibrationPathSpec(
        path_id="local_refuse",
        action_path=[QueryAction.LOCAL_SEARCH, QueryAction.REFUSE],
        route_family="refuse",
        purpose="验证检索后拒答是否只在证据不足场景拿高分。",
    ),
    CalibrationPathSpec(
        path_id="web_answer",
        action_path=[QueryAction.WEB_SEARCH, QueryAction.ANSWER],
        route_family="web_search",
        purpose="验证直接 Web 路线是否只在允许实时信息时合理。",
    ),
    CalibrationPathSpec(
        path_id="ask_direct",
        action_path=[QueryAction.ASK_CLARIFICATION],
        route_family="ask_clarification",
        purpose="验证缺少主体或上下文时直接追问是否高于乱检索。",
    ),
    CalibrationPathSpec(
        path_id="refuse_direct",
        action_path=[QueryAction.REFUSE],
        route_family="refuse",
        purpose="验证不可回答问题直接拒答是否高于乱检索或乱回答。",
    ),
)


FALLBACK_PATH_SPECS: tuple[CalibrationPathSpec, ...] = (
    CalibrationPathSpec(
        path_id="local_web_answer",
        action_path=[QueryAction.LOCAL_SEARCH, QueryAction.WEB_SEARCH, QueryAction.ANSWER],
        route_family="web_fallback",
        purpose="验证本地不足后 Web fallback 的得分边界。",
    ),
    CalibrationPathSpec(
        path_id="local_hyde_web_answer",
        action_path=[
            QueryAction.LOCAL_SEARCH,
            QueryAction.HYDE_SEARCH,
            QueryAction.WEB_SEARCH,
            QueryAction.ANSWER,
        ],
        route_family="multi_step_fallback",
        purpose="验证多步 fallback 是否会因成本过高被合理扣分。",
    ),
    CalibrationPathSpec(
        path_id="local_hyde_refuse",
        action_path=[QueryAction.LOCAL_SEARCH, QueryAction.HYDE_SEARCH, QueryAction.REFUSE],
        route_family="multi_step_fallback",
        purpose="验证本地和 HyDE 都不足后拒答的得分边界。",
    ),
    CalibrationPathSpec(
        path_id="web_refuse",
        action_path=[QueryAction.WEB_SEARCH, QueryAction.REFUSE],
        route_family="web_search",
        purpose="验证 Web 无可靠证据后拒答的得分边界。",
    ),
)


def build_action_path_suite(case: PlannerEvalCase) -> list[CalibrationPathSpec]:
    """
    为单条 dev case 生成校准路线。

    基础路线所有 case 都执行；如果 case 期望 Web，或 acceptable_action_paths 中出现 HyDE/Web
    等 fallback 动作，再追加多步 fallback 路线。这样能满足每个 case 至少 6 条路线，同时
    不把所有 dev case 都膨胀成过多无意义 Web 组合。
    """
    paths = list(BASE_PATH_SPECS)
    acceptable_actions = {
        action
        for path in case.acceptable_action_paths
        for action in path
    }
    needs_fallback_probe = (
        case.expected_behavior.should_call_web
        or QueryAction.HYDE_SEARCH in acceptable_actions
        or QueryAction.WEB_SEARCH in acceptable_actions
    )
    if needs_fallback_probe:
        paths.extend(FALLBACK_PATH_SPECS)
    return paths


__all__ = [
    "ACTION_PATH_SUITE_VERSION",
    "BASE_PATH_SPECS",
    "FALLBACK_PATH_SPECS",
    "CalibrationPathSpec",
    "build_action_path_suite",
]

