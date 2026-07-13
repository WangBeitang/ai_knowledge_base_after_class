"""查询 Planner 的可替换接口和阶段 5 默认规则策略。"""

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.rag.query.contracts import (
    IdentifierResolutionStatus,
    ObservationStatus,
    PlannerContext,
    PlannerDecision,
    PlannerExecutionStatus,
    PlannerReasonCode,
    QueryAction,
    SubjectResolutionStatus,
)
from app.rag.query.config import POLICY_VERSION, REALTIME_PATTERNS_VERSION


# policy version 的中文含义是“策略版本”。只要规则顺序、阈值含义或实时判断发生变化，
# 就应升级版本，使 Trace 和后续训练数据能准确知道当时使用了哪套决策策略。
RULE_BASED_POLICY_VERSION = POLICY_VERSION

# realtime rule version 的中文含义是“实时问题识别规则版本”。它与 Planner 总策略版本
# 分开，是因为实时关键词可以独立迭代；后续 Trace 应同时记录两个版本。
REALTIME_RULE_VERSION = REALTIME_PATTERNS_VERSION

# 这里只识别“明显需要外部最新信息”的保守模式。单独出现“当前报警”或“最新维修方法”
# 不足以认定必须联网，避免设备本地知识问题被关键词误导到 Web。
_REALTIME_QUERY_PATTERNS = (
    re.compile(
        r"(?:今天|今日|刚刚|最新|实时|截至|本周|本月|今年)"
        r".{0,20}(?:公告|通知|召回|新闻|价格|状态|进展|版本|更新|发布|天气|汇率|股价)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:公告|通知|召回|新闻|价格|状态|进展|版本|更新|发布|天气|汇率|股价)"
        r".{0,20}(?:今天|今日|刚刚|最新|实时|截至|本周|本月|今年)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:联网|网上|官网|网页|web)\s*.{0,8}(?:搜索|查询|查找|查一下|看一下)",
        re.IGNORECASE,
    ),
)

# 检索 Action 只能沿单向升级路径前进。外层累计融合可以是 local -> Web，也可以是
# local -> HyDE -> Web，但 Web 执行后不能再退回 HyDE；answer/追问/拒答都是终止动作。
_VALID_HISTORY_TRANSITIONS = {
    QueryAction.LOCAL_SEARCH: {
        QueryAction.HYDE_SEARCH,
        QueryAction.WEB_SEARCH,
        QueryAction.ANSWER,
        QueryAction.ASK_CLARIFICATION,
        QueryAction.REFUSE,
    },
    QueryAction.HYDE_SEARCH: {
        QueryAction.WEB_SEARCH,
        QueryAction.ANSWER,
        QueryAction.ASK_CLARIFICATION,
        QueryAction.REFUSE,
    },
    QueryAction.WEB_SEARCH: {
        QueryAction.ANSWER,
        QueryAction.ASK_CLARIFICATION,
        QueryAction.REFUSE,
    },
}
_RETRIEVAL_ACTIONS = {
    QueryAction.LOCAL_SEARCH,
    QueryAction.HYDE_SEARCH,
    QueryAction.WEB_SEARCH,
}
_TERMINAL_ACTIONS = {
    QueryAction.ANSWER,
    QueryAction.ASK_CLARIFICATION,
    QueryAction.REFUSE,
}
_ANSWERABLE_IDENTIFIER_STATUSES = {
    IdentifierResolutionStatus.NOT_APPLICABLE,
    IdentifierResolutionStatus.EXACT_MATCH,
    IdentifierResolutionStatus.FALLBACK_EXACT_MATCH,
}


@dataclass(frozen=True, slots=True)
class RuleBasedPlannerConfig:
    """
    RuleBasedPlanner 的版本化证据阈值配置。

    ``rerank_evidence_threshold`` 是“允许进入 answer 的最低 reranker 分数”，范围 0～1。
    本类故意不提供默认值：该阈值必须来自设备运维开发集标注，而不能由代码作者凭感觉
    写死。``retrieval_config_version`` 记录这次阈值属于哪版检索配置，便于 Trace 重放。
    """

    rerank_evidence_threshold: float
    retrieval_config_version: str

    def __post_init__(self) -> None:
        threshold = self.rerank_evidence_threshold
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("rerank_evidence_threshold 必须是 0～1 的数字")
        if not 0 <= float(threshold) <= 1:
            raise ValueError("rerank_evidence_threshold 必须位于 0～1")
        normalized_version = str(self.retrieval_config_version or "").strip()
        if not normalized_version:
            raise ValueError("retrieval_config_version 不能为空")
        # frozen dataclass 仍可在初始化校验阶段使用 object.__setattr__ 写入规范值；之后
        # 配置不可变，避免一次 Trace 的决策过程中阈值或版本被其他代码动态修改。
        object.__setattr__(self, "rerank_evidence_threshold", float(threshold))
        object.__setattr__(self, "retrieval_config_version", normalized_version)


@runtime_checkable
class QueryPlanner(Protocol):
    """
    所有查询策略必须满足的最小接口。

    Protocol 是 Python 的结构化接口：实现类不需要显式继承，只要提供 ``policy_version``
    和签名一致的 ``plan`` 方法即可。这样规则 Planner 与后续模型 Planner 可以独立实现，
    LangGraph 和执行节点只依赖接口，不依赖具体策略类或训练框架。

    ``plan`` 必须是无外部 I/O 的决策函数。Milvus、Mongo、HTTP、LLM 工具执行都属于
    环境节点；Planner 只读取 ``PlannerContext`` 并返回经过校验的 ``PlannerDecision``。
    """

    policy_version: str

    def plan(self, context: PlannerContext) -> PlannerDecision:
        """根据当前结构化上下文选择下一步 Action。"""
        ...


class RuleBasedPlanner:
    """
    无外部 I/O、可稳定重放的默认查询策略。

    Planner（规划器）只做决策，不执行 Action：它不会访问 Milvus、Mongo、HTTP、LLM，
    也不会修改传入的 Context。相同 policy/config、相同 Context 必须返回内容相同的
    PlannerDecision，后续 LangGraph 节点再按 decision.action 执行真实检索或答案生成。
    """

    realtime_rule_version = REALTIME_RULE_VERSION

    def __init__(
            self,
            *,
            config: RuleBasedPlannerConfig,
            policy_version: str = RULE_BASED_POLICY_VERSION,
    ) -> None:
        normalized_policy_version = str(policy_version or "").strip()
        if not normalized_policy_version:
            raise ValueError("policy_version 不能为空")
        self.config = config
        self.policy_version = normalized_policy_version

    @staticmethod
    def is_realtime_query(context: PlannerContext) -> bool:
        """使用版本化保守正则判断问题是否明显依赖外部最新信息。"""
        query_text = f"{context.original_query}\n{context.current_query}"
        return any(pattern.search(query_text) for pattern in _REALTIME_QUERY_PATTERNS)

    def plan(self, context: PlannerContext) -> PlannerDecision:
        """按固定优先级选择下一步 Action；任何非法状态都收口到安全拒答。"""
        if not isinstance(context, PlannerContext):
            raise TypeError("context 必须是 PlannerContext")

        # 安全约束和最大步骤优先级最高：一旦触发，不允许后面的证据规则重新放行 answer。
        if context.safe_guard_triggered or context.planner_step >= context.max_steps:
            return self._refuse(context, PlannerReasonCode.SAFE_GUARD_TRIGGERED)

        # history（动作历史）是 Planner 防循环的事实来源。步骤断裂、重复检索、Web 后退回
        # HyDE、Observation 与最新 Action 不一致，都视为不可安全解释的非法转移。
        if not self._history_is_valid(context):
            return self._refuse(context, PlannerReasonCode.SAFE_GUARD_TRIGGERED)

        executed_actions = {item.decision.action for item in context.action_history}
        if executed_actions.intersection(_TERMINAL_ACTIONS):
            return self._refuse(context, PlannerReasonCode.SAFE_GUARD_TRIGGERED)

        subject_status = context.subject_resolution_status
        is_realtime = self.is_realtime_query(context)

        # 主体歧义/未提及时先问用户。即使问题包含“最新”，也不能在不知道设备是谁时
        # 直接把模糊问题发给 Web，避免扩大检索范围并返回无关设备信息。
        if subject_status == SubjectResolutionStatus.AMBIGUOUS:
            return self._decision(
                context,
                QueryAction.ASK_CLARIFICATION,
                PlannerReasonCode.SUBJECT_AMBIGUOUS,
            )
        if subject_status == SubjectResolutionStatus.NO_MENTION:
            return self._decision(
                context,
                QueryAction.ASK_CLARIFICATION,
                PlannerReasonCode.SUBJECT_REQUIRED,
            )

        # CONFIRMED 的业务含义就是已经得到稳定 subject_id。该状态却没有 ID 属于契约
        # 事实矛盾，即使问题看起来实时也不能继续，避免 Trace 记录“已确认”但无法说明主体。
        if subject_status == SubjectResolutionStatus.CONFIRMED and not context.subject_ids:
            return self._refuse(context, PlannerReasonCode.SAFE_GUARD_TRIGGERED)

        # 用户明确提到主体但本地无法映射时，普通知识问题直接拒答；明显实时问题仍允许
        # 走 Web，因为外部最新公告中的主体不一定已经存在于本地知识库。
        if subject_status == SubjectResolutionStatus.NOT_FOUND and not is_realtime:
            return self._refuse(context, PlannerReasonCode.SUBJECT_NOT_FOUND)

        if is_realtime and QueryAction.WEB_SEARCH not in executed_actions:
            if context.web_search_allowed:
                return self._decision(
                    context,
                    QueryAction.WEB_SEARCH,
                    PlannerReasonCode.REALTIME_QUERY,
                )
            return self._refuse(context, PlannerReasonCode.SAFE_GUARD_TRIGGERED)

        # 非实时本地问题的第一步固定为 local_search。CONFIRMED 却没有 subject_id 属于
        # 契约事实矛盾，不能退化为无 subject 的全库搜索。
        direct_realtime_web_completed = (
            is_realtime and QueryAction.WEB_SEARCH in executed_actions
        )
        if QueryAction.LOCAL_SEARCH not in executed_actions and not direct_realtime_web_completed:
            if subject_status != SubjectResolutionStatus.CONFIRMED:
                return self._refuse(context, PlannerReasonCode.SAFE_GUARD_TRIGGERED)
            return self._decision(
                context,
                QueryAction.LOCAL_SEARCH,
                PlannerReasonCode.INITIAL_LOCAL_SEARCH,
            )

        observation = context.latest_observation
        if observation is None:
            return self._refuse(context, PlannerReasonCode.ACTION_EXECUTION_ERROR)

        # 编号安全规则必须先于分数判断。即使 E021 候选分数很高，只要用户输入的是 E020，
        # SUGGESTION_REQUIRED 仍只能追问，不能被高 rerank 分“冲掉”。
        if observation.identifier_resolution_status == IdentifierResolutionStatus.SUGGESTION_REQUIRED:
            return self._decision(
                context,
                QueryAction.ASK_CLARIFICATION,
                PlannerReasonCode.IDENTIFIER_CONFIRMATION_REQUIRED,
            )
        if observation.identifier_resolution_status == IdentifierResolutionStatus.NOT_FOUND:
            return self._decision(
                context,
                QueryAction.ASK_CLARIFICATION,
                PlannerReasonCode.IDENTIFIER_NOT_FOUND,
            )
        if observation.evidence_ambiguous:
            return self._decision(
                context,
                QueryAction.ASK_CLARIFICATION,
                PlannerReasonCode.EVIDENCE_AMBIGUOUS,
            )

        if observation.status == ObservationStatus.FAILED:
            return self._after_failed_action(context, observation.action, executed_actions)

        if self._evidence_is_sufficient(observation):
            return self._decision(
                context,
                QueryAction.ANSWER,
                self._sufficient_evidence_reason(observation.action),
            )

        if observation.action == QueryAction.LOCAL_SEARCH:
            if observation.candidate_count > 0 and QueryAction.HYDE_SEARCH not in executed_actions:
                return self._decision(
                    context,
                    QueryAction.HYDE_SEARCH,
                    PlannerReasonCode.LOCAL_LOW_SCORE,
                )
            if context.web_search_allowed and QueryAction.WEB_SEARCH not in executed_actions:
                reason_code = (
                    PlannerReasonCode.LOCAL_EMPTY
                    if observation.candidate_count == 0
                    else PlannerReasonCode.LOCAL_LOW_SCORE
                )
                return self._decision(context, QueryAction.WEB_SEARCH, reason_code)
            reason_code = (
                PlannerReasonCode.LOCAL_EMPTY
                if observation.candidate_count == 0
                else PlannerReasonCode.LOCAL_LOW_SCORE
            )
            return self._refuse(context, reason_code)

        if observation.action == QueryAction.HYDE_SEARCH:
            if context.web_search_allowed and QueryAction.WEB_SEARCH not in executed_actions:
                return self._decision(
                    context,
                    QueryAction.WEB_SEARCH,
                    PlannerReasonCode.HYDE_STILL_INSUFFICIENT,
                )
            return self._refuse(context, PlannerReasonCode.HYDE_STILL_INSUFFICIENT)

        if observation.action == QueryAction.WEB_SEARCH:
            # Web 虽有候选但 rerank 仍低于阈值时也不能回答；现有 reason code 将空结果、
            # 执行失败和最终低质量统一归入“Web 不可用”的安全终止类别。
            return self._refuse(context, PlannerReasonCode.WEB_EMPTY_OR_FAILED)

        return self._refuse(context, PlannerReasonCode.ACTION_EXECUTION_ERROR)

    def _evidence_is_sufficient(self, observation) -> bool:
        """同时检查执行状态、候选、rerank 阈值和编号确认状态。"""
        return (
            observation.status == ObservationStatus.SUCCESS
            and observation.candidate_count > 0
            and observation.reranked_count > 0
            and observation.top_rerank_score is not None
            and observation.top_rerank_score >= self.config.rerank_evidence_threshold
            and observation.identifier_resolution_status in _ANSWERABLE_IDENTIFIER_STATUSES
        )

    @staticmethod
    def _sufficient_evidence_reason(action: QueryAction) -> PlannerReasonCode:
        """按最后产生充分证据的 Action 选择可聚合的 reason code。"""
        if action == QueryAction.LOCAL_SEARCH:
            return PlannerReasonCode.LOCAL_EVIDENCE_SUFFICIENT
        if action == QueryAction.HYDE_SEARCH:
            return PlannerReasonCode.HYDE_EVIDENCE_SUFFICIENT
        if action == QueryAction.WEB_SEARCH:
            return PlannerReasonCode.WEB_EVIDENCE_AVAILABLE
        return PlannerReasonCode.ACTION_EXECUTION_ERROR

    def _after_failed_action(
            self,
            context: PlannerContext,
            failed_action: QueryAction,
            executed_actions: set[QueryAction],
    ) -> PlannerDecision:
        """本地/HyDE 异常可降级 Web；Web 自身异常只能拒答。"""
        if failed_action == QueryAction.WEB_SEARCH:
            return self._refuse(context, PlannerReasonCode.WEB_EMPTY_OR_FAILED)
        if context.web_search_allowed and QueryAction.WEB_SEARCH not in executed_actions:
            return self._decision(
                context,
                QueryAction.WEB_SEARCH,
                PlannerReasonCode.ACTION_EXECUTION_ERROR,
            )
        return self._refuse(context, PlannerReasonCode.ACTION_EXECUTION_ERROR)

    @staticmethod
    def _history_is_valid(context: PlannerContext) -> bool:
        """验证历史步骤、单向 Action 转移和最新 Observation 的对应关系。"""
        history = context.action_history
        if context.planner_step != len(history):
            return False

        seen_retrieval_actions: set[QueryAction] = set()
        previous_action: QueryAction | None = None
        for expected_step, item in enumerate(history, start=1):
            action = item.decision.action
            if item.step != expected_step or action not in context.allowed_actions:
                return False
            if action in _RETRIEVAL_ACTIONS:
                if action in seen_retrieval_actions:
                    return False
                seen_retrieval_actions.add(action)
            if previous_action is None:
                if action not in {
                    QueryAction.LOCAL_SEARCH,
                    QueryAction.WEB_SEARCH,
                    QueryAction.ASK_CLARIFICATION,
                    QueryAction.REFUSE,
                }:
                    return False
            elif action not in _VALID_HISTORY_TRANSITIONS.get(previous_action, set()):
                return False
            previous_action = action

        observation = context.latest_observation
        if observation is None:
            return not history or history[-1].decision.action not in _RETRIEVAL_ACTIONS
        if not history or history[-1].decision.action != observation.action:
            return False
        latest_execution_status = history[-1].execution_status
        if observation.status == ObservationStatus.FAILED:
            return latest_execution_status == PlannerExecutionStatus.FAILED
        return latest_execution_status == PlannerExecutionStatus.COMPLETED

    def _decision(
            self,
            context: PlannerContext,
            action: QueryAction,
            reason_code: PlannerReasonCode,
    ) -> PlannerDecision:
        """创建强校验 Decision；目标 Action 不在白名单时统一安全拒答。"""
        if action not in context.allowed_actions:
            return self._refuse(context, PlannerReasonCode.SAFE_GUARD_TRIGGERED)
        return PlannerDecision(
            action=action,
            query=context.current_query,
            reason_code=reason_code,
        )

    @staticmethod
    def _refuse(
            context: PlannerContext,
            reason_code: PlannerReasonCode,
    ) -> PlannerDecision:
        """所有异常和非法转移共用的确定性安全出口。"""
        return PlannerDecision(
            action=QueryAction.REFUSE,
            query=context.current_query,
            reason_code=reason_code,
        )
