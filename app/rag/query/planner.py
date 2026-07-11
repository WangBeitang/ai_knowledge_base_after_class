"""查询 Planner 的可替换接口。"""

from typing import Protocol, runtime_checkable

from app.rag.query.contracts import PlannerContext, PlannerDecision


# 规则策略的版本号先作为稳定协议落地。具体 RuleBasedPlanner 决策规则属于阶段 5
# 后续任务，不能在契约阶段放一个永远抛 NotImplementedError 的“假实现”。
RULE_BASED_POLICY_VERSION = "rule-v1"


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

