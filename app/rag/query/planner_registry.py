"""Planner registry（规划器注册表）与运行时选择。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from app.rag.query.config import (
    RERANK_EVIDENCE_THRESHOLD,
    RETRIEVAL_CONFIG_VERSION,
)
from app.rag.query.contracts import PlannerContext, PlannerDecision
from app.rag.query.model_planner import PlannerClient, PROMPT_BUILDER_VERSION
from app.rag.query.planner import QueryPlanner, RuleBasedPlanner, RuleBasedPlannerConfig
from app.shared.config.planner_model_config import PlannerModelConfig, planner_model_config


class PlannerMode(str, Enum):
    """
    Planner mode（规划器模式）枚举。

    同一条线上查询只会使用一个 mode（模式）；保留多个成员是为了 baseline（基线对比）、
    rollback（回滚）和阶段推进，而不是让一次请求混用多个 Planner（规划器）。
    """

    RULE = "rule"  # rule（规则）：确定性 RuleBasedPlanner（规则规划器）。
    LOCAL_BASE = "local_base"  # local_base（未微调基础模型）：同一 HTTP 接口调用基础模型。
    SFT = "sft"  # sft（监督微调模型）：同一 HTTP 接口调用 SFT checkpoint（检查点）。
    GRPO = "grpo"  # grpo（强化训练模型）：同一 HTTP 接口调用 GRPO checkpoint（检查点）。
    HTTP_MOCK = "http_mock"  # http_mock（HTTP 模拟）：仅用于本地协议和错误边界测试。


MODEL_PLANNER_MODES = {
    PlannerMode.LOCAL_BASE,
    PlannerMode.SFT,
    PlannerMode.GRPO,
    PlannerMode.HTTP_MOCK,
}


class PlannerRegistryError(RuntimeError):
    """planner_registry（规划器注册表）无法选择可用 Planner（规划器）时抛出的结构化错误。"""

    def __init__(
            self,
            error_code: str,
            message: str,
            *,
            planner_mode: str = "",
            details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{error_code}: {message}")
        self.error_code = error_code
        self.message = message
        self.planner_mode = planner_mode
        self.details = dict(details or {})


@dataclass(frozen=True, slots=True)
class PlannerRegistryItem:
    """
    一个已注册 Planner（规划器）实现的状态摘要。

    enabled_online（是否可用于线上）只表示当前配置能否真实执行；未选中的模型 mode（模式）
    可以已注册但不可用，原因写入 unavailable_reason（不可用原因），供管理接口和评测脚本展示。
    """

    planner_mode: str
    planner_type: str
    enabled_online: bool
    enabled_for_eval: bool
    unavailable_reason: str = ""
    policy_version: str = ""
    provider: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    prompt_version: str | None = None
    endpoint: str | None = None

    def model_dump(self) -> dict[str, Any]:
        """返回可直接进入 API JSON（结构化响应）的字典。"""

        return {
            "planner_mode": self.planner_mode,
            "planner_type": self.planner_type,
            "enabled_online": self.enabled_online,
            "enabled_for_eval": self.enabled_for_eval,
            "unavailable_reason": self.unavailable_reason,
            "policy_version": self.policy_version,
            "provider": self.provider,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "prompt_version": self.prompt_version,
            "endpoint": self.endpoint,
        }


@dataclass(frozen=True, slots=True)
class PlannerRuntime:
    """
    当前被选中的 Planner（规划器）运行时。

    planner（规划器实例）只负责 plan(context)（根据上下文决策）；mode/type/provider/model_id
    等 runtime metadata（运行元数据）由 registry（注册表）统一提供，避免查询节点靠类名猜测。
    """

    mode: PlannerMode
    planner: QueryPlanner
    planner_type: str
    policy_version: str
    retrieval_config_version: str
    provider: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    prompt_version: str | None = None
    endpoint: str | None = None
    realtime_rule_version: str | None = None

    def plan(self, context: PlannerContext) -> PlannerDecision:
        """调用当前 Planner（规划器）并返回 PlannerDecision（规划器决策）。"""

        return self.planner.plan(context)

    def runtime_metadata(
            self,
            *,
            duration_ms: int,
            error_code: str = "",
            error_message: str = "",
    ) -> dict[str, object]:
        """
        构造 Planner runtime metadata（规划器运行元数据）。

        9.3.3 尚未接入模型服务 token usage（token 用量）返回，因此 token 和费用先固定为 0；
        error_code/error_message（错误码/错误信息）只记录结构化失败原因，不保存模型思维链。
        """

        return {
            "planner_mode": self.mode.value,
            "provider": self.provider,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "prompt_version": self.prompt_version,
            "endpoint": self.endpoint,
            "realtime_rule_version": self.realtime_rule_version,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost": 0,
            "duration_ms": max(0, int(duration_ms)),
            "error_code": error_code,
            "error_message": error_message,
        }

    def registry_item(self) -> PlannerRegistryItem:
        """把当前 runtime（运行时）投影成注册表条目。"""

        return PlannerRegistryItem(
            planner_mode=self.mode.value,
            planner_type=self.planner_type,
            enabled_online=True,
            enabled_for_eval=True,
            policy_version=self.policy_version,
            provider=self.provider,
            model_id=self.model_id,
            model_revision=self.model_revision,
            prompt_version=self.prompt_version,
            endpoint=self.endpoint,
        )


def normalize_planner_mode(raw_mode: str | PlannerMode | None) -> PlannerMode:
    """校验并规范化 PLANNER_MODE（规划器模式）。"""

    if isinstance(raw_mode, PlannerMode):
        return raw_mode
    value = str(raw_mode or PlannerMode.RULE.value).strip().lower()
    try:
        return PlannerMode(value)
    except ValueError as exc:
        supported = ", ".join(mode.value for mode in PlannerMode)
        raise PlannerRegistryError(
            "planner_mode_unknown",
            f"不支持的 PLANNER_MODE={raw_mode!r}，可选值：{supported}",
            planner_mode=str(raw_mode or ""),
        ) from exc


def build_rule_planner() -> RuleBasedPlanner:
    """构造默认 RuleBasedPlanner（规则规划器）。"""

    return RuleBasedPlanner(
        config=RuleBasedPlannerConfig(
            rerank_evidence_threshold=RERANK_EVIDENCE_THRESHOLD,
            retrieval_config_version=RETRIEVAL_CONFIG_VERSION,
        )
    )


def build_planner_runtime(
        *,
        mode: PlannerMode | str,
        config: PlannerModelConfig | None = None,
) -> PlannerRuntime:
    """
    构造某个 mode（模式）的 PlannerRuntime（规划器运行时）。

    rule（规则）不依赖模型服务；local_base/sft/grpo/http_mock（基础模型/监督微调/强化训练/
    HTTP 模拟）当前统一通过 PlannerClient（规划器客户端）调用 OpenAI-compatible（兼容
    OpenAI）的 HTTP 模型服务。
    """

    resolved_mode = normalize_planner_mode(mode)
    resolved_config = config or planner_model_config
    if resolved_mode == PlannerMode.RULE:
        planner = build_rule_planner()
        return PlannerRuntime(
            mode=resolved_mode,
            planner=planner,
            planner_type="rule",
            policy_version=planner.policy_version,
            retrieval_config_version=planner.config.retrieval_config_version,
            realtime_rule_version=planner.realtime_rule_version,
        )
    if resolved_mode in MODEL_PLANNER_MODES:
        endpoint = str(resolved_config.planner_model_endpoint or "").strip()
        model_id = str(resolved_config.planner_model_id or "").strip()
        if not endpoint:
            raise PlannerRegistryError(
                "planner_endpoint_missing",
                "模型 Planner 需要配置 PLANNER_MODEL_ENDPOINT",
                planner_mode=resolved_mode.value,
            )
        if not model_id:
            raise PlannerRegistryError(
                "planner_model_id_missing",
                "模型 Planner 需要配置 PLANNER_MODEL_ID",
                planner_mode=resolved_mode.value,
            )
        client = PlannerClient(config=resolved_config)
        return PlannerRuntime(
            mode=resolved_mode,
            planner=client,
            planner_type="model",
            policy_version=f"{resolved_mode.value}:{client.policy_version}",
            retrieval_config_version=RETRIEVAL_CONFIG_VERSION,
            provider="http",
            model_id=model_id,
            prompt_version=PROMPT_BUILDER_VERSION,
            endpoint=endpoint,
        )
    raise PlannerRegistryError(
        "planner_mode_unhandled",
        f"PLANNER_MODE={resolved_mode.value} 尚未实现",
        planner_mode=resolved_mode.value,
    )


def get_current_planner_runtime(config: PlannerModelConfig | None = None) -> PlannerRuntime:
    """根据当前配置返回业务查询节点应使用的唯一 PlannerRuntime（规划器运行时）。"""

    resolved_config = config or planner_model_config
    return build_planner_runtime(mode=resolved_config.planner_mode, config=resolved_config)


def get_registered_planners(config: PlannerModelConfig | None = None) -> list[PlannerRegistryItem]:
    """返回所有已知 Planner（规划器）模式的可用性摘要。"""

    resolved_config = config or planner_model_config
    try:
        current_mode: PlannerMode | None = normalize_planner_mode(resolved_config.planner_mode)
    except PlannerRegistryError:
        current_mode = None
    items: list[PlannerRegistryItem] = []
    for mode in PlannerMode:
        try:
            runtime = build_planner_runtime(mode=mode, config=resolved_config)
        except PlannerRegistryError as error:
            items.append(PlannerRegistryItem(
                planner_mode=mode.value,
                planner_type="model" if mode in MODEL_PLANNER_MODES else "rule",
                enabled_online=False,
                enabled_for_eval=False,
                unavailable_reason=error.message,
                provider="http" if mode in MODEL_PLANNER_MODES else None,
                model_id=resolved_config.planner_model_id if mode in MODEL_PLANNER_MODES else None,
                prompt_version=PROMPT_BUILDER_VERSION if mode in MODEL_PLANNER_MODES else None,
                endpoint=resolved_config.planner_model_endpoint if mode in MODEL_PLANNER_MODES else None,
            ))
            continue
        item = runtime.registry_item()
        items.append(PlannerRegistryItem(
            planner_mode=item.planner_mode,
            planner_type=item.planner_type,
            # 线上只启用当前 mode（模式）；其他可构造 mode 用于评测和受控切换。
            enabled_online=(mode == current_mode),
            enabled_for_eval=True,
            unavailable_reason="",
            policy_version=item.policy_version,
            provider=item.provider,
            model_id=item.model_id,
            model_revision=item.model_revision,
            prompt_version=item.prompt_version,
            endpoint=item.endpoint,
        ))
    return items


def get_planner_registry_status(config: PlannerModelConfig | None = None) -> dict[str, object]:
    """返回 /planner/status（规划器状态接口）使用的 registry（注册表）摘要。"""

    resolved_config = config or planner_model_config
    try:
        current_runtime = get_current_planner_runtime(resolved_config)
    except PlannerRegistryError as error:
        return {
            "online_mode": error.planner_mode or str(resolved_config.planner_mode or ""),
            "planner_type": "unavailable",
            "policy_version": "planner-registry-error",
            "provider": None,
            "model_id": None,
            "model_revision": None,
            "prompt_version": None,
            "endpoint": None,
            "current_unavailable_reason": error.message,
            "registered_planners": [
                item.model_dump() for item in get_registered_planners(resolved_config)
            ],
        }
    return {
        "online_mode": current_runtime.mode.value,
        "planner_type": current_runtime.planner_type,
        "policy_version": current_runtime.policy_version,
        "provider": current_runtime.provider,
        "model_id": current_runtime.model_id,
        "model_revision": current_runtime.model_revision,
        "prompt_version": current_runtime.prompt_version,
        "endpoint": current_runtime.endpoint,
        "current_unavailable_reason": "",
        "registered_planners": [
            item.model_dump() for item in get_registered_planners(resolved_config)
        ],
    }
