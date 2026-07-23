"""模型版 Planner（规划器）适配器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from app.rag.query.contracts import PlannerContext, PlannerDecision


class ModelPlannerOutputError(ValueError):
    """模型输出无法解析为合法 PlannerDecision（规划器决策）。"""


class ModelDecisionGenerator(Protocol):
    """
    模型生成函数协议。

    Generator（生成器）输入已经构造好的 prompt（提示词）和 PlannerContext（规划器上下文），
    返回模型原始文本；它不执行 Milvus（向量数据库）、Web（网页检索）或答案生成。
    """

    def __call__(
            self,
            *,
            prompt: str,
            context: PlannerContext,
            prompt_hash: str,
            prompt_payload: dict[str, Any],
    ) -> str:
        ...


class ModelPlanner:
    """
    让 SFT/GRPO checkpoint（监督微调/强化训练检查点）符合 QueryPlanner 协议。

    ModelPlanner（模型规划器）只做决策：读取 PlannerContext（规划器上下文），调用模型生成
    JSON，再用 decision_codec（决策编解码器）校验成 PlannerDecision（规划器决策）。它
    不直接访问检索、数据库或 Web。
    """

    def __init__(
            self,
            *,
            policy_version: str,
            generate_text: ModelDecisionGenerator,
            metadata: dict[str, Any] | None = None,
    ) -> None:
        normalized_policy_version = str(policy_version or "").strip()
        if not normalized_policy_version:
            raise ValueError("policy_version 不能为空")
        self.policy_version = normalized_policy_version
        self._generate_text = generate_text
        self.metadata = dict(metadata or {})

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str | Path) -> "ModelPlanner":
        """
        从 checkpoint（检查点）创建 ModelPlanner（模型规划器）。

        这里延迟导入 evaluation.stage9，是为了让线上业务模块在未启用模型 Planner 时不会
        因训练依赖加载失败而影响规则 Planner。
        """

        from evaluation.stage9.model_planner.sft_infer import load_checkpoint_runtime

        runtime = load_checkpoint_runtime(checkpoint_dir)

        def _generate(
                *,
                prompt: str,
                context: PlannerContext,
                prompt_hash: str,
                prompt_payload: dict[str, Any],
        ) -> str:
            # runtime（运行时）会重新用结构化 payload 查找或生成；prompt 参数保留给未来
            # 接入直接文本生成的 checkpoint。
            del prompt, prompt_hash, prompt_payload
            return runtime.predict(context).raw_output

        return cls(
            policy_version=runtime.policy_version,
            generate_text=_generate,
            metadata=runtime.manifest.model_dump(mode="json"),
        )

    def plan(self, context: PlannerContext) -> PlannerDecision:
        """输出下一步 PlannerDecision（规划器决策）。"""

        if not isinstance(context, PlannerContext):
            raise TypeError("context 必须是 PlannerContext")

        from evaluation.stage9.model_planner.decision_codec import decode_decision
        from evaluation.stage9.model_planner.prompt_builder import build_planner_prompt

        prompt = build_planner_prompt(context)
        raw_output = self._generate_text(
            prompt=prompt.prompt,
            context=context,
            prompt_hash=prompt.payload_hash,
            prompt_payload=prompt.payload,
        )
        result = decode_decision(raw_output, allowed_actions=context.allowed_actions)
        if not result.success or result.decision is None:
            raise ModelPlannerOutputError(
                f"{result.error_code}: {result.error_message}; raw={result.raw_output_excerpt}"
            )
        return result.decision
