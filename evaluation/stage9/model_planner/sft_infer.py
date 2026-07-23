"""阶段 9 Planner（规划器）checkpoint（检查点）推理入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.query.contracts import PlannerContext, PlannerDecision, PlannerReasonCode, QueryAction  # noqa: E402
from evaluation.stage9.model_planner.checkpoint_io import (  # noqa: E402
    CheckpointManifest,
    TrainingBackend,
    load_checkpoint_manifest,
)
from evaluation.stage9.model_planner.decision_codec import DecisionDecodeResult, decode_decision, encode_decision  # noqa: E402
from evaluation.stage9.model_planner.prompt_builder import PlannerPrompt, PlannerPromptConfig, build_planner_prompt  # noqa: E402


class InferenceModel(BaseModel):
    """inference（推理）内部 schema（结构）基类。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


class PlannerInferenceResult(InferenceModel):
    """
    单次 Planner（规划器）推理结果。

    raw_output（原始输出）用于审计模型是否遵守 JSON 格式；decision（规划器决策）只有在
    decode_result.success（解析成功）为 True 时才允许进入环境执行。
    """

    success: bool
    decision: PlannerDecision | None = None
    raw_output: str = ""
    decode_result: DecisionDecodeResult
    prompt_hash: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    training_backend: TrainingBackend


class PlannerCheckpointRuntime:
    """
    checkpoint（检查点）运行时。

    Runtime（运行时）只负责从 checkpoint 加载模型或调试策略并生成 JSON 字符串；Action
    （动作）合法性仍由 decision_codec（决策编解码器）按 allowed_actions（允许动作）校验。
    """

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.manifest = load_checkpoint_manifest(self.checkpoint_dir)
        self.prompt_config = PlannerPromptConfig()
        self._debug_policy: dict[str, Any] | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        if self.manifest.training_backend == TrainingBackend.DEBUG_MEMORIZED:
            self._debug_policy = json.loads(
                (PROJECT_ROOT / self.manifest.model_path / "debug_memorized_policy.json").read_text(encoding="utf-8")
            )

    @property
    def policy_version(self) -> str:
        return self.manifest.policy_version

    def predict(self, context: PlannerContext | dict[str, Any]) -> PlannerInferenceResult:
        """对单条上下文执行推理并解析成 PlannerDecision（规划器决策）。"""

        prompt = build_planner_prompt(context, config=self.prompt_config)
        raw_output = self.generate_text(prompt)
        allowed_actions = prompt.payload["allowed_actions"]
        decode_result = decode_decision(raw_output, allowed_actions=allowed_actions)
        return PlannerInferenceResult(
            success=decode_result.success,
            decision=decode_result.decision,
            raw_output=raw_output,
            decode_result=decode_result,
            prompt_hash=prompt.payload_hash,
            policy_version=self.policy_version,
            training_backend=self.manifest.training_backend,
        )

    def generate_text(self, prompt: PlannerPrompt) -> str:
        """生成原始 JSON 文本；debug 和 transformers 共用同一接口。"""

        if self.manifest.training_backend == TrainingBackend.DEBUG_MEMORIZED:
            return self._generate_debug_memorized(prompt)
        return self._generate_transformers(prompt)

    def _generate_debug_memorized(self, prompt: PlannerPrompt) -> str:
        policy = self._debug_policy or {}
        mapping = policy.get("context_key_to_target", {})
        matched = mapping.get(prompt.context_key)
        if matched:
            return str(matched["target_json"])
        fallback = PlannerDecision(
            action=QueryAction.REFUSE,
            query=str(prompt.payload.get("current_query") or prompt.payload.get("original_query") or "无法匹配上下文"),
            reason_code=PlannerReasonCode.SAFE_GUARD_TRIGGERED,
        )
        return encode_decision(fallback)

    def _generate_transformers(self, prompt: PlannerPrompt) -> str:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._tokenizer is None or self._model is None:
            tokenizer_path = PROJECT_ROOT / (self.manifest.tokenizer_path or self.manifest.model_path)
            model_path = PROJECT_ROOT / self.manifest.model_path
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self._model = AutoModelForCausalLM.from_pretrained(model_path)
            self._model.eval()
        tokenizer = self._tokenizer
        model = self._model
        inputs = tokenizer(prompt.prompt, return_tensors="pt", truncation=True, max_length=self.manifest.max_input_tokens)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.manifest.max_target_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def load_checkpoint_runtime(checkpoint_dir: str | Path) -> PlannerCheckpointRuntime:
    """加载 checkpoint（检查点）运行时。"""

    return PlannerCheckpointRuntime(checkpoint_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段 9 Planner checkpoint 推理。")
    parser.add_argument("--checkpoint", required=True, type=Path, help="checkpoint 目录。")
    parser.add_argument("--context-json", required=True, type=Path, help="PlannerContext 或 input_context JSON 文件。")
    args = parser.parse_args(argv)
    runtime = load_checkpoint_runtime(args.checkpoint)
    context_payload = json.loads(args.context_json.read_text(encoding="utf-8"))
    result = runtime.predict(context_payload)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
