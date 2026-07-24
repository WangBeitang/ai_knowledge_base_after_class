"""模型 Planner（规划器）checkpoint（检查点）推理运行时。"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.rag.query.contracts import PlannerContext, PlannerDecision, PlannerReasonCode, QueryAction
from app.rag.query.model_planner.decision_codec import DecisionDecodeResult, decode_decision, encode_decision
from app.rag.query.model_planner.prompt_builder import PlannerPrompt, PlannerPromptConfig, build_planner_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[4]


class TrainingBackend(str, Enum):
    """
    训练后端枚举。

    DEBUG_MEMORIZED（调试记忆）只验证数据、prompt（提示词）、codec（编解码器）、
    checkpoint（检查点）和推理链路；TRANSFORMERS_CAUSAL_LM（transformers 因果语言模型）
    才是云端 GPU（显卡算力）实际训练入口。
    """

    DEBUG_MEMORIZED = "debug_memorized"
    TRANSFORMERS_CAUSAL_LM = "transformers_causal_lm"


class RuntimeModel(BaseModel):
    """Planner runtime（规划器运行时）schema（结构）基类。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


class ModelProfileSnapshot(RuntimeModel):
    """
    model profile（模型配置档案）快照。

    checkpoint manifest（检查点清单）保存的是训练当时的 profile 内容，而不是只保存
    profile_id（配置档案身份）。这样后续 profile 文件被修改时，旧 checkpoint 仍能说明
    自己当时到底以哪个模型、模板和 token（分词单元）边界训练。
    """

    profile_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    model_family: str = Field(min_length=1)
    role: str = Field(min_length=1)
    auto_train_enabled: bool
    base_model_id: str = Field(min_length=1)
    training_model_id: str = Field(min_length=1)
    serving_model_id: str = Field(min_length=1)
    parameter_count_b: float | None = Field(default=None, gt=0)
    chat_template: str = Field(min_length=1)
    enable_thinking: bool
    max_context_tokens: int = Field(ge=1)
    max_target_tokens: int = Field(ge=1)
    recommended_backend: str = Field(min_length=1)
    recommended_training_backend: str = ""
    recommended_inference_backend: str = ""
    quantization: str = ""
    license: str = ""


class CheckpointManifest(RuntimeModel):
    """
    checkpoint manifest（检查点清单）。

    它是训练产物的机器可读索引：后续 eval（评测）、SFT（监督微调）对比和 GRPO（组相对
    策略优化强化训练）必须从这里追踪模型、数据、Reward（奖励）和代码版本。该 schema
    放在 app（业务模块）下，是因为线上推理只需要读取 manifest（清单），不能依赖
    evaluation/stage9（阶段实验目录）。
    """

    run_id: str = Field(min_length=1)
    run_name: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    training_backend: TrainingBackend
    base_model_id: str = Field(min_length=1)
    model_profile_id: str = ""
    model_profile: ModelProfileSnapshot | None = None
    train_data: str = Field(min_length=1)
    train_manifest: str = Field(min_length=1)
    reward_profile: str = Field(min_length=1)
    snapshot_id: str = Field(min_length=1)
    code_version: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    seed: int = Field(ge=0)
    framework_versions: dict[str, str] = Field(default_factory=dict)
    prompt_builder_version: str = Field(min_length=1)
    decision_codec_version: str = Field(min_length=1)
    model_path: str = Field(min_length=1)
    tokenizer_path: str = ""
    train_metrics_path: str = Field(min_length=1)
    eval_metrics_path: str = ""
    training_config_path: str = Field(min_length=1)
    sample_count: int = Field(ge=0)
    source_case_count: int = Field(ge=0)
    action_counts: dict[str, int] = Field(default_factory=dict)
    reason_code_counts: dict[str, int] = Field(default_factory=dict)
    max_input_tokens: int = Field(ge=1)
    max_target_tokens: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_model_profile(self) -> "CheckpointManifest":
        """校验 checkpoint（检查点）记录的模型身份和 profile（配置档案）一致。"""

        if self.model_profile is None:
            return self
        if self.model_profile_id and self.model_profile.profile_id != self.model_profile_id:
            raise ValueError("model_profile_id 与 model_profile.profile_id 不一致")
        if self.model_profile.base_model_id != self.base_model_id:
            raise ValueError("base_model_id 与 model_profile.base_model_id 不一致")
        if self.max_target_tokens > self.model_profile.max_target_tokens:
            raise ValueError("max_target_tokens 不能超过 model_profile.max_target_tokens")
        if self.max_input_tokens > self.model_profile.max_context_tokens:
            raise ValueError("max_input_tokens 不能超过 model_profile.max_context_tokens")
        return self


class PlannerInferenceResult(RuntimeModel):
    """
    单次 Planner（规划器）推理结果。

    raw_output（原始输出）用于审计模型是否遵守 JSON（结构化数据）格式；decision（规划器
    决策）只有在 decode_result.success（解析成功）为 True 时才允许进入环境执行。
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
                _resolve_project_path(self.manifest.model_path, base_dir=self.checkpoint_dir)
                .joinpath("debug_memorized_policy.json")
                .read_text(encoding="utf-8")
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
            tokenizer_path = _resolve_project_path(self.manifest.tokenizer_path or self.manifest.model_path)
            model_path = _resolve_project_path(self.manifest.model_path)
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


def load_checkpoint_manifest(checkpoint_dir: str | Path) -> CheckpointManifest:
    """读取 checkpoint_manifest.json（检查点清单）。"""

    manifest_path = Path(checkpoint_dir) / "checkpoint_manifest.json"
    return CheckpointManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))


def load_checkpoint_runtime(checkpoint_dir: str | Path) -> PlannerCheckpointRuntime:
    """加载 checkpoint（检查点）运行时。"""

    return PlannerCheckpointRuntime(checkpoint_dir)


def _resolve_project_path(raw_path: str | Path, *, base_dir: Path | None = None) -> Path:
    """
    解析 checkpoint manifest（检查点清单）里的路径。

    训练脚本会优先写项目内相对路径；测试场景可能写临时目录绝对路径。base_dir（基础目录）
    只用于兼容少量 checkpoint 内部相对模型目录，常规项目相对路径仍以 PROJECT_ROOT
    （项目根目录）为准。
    """

    path = Path(raw_path)
    if path.is_absolute():
        return path
    project_path = PROJECT_ROOT / path
    if project_path.exists() or base_dir is None:
        return project_path
    return base_dir / path
