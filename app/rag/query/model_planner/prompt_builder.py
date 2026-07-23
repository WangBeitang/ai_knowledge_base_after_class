"""
Planner（规划器）训练和推理 Prompt（提示词）构造。

PromptBuilder（提示词构造器）只把 SftPlannerSample.input_context（监督微调输入上下文）
或 PlannerContext（规划器上下文）转换成模型可读文本，不读取 Milvus（向量数据库）、
Mongo（文档数据库）或 Web（网页检索）。
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.rag.query.contracts import PlannerContext, PlannerReasonCode, QueryAction


PROMPT_BUILDER_VERSION = "stage9-planner-prompt-v1"


class PlannerPromptModel(BaseModel):
    """Prompt（提示词）相关 schema（结构）基类。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


class PlannerPromptConfig(PlannerPromptModel):
    """
    Prompt（提示词）构造配置。

    max_input_chars（最大输入字符数）用于本地 smoke（冒烟）和无 tokenizer（分词器）场景；
    真正 transformers（大模型训练框架）训练时还会在 tokenizer 层再按 max_input_tokens
    （最大输入 token 数）截断。
    """

    max_input_chars: int = Field(default=12_000, ge=1)
    max_history_items: int = Field(default=8, ge=0)
    max_observation_items: int = Field(default=5, ge=0)
    max_text_field_chars: int = Field(default=800, ge=80)


class PlannerPrompt(PlannerPromptModel):
    """
    单次模型调用的 Prompt（提示词）产物。

    payload_hash（完整载荷哈希）用于审计 prompt（提示词）输入是否变化；context_key
    （上下文键）只使用运行时 PlannerContext（规划器上下文）能看到的字段，供本地 smoke
    （冒烟）checkpoint（检查点）做确定性查找。
    """

    prompt: str = Field(min_length=1)
    payload: dict[str, Any]
    payload_hash: str = Field(min_length=1)
    context_key: str = Field(min_length=1)
    prompt_builder_version: str = PROMPT_BUILDER_VERSION
    truncation_applied: bool = False


def build_planner_prompt(
        context: PlannerContext | dict[str, Any],
        config: PlannerPromptConfig | None = None,
) -> PlannerPrompt:
    """构造给 Planner（规划器）模型的稳定文本输入。"""

    active_config = config or PlannerPromptConfig()
    payload = context_to_prompt_payload(context, config=active_config)
    payload_hash = stable_payload_hash(payload)
    context_key = stable_context_key(payload)
    prompt, truncation_applied = _render_prompt(payload, active_config)
    return PlannerPrompt(
        prompt=prompt,
        payload=payload,
        payload_hash=payload_hash,
        context_key=context_key,
        truncation_applied=truncation_applied,
    )


def context_to_prompt_payload(
        context: PlannerContext | dict[str, Any],
        config: PlannerPromptConfig | None = None,
) -> dict[str, Any]:
    """
    把训练样本或运行时 PlannerContext（规划器上下文）规整成同一份 payload（结构载荷）。

    训练样本早期字段使用 query（用户问题），运行时契约使用 original_query（原始问题）。
    这里统一成 original_query/current_query，保证本地 smoke（冒烟）和云端训练入口读取同一
    份语义，而不是靠字段名差异分叉。
    """

    active_config = config or PlannerPromptConfig()
    raw = (
        context.model_dump(mode="json")
        if isinstance(context, PlannerContext)
        else copy.deepcopy(dict(context))
    )
    subject_ids = _text_list(raw.get("subject_ids"))
    subject_status = _text(
        raw.get("subject_resolution_status")
        or ("confirmed" if subject_ids else "no_mention")
    )
    allowed_actions = _action_values(raw.get("allowed_actions"))
    if QueryAction.REFUSE.value not in allowed_actions:
        allowed_actions.append(QueryAction.REFUSE.value)

    payload = {
        "original_query": _truncate(
            _text(raw.get("original_query") or raw.get("query")),
            active_config.max_text_field_chars,
        ),
        "current_query": _truncate(
            _text(raw.get("current_query") or raw.get("query") or raw.get("original_query")),
            active_config.max_text_field_chars,
        ),
        "dataset_ids": _text_list(raw.get("dataset_ids")),
        "subject_resolution_status": subject_status,
        "subject_ids": subject_ids,
        "subject_candidates": _text_list(raw.get("subject_candidates")),
        "standard_subject_names": _text_list(raw.get("standard_subject_names")),
        "clarification_question": _optional_text(raw.get("clarification_question")),
        "query_identifiers": _identifier_mapping(raw.get("query_identifiers")),
        "web_search_allowed": bool(raw.get("web_search_allowed", False)),
        "safe_guard_triggered": bool(raw.get("safe_guard_triggered", False)),
        "planner_step": _non_negative_int(raw.get("planner_step")),
        "max_steps": max(1, _non_negative_int(raw.get("max_steps") or raw.get("planner_max_steps") or 4)),
        "allowed_actions": allowed_actions,
        "action_history": _compact_history(
            raw.get("action_history"),
            max_items=active_config.max_history_items,
        ),
        "latest_observation": _compact_observation(
            raw.get("latest_observation"),
            max_items=active_config.max_observation_items,
            max_chars=active_config.max_text_field_chars,
        ),
    }
    return payload


def stable_payload_hash(payload: dict[str, Any]) -> str:
    """生成 payload（结构载荷）的稳定哈希，用于审计完整 prompt（提示词）输入。"""

    raw_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw_json.encode("utf-8")).hexdigest()


def stable_context_key(payload: dict[str, Any]) -> str:
    """
    生成运行时上下文键。

    当前 PlannerContext（规划器上下文）还不携带 dataset_ids（知识库 ID 列表）和
    standard_subject_names（标准主体名），所以本地 smoke（冒烟）查找键不能依赖这些只在
    SFT 样本里存在的字段。真实 transformers（大模型训练框架）训练仍会看到完整 prompt。
    """

    runtime_visible_fields = (
        "original_query",
        "current_query",
        "subject_resolution_status",
        "subject_ids",
        "subject_candidates",
        "clarification_question",
        "query_identifiers",
        "web_search_allowed",
        "safe_guard_triggered",
        "planner_step",
        "max_steps",
        "allowed_actions",
        "action_history",
        "latest_observation",
    )
    runtime_payload = {
        field: payload.get(field)
        for field in runtime_visible_fields
    }
    raw_json = json.dumps(runtime_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw_json.encode("utf-8")).hexdigest()


def _render_prompt(payload: dict[str, Any], config: PlannerPromptConfig) -> tuple[str, bool]:
    reason_codes = [item.value for item in PlannerReasonCode]
    instruction = (
        "你是设备知识库 Planner（规划器）。你只选择下一步 Action（动作），不生成最终答案正文。\n"
        "输出必须是严格 JSON 对象，只允许 action、query、reason_code 三个字段。\n"
        "action 必须来自 allowed_actions（允许动作）；reason_code 必须来自 reason_code_options（原因码选项）。\n"
        "不要输出 Markdown、解释文字、私有思维链或多余字段。"
    )
    context_json = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    reason_json = json.dumps(reason_codes, ensure_ascii=False)
    prompt = (
        f"{instruction}\n\n"
        f"reason_code_options（原因码选项）:\n{reason_json}\n\n"
        f"context（上下文）:\n{context_json}\n\n"
        "只输出 JSON:"
    )
    if len(prompt) <= config.max_input_chars:
        return prompt, False
    clipped_context = context_json[: max(100, config.max_input_chars - len(instruction) - len(reason_json) - 80)]
    prompt = (
        f"{instruction}\n\n"
        f"reason_code_options（原因码选项）:\n{reason_json}\n\n"
        f"context（上下文，已截断）:\n{clipped_context}\n\n"
        "只输出 JSON:"
    )
    return prompt[:config.max_input_chars], True


def _compact_history(raw_history: Any, *, max_items: int) -> list[dict[str, Any]]:
    history = raw_history if isinstance(raw_history, list) else []
    compacted: list[dict[str, Any]] = []
    for index, item in enumerate(history[-max_items:] if max_items else [], start=1):
        data = item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item or {})
        decision = data.get("decision") if isinstance(data.get("decision"), dict) else data
        compacted.append({
            "step": _non_negative_int(data.get("step") or index),
            "action": _text(decision.get("action")),
            "execution_status": _text(data.get("execution_status") or data.get("status") or "completed"),
        })
    return compacted


def _compact_observation(raw_observation: Any, *, max_items: int, max_chars: int) -> dict[str, Any] | None:
    if raw_observation is None:
        return None
    data = raw_observation.model_dump(mode="json") if hasattr(raw_observation, "model_dump") else dict(raw_observation)
    summaries = data.get("evidence_summaries") if isinstance(data.get("evidence_summaries"), list) else []
    compact_summaries: list[dict[str, Any]] = []
    for item in summaries[:max_items]:
        summary = item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item or {})
        compact_summaries.append({
            "document_id": _optional_text(summary.get("document_id")),
            "chunk_id": summary.get("chunk_id"),
            "title": _truncate(_text(summary.get("title")), max_chars),
            "source_type": _text(summary.get("source_type")),
            "rerank_score": summary.get("rerank_score"),
            "matched_identifiers": _identifier_mapping(summary.get("matched_identifiers")),
            "content_excerpt": _truncate(_text(summary.get("content_excerpt")), max_chars),
        })
    return {
        "action": _text(data.get("action")),
        "status": _text(data.get("status")),
        "candidate_count": _non_negative_int(data.get("candidate_count")),
        "reranked_count": _non_negative_int(data.get("reranked_count")),
        "top_rerank_score": data.get("top_rerank_score"),
        "requested_identifiers": _identifier_mapping(data.get("requested_identifiers")),
        "matched_identifiers": _identifier_mapping(data.get("matched_identifiers")),
        "identifier_resolution_status": _text(data.get("identifier_resolution_status")),
        "clarification_question": _optional_text(data.get("clarification_question")),
        "retrieved_chunk_ids": data.get("retrieved_chunk_ids") or [],
        "citation_chunk_ids": data.get("citation_chunk_ids") or [],
        "contains_full_chunk_content": bool(data.get("contains_full_chunk_content", False)),
        "evidence_summaries": compact_summaries,
    }


def _identifier_mapping(raw_mapping: Any) -> dict[str, list[str]]:
    if not isinstance(raw_mapping, dict):
        return {}
    normalized: dict[str, list[str]] = {}
    for key, values in raw_mapping.items():
        clean_key = _text(key)
        if not clean_key:
            continue
        clean_values = _text_list(values)
        if clean_values:
            normalized[clean_key] = clean_values
    return normalized


def _action_values(raw_actions: Any) -> list[str]:
    values = _text_list(raw_actions)
    return values or [action.value for action in QueryAction]


def _text_list(raw_values: Any) -> list[str]:
    if raw_values is None:
        return []
    values = raw_values if isinstance(raw_values, list) else [raw_values]
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _text(value.value if hasattr(value, "value") else value)
        if text and text not in seen:
            normalized.append(text)
            seen.add(text)
    return normalized


def _optional_text(value: Any) -> str | None:
    text = _text(value)
    return text or None


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 20)] + "...[truncated]"


def _non_negative_int(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, number)
