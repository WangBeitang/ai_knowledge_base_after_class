"""
阶段 9 PlannerDecision（规划器决策）JSON 编解码。

Codec（编解码器）的边界是模型输出和业务契约之间的闸门：训练目标写成最小 JSON，
推理输出必须重新解析并校验，非法 Action（动作）不会进入离线环境或线上路由。
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

from app.rag.query.contracts import PlannerDecision, PlannerReasonCode, QueryAction


DECISION_CODEC_VERSION = "stage9-decision-codec-v1"
_DECISION_FIELDS = ("action", "query", "reason_code")


class DecisionCodecModel(BaseModel):
    """Codec（编解码器）内部 schema（结构）基类，拒绝未知字段防止结果漂移。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


class DecisionDecodeResult(DecisionCodecModel):
    """
    模型输出解析结果。

    success（是否成功）为 False 时，decision（规划器决策）必须为空，调用方应把它记录成
    结构化 Planner（规划器）输出错误，而不是继续执行坏 Action（动作）。
    """

    success: bool
    decision: PlannerDecision | None = None
    normalized_json: str = ""
    raw_output_excerpt: str = Field(default="", max_length=500)
    error_code: str = ""
    error_message: str = ""


def encode_decision(decision: PlannerDecision | dict[str, Any]) -> str:
    """
    把 PlannerDecision（规划器决策）写成最小 JSON。

    只保留 action（动作）、query（动作查询文本）、reason_code（原因码）三个字段，训练
    目标不包含最终答案、解释性长文本或模型私有思维链。
    """

    validated = decision if isinstance(decision, PlannerDecision) else PlannerDecision.model_validate(decision)
    payload = validated.model_dump(mode="json")
    minimal_payload = {field: payload[field] for field in _DECISION_FIELDS}
    return json.dumps(minimal_payload, ensure_ascii=False, separators=(",", ":"))


def decode_decision(
        raw_output: str,
        *,
        allowed_actions: Iterable[QueryAction | str] | None = None,
        allow_json_object_extraction: bool = True,
) -> DecisionDecodeResult:
    """
    从模型原始输出解析 PlannerDecision（规划器决策）。

    allowed_actions（允许动作）来自 PlannerContext（规划器上下文），用于阻止模型输出当前
    State（状态）不可执行的 Action（动作）。allow_json_object_extraction（允许抽取 JSON
    对象）用于处理模型偶尔回显 prompt（提示词）或包上代码块的情况，但抽出的对象仍必须
    只有三个合法字段。
    """

    raw_text = str(raw_output or "").strip()
    if not raw_text:
        return _decode_error(
            "empty_output",
            "模型输出为空，无法解析 PlannerDecision",
            raw_output,
        )

    try:
        payload = _loads_json_object(raw_text, allow_extraction=allow_json_object_extraction)
    except ValueError as exc:
        return _decode_error("json_parse_failed", str(exc), raw_output)

    if not isinstance(payload, dict):
        return _decode_error("json_not_object", "模型输出 JSON 必须是对象", raw_output)

    fields = set(payload)
    expected_fields = set(_DECISION_FIELDS)
    if fields != expected_fields:
        unknown = sorted(fields - expected_fields)
        missing = sorted(expected_fields - fields)
        details: list[str] = []
        if unknown:
            details.append(f"未知字段：{', '.join(unknown)}")
        if missing:
            details.append(f"缺少字段：{', '.join(missing)}")
        return _decode_error("decision_fields_invalid", "；".join(details), raw_output)

    action = str(payload.get("action", "")).strip()
    allowed_action_values = _allowed_action_values(allowed_actions)
    if allowed_action_values is not None and action not in allowed_action_values:
        return _decode_error(
            "action_not_allowed",
            f"Action={action} 不在当前 allowed_actions 中",
            raw_output,
        )

    reason_code = str(payload.get("reason_code", "")).strip()
    if reason_code not in {item.value for item in PlannerReasonCode}:
        return _decode_error(
            "reason_code_unknown",
            f"未知 reason_code={reason_code}",
            raw_output,
        )

    try:
        decision = PlannerDecision.model_validate(payload)
    except Exception as exc:
        return _decode_error("planner_decision_invalid", str(exc), raw_output)

    return DecisionDecodeResult(
        success=True,
        decision=decision,
        normalized_json=encode_decision(decision),
        raw_output_excerpt=_excerpt(raw_output),
    )


def _loads_json_object(raw_text: str, *, allow_extraction: bool) -> Any:
    stripped = _strip_code_fence(raw_text)
    if not allow_extraction:
        return json.loads(stripped)

    decoder = json.JSONDecoder()
    start = stripped.find("{")
    if start < 0:
        raise ValueError("模型输出中没有 JSON 对象起始符 {")
    try:
        payload, _ = decoder.raw_decode(stripped[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"模型输出不是合法 JSON：{exc}") from exc
    return payload


def _strip_code_fence(raw_text: str) -> str:
    lines = raw_text.strip().splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return raw_text.strip()


def _allowed_action_values(
        allowed_actions: Iterable[QueryAction | str] | None,
) -> set[str] | None:
    if allowed_actions is None:
        return None
    values = {
        action.value if isinstance(action, QueryAction) else str(action).strip()
        for action in allowed_actions
    }
    return {value for value in values if value}


def _decode_error(error_code: str, error_message: str, raw_output: str) -> DecisionDecodeResult:
    return DecisionDecodeResult(
        success=False,
        error_code=error_code,
        error_message=error_message,
        raw_output_excerpt=_excerpt(raw_output),
    )


def _excerpt(raw_output: str) -> str:
    return str(raw_output or "").strip()[:500]
