"""整轮生成式 LLM Token 可用性聚合（Stage 4 首轮复核决策 2）。

``token_usage`` 的定义是“本次 Query 能够证明完整的全部生成式 LLM Token”，
覆盖实际执行的：subject rewrite LLM、HyDE LLM、answer LLM。

规则：
- 所有实际执行的 LLM 调用都有可信 usage → 求和，available=true；
- 任一实际调用 usage 缺失/不完整 → 整轮 available=false（平台 input/output/total 全 NULL）；
- provider 明确返回真实 0 → 仍保存 0；
- partial usage 不能把缺失 input/output 擅自补 0 后宣称 available=true；
- 不做大规模重构；宁可 available=false，也不能低估成“真实 Token”。
"""

from __future__ import annotations

from typing import Any

from app.rag.query.contracts import UsageMetrics


def extract_usage_metadata(message) -> dict[str, int | bool]:
    """兼容不同 LangChain provider 的 token 用量字段，并区分“缺失”与“真实 0”。

    返回 dict 额外携带 ``answer_usage_available``：
    - provider 明确存在 usage metadata（任一 token 字段非 None，即使真实值为 0）→ True；
    - provider 完全没有返回 usage → False（数值 0 只是兼容占位，不是真实用量）。
    """
    usage = getattr(message, "usage_metadata", None) or {}
    response_metadata = getattr(message, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}

    def _first(*names: str) -> int | None:
        for source in (usage, token_usage):
            for name in names:
                value = source.get(name)
                if value is not None:
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        return None
        return None

    raw_input = _first("input_tokens", "prompt_tokens")
    raw_output = _first("output_tokens", "completion_tokens")
    raw_total = _first("total_tokens")

    available = raw_input is not None or raw_output is not None or raw_total is not None
    input_tokens = max(0, raw_input) if raw_input is not None else 0
    output_tokens = max(0, raw_output) if raw_output is not None else 0
    if raw_total is not None:
        total_tokens = max(0, raw_total)
    else:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "answer_usage_available": available,
    }


def _part_available(metadata: dict[str, Any] | None) -> bool:
    if not metadata:
        return False
    return bool(metadata.get("answer_usage_available", False))


def _part_numbers(metadata: dict[str, Any] | None) -> tuple[int, int, int]:
    metadata = dict(metadata or {})
    return (
        max(0, int(metadata.get("input_tokens") or 0)),
        max(0, int(metadata.get("output_tokens") or 0)),
        max(0, int(metadata.get("total_tokens") or 0)),
    )


def aggregate_query_usage(state: dict[str, Any]) -> UsageMetrics:
    """聚合整轮实际执行的生成式 LLM Token。

    参与者：
    - subject rewrite：每轮必执行（node_subject_name_confirm 是图入口）；
    - HyDE：仅当 state 已写入 hyde_usage_metadata（该检索节点实际被路由）时参与；
    - answer：确定性终态（澄清/拒答不调用答案模型）为确定 0（available=true）；
      真实调用时按 answer_runtime_metadata 的 availability。

    任一实际调用缺失/不可信 → 整轮 available=false（数值保留 0 仅为兼容占位）。
    """
    parts: list[tuple[bool, int, int, int]] = []

    # subject rewrite（必执行）
    subject_metadata = state.get("subject_rewrite_usage_metadata")
    if isinstance(subject_metadata, dict) and subject_metadata:
        parts.append(
            (_part_available(subject_metadata), *_part_numbers(subject_metadata))
        )
    else:
        # 缺少 subject usage（本轮没有捕获）：整轮不可证明
        return UsageMetrics(available=False)

    # HyDE（仅实际执行时写入非空 dict；default {} 表示未执行）
    hyde_metadata = state.get("hyde_usage_metadata")
    if isinstance(hyde_metadata, dict) and hyde_metadata:
        parts.append((_part_available(hyde_metadata), *_part_numbers(hyde_metadata)))
    elif hyde_metadata is not None and not isinstance(hyde_metadata, dict):
        # 结构异常：不可证明
        return UsageMetrics(available=False)

    # answer：确定性终态（无 answer_runtime_metadata）→ 确定 0；否则按实际
    answer_metadata = state.get("answer_runtime_metadata")
    if isinstance(answer_metadata, dict) and answer_metadata:
        parts.append(
            (_part_available(answer_metadata), *_part_numbers(answer_metadata))
        )
    else:
        # 澄清/拒答等确定性终态：答案模型调用本身不存在，token 为确定的 0
        parts.append((True, 0, 0, 0))

    if not all(available for available, _, _, _ in parts):
        return UsageMetrics(available=False)

    input_tokens = sum(part[1] for part in parts)
    output_tokens = sum(part[2] for part in parts)
    total_tokens = sum(part[3] for part in parts)
    return UsageMetrics(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        available=True,
    )
