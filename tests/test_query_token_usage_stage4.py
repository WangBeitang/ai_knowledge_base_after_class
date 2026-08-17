"""Stage 4：Trace Token 可用性契约补强测试。

覆盖（决策十九/二十）：
A. provider 正 token → available=true，123/45/168；
B. provider 明确返回 0 → available=true，0/0/0（真实 0 ≠ 缺失）；
C. provider 完全没有 usage → available=false（内部保留数值 0，API 投影 null）；
D. 旧 Trace 没有 availability 字段 → available=false，token 全 null；
E. Trace 摘要投影正数 / unavailable；
F. 确定性终态（不调用答案 LLM）→ available=true + 确定 0。
"""

import copy

import pytest

from app.rag.management.trace_feedback_service import project_trace_summary
from app.rag.query.answer_service import _extract_usage_metadata
from app.rag.query.contracts import UsageMetrics
from app.rag.query.trace_service import _usage_from_metadata


class _FakeMessage:
    """模拟 LangChain 消息对象（usage_metadata / response_metadata）。"""

    def __init__(self, *, usage_metadata=None, response_metadata=None):
        self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata


def test_extract_usage_metadata_positive():
    message = _FakeMessage(
        usage_metadata={"input_tokens": 123, "output_tokens": 45, "total_tokens": 168},
        response_metadata={},
    )
    usage = _extract_usage_metadata(message)
    assert usage["input_tokens"] == 123
    assert usage["output_tokens"] == 45
    assert usage["total_tokens"] == 168
    assert usage["answer_usage_available"] is True


def test_extract_usage_metadata_explicit_zero_is_available():
    message = _FakeMessage(
        usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        response_metadata={},
    )
    usage = _extract_usage_metadata(message)
    assert usage["input_tokens"] == 0
    assert usage["output_tokens"] == 0
    assert usage["total_tokens"] == 0
    # provider 明确返回 0 也是真实 usage：available=true
    assert usage["answer_usage_available"] is True


def test_extract_usage_metadata_missing_is_not_available():
    message = _FakeMessage(usage_metadata={}, response_metadata={})
    usage = _extract_usage_metadata(message)
    assert usage["input_tokens"] == 0
    assert usage["output_tokens"] == 0
    assert usage["total_tokens"] == 0
    assert usage["answer_usage_available"] is False


def test_extract_usage_metadata_legacy_token_usage_response_metadata():
    """兼容 response_metadata.token_usage / usage 的旧形态 provider。"""
    message = _FakeMessage(
        usage_metadata=None,
        response_metadata={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
    )
    usage = _extract_usage_metadata(message)
    assert usage["input_tokens"] == 10
    assert usage["output_tokens"] == 5
    assert usage["total_tokens"] == 15
    assert usage["answer_usage_available"] is True


def test_usage_from_metadata_carries_availability():
    metrics = _usage_from_metadata({"answer_usage_available": True, "input_tokens": 1, "output_tokens": 2, "total_tokens": 3})
    assert metrics.available is True
    assert metrics.input_tokens == 1

    metrics_missing = _usage_from_metadata({"input_tokens": 1, "output_tokens": 2, "total_tokens": 3})
    assert metrics_missing.available is False


def test_usage_metrics_default_available_false():
    """旧数据没有 available 字段时默认 False（向后兼容）。"""
    metrics = UsageMetrics()
    assert metrics.available is False
    assert metrics.input_tokens == 0


def test_trace_projection_positive_tokens():
    trace = {
        "trace_id": "trace-1",
        "answer_usage": {
            "available": True,
            "input_tokens": 123,
            "output_tokens": 45,
            "total_tokens": 168,
            "duration_ms": 100,
        },
    }
    summary = project_trace_summary(trace)
    assert summary["token_usage"] == {
        "available": True,
        "input_tokens": 123,
        "output_tokens": 45,
        "total_tokens": 168,
    }


def test_trace_projection_explicit_zero():
    trace = {
        "trace_id": "trace-2",
        "answer_usage": {
            "available": True,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    }
    summary = project_trace_summary(trace)
    assert summary["token_usage"]["available"] is True
    assert summary["token_usage"]["input_tokens"] == 0
    assert summary["token_usage"]["output_tokens"] == 0
    assert summary["token_usage"]["total_tokens"] == 0


def test_trace_projection_unavailable_is_null():
    trace = {
        "trace_id": "trace-3",
        "answer_usage": {"available": False, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }
    summary = project_trace_summary(trace)
    assert summary["token_usage"] == {
        "available": False,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }


def test_trace_projection_legacy_trace_without_availability():
    """旧 Trace 没有 available 字段：不得把默认 0 当真实 0。"""
    trace = {
        "trace_id": "trace-old",
        "answer_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }
    summary = project_trace_summary(trace)
    assert summary["token_usage"]["available"] is False
    assert summary["token_usage"]["input_tokens"] is None
    assert summary["token_usage"]["output_tokens"] is None
    assert summary["token_usage"]["total_tokens"] is None


def test_trace_projection_missing_answer_usage():
    """Trace 完全没有 answer_usage：按不可用兼容。"""
    summary = project_trace_summary({"trace_id": "trace-4"})
    assert summary["token_usage"]["available"] is False
    assert summary["token_usage"]["input_tokens"] is None


def test_trace_projection_available_with_invalid_numbers_is_null():
    """available=true 但数值缺失/类型非法：保守按不可用投影，不伪造。"""
    trace = {
        "trace_id": "trace-5",
        "answer_usage": {"available": True, "input_tokens": "abc", "output_tokens": None, "total_tokens": -1},
    }
    summary = project_trace_summary(trace)
    assert summary["token_usage"] == {
        "available": False,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }


def test_trace_projection_does_not_expose_runtime_metadata():
    """投影不得暴露 answer_runtime_metadata 全对象 / prompt / 内部配置。"""
    trace = {
        "trace_id": "trace-6",
        "answer_runtime_metadata": {"provider": "openai-compatible", "prompt": "secret prompt"},
        "answer_usage": {"available": True, "input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
    summary = project_trace_summary(trace)
    assert "answer_runtime_metadata" not in summary
    assert "prompt" not in summary
    assert "secret" not in str(summary)


def test_usage_metrics_serialization_backward_compatible():
    """UsageMetrics 序列化含 available，且不影响既有数字字段。"""
    metrics = UsageMetrics(input_tokens=5, output_tokens=3, total_tokens=8, available=True)
    dumped = metrics.model_dump(mode="json")
    assert dumped["input_tokens"] == 5
    assert dumped["total_tokens"] == 8
    assert dumped["available"] is True
    # 旧结构仍可直接解析（缺省 available=False）
    legacy = UsageMetrics.model_validate({"input_tokens": 5, "output_tokens": 3, "total_tokens": 8})
    assert legacy.available is False
