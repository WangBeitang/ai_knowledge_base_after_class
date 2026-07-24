"""Planner（规划器）管理服务。"""

from __future__ import annotations

from app.rag.query.config import (
    PLANNER_MAX_STEPS,
    REALTIME_PATTERNS_VERSION,
    RERANK_EVIDENCE_THRESHOLD,
    RETRIEVAL_CONFIG_VERSION,
    WEB_FALLBACK_ENABLED,
)
from app.rag.query.planner_registry import get_planner_registry_status


def get_planner_status() -> dict[str, object]:
    """返回当前线上 Planner（规划器）和 registry（注册表）实现摘要。"""
    registry_status = get_planner_registry_status()
    return {
        "code": 200,
        **registry_status,
        "retrieval_config_version": RETRIEVAL_CONFIG_VERSION,
        "max_steps": PLANNER_MAX_STEPS,
        "web_fallback_enabled": WEB_FALLBACK_ENABLED,
        "rule_config": {
            "rerank_evidence_threshold": RERANK_EVIDENCE_THRESHOLD,
            "realtime_rule_version": REALTIME_PATTERNS_VERSION,
        },
    }
