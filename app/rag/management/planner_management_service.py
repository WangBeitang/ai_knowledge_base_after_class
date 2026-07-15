"""阶段 7 Planner 管理服务。"""

from __future__ import annotations

from app.rag.query.config import (
    PLANNER_MAX_STEPS,
    POLICY_VERSION,
    REALTIME_PATTERNS_VERSION,
    RERANK_EVIDENCE_THRESHOLD,
    RETRIEVAL_CONFIG_VERSION,
    WEB_FALLBACK_ENABLED,
)


def get_planner_status() -> dict[str, object]:
    """返回当前线上 Planner 和预注册实现摘要。"""
    return {
        "code": 200,
        "online_mode": "rule",
        "policy_version": POLICY_VERSION,
        "retrieval_config_version": RETRIEVAL_CONFIG_VERSION,
        "max_steps": PLANNER_MAX_STEPS,
        "web_fallback_enabled": WEB_FALLBACK_ENABLED,
        "rule_config": {
            "rerank_evidence_threshold": RERANK_EVIDENCE_THRESHOLD,
            "realtime_rule_version": REALTIME_PATTERNS_VERSION,
        },
        "registered_planners": [
            {"planner_mode": "rule", "enabled_online": True, "enabled_for_eval": True, "unavailable_reason": ""},
            {"planner_mode": "api", "enabled_online": False, "enabled_for_eval": False, "unavailable_reason": "阶段 7 只注册名称，不加载 API Planner"},
            {"planner_mode": "local_base", "enabled_online": False, "enabled_for_eval": False, "unavailable_reason": "阶段 8 离线评测预留"},
            {"planner_mode": "sft", "enabled_online": False, "enabled_for_eval": False, "unavailable_reason": "阶段 9 训练后启用"},
            {"planner_mode": "grpo", "enabled_online": False, "enabled_for_eval": False, "unavailable_reason": "阶段 9 训练后启用"},
        ],
    }
