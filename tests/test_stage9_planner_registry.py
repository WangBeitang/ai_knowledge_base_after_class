import pytest

from app.rag.query.contracts import QueryAction
from app.rag.query.planner import RuleBasedPlanner
from app.rag.query.planner_registry import (
    PlannerMode,
    PlannerRegistryError,
    build_planner_runtime,
    get_planner_registry_status,
    get_registered_planners,
    normalize_planner_mode,
)
from app.shared.config.planner_model_config import PlannerModelConfig


def _config(
        *,
        mode: str = "rule",
        endpoint: str = "http://localhost:11434/v1/chat/completions",
        model_id: str = "qwen3.5:4b",
) -> PlannerModelConfig:
    return PlannerModelConfig(
        planner_mode=mode,
        planner_backend="http",
        planner_model_endpoint=endpoint,
        planner_model_id=model_id,
        planner_timeout_seconds=30.0,
        planner_max_new_tokens=128,
        planner_temperature=0.0,
        planner_enable_thinking=False,
    )


def test_planner_registry_defaults_to_rule_runtime():
    runtime = build_planner_runtime(mode="rule", config=_config())

    assert runtime.mode is PlannerMode.RULE
    assert runtime.planner_type == "rule"
    assert runtime.policy_version == "rule-v1"
    assert runtime.provider is None
    assert isinstance(runtime.planner, RuleBasedPlanner)
    assert runtime.runtime_metadata(duration_ms=7)["planner_mode"] == "rule"


def test_planner_registry_builds_http_model_runtime():
    runtime = build_planner_runtime(mode="sft", config=_config(mode="sft"))

    assert runtime.mode is PlannerMode.SFT
    assert runtime.planner_type == "model"
    assert runtime.provider == "http"
    assert runtime.model_id == "qwen3.5:4b"
    assert runtime.endpoint == "http://localhost:11434/v1/chat/completions"
    assert runtime.prompt_version.startswith("stage9-")
    assert runtime.policy_version == "sft:http:qwen3.5:4b"


def test_planner_registry_rejects_unknown_mode():
    with pytest.raises(PlannerRegistryError) as exc_info:
        normalize_planner_mode("unknown")

    assert exc_info.value.error_code == "planner_mode_unknown"


def test_planner_registry_marks_model_modes_unavailable_without_endpoint():
    config = _config(mode="local_base", endpoint="")

    status = get_planner_registry_status(config)
    entries = {item["planner_mode"]: item for item in status["registered_planners"]}

    assert status["online_mode"] == "local_base"
    assert status["planner_type"] == "unavailable"
    assert status["current_unavailable_reason"] == "模型 Planner 需要配置 PLANNER_MODEL_ENDPOINT"
    assert entries["rule"]["enabled_for_eval"] is True
    assert entries["local_base"]["enabled_online"] is False
    assert entries["local_base"]["unavailable_reason"] == "模型 Planner 需要配置 PLANNER_MODEL_ENDPOINT"


def test_registered_planners_enable_only_current_mode_online():
    items = {item.planner_mode: item for item in get_registered_planners(_config(mode="grpo"))}

    assert items["rule"].enabled_online is False
    assert items["grpo"].enabled_online is True
    assert items["grpo"].planner_type == "model"
    assert QueryAction.REFUSE.value == "refuse"
