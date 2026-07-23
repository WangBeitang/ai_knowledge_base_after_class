import json

from app.rag.query.contracts import PlannerDecision, PlannerReasonCode, QueryAction
from app.rag.query.model_planner.decision_codec import decode_decision, encode_decision


def test_encode_decision_writes_minimal_json():
    decision = PlannerDecision(
        action=QueryAction.LOCAL_SEARCH,
        query="HAK 180 E020 如何处理？",
        reason_code=PlannerReasonCode.INITIAL_LOCAL_SEARCH,
    )

    payload = json.loads(encode_decision(decision))

    assert payload == {
        "action": "local_search",
        "query": "HAK 180 E020 如何处理？",
        "reason_code": "initial_local_search",
    }


def test_decode_decision_accepts_code_fenced_json():
    result = decode_decision(
        '```json\n{"action":"refuse","query":"证据不足","reason_code":"safe_guard_triggered"}\n```',
        allowed_actions=[QueryAction.REFUSE],
    )

    assert result.success is True
    assert result.decision is not None
    assert result.decision.action == QueryAction.REFUSE


def test_decode_decision_rejects_unknown_field():
    result = decode_decision(
        '{"action":"local_search","query":"q","reason_code":"initial_local_search","note":"extra"}',
        allowed_actions=[QueryAction.LOCAL_SEARCH, QueryAction.REFUSE],
    )

    assert result.success is False
    assert result.error_code == "decision_fields_invalid"


def test_decode_decision_rejects_action_outside_allowed_actions():
    result = decode_decision(
        '{"action":"web_search","query":"q","reason_code":"realtime_query"}',
        allowed_actions=[QueryAction.LOCAL_SEARCH, QueryAction.REFUSE],
    )

    assert result.success is False
    assert result.error_code == "action_not_allowed"


def test_decode_decision_rejects_unknown_reason_code():
    result = decode_decision(
        '{"action":"refuse","query":"q","reason_code":"made_up"}',
        allowed_actions=[QueryAction.REFUSE],
    )

    assert result.success is False
    assert result.error_code == "reason_code_unknown"
