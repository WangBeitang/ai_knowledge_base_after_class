import copy
from datetime import datetime
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.http import query_server
from app.api.schema.query_schema import QueryRequestParam
from app.shared.config.knowledge_base_config import DEFAULT_DATASET_ID, DEFAULT_TENANT_ID
from app.shared.runtime.logger import _trace_id


client = TestClient(query_server.app)


def _query_body(**overrides):
    body = {
        "query": "HAK 180 烫金机如何开机？",
        "session_id": "session-stage5-part1",
        "is_stream": False,
    }
    body.update(overrides)
    return body


def test_query_request_uses_default_dataset_when_field_is_omitted():
    request_param = QueryRequestParam(**_query_body())

    assert request_param.dataset_ids == [DEFAULT_DATASET_ID]


def test_query_request_normalizes_and_deduplicates_dataset_ids_in_order():
    request_param = QueryRequestParam(
        **_query_body(
            dataset_ids=[
                " dataset_ops_a ",
                "",
                "dataset_ops_b",
                "dataset_ops_a",
                "   ",
            ]
        )
    )

    assert request_param.dataset_ids == ["dataset_ops_a", "dataset_ops_b"]


@pytest.mark.parametrize("dataset_ids", [[], [""], ["  ", "\n"]])
def test_query_request_rejects_explicitly_empty_dataset_scope(dataset_ids):
    with pytest.raises(ValidationError, match="dataset_ids 至少包含一个非空知识库 ID"):
        QueryRequestParam(**_query_body(dataset_ids=dataset_ids))


def test_query_request_rejects_more_than_ten_unique_datasets():
    with pytest.raises(ValidationError, match="dataset_ids 最多包含 10 个不同的知识库 ID"):
        QueryRequestParam(
            **_query_body(dataset_ids=[f"dataset_{index}" for index in range(11)])
        )


def test_query_request_allows_exactly_ten_unique_datasets():
    request_param = QueryRequestParam(
        **_query_body(dataset_ids=[f"dataset_{index}" for index in range(10)])
    )

    assert request_param.dataset_ids == [f"dataset_{index}" for index in range(10)]


@pytest.mark.parametrize(
    ("headers", "is_stream"),
    [
        ({}, False),
        ({"X-User-Id": "   "}, True),
    ],
)
def test_query_requires_non_blank_user_header_before_graph_execution(
        monkeypatch,
        headers,
        is_stream,
):
    graph_calls = []
    sse_queue_calls = []
    monkeypatch.setattr(
        query_server,
        "query_graph_invoke",
        lambda **kwargs: graph_calls.append(kwargs),
    )
    monkeypatch.setattr(
        query_server,
        "create_sse_queue",
        lambda session_id: sse_queue_calls.append(session_id),
    )

    response = client.post(
        "/query",
        json=_query_body(is_stream=is_stream),
        headers=headers,
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "缺少 X-User-Id 请求头"
    assert graph_calls == []
    assert sse_queue_calls == []


@pytest.mark.parametrize(
    "dataset_ids",
    [
        [],
        [f"dataset_{index}" for index in range(11)],
    ],
)
def test_query_rejects_invalid_dataset_scope_before_graph_execution(monkeypatch, dataset_ids):
    graph_calls = []
    monkeypatch.setattr(
        query_server,
        "query_graph_invoke",
        lambda **kwargs: graph_calls.append(kwargs),
    )

    response = client.post(
        "/query",
        json=_query_body(dataset_ids=dataset_ids),
        headers={"X-User-Id": "user_a"},
    )

    assert response.status_code == 422
    assert graph_calls == []


def test_query_api_passes_normalized_context_to_sync_graph(monkeypatch):
    graph_calls = []
    dataset_scope_calls = []

    def fake_query_graph_invoke(**kwargs):
        graph_calls.append(copy.deepcopy(kwargs))
        return {
            "answer": "测试答案",
            "image_urls": [],
            "trace_id": "trace-api-test",
            "citations": [],
            "terminal_reason_code": "local_evidence_sufficient",
        }

    monkeypatch.setattr(query_server, "query_graph_invoke", fake_query_graph_invoke)
    monkeypatch.setattr(
        query_server,
        "_require_dataset_read_scope",
        lambda user_id, dataset_ids: dataset_scope_calls.append((user_id, list(dataset_ids))),
    )

    response = client.post(
        "/query",
        json=_query_body(dataset_ids=[" dataset_b ", "dataset_a", "dataset_b"]),
        headers={"X-User-Id": "  user_a  "},
    )

    assert response.status_code == 200
    assert graph_calls == [
        {
            "session_id": "session-stage5-part1",
            "query": "HAK 180 烫金机如何开机？",
            "is_stream": False,
            "owner_user_id": "user_a",
            "dataset_ids": ["dataset_b", "dataset_a"],
        }
    ]
    assert dataset_scope_calls == [("user_a", ["dataset_b", "dataset_a"])]


def test_query_api_passes_identity_and_default_dataset_to_stream_task(monkeypatch):
    graph_calls = []
    monkeypatch.setattr(query_server, "create_sse_queue", lambda session_id: None)
    monkeypatch.setattr(
        query_server,
        "query_graph_invoke",
        lambda **kwargs: graph_calls.append(copy.deepcopy(kwargs)),
    )

    response = client.post(
        "/query",
        json=_query_body(is_stream=True),
        headers={"X-User-Id": "user_stream"},
    )

    assert response.status_code == 200
    assert graph_calls[0]["owner_user_id"] == "user_stream"
    assert graph_calls[0]["dataset_ids"] == [DEFAULT_DATASET_ID]


def test_query_graph_state_keeps_different_owner_context_for_same_query(monkeypatch):
    captured_states = []

    class FakeQueryGraph:
        def invoke(self, state):
            captured_states.append(copy.deepcopy(state))
            return {**state, "answer": "测试答案", "image_urls": []}

    monkeypatch.setattr(query_server, "query_graph_app", FakeQueryGraph())
    monkeypatch.setattr(query_server, "clear_task", lambda session_id: None)
    monkeypatch.setattr(query_server, "update_task_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(query_server, "safe_create_running_trace", lambda state: None)

    for user_id in ("user_a", "user_b"):
        query_server.query_graph_invoke(
            session_id=f"session-{user_id}",
            query="HAK-180 的 E020 怎么处理？",
            is_stream=False,
            owner_user_id=user_id,
            dataset_ids=[DEFAULT_DATASET_ID],
        )

    assert [state["owner_user_id"] for state in captured_states] == ["user_a", "user_b"]
    assert all(state["tenant_id"] == DEFAULT_TENANT_ID for state in captured_states)
    assert all(state["dataset_ids"] == [DEFAULT_DATASET_ID] for state in captured_states)
    # 同一个 session 可以执行多次查询，所以 trace_id 必须是每次执行独立生成的 UUID，
    # 不能直接复用用户 ID 或会话 ID。
    assert len({state["trace_id"] for state in captured_states}) == 2
    assert all(str(UUID(state["trace_id"])) == state["trace_id"] for state in captured_states)
    # query_started_at 使用带 UTC 时区的 ISO 8601 字符串，后续 Trace 计算耗时不依赖本地时区。
    assert all(datetime.fromisoformat(state["query_started_at"]).utcoffset() is not None for state in captured_states)
    # 查询入口必须在创建 running Trace 前冻结策略/检索版本；planner_step 仍为 0，表示
    # Planner 尚未真正做出第一步 Decision。
    assert all(state["planner_step"] == 0 for state in captured_states)
    assert all(state["policy_version"] == "rule-v1" for state in captured_states)
    assert all(state["retrieval_config_version"] == "retrieval-stage5-final-v1" for state in captured_states)
    assert all(state["retrieval_config_snapshot"]["rrf_k"] == 60 for state in captured_states)
    assert all(state["chunk_status_filter_enabled"] is True for state in captured_states)
    assert all(state["disabled_chunk_ids"] == [] for state in captured_states)
    # 标识提取在进入 LangGraph 前完成，并且不同用户得到彼此独立但内容一致的字典。
    assert all(
        state["query_identifiers"] == {
            "equipment_model": ["HAK 180"],
            "alarm_code": ["E020"],
        }
        for state in captured_states
    )


def test_query_graph_invoke_accepts_retrieval_test_config_overrides(monkeypatch):
    captured_states = []

    class FakeQueryGraph:
        def invoke(self, state):
            captured_states.append(copy.deepcopy(state))
            return {**state, "answer": "测试答案", "image_urls": []}

    monkeypatch.setattr(query_server, "query_graph_app", FakeQueryGraph())
    monkeypatch.setattr(query_server, "clear_task", lambda session_id: None)
    monkeypatch.setattr(query_server, "update_task_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(query_server, "safe_create_running_trace", lambda state: None)

    query_server.query_graph_invoke(
        session_id="retrieval-test-config",
        query="HAK 180 怎么开机？",
        is_stream=False,
        owner_user_id="user_a",
        dataset_ids=[DEFAULT_DATASET_ID],
        history_persistence_enabled=False,
        execution_source="retrieval_test",
        retrieval_mode="dense_bm25",
        web_fallback_enabled=False,
    )

    assert captured_states[0]["execution_source"] == "retrieval_test"
    assert captured_states[0]["history_persistence_enabled"] is False
    assert captured_states[0]["retrieval_mode"] == "dense_bm25"
    assert captured_states[0]["retrieval_config_snapshot"]["retrieval_mode"] == "dense_bm25"
    assert captured_states[0]["web_search_allowed"] is False
    assert captured_states[0]["retrieval_config_snapshot"]["web_fallback_enabled"] is False


def test_query_log_trace_prefers_single_execution_id_and_keeps_legacy_fallbacks():
    # trace_id 表示单次查询执行，应优先于可能覆盖多次查询的 session_id。
    assert _trace_id(
        {
            "trace_id": "trace-query-1",
            "task_id": "task-import-1",
            "session_id": "session-1",
        }
    ) == "trace-query-1"
    # 导入 State 暂时没有 trace_id，日志装饰器仍需兼容原有 task_id。
    assert _trace_id({"task_id": "task-import-1", "session_id": "session-1"}) == "task-import-1"
    # 旧查询调用或测试没有新字段时，最后回退 session_id，而不是产生异常。
    assert _trace_id({"session_id": "session-1"}) == "session-1"


def test_query_graph_invoke_reraises_programming_error_instead_of_disguising_no_evidence(monkeypatch):
    statuses = []
    errors = []

    class BrokenQueryGraph:
        def invoke(self, state):
            raise ValueError("planner state contract broken")

    monkeypatch.setattr(query_server, "query_graph_app", BrokenQueryGraph())
    monkeypatch.setattr(query_server, "clear_task", lambda session_id: None)
    monkeypatch.setattr(
        query_server,
        "update_task_status",
        lambda session_id, status, is_stream: statuses.append(status),
    )
    monkeypatch.setattr(
        query_server,
        "push_to_session",
        lambda session_id, event, payload: errors.append(payload),
    )

    with pytest.raises(ValueError, match="planner state contract broken"):
        query_server.query_graph_invoke(
            session_id="session-broken",
            query="HAK 180 怎么开机？",
            is_stream=False,
            owner_user_id="user-a",
            dataset_ids=[DEFAULT_DATASET_ID],
        )

    assert statuses == [query_server.TASK_STATUS_PROCESSING, query_server.TASK_STATUS_FAILED]
    assert errors == [{"error": "planner state contract broken"}]
