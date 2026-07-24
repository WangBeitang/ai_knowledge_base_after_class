import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest
import requests

from app.rag.query.contracts import (
    PlannerContext,
    PlannerDecision,
    PlannerReasonCode,
    QueryAction,
    SubjectResolutionStatus,
)
from app.rag.query.model_planner.decision_codec import encode_decision
from app.rag.query.model_planner.http_client import PlannerClient, PlannerClientError
from app.shared.config.planner_model_config import PlannerModelConfig
from evaluation.stage9.model_planner.mock_planner_server import make_handler


def _context() -> PlannerContext:
    return PlannerContext(
        original_query="HAK 180 E020 如何处理？",
        current_query="HAK 180 E020 如何处理？",
        subject_resolution_status=SubjectResolutionStatus.CONFIRMED,
        subject_ids=["subject_hak_180"],
        query_identifiers={"alarm_code": ["E020"]},
        web_search_allowed=False,
        planner_step=0,
        max_steps=4,
        allowed_actions=[QueryAction.LOCAL_SEARCH, QueryAction.REFUSE],
    )


def _config(endpoint: str) -> PlannerModelConfig:
    return PlannerModelConfig(
        planner_backend="http",
        planner_model_endpoint=endpoint,
        planner_model_id="qwen3.5:4b",
        planner_timeout_seconds=3.0,
        planner_max_new_tokens=128,
        planner_temperature=0.0,
        planner_enable_thinking=False,
    )


def _openai_response(content: str) -> dict[str, Any]:
    return {
        "id": "test-chat-completion",
        "object": "chat.completion",
        "model": "qwen3.5:4b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


class _RunningServer:
    def __init__(self, *, status_code: int = 200, body: Any = None, raw_body: bytes | None = None) -> None:
        self.captured_requests: list[dict[str, Any]] = []

        captured_requests = self.captured_requests

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                content_length = int(self.headers.get("Content-Length", "0"))
                request_body = self.rfile.read(content_length) if content_length else b"{}"
                captured_requests.append(json.loads(request_body.decode("utf-8")))
                if raw_body is not None:
                    response_body = raw_body
                else:
                    response_body = json.dumps(body or {}, ensure_ascii=False).encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.endpoint = f"http://127.0.0.1:{self._server.server_port}/v1/chat/completions"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "_RunningServer":
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._server.shutdown()
        self._thread.join(timeout=3)
        self._server.server_close()


class _RunningHandlerServer:
    def __init__(self, handler: type[BaseHTTPRequestHandler]) -> None:
        self._server = HTTPServer(("127.0.0.1", 0), handler)
        self.endpoint = f"http://127.0.0.1:{self._server.server_port}/v1/chat/completions"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "_RunningHandlerServer":
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._server.shutdown()
        self._thread.join(timeout=3)
        self._server.server_close()


def test_planner_client_calls_openai_compatible_chat_completion():
    target_json = encode_decision({
        "action": "local_search",
        "query": "HAK 180 E020 如何处理？",
        "reason_code": "initial_local_search",
    })
    with _RunningServer(body=_openai_response(target_json)) as server:
        client = PlannerClient(config=_config(server.endpoint))

        result = client.request_decision(_context())

    assert result.decision.action == QueryAction.LOCAL_SEARCH
    assert result.decode_result.success is True
    assert result.model_id == "qwen3.5:4b"
    request_payload = server.captured_requests[0]
    assert request_payload["model"] == "qwen3.5:4b"
    assert request_payload["temperature"] == 0.0
    assert request_payload["max_tokens"] == 128
    assert request_payload["reasoning_effort"] == "none"
    assert request_payload["enable_thinking"] is False
    assert request_payload["chat_template_kwargs"]["enable_thinking"] is False
    assert "Planner（规划器）" in request_payload["messages"][0]["content"]
    assert "allowed_actions" in request_payload["messages"][1]["content"]


def test_planner_client_reports_non_2xx_status():
    with _RunningServer(status_code=503, body={"error": "unavailable"}) as server:
        client = PlannerClient(config=_config(server.endpoint))

        with pytest.raises(PlannerClientError) as exc_info:
            client.request_decision(_context())

    assert exc_info.value.error_code == "http_status_error"
    assert exc_info.value.details["status_code"] == 503


def test_planner_client_reports_invalid_response_json():
    with _RunningServer(raw_body=b"not-json") as server:
        client = PlannerClient(config=_config(server.endpoint))

        with pytest.raises(PlannerClientError) as exc_info:
            client.request_decision(_context())

    assert exc_info.value.error_code == "response_json_invalid"


def test_planner_client_reports_empty_chat_content():
    with _RunningServer(body={"choices": [{"message": {"content": ""}}]}) as server:
        client = PlannerClient(config=_config(server.endpoint))

        with pytest.raises(PlannerClientError) as exc_info:
            client.request_decision(_context())

    assert exc_info.value.error_code == "empty_model_content"


def test_planner_client_reports_unknown_action_from_model():
    target_json = encode_decision({
        "action": "web_search",
        "query": "查官网实时信息",
        "reason_code": "realtime_query",
    })
    with _RunningServer(body=_openai_response(target_json)) as server:
        client = PlannerClient(config=_config(server.endpoint))

        with pytest.raises(PlannerClientError) as exc_info:
            client.request_decision(_context())

    assert exc_info.value.error_code == "action_not_allowed"
    assert "raw_output_excerpt" in exc_info.value.details


def test_planner_client_reports_timeout():
    class TimeoutSession:
        def post(self, *_: Any, **__: Any) -> Any:
            raise requests.Timeout("timeout")

    client = PlannerClient(
        config=_config("http://127.0.0.1:1/v1/chat/completions"),
        session=TimeoutSession(),  # type: ignore[arg-type]
    )

    with pytest.raises(PlannerClientError) as exc_info:
        client.request_decision(_context())

    assert exc_info.value.error_code == "http_timeout"


def test_mock_planner_server_returns_fixed_decision():
    decision = PlannerDecision(
        action=QueryAction.REFUSE,
        query="mock refusal",
        reason_code=PlannerReasonCode.SAFE_GUARD_TRIGGERED,
    )
    with _RunningHandlerServer(make_handler(decision)) as server:
        client = PlannerClient(config=_config(server.endpoint))

        result = client.request_decision(PlannerContext(
            original_query="无法确认设备",
            current_query="无法确认设备",
            subject_resolution_status=SubjectResolutionStatus.NO_MENTION,
            web_search_allowed=False,
            planner_step=0,
            max_steps=4,
            allowed_actions=[QueryAction.REFUSE],
        ))

    assert result.decision == decision
