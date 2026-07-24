import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from app.rag.query.contracts import PlannerDecision, PlannerReasonCode, QueryAction
from app.rag.query.model_planner.decision_codec import encode_decision
from scripts.planner_model_server.mock_planner_server import make_handler
from scripts.planner_model_server.healthcheck_planner_server import main as healthcheck_main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_SERVER_DIR = PROJECT_ROOT / "deploy/planner_model_server"


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


def _fixed_response_handler(*, model_id: str, decision: PlannerDecision) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            response_body = json.dumps({
                "id": "stage9-contract-test",
                "object": "chat.completion",
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": encode_decision(decision)},
                        "finish_reason": "stop",
                    }
                ],
            }, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    return Handler


def test_vllm_script_is_primary_planner_model_server_entry():
    script = (MODEL_SERVER_DIR / "run_vllm_planner_server.sh").read_text(encoding="utf-8")

    assert "vllm" in script
    assert "serve" in script
    assert "--served-model-name" in script
    assert "--enable-lora" in script
    assert "--lora-modules" in script
    assert "PLANNER_API_KEY" in script


def test_sglang_entry_is_documented_as_deferred_not_required():
    readme = (MODEL_SERVER_DIR / "README.md").read_text(encoding="utf-8")

    assert "SGLang（大模型推理服务框架）" in readme
    assert "后续可选对照" in readme
    assert not (MODEL_SERVER_DIR / "run_sglang_planner_server.sh").exists()


def test_healthcheck_uses_planner_client_protocol_against_mock_server(capsys):
    decision = PlannerDecision(
        action=QueryAction.REFUSE,
        query="mock refusal",
        reason_code=PlannerReasonCode.SAFE_GUARD_TRIGGERED,
    )
    with _RunningHandlerServer(make_handler(decision)) as server:
        exit_code = healthcheck_main([
            "--endpoint", server.endpoint,
            "--model-id", "qwen3.5:4b",
            "--expected-action", "refuse",
        ])

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert exit_code == 0
    assert output["ok"] is True
    assert output["model_id"] == "qwen3.5:4b"
    assert output["response_model_id"] == "qwen3.5:4b"
    assert output["action"] == "refuse"
    assert output["reason_code"] == "safe_guard_triggered"


def test_healthcheck_fails_when_response_model_id_does_not_match(capsys):
    decision = PlannerDecision(
        action=QueryAction.REFUSE,
        query="mock refusal",
        reason_code=PlannerReasonCode.SAFE_GUARD_TRIGGERED,
    )
    handler = _fixed_response_handler(model_id="wrong-model", decision=decision)
    with _RunningHandlerServer(handler) as server:
        exit_code = healthcheck_main([
            "--endpoint", server.endpoint,
            "--model-id", "qwen3.5:4b",
        ])

    captured = capsys.readouterr()
    output = json.loads(captured.err)
    assert exit_code == 1
    assert output["ok"] is False
    assert output["error_code"] == "model_id_mismatch"
    assert output["expected_model_id"] == "qwen3.5:4b"
    assert output["response_model_id"] == "wrong-model"
