"""PlannerModelServer（规划器模型服务）mock（模拟）入口。"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.query.contracts import PlannerDecision, PlannerReasonCode, QueryAction  # noqa: E402
from app.rag.query.model_planner.decision_codec import encode_decision  # noqa: E402


MOCK_SERVER_VERSION = "planner-model-mock-server-v1"


def make_handler(decision: PlannerDecision) -> type[BaseHTTPRequestHandler]:
    """
    创建固定返回 PlannerDecision（规划器决策）的 HTTP handler（请求处理器）。

    mock（模拟）服务只验证 OpenAI-compatible chat completions（兼容 OpenAI 的聊天补全）
    协议和 decision_codec（决策编解码器）边界，不代表真实模型能力。
    """

    class MockPlannerHandler(BaseHTTPRequestHandler):
        server_version = MOCK_SERVER_VERSION

        def do_POST(self) -> None:  # noqa: N802
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                content_length = 0
            raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                request_payload = json.loads(raw_body.decode("utf-8"))
            except json.JSONDecodeError:
                self._write_json(400, {"error": {"code": "request_json_invalid"}})
                return

            response = {
                "id": "planner-model-mock",
                "object": "chat.completion",
                "model": str(request_payload.get("model") or "planner-model-mock"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": encode_decision(decision),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": len(encode_decision(decision)),
                    "total_tokens": 0,
                },
            }
            self._write_json(200, response)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            # mock（模拟）服务默认不刷访问日志，避免测试输出噪音。
            return

        def _write_json(self, status_code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return MockPlannerHandler


def run_server(
        *,
        host: str,
        port: int,
        decision: PlannerDecision,
) -> None:
    """启动阻塞式 mock PlannerModelServer（模拟规划器模型服务）。"""

    server = HTTPServer((host, port), make_handler(decision))
    print(f"mock_planner_server={host}:{server.server_port}")
    print(f"decision={encode_decision(decision)}")
    server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="mock PlannerModelServer（模拟规划器模型服务）。")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8019)
    parser.add_argument("--action", default=QueryAction.REFUSE.value)
    parser.add_argument("--query", default="mock planner decision")
    parser.add_argument("--reason-code", default=PlannerReasonCode.SAFE_GUARD_TRIGGERED.value)
    args = parser.parse_args(argv)
    decision = PlannerDecision(
        action=QueryAction(args.action),
        query=args.query,
        reason_code=PlannerReasonCode(args.reason_code),
    )
    run_server(host=args.host, port=args.port, decision=decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
