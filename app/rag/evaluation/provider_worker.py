"""为训练进程提供真实 ActionProvider（动作执行器）的本机 HTTP 服务。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import metadata
from pathlib import Path
from typing import Any

from app.rag.evaluation.action_providers import MilvusActionProvider
from app.rag.evaluation.case_schema import EnvironmentSnapshot
from app.rag.evaluation.offline_environment import OfflineState
from app.rag.query.contracts import PlannerDecision, QueryAction


class ProviderWorker:
    """
    常驻真实 Provider（动作执行器）。

    snapshot（环境快照）在进程启动时只加载一次；每个请求携带模型真实生成的
    PlannerDecision（规划器决策）和独立 OfflineState（离线状态）。strict_errors（严格
    失败模式）保证 Milvus/Web/HyDE 异常返回失败，而不是伪装成空 Observation（观察结果）。
    """

    def __init__(self, snapshot_path: str | Path, *, preload_embedding: bool = True) -> None:
        self.snapshot_path = Path(snapshot_path)
        self.snapshot = EnvironmentSnapshot.model_validate_json(
            self.snapshot_path.read_text(encoding="utf-8")
        )
        self.provider = MilvusActionProvider(
            chunk_status_filter_enabled=True,
            strict_errors=True,
        )
        self.health_payload = self._preflight(preload_embedding=preload_embedding)

    def _preflight(self, *, preload_embedding: bool) -> dict[str, Any]:
        """在监听端口前验证真实检索、HyDE 和 Web 所需身份。"""

        required_env = (
            "MILVUS_URL",
            "MILVUS_TOKEN",
            "CHUNKS_COLLECTION",
            "MONGO_URL",
            "MONGO_DB_NAME",
            "OPENAI_BASE_URL",
            "OPENAI_API_KEY",
            "LLM_DEFAULT_MODEL",
            "MCP_DASHSCOPE_BASE_URL",
        )
        missing = [name for name in required_env if not str(os.getenv(name) or "").strip()]
        if not (str(os.getenv("BGE_M3_PATH") or "").strip() or str(os.getenv("BGE_M3") or "").strip()):
            missing.append("BGE_M3_PATH|BGE_M3")
        if missing:
            raise RuntimeError(f"真实 Provider 缺少环境变量：{missing}")

        # HyDE（假设式改写检索）与 Web（网页检索）在 75 条正式 case 中均可能被模型选择；
        # 启动前只验证依赖可导入，不提前调用外部 LLM/Web 服务。
        from agents.mcp import MCPServerStreamableHttp  # noqa: F401
        from langchain_core.messages import HumanMessage  # noqa: F401

        from app.infra.vectorstore.milvus_gateway import milvus_gateway

        client = milvus_gateway.client
        if client is None:
            raise RuntimeError("Milvus 客户端不可用")
        collection = milvus_gateway.chunk_collection_name
        description = client.describe_collection(collection_name=collection)
        if not description:
            raise RuntimeError(f"Milvus collection 不可读取：{collection}")

        if preload_embedding:
            from app.infra.llm.providers import llm_provider

            probe = llm_provider.embed_documents(["设备运维检索 Provider 正式训练前检查"])
            if not probe.get("dense") or not probe.get("sparse"):
                raise RuntimeError("BGE-M3 embedding（嵌入模型）预加载结果为空")

        return {
            "ready": True,
            "provider": self.provider.provider_name,
            "strict_errors": True,
            "snapshot_id": self.snapshot.snapshot_id,
            "snapshot_sha256": hashlib.sha256(self.snapshot_path.read_bytes()).hexdigest(),
            "retrieval_config_version": self.snapshot.retrieval_config_version,
            "milvus_collection": collection,
            "embedding_preloaded": preload_embedding,
            "embedding_model": os.getenv("BGE_M3_PATH") or os.getenv("BGE_M3") or "BAAI/bge-m3",
            "embedding_device": os.getenv("BGE_DEVICE") or "cpu",
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "package_versions": {
                package: _package_version(package)
                for package in ("pydantic", "pymilvus", "pymongo", "requests", "transformers")
            },
        }

    def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        """执行模型实际 Action（动作）并返回可校验候选。"""

        state_payload = dict(payload.get("state") or {})
        if state_payload.get("snapshot_id") != self.snapshot.snapshot_id:
            raise ValueError("请求 snapshot_id 与 Provider Worker 冻结快照不一致")
        state_payload["snapshot"] = self.snapshot.model_dump(mode="json")
        state = OfflineState.model_validate(state_payload)
        decision = PlannerDecision.model_validate(payload.get("decision"))
        method = {
            QueryAction.LOCAL_SEARCH: self.provider.local_search,
            QueryAction.HYDE_SEARCH: self.provider.hyde_search,
            QueryAction.WEB_SEARCH: self.provider.web_search,
        }.get(decision.action)
        if method is None:
            raise ValueError(f"Provider Worker 不执行终态 Action={decision.action.value}")
        candidates = method(state, decision)
        return {
            "ok": True,
            "action": decision.action.value,
            "query": decision.query,
            "candidate_count": len(candidates),
            "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
        }


def build_handler(worker: ProviderWorker) -> type[BaseHTTPRequestHandler]:
    """为固定 Worker 构造只暴露 health/execute 的请求处理器。"""

    class Handler(BaseHTTPRequestHandler):
        server_version = "Stage9RealProvider/1.0"

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._json_response(404, {"ok": False, "error": "not_found"})
                return
            self._json_response(200, worker.health_payload)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/execute":
                self._json_response(404, {"ok": False, "error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or "0")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                self._json_response(200, worker.execute(payload))
            except Exception as exc:
                self._json_response(
                    500,
                    {"ok": False, "error": exc.__class__.__name__, "message": str(exc)[:1000]},
                )

        def log_message(self, format_text: str, *args: object) -> None:
            # HTTP access log（访问日志）写 stderr，stdout 保留给启动身份和机器读取。
            sys.stderr.write("provider-worker: " + (format_text % args) + "\n")

        def _json_response(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _package_version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unavailable"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8021, type=int)
    parser.add_argument("--skip-embedding-preload", action="store_true")
    args = parser.parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost"}:
        raise ValueError("Provider Worker 只允许绑定本机地址")
    worker = ProviderWorker(
        args.snapshot,
        preload_embedding=not args.skip_embedding_preload,
    )
    server = ThreadingHTTPServer((args.host, args.port), build_handler(worker))
    print(json.dumps({**worker.health_payload, "endpoint": f"http://{args.host}:{args.port}"}, ensure_ascii=False))
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
