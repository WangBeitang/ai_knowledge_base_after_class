"""PlannerModelServer（规划器模型服务）healthcheck（健康检查）入口。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.query.contracts import (  # noqa: E402
    PlannerContext,
    QueryAction,
    SubjectResolutionStatus,
)
from app.rag.query.model_planner.http_client import PlannerClient, PlannerClientError  # noqa: E402
from app.shared.config.planner_model_config import PlannerModelConfig  # noqa: E402


def build_healthcheck_context() -> PlannerContext:
    """
    构造最小 PlannerContext（规划器上下文）探针。

    healthcheck（健康检查）只验证服务协议和结构化输出，不评估模型业务正确性；因此使用
    固定问题和全量 allowed_actions（允许动作），避免健康检查把某个真实 Action（动作）误判
    成“不在白名单内”。
    """

    return PlannerContext(
        original_query="HAK 180 E020 如何处理？",
        current_query="HAK 180 E020 如何处理？",
        subject_resolution_status=SubjectResolutionStatus.CONFIRMED,
        subject_ids=["subject_hak_180"],
        query_identifiers={"equipment_model": ["HAK 180"], "alarm_code": ["E020"]},
        web_search_allowed=False,
        planner_step=0,
        max_steps=4,
        allowed_actions=[
            QueryAction.LOCAL_SEARCH,
            QueryAction.HYDE_SEARCH,
            QueryAction.WEB_SEARCH,
            QueryAction.ANSWER,
            QueryAction.ASK_CLARIFICATION,
            QueryAction.REFUSE,
        ],
    )


def build_config(args: argparse.Namespace) -> PlannerModelConfig:
    """
    从 CLI（命令行）参数构造 PlannerModelConfig（规划器模型配置）。

    api_key（接口密钥）只进入客户端请求头；healthcheck（健康检查）输出不会回显密钥。
    """

    return PlannerModelConfig(
        planner_mode="sft",
        planner_backend="http",
        planner_model_endpoint=args.endpoint,
        planner_model_id=args.model_id,
        planner_timeout_seconds=args.timeout_seconds,
        planner_max_new_tokens=args.max_new_tokens,
        planner_temperature=args.temperature,
        planner_enable_thinking=args.enable_thinking,
        planner_api_key=args.api_key,
    )


def _json_line(payload: dict[str, Any]) -> str:
    """稳定输出单行 JSON（结构化数据），方便 shell（命令行）和 CI（持续集成）解析。"""

    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _parse_action(value: str | None) -> QueryAction | None:
    if not value:
        return None
    try:
        return QueryAction(value)
    except ValueError as exc:
        allowed = ", ".join(action.value for action in QueryAction)
        raise argparse.ArgumentTypeError(f"--expected-action 必须是以下值之一：{allowed}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查 PlannerModelServer（规划器模型服务）是否可用。")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("PLANNER_MODEL_ENDPOINT", "http://127.0.0.1:8019/v1/chat/completions"),
        help="OpenAI-compatible chat completions（兼容 OpenAI 的聊天补全）地址。",
    )
    parser.add_argument(
        "--model-id",
        default=os.environ.get("PLANNER_MODEL_ID", "qwen3_5_4b_base"),
        help="期望请求和响应一致的 model_id（模型身份）。",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("PLANNER_API_KEY", ""),
        help="可选 api_key（接口密钥），只写入 Authorization（鉴权）请求头。",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(os.environ.get("PLANNER_TIMEOUT_SECONDS", "30")),
        help="HTTP（超文本传输协议）超时时间，单位秒。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.environ.get("PLANNER_MAX_NEW_TOKENS", "128")),
        help="最大生成 token（分词单元）数量，客户端会映射为 max_tokens。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("PLANNER_TEMPERATURE", "0")),
        help="采样温度；Planner（规划器）健康检查默认 0。",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=os.environ.get("PLANNER_ENABLE_THINKING", "0").lower() in {"1", "true", "yes", "on"},
        help="启用 thinking（思考模式）；阶段 9 Planner 默认关闭。",
    )
    parser.add_argument(
        "--expected-action",
        type=_parse_action,
        default=None,
        help="可选：要求模型返回指定 Action（动作），常用于 mock（模拟）服务测试。",
    )
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="跳过 response model（响应模型身份）校验，只检查协议和结构化输出。",
    )
    args = parser.parse_args(argv)

    config = build_config(args)
    client = PlannerClient(config=config)
    try:
        result = client.request_decision(build_healthcheck_context())
        if not args.skip_model_check and result.response_model_id != args.model_id:
            print(
                _json_line({
                    "ok": False,
                    "error_code": "model_id_mismatch",
                    "message": "PlannerModelServer 响应 model_id 与请求不一致",
                    "expected_model_id": args.model_id,
                    "response_model_id": result.response_model_id,
                    "endpoint": args.endpoint,
                }),
                file=sys.stderr,
            )
            return 1
        if args.expected_action is not None and result.decision.action != args.expected_action:
            print(
                _json_line({
                    "ok": False,
                    "error_code": "action_mismatch",
                    "message": "PlannerModelServer 返回 Action 与期望不一致",
                    "expected_action": args.expected_action.value,
                    "actual_action": result.decision.action.value,
                    "endpoint": args.endpoint,
                    "model_id": args.model_id,
                }),
                file=sys.stderr,
            )
            return 1
        print(_json_line({
            "ok": True,
            "endpoint": args.endpoint,
            "model_id": args.model_id,
            "response_model_id": result.response_model_id,
            "action": result.decision.action.value,
            "reason_code": result.decision.reason_code.value,
            "prompt_hash": result.prompt_hash,
        }))
        return 0
    except PlannerClientError as exc:
        print(
            _json_line({
                "ok": False,
                "error_code": exc.error_code,
                "message": exc.message,
                "details": exc.details,
                "endpoint": args.endpoint,
                "model_id": args.model_id,
            }),
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # pragma: no cover - 兜底输出用于云端脚本定位未知异常。
        print(
            _json_line({
                "ok": False,
                "error_code": "healthcheck_failed",
                "message": str(exc),
                "endpoint": args.endpoint,
                "model_id": args.model_id,
            }),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
