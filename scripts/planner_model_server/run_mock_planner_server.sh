#!/usr/bin/env bash
set -euo pipefail

# mock server（模拟模型服务）只验证 HTTP（超文本传输协议）和 decision_codec（决策编解码器）
# 契约，不代表 Planner（规划器）模型能力。

PLANNER_HOST="${PLANNER_HOST:-127.0.0.1}"
PLANNER_PORT="${PLANNER_PORT:-8019}"
PLANNER_MOCK_ACTION="${PLANNER_MOCK_ACTION:-refuse}"
PLANNER_MOCK_QUERY="${PLANNER_MOCK_QUERY:-mock planner decision}"
PLANNER_MOCK_REASON_CODE="${PLANNER_MOCK_REASON_CODE:-safe_guard_triggered}"

exec uv run python scripts/planner_model_server/mock_planner_server.py \
  --host "$PLANNER_HOST" \
  --port "$PLANNER_PORT" \
  --action "$PLANNER_MOCK_ACTION" \
  --query "$PLANNER_MOCK_QUERY" \
  --reason-code "$PLANNER_MOCK_REASON_CODE"
