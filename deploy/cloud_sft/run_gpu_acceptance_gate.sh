#!/usr/bin/env bash
set -euo pipefail

# 9.3.10 一键 GPU 验收：
# preflight -> 启动 vLLM -> 等待 /health -> 六类 Action/Web 边界 HTTP 探针 -> 停服释放显存。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

load_env() {
  local env_file="${CLOUD_SFT_ENV_FILE:-}"
  if [[ -z "$env_file" ]]; then
    return
  fi
  if [[ ! -f "$env_file" && -f "$PROJECT_ROOT/$env_file" ]]; then
    env_file="$PROJECT_ROOT/$env_file"
  fi
  if [[ ! -f "$env_file" ]]; then
    echo "CLOUD_SFT_ENV_FILE（云端环境文件）不存在：$env_file" >&2
    exit 2
  fi
  set -a
  source "$env_file"
  set +a
}

load_env
APP_ROOT="${APP_ROOT:-$PROJECT_ROOT}"
cd "$APP_ROOT"
CLOUD_RUN_ROOT="${CLOUD_RUN_ROOT:-evaluation/stage9/artifacts/cloud_runs}"
timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
RUN_DIR="$CLOUD_RUN_ROOT/gpu_acceptance_gate_$timestamp"
mkdir -p "$RUN_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export REQUIRE_CUDA=1
export REQUIRE_GPU_PREFLIGHT=1
export CLOUD_PREFLIGHT_OUTPUT="$RUN_DIR/preflight.json"

health_url="http://${PLANNER_HOST:-127.0.0.1}:${PLANNER_PORT:-8019}/health"
timeout_seconds="${CLOUD_SERVER_START_TIMEOUT_SECONDS:-600}"
server_pid=""
server_process_group=0

cleanup() {
  local exit_code=$?
  trap - EXIT
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    printf 'planner_server（规划器服务）status=stopping pid=%s\n' "$server_pid"
    if [[ "$server_process_group" == "1" ]]; then
      kill -TERM -- "-$server_pid" 2>/dev/null || true
    else
      kill -TERM "$server_pid" 2>/dev/null || true
    fi
    for _ in {1..30}; do
      kill -0 "$server_pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$server_pid" 2>/dev/null; then
      if [[ "$server_process_group" == "1" ]]; then
        kill -KILL -- "-$server_pid" 2>/dev/null || true
      else
        kill -KILL "$server_pid" 2>/dev/null || true
      fi
    fi
    wait "$server_pid" 2>/dev/null || true
  fi
  nvidia-smi \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader 2>/dev/null \
    > "$RUN_DIR/gpu_processes_after_stop.txt" || true
  exit "$exit_code"
}
on_signal() {
  exit 130
}
trap cleanup EXIT
trap on_signal INT TERM

printf '%s\n' \
  "CLOUD_SFT_ENV_FILE=${CLOUD_SFT_ENV_FILE:-} bash deploy/cloud_sft/run_gpu_acceptance_gate.sh" \
  > "$RUN_DIR/command.txt"

printf 'gpu_acceptance_gate（GPU 验收门禁）status=starting_server\n'
if command -v setsid >/dev/null 2>&1; then
  setsid bash deploy/cloud_sft/run_planner_server.sh > "$RUN_DIR/planner_server.log" 2>&1 &
  server_pid=$!
  server_process_group=1
else
  bash deploy/cloud_sft/run_planner_server.sh > "$RUN_DIR/planner_server.log" 2>&1 &
  server_pid=$!
fi
printf '%s\n' "$server_pid" > "$RUN_DIR/planner_server.pid"

deadline=$((SECONDS + timeout_seconds))
wait_started_at=$SECONDS
while ! curl --fail --silent --show-error --max-time 5 "$health_url" >/dev/null 2>&1; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "vLLM 在 /health 就绪前退出；查看 $RUN_DIR/planner_server.log" >&2
    tail -80 "$RUN_DIR/planner_server.log" >&2 || true
    exit 2
  fi
  if (( SECONDS >= deadline )); then
    echo "等待 vLLM /health 超时（${timeout_seconds}s）；查看 $RUN_DIR/planner_server.log" >&2
    tail -80 "$RUN_DIR/planner_server.log" >&2 || true
    exit 2
  fi
  printf 'gpu_acceptance_gate（GPU 验收门禁）status=waiting_health elapsed_seconds=%s\n' \
    "$((SECONDS - wait_started_at))"
  sleep 5
done

printf 'gpu_acceptance_gate（GPU 验收门禁）status=health_ready\n'
CLOUD_SFT_ENV_FILE="" \
  bash deploy/cloud_sft/run_planner_action_probe.sh \
  2>&1 | tee "$RUN_DIR/planner_action_probe.log"

printf 'gpu_acceptance_gate（GPU 验收门禁）status=completed\n'
printf 'run_dir（运行目录）=%s\n' "$RUN_DIR"
