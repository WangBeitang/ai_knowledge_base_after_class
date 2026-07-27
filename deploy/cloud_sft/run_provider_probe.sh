#!/usr/bin/env bash
set -euo pipefail

# 运行一次真实 Web Provider（网页执行器）探针并保留 JSONL 审计记录。
# 该步骤不依赖 GPU，但依赖业务环境中的 Web/Milvus 配置；默认只跑一条已审核 Web 路线。

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
RUN_DIR="$CLOUD_RUN_ROOT/provider_probe_$timestamp"
PROBE_OUTPUT="${CLOUD_PROVIDER_PROBE_OUTPUT:-evaluation/stage9/artifacts/provider_records/cloud_smoke_provider_observations.jsonl}"
mkdir -p "$RUN_DIR"

args=(
  run
  --frozen
  --no-sync
  python
  evaluation/stage9/providers/record_provider_observations.py
  --case-id "${CLOUD_PROVIDER_PROBE_CASE_ID:-stage9-route-web-refuse-001}"
  --max-cases 1
  --output "$PROBE_OUTPUT"
)
if [[ "${CLOUD_PROBE_OVERWRITE:-0}" == "1" ]]; then
  args+=(--overwrite)
fi
if [[ "${CLOUD_PROVIDER_DISABLE_CHUNK_STATUS_FILTER:-0}" == "1" ]]; then
  args+=(--disable-chunk-status-filter)
fi

printf 'uv %s\n' "${args[*]}" > "$RUN_DIR/command.txt"
printf 'provider_probe（真实执行器探针）status=running\n'
uv "${args[@]}" 2>&1 | tee "$RUN_DIR/provider_probe.log"
uv run --frozen --no-sync python - "$PROBE_OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
records = [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if not records:
    raise SystemExit("Provider 探针未生成记录")
actions = sorted({str(record["action"]) for record in records})
print(json.dumps({
    "ok": True,
    "output": str(path),
    "record_count": len(records),
    "actions": actions,
}, ensure_ascii=False, sort_keys=True))
PY
printf 'provider_probe（真实执行器探针）status=completed\n'
printf 'run_dir（运行目录）=%s\n' "$RUN_DIR"
printf 'output（记录）=%s\n' "$PROBE_OUTPUT"
