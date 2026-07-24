# PlannerModelServer（规划器模型服务）部署入口

本目录是正式部署资产，不属于某个评测阶段。业务服务通过
`PlannerClient（规划器客户端） -> HTTP（超文本传输协议） -> /v1/chat/completions`
调用 PlannerModelServer（规划器模型服务），避免每个业务 worker（服务工作进程）重复加载模型。

第一版正式入口选择 `vLLM（大模型推理服务框架）`。`SGLang（大模型推理服务框架）`
作为后续可选对照，只有在性能或兼容性评估需要时再接入，当前不是必需部署入口。

## 启动 base model（基础模型）

```bash
PLANNER_MODEL_PATH=Qwen/Qwen3.5-4B \
PLANNER_MODEL_ID=qwen3_5_4b_base \
PLANNER_HOST=127.0.0.1 \
PLANNER_PORT=8019 \
bash deploy/planner_model_server/run_vllm_planner_server.sh
```

## 启动 SFT adapter（监督微调适配器）

```bash
PLANNER_MODEL_PATH=Qwen/Qwen3.5-4B \
PLANNER_BASE_MODEL_ID=qwen3_5_4b_base \
PLANNER_MODEL_ID=qwen3_5_4b_sft_stage9 \
PLANNER_ADAPTER_PATH=/path/to/checkpoint/adapter \
PLANNER_HOST=127.0.0.1 \
PLANNER_PORT=8019 \
bash deploy/planner_model_server/run_vllm_planner_server.sh
```

## 业务侧配置

```dotenv
PLANNER_MODE=sft
PLANNER_BACKEND=http
PLANNER_MODEL_ENDPOINT=http://127.0.0.1:8019/v1/chat/completions
PLANNER_MODEL_ID=qwen3_5_4b_sft_stage9
PLANNER_API_KEY=
PLANNER_ENABLE_THINKING=0
```

`PLANNER_API_KEY（规划器接口密钥）`为空时不发送鉴权头；如果 vLLM（大模型推理服务框架）
启动时设置了 `--api-key`，业务侧必须设置同一个值。

## 健康检查

```bash
uv run python scripts/planner_model_server/healthcheck_planner_server.py \
  --endpoint http://127.0.0.1:8019/v1/chat/completions \
  --model-id qwen3_5_4b_sft_stage9
```

本地无模型时可用 mock server（模拟模型服务）验证协议：

```bash
bash scripts/planner_model_server/run_mock_planner_server.sh
```
