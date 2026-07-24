# PlannerModelServer（规划器模型服务）启动入口

本目录只负责阶段 9 的模型服务启动和健康检查。业务查询仍通过
`PlannerClient（规划器客户端） -> HTTP（超文本传输协议） -> /v1/chat/completions`
调用模型，不在每个业务 worker（服务工作进程）里直接加载大模型。

## 第一版选择

第一版只把 `vLLM（大模型推理服务框架）`作为正式入口落地。`SGLang（大模型推理服务框架）`
保留为后续可选对照，不作为阶段 9.3.6 的必做代码，避免同时维护两套启动脚本。

`vLLM（大模型推理服务框架）`会把本地或云端的 Qwen 模型封装成 OpenAI-compatible chat
completions（兼容 OpenAI 的聊天补全接口）。如果已经完成 SFT（监督微调），启动时在
base model（基础模型）上挂载 LoRA adapter（低秩适配器）；如果还没微调，就只启动
base model（基础模型）。

## 启动真实模型服务

未挂载 adapter（适配器）时：

```bash
PLANNER_MODEL_PATH=Qwen/Qwen3.5-4B \
PLANNER_MODEL_ID=qwen3_5_4b_base \
PLANNER_HOST=127.0.0.1 \
PLANNER_PORT=8019 \
bash evaluation/stage9/model_server/run_vllm_planner_server.sh
```

挂载 SFT LoRA adapter（监督微调低秩适配器）时：

```bash
PLANNER_MODEL_PATH=Qwen/Qwen3.5-4B \
PLANNER_BASE_MODEL_ID=qwen3_5_4b_base \
PLANNER_MODEL_ID=qwen3_5_4b_sft_stage9 \
PLANNER_ADAPTER_PATH=/path/to/checkpoint/adapter \
PLANNER_HOST=127.0.0.1 \
PLANNER_PORT=8019 \
bash evaluation/stage9/model_server/run_vllm_planner_server.sh
```

如果服务设置了 `PLANNER_API_KEY（规划器接口密钥）`，业务侧 `.env` 也必须设置同一个值：

```dotenv
PLANNER_MODE=sft
PLANNER_BACKEND=http
PLANNER_MODEL_ENDPOINT=http://127.0.0.1:8019/v1/chat/completions
PLANNER_MODEL_ID=qwen3_5_4b_sft_stage9
PLANNER_API_KEY=replace-with-local-secret
PLANNER_ENABLE_THINKING=0
```

## 本地 mock（模拟）服务

mock server（模拟模型服务）只用于本地无模型或 CI（持续集成）验证协议，不代表模型能力：

```bash
bash evaluation/stage9/model_server/run_mock_planner_server.sh
```

## 健康检查

healthcheck（健康检查）用业务侧同一份 `PlannerClient（规划器客户端）` 调服务，并检查：

- HTTP（超文本传输协议）可访问。
- 响应顶层 `model（模型身份）` 与期望值一致。
- `message.content（消息正文）` 能被 `decision_codec（决策编解码器）`解析成合法 `PlannerDecision（规划器决策）`。

```bash
uv run python evaluation/stage9/model_server/healthcheck_planner_server.py \
  --endpoint http://127.0.0.1:8019/v1/chat/completions \
  --model-id qwen3_5_4b_sft_stage9
```

本地 Ollama（本地大模型运行器）也可以使用同一个 healthcheck（健康检查），只要它暴露
OpenAI-compatible（兼容 OpenAI）的 `/v1/chat/completions`：

```bash
uv run python evaluation/stage9/model_server/healthcheck_planner_server.py \
  --endpoint http://localhost:11434/v1/chat/completions \
  --model-id qwen3.5:4b
```

## 云端边界

GPU（显卡算力）服务器上可以同时部署业务服务、Milvus（向量数据库）、Web provider
（联网检索服务）和 PlannerModelServer（规划器模型服务），但模型仍作为独立 HTTP 服务进程
存在。这样可以避免多个业务 worker（服务工作进程）重复加载模型权重，并且便于单独重启、
健康检查和记录 `model_id（模型身份）`。
