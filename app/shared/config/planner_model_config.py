"""
模型 Planner（规划器）服务配置。

这些配置描述业务服务如何选择 Planner（规划器）模式，以及如何调用 PlannerModelServer
（规划器模型服务）。它不负责加载模型权重；真正的实例选择和可用性判断由
planner_registry（规划器注册表）处理。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.shared.config.common import env_bool, env_float, env_str


@dataclass(frozen=True)
class PlannerModelConfig:
    """
    PlannerModelServer（规划器模型服务）HTTP（超文本传输协议）调用配置。

    endpoint（调用地址）通常是同机 `http://127.0.0.1:<port>/v1/chat/completions`；
    model_id（模型身份）必须进入请求和 Trace（轨迹记录），用于区分本地 mock（模拟服务）、
    base model（基础模型）和 SFT checkpoint（监督微调检查点）。
    """

    # planner_mode 的中文含义是“规划器模式”。它决定业务查询节点当前选择哪一种
    # QueryPlanner（查询规划器）：rule/local_base/sft/grpo/http_mock。默认 rule，避免
    # 开发环境缺少模型服务时影响现有查询链路。
    planner_mode: str
    # planner_backend 的中文含义是“规划器后端”。9.3.2 只实现 http 客户端；默认 rule
    # 保持现有业务启动不依赖外部模型服务。
    planner_backend: str
    # HTTP endpoint（调用地址）。9.3.2 用它连接本地 mock 或云端同机模型服务。
    planner_model_endpoint: str
    # model_id（模型身份）。写入请求体，模型服务和审计日志用它识别当前 Planner 模型。
    planner_model_id: str
    # timeout（超时时间）。单位秒，避免模型服务无响应时阻塞查询链路。
    planner_timeout_seconds: float
    # max_new_tokens（最大生成 token 数）。Planner 只输出短 JSON，默认 128 足够。
    planner_max_new_tokens: int
    # temperature（采样温度）。Planner 需要稳定结构化决策，默认 0。
    planner_temperature: float
    # enable_thinking（是否启用思考模式）。Planner 不保存私有思维链，默认关闭。
    planner_enable_thinking: bool
    # api_key（接口密钥）。本地 Ollama/mock 可以为空；vLLM 云端服务如启用 --api-key，
    # PlannerClient 会把它写入 Authorization Bearer 头。
    planner_api_key: str = ""


def _env_int(name: str, default: int) -> int:
    value = env_str(name)
    if value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


planner_model_config = PlannerModelConfig(
    planner_mode=env_str("PLANNER_MODE", "rule"),
    planner_backend=env_str("PLANNER_BACKEND", "rule"),
    planner_model_endpoint=env_str("PLANNER_MODEL_ENDPOINT"),
    planner_model_id=env_str("PLANNER_MODEL_ID", "qwen3.5:4b"),
    planner_timeout_seconds=env_float("PLANNER_TIMEOUT_SECONDS", 30.0),
    planner_max_new_tokens=_env_int("PLANNER_MAX_NEW_TOKENS", 128),
    planner_temperature=env_float("PLANNER_TEMPERATURE", 0.0),
    planner_enable_thinking=env_bool("PLANNER_ENABLE_THINKING", False),
    planner_api_key=env_str("PLANNER_API_KEY"),
)
