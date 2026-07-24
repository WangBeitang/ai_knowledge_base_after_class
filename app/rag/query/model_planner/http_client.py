"""PlannerClient（规划器客户端）HTTP（超文本传输协议）接入。"""

from __future__ import annotations

from typing import Any

import requests
from pydantic import BaseModel, ConfigDict, Field

from app.rag.query.contracts import PlannerContext, PlannerDecision
from app.rag.query.model_planner.decision_codec import DecisionDecodeResult, decode_decision
from app.rag.query.model_planner.prompt_builder import PlannerPrompt, build_planner_prompt
from app.shared.config.planner_model_config import PlannerModelConfig, planner_model_config


class PlannerClientError(RuntimeError):
    """
    PlannerClient（规划器客户端）结构化错误。

    error_code（错误码）用于 Trace（轨迹记录）和后续 Reward（奖励函数）扣分归因；details
    （错误详情）只保存可审计的状态码、响应摘要或解析失败原因，不保存模型私有思维链。
    """

    def __init__(self, error_code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(f"{error_code}: {message}")
        self.error_code = error_code
        self.message = message
        self.details = dict(details or {})


class PlannerHttpModel(BaseModel):
    """Planner HTTP（规划器 HTTP）内部 schema（结构）基类。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


class PlannerHttpResult(PlannerHttpModel):
    """
    一次 PlannerClient（规划器客户端）调用结果。

    raw_output（原始输出）来自模型服务；decision（规划器决策）只有在 decode_result.success
    （解析成功）为 True 时才可以交给查询图执行。
    """

    decision: PlannerDecision
    raw_output: str = Field(min_length=1)
    decode_result: DecisionDecodeResult
    model_id: str = Field(min_length=1)
    endpoint: str = Field(min_length=1)
    prompt_hash: str = Field(min_length=1)
    request_payload: dict[str, Any]


class PlannerClient:
    """
    调用 PlannerModelServer（规划器模型服务）的同步 HTTP 客户端。

    9.3.2 只负责“发 prompt（提示词）并解析 JSON（结构化数据）”。它不执行 Milvus（向量
    数据库）、Mongo（文档数据库）或 Web（网页检索），也不负责在 rule/local_base/sft
    （规则/未微调基础模型/监督微调模型）之间切换。
    """

    def __init__(
            self,
            config: PlannerModelConfig | None = None,
            *,
            session: requests.Session | None = None,
    ) -> None:
        self.config = config or planner_model_config
        self._session = session or requests.Session()

    @property
    def policy_version(self) -> str:
        """策略版本；9.3.2 先用 HTTP endpoint（调用地址）和 model_id（模型身份）标识。"""

        return f"http:{self.config.planner_model_id}"

    def plan(self, context: PlannerContext) -> PlannerDecision:
        """调用模型服务并返回 PlannerDecision（规划器决策）。"""

        return self.request_decision(context).decision

    def request_decision(self, context: PlannerContext) -> PlannerHttpResult:
        """
        调用模型服务，解析并校验输出。

        HTTP（超文本传输协议）错误、空响应、非法 JSON（结构化数据）和未知 Action（动作）
        都转换为 PlannerClientError（规划器客户端错误），避免坏输出进入 Action 路由。
        """

        if not isinstance(context, PlannerContext):
            raise TypeError("context 必须是 PlannerContext")
        prompt = build_planner_prompt(context)
        request_payload = self.build_request_payload(prompt)
        raw_output = self._post_chat_completion(request_payload)
        decode_result = decode_decision(raw_output, allowed_actions=context.allowed_actions)
        if not decode_result.success or decode_result.decision is None:
            raise PlannerClientError(
                decode_result.error_code or "planner_output_invalid",
                decode_result.error_message or "模型输出无法解析为 PlannerDecision",
                details={
                    "raw_output_excerpt": decode_result.raw_output_excerpt,
                    "model_id": self.config.planner_model_id,
                    "prompt_hash": prompt.payload_hash,
                },
            )
        return PlannerHttpResult(
            decision=decode_result.decision,
            raw_output=raw_output,
            decode_result=decode_result,
            model_id=self.config.planner_model_id,
            endpoint=self.config.planner_model_endpoint,
            prompt_hash=prompt.payload_hash,
            request_payload=request_payload,
        )

    def generate_text(
            self,
            *,
            prompt: str,
            context: PlannerContext,
            prompt_hash: str,
            prompt_payload: dict[str, Any],
    ) -> str:
        """
        作为 ModelPlanner（模型规划器）的 generate_text（文本生成函数）使用。

        context/prompt_hash/prompt_payload（上下文/提示词哈希/提示词载荷）保留给审计和未来
        registry（注册表）接入；本函数只负责返回模型服务原始文本。
        """

        del context, prompt_hash, prompt_payload
        request_payload = self.build_request_payload_from_text(prompt)
        return self._post_chat_completion(request_payload)

    def build_request_payload(self, prompt: PlannerPrompt) -> dict[str, Any]:
        """从 PlannerPrompt（规划器提示词）构造 OpenAI-compatible（兼容 OpenAI）请求体。"""

        return self.build_request_payload_from_text(prompt.prompt)

    def build_request_payload_from_text(self, prompt_text: str) -> dict[str, Any]:
        """
        构造 chat completions（聊天补全）请求体。

        max_tokens（最大生成 token 数）是 OpenAI-compatible（兼容 OpenAI）字段；配置名
        保留 max_new_tokens 是为了和训练配置语义一致。

        enable_thinking（是否启用思考模式）会同时映射到 reasoning_effort（推理强度）、
        顶层字段和 chat_template_kwargs（聊天模板参数），让 Ollama（本地大模型运行器）、
        vLLM/SGLang（大模型推理服务框架）尽量使用同一份业务侧调用代码。
        """

        model_id = str(self.config.planner_model_id or "").strip()
        if not model_id:
            raise PlannerClientError("model_id_missing", "PLANNER_MODEL_ID 不能为空")
        return {
            "model": model_id,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是设备知识库 Planner（规划器）。只输出严格 JSON（结构化数据），"
                        "不要输出解释、Markdown 或私有思维链。enable_thinking=false。"
                    ),
                },
                {"role": "user", "content": prompt_text},
            ],
            "temperature": self.config.planner_temperature,
            "max_tokens": self.config.planner_max_new_tokens,
            "stream": False,
            "reasoning_effort": "low" if self.config.planner_enable_thinking else "none",
            "enable_thinking": self.config.planner_enable_thinking,
            "chat_template_kwargs": {
                "enable_thinking": self.config.planner_enable_thinking,
            },
        }

    def _post_chat_completion(self, request_payload: dict[str, Any]) -> str:
        endpoint = str(self.config.planner_model_endpoint or "").strip()
        if not endpoint:
            raise PlannerClientError("endpoint_missing", "PLANNER_MODEL_ENDPOINT 不能为空")
        try:
            response = self._session.post(
                endpoint,
                json=request_payload,
                timeout=self.config.planner_timeout_seconds,
            )
        except requests.Timeout as exc:
            raise PlannerClientError(
                "http_timeout",
                f"PlannerModelServer 请求超时：{self.config.planner_timeout_seconds}s",
                details={"endpoint": endpoint, "model_id": self.config.planner_model_id},
            ) from exc
        except requests.RequestException as exc:
            raise PlannerClientError(
                "http_request_failed",
                f"PlannerModelServer 请求失败：{exc}",
                details={"endpoint": endpoint, "model_id": self.config.planner_model_id},
            ) from exc

        if not 200 <= int(response.status_code) < 300:
            raise PlannerClientError(
                "http_status_error",
                f"PlannerModelServer 返回非 2xx 状态码：{response.status_code}",
                details={
                    "endpoint": endpoint,
                    "status_code": response.status_code,
                    "response_excerpt": str(response.text or "")[:500],
                },
            )

        try:
            response_payload = response.json()
        except ValueError as exc:
            raise PlannerClientError(
                "response_json_invalid",
                "PlannerModelServer 响应不是合法 JSON",
                details={"response_excerpt": str(response.text or "")[:500]},
            ) from exc
        raw_output = _extract_openai_chat_content(response_payload)
        if not raw_output.strip():
            raise PlannerClientError(
                "empty_model_content",
                "PlannerModelServer 响应没有可用 message.content",
                details={"response_payload_excerpt": str(response_payload)[:500]},
            )
        return raw_output


def _extract_openai_chat_content(response_payload: Any) -> str:
    """
    提取 OpenAI-compatible chat completions（兼容 OpenAI 聊天补全）正文。

    优先读取 choices[0].message.content；同时兼容 choices[0].text 和 content parts（分段正文）。
    """

    if not isinstance(response_payload, dict):
        return ""
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0] if isinstance(choices[0], dict) else {}
    message = first_choice.get("message") if isinstance(first_choice.get("message"), dict) else {}
    content = message.get("content")
    if content is None:
        content = first_choice.get("text")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_content_part_text(item) for item in content)
    return ""


def _content_part_text(part: Any) -> str:
    if isinstance(part, str):
        return part
    if not isinstance(part, dict):
        return ""
    text = part.get("text")
    if isinstance(text, str):
        return text
    if part.get("type") == "text" and isinstance(part.get("content"), str):
        return str(part["content"])
    return ""
