"""统一执行 answer、ask_clarification 和 refuse 三种终止 Action。"""

from app.process.query.agent.state import QueryGraphState
from app.rag.query import answer_service
from app.rag.query.contracts import PlannerReasonCode, QueryAction
from app.shared.runtime.logger import logger, node_log
from app.shared.utils.task_utils import add_done_task, add_running_task

from app.process.query.agent.nodes.node_retrieval_observation import (
    current_decision,
    is_expected_external_error,
)


_REFUSE_TEXT = {
    PlannerReasonCode.SUBJECT_NOT_FOUND: "当前知识库中没有找到您提到的设备，请核对设备型号或标准名称后再试。",
    PlannerReasonCode.LOCAL_EMPTY: "当前知识库范围内没有找到可核验的相关资料，暂时无法可靠回答。",
    PlannerReasonCode.LOCAL_LOW_SCORE: "当前检索证据相关性不足，暂时无法可靠回答。",
    PlannerReasonCode.HYDE_STILL_INSUFFICIENT: "扩展检索后证据仍然不足，暂时无法可靠回答。",
    PlannerReasonCode.WEB_EMPTY_OR_FAILED: "本地和联网检索都没有得到可靠证据，暂时无法回答。",
    PlannerReasonCode.ACTION_EXECUTION_ERROR: "检索服务暂时不可用，请稍后重试。",
    PlannerReasonCode.SAFE_GUARD_TRIGGERED: "当前查询无法在安全约束内完成，请调整问题后重试。",
}


def _clarification_text(state: QueryGraphState, reason_code: PlannerReasonCode) -> str:
    """优先返回上游确定性追问；缺失时按 reason code 生成稳定兜底文本。"""
    if state.get("clarification_question"):
        return str(state["clarification_question"])
    if reason_code == PlannerReasonCode.SUBJECT_AMBIGUOUS:
        candidates = "、".join(state.get("subject_candidates", []))
        return f"请确认您要咨询的设备：{candidates}。" if candidates else "请确认您要咨询的具体设备。"
    if reason_code == PlannerReasonCode.SUBJECT_REQUIRED:
        return "请说明您要咨询的设备型号或标准设备名称。"
    return "当前信息存在歧义，请补充更明确的设备或问题描述。"


def _deliver_existing_text(state: QueryGraphState, text: str) -> None:
    """复用答案服务的 SSE/同步交付和历史保存能力，但不调用答案 LLM。"""
    state["answer"] = text
    state["image_urls"] = []
    state["citations"] = []
    answer_service.try_return_existing_answer(state)
    answer_service.save_assistant_message(state)


@node_log("node_terminal_response")
def node_terminal_response(state: QueryGraphState) -> dict:
    """根据已经校验的终止 Decision 生成答案、追问或拒答文本。"""
    decision = current_decision(state)
    if decision.action not in {
        QueryAction.ANSWER,
        QueryAction.ASK_CLARIFICATION,
        QueryAction.REFUSE,
    }:
        raise ValueError("node_terminal_response 只能执行终止 Action")

    add_running_task(state["session_id"], "node_terminal_response", state.get("is_stream", False))
    terminal_reason_code = decision.reason_code
    # 历史消息保存发生在 answer_service 内部，因此进入具体分支前先写入本次终止原因；
    # 外部答案模型失败时会在调用兜底保存前覆盖成 ACTION_EXECUTION_ERROR。
    state["terminal_reason_code"] = terminal_reason_code
    try:
        if decision.action == QueryAction.ANSWER:
            try:
                result_state = answer_service.generate_answer(state)
            except Exception as error:
                if not is_expected_external_error(error):
                    raise
                logger.warning(
                    f"答案模型可预期失败，trace_id={state.get('trace_id')}, "
                    f"error_type={type(error).__name__}"
                )
                terminal_reason_code = PlannerReasonCode.ACTION_EXECUTION_ERROR
                state["terminal_reason_code"] = terminal_reason_code
                _deliver_existing_text(state, _REFUSE_TEXT[terminal_reason_code])
                result_state = state
        elif decision.action == QueryAction.ASK_CLARIFICATION:
            _deliver_existing_text(state, _clarification_text(state, decision.reason_code))
            result_state = state
        else:
            _deliver_existing_text(
                state,
                _REFUSE_TEXT.get(decision.reason_code, "当前没有足够证据可靠回答这个问题。"),
            )
            result_state = state
    finally:
        add_done_task(state["session_id"], "node_terminal_response", state.get("is_stream", False))

    return {
        "answer": result_state.get("answer", ""),
        "image_urls": result_state.get("image_urls", []),
        "citations": result_state.get("citations", []),
        "answer_runtime_metadata": result_state.get("answer_runtime_metadata", {}),
        "prompt": result_state.get("prompt", ""),
        "clarification_question": result_state.get("clarification_question"),
        "terminal_reason_code": terminal_reason_code,
    }
