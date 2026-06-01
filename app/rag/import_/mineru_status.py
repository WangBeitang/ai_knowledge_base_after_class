from enum import StrEnum
from typing import Any


class MinerUExtractState(StrEnum):
    WAITING_FILE = "waiting-file"
    PENDING = "pending"
    RUNNING = "running"
    CONVERTING = "converting"
    DONE = "done"
    FAILED = "failed"


MINERU_RUNNING_STATES = {
    MinerUExtractState.WAITING_FILE,
    MinerUExtractState.PENDING,
    MinerUExtractState.RUNNING,
    MinerUExtractState.CONVERTING,
}

MINERU_ERROR_CODES: dict[str, tuple[str, str]] = {
    "A0202": ("Token 错误", "检查 Token 是否正确，请检查是否有 Bearer 前缀或者更换新 Token"),
    "A0211": ("Token 过期", "更换新 Token"),
    "-500": ("传参错误", "请确保参数类型及 Content-Type 正确"),
    "-10001": ("服务异常", "请稍后再试"),
    "-10002": ("请求参数错误", "检查请求参数格式"),
    "-60001": ("生成上传 URL 失败", "请稍后再试"),
    "-60002": (
        "获取匹配的文件格式失败",
        "检测文件类型失败，请求的文件名及链接中需带有正确的后缀名，且文件为 pdf、doc、docx、ppt、pptx、xls、xlsx、png、jpg、jpeg 中的一种",
    ),
    "-60003": ("文件读取失败", "请检查文件是否损坏并重新上传"),
    "-60004": ("空文件", "请上传有效文件"),
    "-60005": ("文件大小超出限制", "检查文件大小，最大支持 200MB"),
    "-60006": ("文件页数超过限制", "请拆分文件后重试"),
    "-60007": ("模型服务暂时不可用", "请稍后重试或联系技术支持"),
    "-60008": ("文件读取超时", "检查 URL 可访问"),
    "-60009": ("任务提交队列已满", "请稍后再试"),
    "-60010": ("解析失败", "请稍后再试"),
    "-60011": ("获取有效文件失败", "请确保文件已上传"),
    "-60012": ("找不到任务", "请确保 task_id 有效且未删除"),
    "-60013": ("没有权限访问该任务", "只能访问自己提交的任务"),
    "-60014": ("删除运行中的任务", "运行中的任务暂不支持删除"),
    "-60015": ("文件转换失败", "可以手动转为 pdf 再上传"),
    "-60016": ("文件转换失败", "文件转换为指定格式失败，可以尝试其他格式导出或重试"),
    "-60017": ("重试次数达到上限", "等后续模型升级后重试"),
    "-60018": ("每日解析任务数量已达上限", "明日再来"),
    "-60019": ("html 文件解析额度不足", "明日再来"),
    "-60020": ("文件拆分失败", "请稍后重试"),
    "-60021": ("读取文件页数失败", "请稍后重试"),
    "-60022": ("网页读取失败", "可能因网络问题或者限频导致读取失败，请稍后重试"),
}


def _normalize_code(code: Any) -> str:
    return str(code).strip()


def is_success_code(code: Any) -> bool:
    return _normalize_code(code) == "0"


def get_error_message(code: Any, msg: str | None = None) -> str:
    code_key = _normalize_code(code)
    error_info = MINERU_ERROR_CODES.get(code_key)

    if error_info:
        description, suggestion = error_info
        message = f"业务状态码为{code_key}，说明：{description}，解决建议：{suggestion}"
    else:
        message = f"业务状态码为{code_key}"

    if msg:
        message = f"{message}，接口信息：{msg}"

    return message


def parse_extract_state(value: Any) -> MinerUExtractState:
    state_value = str(value).strip()
    try:
        return MinerUExtractState(state_value)
    except ValueError as exc:
        raise ValueError(f"未知 MinerU 解析状态：{value}") from exc


def is_running_state(state: MinerUExtractState) -> bool:
    return state in MINERU_RUNNING_STATES
