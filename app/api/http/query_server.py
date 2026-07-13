import sys
import uuid
from datetime import datetime, timezone
from mimetypes import guess_type
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from app.api.schema.query_schema import QueryRequestParam, QueryStreamResponse, QueryNotStreamResponse, HistoryItem, \
    HistoryResponse, ClearHistoryResponse, QueryTaskStatusResponse
from app.api.http.request_context import get_current_user_id
from app.infra.persistence.history_repository import history_repository
from app.shared.config.knowledge_base_config import DEFAULT_TENANT_ID
from app.shared.runtime.logger import PROJECT_ROOT, logger, step_log
from app.infra.config.providers import settings
from app.process.query.agent.main_graph import query_graph_app
from app.process.query.agent.state import create_query_default_state,QueryGraphState
from app.rag.query.query_identifier_service import extract_query_identifiers
from app.rag.query.config import (
    POLICY_VERSION,
    RETRIEVAL_CONFIG_VERSION,
    RETRIEVAL_DEFAULT_MODE,
    WEB_FALLBACK_ENABLED,
    build_retrieval_config_snapshot,
)
from app.rag.query.contracts import Citation, PlannerReasonCode
from app.rag.query.trace_service import safe_create_running_trace, safe_fail_trace
from app.shared.utils.sse_utils import SSEEvent, create_sse_queue, push_to_session, sse_generator
from app.shared.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    clear_task,
    get_done_task_list,
    get_task_result,
    get_running_task_list,
    get_task_status,
    set_task_result,
    update_task_status,
)

# 定义fastapi对象
app = FastAPI(
    title=settings.query_app_name,
    description="描述,进行rag查询的服务对象",
    version="0.2.0"
)

# 跨域处理
app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'],
    allow_methods = ['*'],
    allow_headers = ['*']
)

@app.get("/html")
def html():
    chat_html_path_obj = PROJECT_ROOT/"app/resources/http/chat.html"
    return FileResponse(
        path=chat_html_path_obj
    )

# 健康检查接口
@app.get("/health")
def health():
    return {"status": "ok"}


# /stream/{session_id} 流式响应接口
@app.get("/stream/{session_id}")
def stream(session_id: str, request: Request):
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream"
    )


def query_graph_invoke(
        session_id: str,
        query: str,
        is_stream: bool,
        owner_user_id: str,
        dataset_ids: list[str],
        tenant_id: str = DEFAULT_TENANT_ID,
):
    """
    构造一次查询运行时上下文并执行 LangGraph。

    用户身份和 dataset 范围必须由 HTTP 边界完成校验后显式传入，不能在图内再读取
    Request，也不能回退到 anonymous_user。这样每个节点看到的都是同一份稳定上下文，
    后续权限过滤、Planner Trace 和评测重放都可以直接复用这些字段。
    """
    retrieval_config_snapshot = build_retrieval_config_snapshot(
        retrieval_mode=RETRIEVAL_DEFAULT_MODE,
        web_fallback_enabled=WEB_FALLBACK_ENABLED,
    )
    state = create_query_default_state(
        session_id=session_id,
        original_query=query,
        is_stream=is_stream,
        owner_user_id=owner_user_id,
        tenant_id=tenant_id,
        # Pydantic 已完成规范化，这里再复制一次，避免后台任务和调用方意外共享可变 list。
        dataset_ids=list(dataset_ids),
        # trace 的中文含义是“追踪”。它标识一次完整查询执行，不能复用 session_id，
        # 因为同一聊天会话可以连续发起多次独立查询。
        trace_id=str(uuid.uuid4()),
        # 使用带 UTC 时区的 ISO 8601 时间，后续计算耗时、持久化 Trace 和离线重放时
        # 不依赖部署机器的本地时区。
        query_started_at=datetime.now(timezone.utc).isoformat(),
        # 在进入 LangGraph 前从用户原始问题提取确定性设备标识。该字段忠实保存用户输入，
        # 后续检索发现的 E021 等相近候选只能进入 suggested_identifiers，不能覆盖 E020。
        query_identifiers=extract_query_identifiers(query),
        policy_version=POLICY_VERSION,
        planner_type="rule",
        retrieval_config_version=RETRIEVAL_CONFIG_VERSION,
        retrieval_mode=RETRIEVAL_DEFAULT_MODE.value,
        retrieval_config_snapshot=retrieval_config_snapshot,
        web_search_allowed=WEB_FALLBACK_ENABLED,
        # 只有真实查询入口开启持久化；直接调用 graph 的单元测试和离线重放保持无 Mongo I/O。
        trace_persistence_enabled=True,
    )

    # 清空task_utils的数据
    clear_task(session_id)
    safe_create_running_trace(state)

    try:
        update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)
        logger.info(f"开始执行,执行参数为:{state}")
        result_state = query_graph_app.invoke(state)
        logger.info(f"执行结束,执行结果为:{result_state}")
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)

        terminal_reason_code = result_state.get("terminal_reason_code")
        terminal_reason_value = (
            terminal_reason_code.value
            if hasattr(terminal_reason_code, "value")
            else str(terminal_reason_code or "")
        )
        citations = [
            item if isinstance(item, Citation) else Citation.model_validate(item)
            for item in result_state.get("citations") or []
        ]
        # 后台轮询与 SSE FINAL 使用同一份结构化最终结果，避免两条交付路径字段漂移。
        for key, value in {
            "answer": result_state.get("answer", ""),
            "image_urls": result_state.get("image_urls", []),
            "trace_id": result_state.get("trace_id", state["trace_id"]),
            "citations": [item.model_dump(mode="json") for item in citations],
            "terminal_reason_code": terminal_reason_value,
            "error": "",
        }.items():
            set_task_result(session_id, key, value)

        if is_stream:
            push_to_session(
                session_id,
                SSEEvent.FINAL,  # 显示图片
                {
                    "answer": result_state['answer'],
                    "status": "completed",
                    "image_urls": result_state.get("image_urls", []),
                    "trace_id": result_state.get("trace_id", state["trace_id"]),
                    "citations": [item.model_dump(mode="json") for item in citations],
                    "terminal_reason_code": terminal_reason_value,
                }
            )
        # 同步执行返回结果
        return result_state
    except Exception as e:
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})
        set_task_result(session_id, "error", str(e))
        set_task_result(session_id, "trace_id", state.get("trace_id", ""))
        safe_fail_trace(state, e)
        # 编程错误必须继续抛给 FastAPI/后台任务框架形成 500 和失败日志，不能返回 None 后
        # 再被包装成“证据不足”。阶段 10 会在这里追加持久化 Trace failed；阶段 9 先保证
        # 异常语义正确且日志始终携带本次独立 trace_id。
        logger.exception(f"执行失败,trace_id={state.get('trace_id')},错误信息为:{e}")
        raise

@app.post("/query")
def query(request: Request, back_ground_tasks: BackgroundTasks, request_param: QueryRequestParam):
    # 身份校验必须发生在创建 SSE 队列或调度后台任务之前。缺失/空白 header 会在这里
    # 直接返回 400，保证无身份请求不会进入 LangGraph，也不会留下伪任务状态。
    owner_user_id = get_current_user_id(request)
    session_id = request_param.session_id or str(uuid.uuid4())
    is_stream = request_param.is_stream
    query = request_param.query
    dataset_ids = request_param.dataset_ids

    if is_stream:
        # 流式响应
        # 创建 SSE 队列
        create_sse_queue(session_id)

        # 异步执行
        back_ground_tasks.add_task(
            query_graph_invoke,
            session_id=session_id,
            query=query,
            is_stream=is_stream,
            owner_user_id=owner_user_id,
            dataset_ids=dataset_ids,
        )

        return QueryStreamResponse(
            session_id=session_id,
            message=f"查询任务{session_id}开始执行，请稍后..."
        )
    else:
        # 同步响应
        result: QueryGraphState = query_graph_invoke(
            session_id=session_id,
            query=query,
            is_stream=is_stream,
            owner_user_id=owner_user_id,
            dataset_ids=dataset_ids,
        )

        return QueryNotStreamResponse(
            session_id=session_id,
            answer=result.get("answer"),
            done_list=get_done_task_list(session_id),
            image_urls=result.get("image_urls", []),
            trace_id=result.get("trace_id", ""),
            citations=result.get("citations", []),
            terminal_reason_code=result.get("terminal_reason_code"),
            message=f"{session_id}查询结束"
        )


@app.get("/status/{session_id}", response_model=QueryTaskStatusResponse)
def query_status(session_id: str):
    """返回进程内查询任务快照；用于 SSE 断线后的轻量轮询兜底。"""
    reason_value = get_task_result(session_id, "terminal_reason_code", "")
    return QueryTaskStatusResponse(
        session_id=session_id,
        status=get_task_status(session_id),
        done_list=get_done_task_list(session_id),
        running_list=get_running_task_list(session_id),
        answer=get_task_result(session_id, "answer", ""),
        error=get_task_result(session_id, "error", ""),
        image_urls=get_task_result(session_id, "image_urls", []),
        trace_id=get_task_result(session_id, "trace_id", ""),
        citations=get_task_result(session_id, "citations", []),
        terminal_reason_code=(PlannerReasonCode(reason_value) if reason_value else None),
    )

@app.get("/history/{session_id}")
def history(session_id: str, limit: int = 10):
    message_list = history_repository.list_recent(session_id, limit)
    message_list.reverse()
    items = [
        HistoryItem(
            id=str(message.get("_id")),
            session_id=message.get("session_id"),
            role=message.get("role"),
            text=message.get("text"),
            rewritten_query=message.get("rewritten_query", ""),
            standard_subject_names=message.get("standard_subject_names", []),
            image_urls=message.get("image_urls", []),
            citations=message.get("citations", []),
            trace_id=message.get("trace_id", ""),
            terminal_reason_code=(
                PlannerReasonCode(message["terminal_reason_code"])
                if message.get("terminal_reason_code")
                else None
            ),
            ts=message.get("ts")
        )
        for message in message_list
    ]
    return HistoryResponse(session_id=session_id, items=items)

@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    delete_count = history_repository.clear_session(session_id)
    return ClearHistoryResponse(
        message=f"删除:{session_id}会话对应的聊天记录成功!!",
        deleted_count=delete_count
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.app_host, port=settings.query_app_port)
