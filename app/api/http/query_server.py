import sys
import uuid
from datetime import datetime, timezone
from mimetypes import guess_type
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from app.api.schema.conversation_schema import (
    ConversationDeleteSchema,
    ConversationDetailSchema,
    ConversationListSchema,
    ConversationMessageSchema,
)
from app.api.schema.planner_schema import PlannerStatusSchema
from app.api.schema.query_schema import QueryRequestParam, QueryStreamResponse, QueryNotStreamResponse, HistoryItem, \
    HistoryResponse, ClearHistoryResponse, QueryTaskStatusResponse
from app.api.schema.retrieval_test_schema import (
    RetrievalReplayRequest,
    RetrievalReplayResponse,
    RetrievalTestRunRequest,
    RetrievalTestRunResponse,
)
from app.api.schema.trace_feedback_schema import (
    TraceFeedbackCreateRequest,
    TraceFeedbackListSchema,
    TraceFeedbackSchema,
)
from app.api.http.request_context import get_current_user_id
from app.infra.persistence.history_repository import history_repository
from app.infra.persistence.retrieval_trace_repository import get_retrieval_trace_repository
from app.shared.config.knowledge_base_config import DEFAULT_DATASET_ID, DEFAULT_TENANT_ID
from app.shared.runtime.logger import PROJECT_ROOT, logger, step_log
from app.infra.config.providers import settings
from app.process.query.agent.main_graph import query_graph_app
from app.process.query.agent.state import create_query_default_state,QueryGraphState
from app.rag.query.query_identifier_service import extract_query_identifiers
from app.rag.query.config import (
    POLICY_VERSION,
    RETRIEVAL_CONFIG_VERSION,
    WEB_FALLBACK_ENABLED,
    build_retrieval_config_snapshot,
    normalize_retrieval_mode,
)
from app.rag.query.contracts import Citation, PlannerReasonCode
from app.rag.query.trace_service import safe_create_running_trace, safe_fail_trace
from app.rag.management.access_control_service import (
    AccessControlError,
    PermissionDeniedError,
    get_access_control_service,
)
from app.rag.management.conversation_management_service import get_conversation_management_service
from app.rag.management.planner_management_service import get_planner_status
from app.rag.management.retrieval_test_service import RetrievalTestError, run_retrieval_test
from app.rag.management.trace_feedback_service import TraceNotFoundError, get_trace_feedback_service
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
        history_persistence_enabled: bool = True,
        execution_source: str = "chat",
        replay_of_trace_id: str | None = None,
        config_match_status: str = "unknown",
        corpus_match_status: str = "unknown",
        retrieval_mode: str | None = None,
        web_fallback_enabled: bool | None = None,
):
    """
    构造一次查询运行时上下文并执行 LangGraph。

    用户身份和 dataset 范围必须由 HTTP 边界完成校验后显式传入，不能在图内再读取
    Request，也不能回退到 anonymous_user。这样每个节点看到的都是同一份稳定上下文，
    后续权限过滤、Planner Trace 和评测重放都可以直接复用这些字段。
    """
    normalized_retrieval_mode = normalize_retrieval_mode(retrieval_mode)
    effective_web_fallback_enabled = (
        WEB_FALLBACK_ENABLED
        if web_fallback_enabled is None
        else bool(web_fallback_enabled)
    )
    retrieval_config_snapshot = build_retrieval_config_snapshot(
        retrieval_mode=normalized_retrieval_mode,
        web_fallback_enabled=effective_web_fallback_enabled,
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
        retrieval_mode=normalized_retrieval_mode.value,
        retrieval_config_snapshot=retrieval_config_snapshot,
        web_search_allowed=effective_web_fallback_enabled,
        # 阶段 6 路线 B：真实查询必须读取 Mongo 人工禁用覆盖层，并在 Milvus expr 中
        # 追加 chunk_id not in [...]，否则管理端禁用不会影响召回。
        chunk_status_filter_enabled=True,
        # 只有真实查询入口开启持久化；直接调用 graph 的单元测试和离线重放保持无 Mongo I/O。
        trace_persistence_enabled=True,
        history_persistence_enabled=history_persistence_enabled,
        execution_source=execution_source,
        replay_of_trace_id=replay_of_trace_id,
        config_match_status=config_match_status,
        corpus_match_status=corpus_match_status,
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


def _require_dataset_read_scope(user_id: str, dataset_ids: list[str]) -> None:
    try:
        access = get_access_control_service()
    except Exception as error:
        if all(dataset_id == DEFAULT_DATASET_ID for dataset_id in dataset_ids):
            logger.warning("Mongo 配置不可用，默认 dataset 权限校验在本地测试中跳过")
            return
        raise
    for dataset_id in dataset_ids:
        try:
            access.require_dataset_read(
                dataset_id=dataset_id,
                user_id=user_id,
                tenant_id=DEFAULT_TENANT_ID,
            )
        except Exception as error:
            if dataset_id == DEFAULT_DATASET_ID:
                logger.warning("Mongo 配置不可用，默认 dataset 权限校验在本地测试中跳过")
                continue
            raise


def _raise_management_error(error: Exception) -> None:
    if isinstance(error, TraceNotFoundError):
        raise HTTPException(status_code=404, detail=str(error)) from error
    if isinstance(error, PermissionDeniedError):
        raise HTTPException(status_code=403, detail=str(error)) from error
    if isinstance(error, AccessControlError):
        raise HTTPException(status_code=404, detail=str(error)) from error
    if isinstance(error, RetrievalTestError):
        raise HTTPException(status_code=422, detail=str(error)) from error
    raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/query")
def query(request: Request, back_ground_tasks: BackgroundTasks, request_param: QueryRequestParam):
    # 身份校验必须发生在创建 SSE 队列或调度后台任务之前。缺失/空白 header 会在这里
    # 直接返回 400，保证无身份请求不会进入 LangGraph，也不会留下伪任务状态。
    owner_user_id = get_current_user_id(request)
    session_id = request_param.session_id or str(uuid.uuid4())
    is_stream = request_param.is_stream
    query = request_param.query
    dataset_ids = request_param.dataset_ids
    try:
        _require_dataset_read_scope(owner_user_id, dataset_ids)
    except Exception as error:
        _raise_management_error(error)

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
def history(session_id: str, request: Request, limit: int = 10):
    owner_user_id = get_current_user_id(request)
    message_list = history_repository.list_recent(session_id, limit, user_id=owner_user_id)
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
def clear_history(session_id: str, request: Request):
    owner_user_id = get_current_user_id(request)
    delete_count = history_repository.clear_session(session_id, user_id=owner_user_id)
    return ClearHistoryResponse(
        message=f"删除:{session_id}会话对应的聊天记录成功!!",
        deleted_count=delete_count
    )


@app.get("/conversations", response_model=ConversationListSchema)
def list_conversations(request: Request, limit: int = Query(default=50, ge=1, le=100)):
    owner_user_id = get_current_user_id(request)
    return ConversationListSchema(**get_conversation_management_service().list_conversations(
        user_id=owner_user_id,
        limit=limit,
    ))


@app.get("/conversations/{session_id}", response_model=ConversationDetailSchema)
def conversation_detail(
        request: Request,
        session_id: str,
        limit: int = Query(default=50, ge=1, le=100),
):
    owner_user_id = get_current_user_id(request)
    result = get_conversation_management_service().get_conversation(
        user_id=owner_user_id,
        session_id=session_id,
        limit=limit,
    )
    return ConversationDetailSchema(
        code=200,
        session_id=session_id,
        items=[
            ConversationMessageSchema(
                id=str(message.get("_id", message.get("id", ""))),
                user_id=str(message.get("user_id") or ""),
                session_id=str(message.get("session_id") or ""),
                role=str(message.get("role") or ""),
                text=str(message.get("text") or ""),
                rewritten_query=str(message.get("rewritten_query") or ""),
                standard_subject_names=list(message.get("standard_subject_names") or []),
                image_urls=list(message.get("image_urls") or []),
                citations=list(message.get("citations") or []),
                trace_id=str(message.get("trace_id") or ""),
                terminal_reason_code=(
                    PlannerReasonCode(message["terminal_reason_code"])
                    if message.get("terminal_reason_code")
                    else None
                ),
                ts=message.get("ts"),
            )
            for message in result["items"]
        ],
    )


@app.delete("/conversations/{session_id}", response_model=ConversationDeleteSchema)
def delete_conversation(request: Request, session_id: str):
    owner_user_id = get_current_user_id(request)
    result = get_conversation_management_service().hide_conversation(
        user_id=owner_user_id,
        session_id=session_id,
    )
    return ConversationDeleteSchema(**result)


@app.get("/planner/status", response_model=PlannerStatusSchema)
def planner_status():
    return PlannerStatusSchema(**get_planner_status())


@app.get("/traces")
def list_traces(
        request: Request,
        session_id: str | None = None,
        dataset_id: str | None = None,
        execution_source: str | None = None,
        limit: int = Query(default=50, ge=1, le=100),
):
    owner_user_id = get_current_user_id(request)
    return get_trace_feedback_service().list_traces(
        user_id=owner_user_id,
        session_id=session_id,
        dataset_id=dataset_id,
        execution_source=execution_source,
        limit=limit,
    )


@app.get("/traces/{trace_id}")
def get_trace(request: Request, trace_id: str):
    owner_user_id = get_current_user_id(request)
    try:
        return get_trace_feedback_service().get_trace(trace_id=trace_id, user_id=owner_user_id)
    except Exception as error:
        _raise_management_error(error)


@app.post("/traces/{trace_id}/feedback", response_model=TraceFeedbackSchema)
def create_trace_feedback(request: Request, trace_id: str, payload: TraceFeedbackCreateRequest):
    owner_user_id = get_current_user_id(request)
    try:
        return TraceFeedbackSchema(**get_trace_feedback_service().create_feedback(
            trace_id=trace_id,
            user_id=owner_user_id,
            payload=payload.model_dump(mode="json"),
        ))
    except Exception as error:
        _raise_management_error(error)


@app.get("/traces/{trace_id}/feedback", response_model=TraceFeedbackListSchema)
def list_trace_feedback(
        request: Request,
        trace_id: str,
        limit: int = Query(default=50, ge=1, le=100),
):
    owner_user_id = get_current_user_id(request)
    try:
        return TraceFeedbackListSchema(**get_trace_feedback_service().list_feedbacks(
            trace_id=trace_id,
            user_id=owner_user_id,
            limit=limit,
        ))
    except Exception as error:
        _raise_management_error(error)


@app.post("/retrieval-tests/runs", response_model=RetrievalTestRunResponse)
def retrieval_test_run(request: Request, payload: RetrievalTestRunRequest):
    owner_user_id = get_current_user_id(request)
    test_user_id = payload.test_user_id or owner_user_id
    if test_user_id != owner_user_id:
        raise HTTPException(status_code=403, detail="阶段 7 第一版不允许替其他用户运行 Retrieval Test")
    try:
        _require_dataset_read_scope(test_user_id, payload.dataset_ids)
        return RetrievalTestRunResponse(**run_retrieval_test(
            query_graph_runner=query_graph_invoke,
            user_id=test_user_id,
            query=payload.query,
            dataset_ids=payload.dataset_ids,
            planner_mode=payload.planner_mode,
            retrieval_mode=payload.retrieval_mode,
            web_fallback_enabled=payload.web_fallback_enabled,
        ))
    except Exception as error:
        _raise_management_error(error)


@app.post("/retrieval-tests/traces/{trace_id}/replay", response_model=RetrievalReplayResponse)
def retrieval_test_replay(request: Request, trace_id: str, payload: RetrievalReplayRequest):
    owner_user_id = get_current_user_id(request)
    try:
        get_trace_feedback_service().get_trace(trace_id=trace_id, user_id=owner_user_id)
        original_trace = get_retrieval_trace_repository().get_trace(trace_id, owner_user_id=owner_user_id)
        if payload.planner_mode != "rule":
            raise RetrievalTestError(f"planner_mode={payload.planner_mode} 尚未注册或未启用")
        if payload.strict_config_match:
            # 当前阶段只做配置和语料状态标记，不伪装成离线环境的完全重放。
            return RetrievalReplayResponse(
                original_trace_id=trace_id,
                replay_trace_id="",
                config_match_status="unknown",
                corpus_match_status="unknown",
                message="阶段 7 已提供 replay API 契约；严格离线环境执行属于阶段 8",
            )
        original_config_snapshot = original_trace.get("retrieval_config_snapshot") or {}
        result = query_graph_invoke(
            session_id=f"replay_{uuid.uuid4().hex}",
            query=str(original_trace.get("original_query") or ""),
            is_stream=False,
            owner_user_id=owner_user_id,
            dataset_ids=list(original_trace.get("dataset_ids") or []),
            history_persistence_enabled=False,
            execution_source="replay",
            replay_of_trace_id=trace_id,
            retrieval_mode=str(original_config_snapshot.get("retrieval_mode") or ""),
            web_fallback_enabled=original_config_snapshot.get("web_fallback_enabled"),
            config_match_status="unknown",
            corpus_match_status="unknown",
        )
        return RetrievalReplayResponse(
            original_trace_id=trace_id,
            replay_trace_id=str(result.get("trace_id") or ""),
            config_match_status="unknown",
            corpus_match_status="unknown",
            message="已按当前在线配置执行一次 replay",
        )
    except Exception as error:
        _raise_management_error(error)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.app_host, port=settings.query_app_port)
