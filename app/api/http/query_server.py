from mimetypes import guess_type
from pathlib import Path
import sys
import uuid

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from app.api.schema.query_schema import QueryRequestParam, QueryStreamResponse, QueryNotStreamResponse
from app.shared.runtime.logger import PROJECT_ROOT, logger, step_log
from app.infra.config.providers import settings
from app.process.query.agent.main_graph import query_graph_app
from app.process.query.agent.state import create_query_default_state,QueryGraphState
from app.shared.utils.sse_utils import SSEEvent, create_sse_queue, push_to_session, sse_generator
from app.shared.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    clear_task,
    get_done_task_list,
    get_task_result,
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


def query_graph_invoke(session_id: str, query: str, is_stream: bool):
    state = create_query_default_state(
        session_id=session_id,
        query=query,
        is_stream=is_stream
    )

    # 清空task_utils的数据
    clear_task(session_id)

    try:
        update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)
        logger.info(f"开始执行,执行参数为:{state}")
        result_state = query_graph_app.invoke(state)
        logger.info(f"执行结束,执行结果为:{result_state}")
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)

        image_urls = ["https://q1.itc.cn/images01/20260529/75a13a50d752432db9b9b751b846ecb6.jpeg"]
        result_state["image_urls"] = image_urls

        push_to_session(
            session_id,
            SSEEvent.FINAL,
            {
                "answer": result_state['answer'],
                "status": "completed",
                "image_urls": image_urls
            }
        )
        # 同步执行返回结果
        return result_state
    except Exception as e:
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})
        logger.exception(f"执行失败,错误信息为:{e}")

@app.post("/query")
def query(back_ground_tasks: BackgroundTasks, request_param: QueryRequestParam):
    session_id = request_param.session_id or str(uuid.uuid4())
    is_stream = request_param.is_stream
    query = request_param.query

    if is_stream:
        # 流式响应
        # 创建 SSE 队列
        create_sse_queue(session_id)

        # 异步执行
        back_ground_tasks.add_task(
            query_graph_invoke,
            session_id=session_id,
            query=query,
            is_stream=is_stream
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
            is_stream=is_stream
        )

        return QueryNotStreamResponse(
            session_id=session_id,
            answer=result.get("answer"),
            done_list=get_done_task_list(session_id),
            image_urls=result.get("image_urls"),
            message=f"{session_id}查询结束"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.app_host, port=settings.query_app_port)