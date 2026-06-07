import shutil
import uuid
from datetime import datetime
from mimetypes import guess_type
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware

from app.api.schema.import_schema import TaskStatusSchema, UploadSchema
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.process.import_.agent.main_graph import kb_import_app
from app.process.import_.agent.state import get_default_state, ImportGraphState, create_default_state
from app.infra.config.providers import settings
from app.shared.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    get_done_task_list,
    get_running_task_list,
    get_task_status,
    update_task_status, add_running_task, add_done_task,
)

app = FastAPI(
    title=settings.import_app_name,
    description="企业化 RAG 导入服务，负责文件上传、导入执行与状态查询。",
    version="0.3.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins) or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/html")
def html():
    html_path_obj = PROJECT_ROOT / "app" / "resources" / "http" / "import.html"
    return FileResponse(path=html_path_obj)

# 返回task_id对应的任务状态列表
@app.get("/status/{task_id}")
def status(task_id: str) -> TaskStatusSchema:
    return TaskStatusSchema(
        code=200,
        task_id=task_id,
        status=get_task_status(task_id),
        done_list=get_done_task_list(task_id),
        running_list=get_running_task_list(task_id)
    )

def invoke_graph(task_id: str, local_file_path_obj: Path, local_dir_path_obj: Path):
    state = create_default_state(
        task_id=task_id,
        local_file_path=str(local_file_path_obj),
        local_dir=str(local_dir_path_obj)
    )

    try:
        logger.info(f"{task_id}对应的文件解析任务开始!参数：{state}")
        update_task_status(task_id, TASK_STATUS_PROCESSING)

        final_state = kb_import_app.invoke(state)

        logger.info(f"{task_id}对应的文件解析任务完成!最终结果: {final_state}")
        update_task_status(task_id, TASK_STATUS_COMPLETED)
    except Exception as e:
        update_task_status(task_id, TASK_STATUS_FAILED)
        logger.exception(f"===== 全流程测试运行失败 =====,错误信息：{e}")



# 上传文件
@app.post("/upload")
def upload(background_tasks: BackgroundTasks, files: UploadFile = File(...)):
    # 1.相关参数
    task_id = str(uuid.uuid4())
    add_running_task(task_id, "upload_file")
    local_dir_path_obj = PROJECT_ROOT / "output" / datetime.now().strftime("%Y%m%d") / task_id
    local_dir_path_obj.mkdir(parents=True, exist_ok=True)
    file_path_obj = local_dir_path_obj / files.filename

    # 2.保存文件
    with file_path_obj.open("wb") as buffer:
        shutil.copyfileobj(files.file, buffer)
    add_done_task(task_id, "upload_file")

    # 3.异步调用
    background_tasks.add_task(
        invoke_graph,
        task_id=task_id,
        local_file_path_obj=file_path_obj,
        local_dir_path_obj=local_dir_path_obj
    )

    # 4.主线程返回结果
    return UploadSchema(
        code=200,
        message="上传成功，正在处理中...",
        task_ids=[task_id]
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.app_host, port=settings.import_app_port)
