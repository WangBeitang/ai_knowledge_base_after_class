import shutil
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware

from app.api.schema.import_schema import (
    DeleteDocumentSchema,
    DocumentListSchema,
    DocumentStatusSchema,
    RebuildDocumentSchema,
    TaskHistorySchema,
    TaskStatusSchema,
    UploadSchema,
)
from app.infra.persistence.import_metadata_repository import (
    DEFAULT_DATASET_ID,
    DEFAULT_TENANT_ID,
    DEFAULT_VISIBILITY,
    get_import_metadata_repository,
    safe_mark_import_completed,
    safe_mark_import_failed,
)
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.process.import_.agent.main_graph import kb_import_app
from app.process.import_.agent.state import create_default_state
from app.infra.config.providers import settings
from app.rag.import_.document_lifecycle_service import (
    DocumentNotFoundError,
    DocumentStateError,
    delete_document as delete_document_service,
    prepare_document_rebuild,
)
from app.shared.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    get_done_task_list,
    get_last_running_task_node_name,
    get_persistent_task_metadata,
    get_running_task_list,
    get_task_status,
    register_persistent_task,
    to_display_node_list,
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


def get_current_user_id(request: Request) -> str:
    user_id = request.headers.get("X-User-Id", "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="缺少 X-User-Id 请求头")
    return user_id


def _task_status_from_record(task: dict, code: int = 200) -> TaskStatusSchema:
    return TaskStatusSchema(
        code=code,
        task_id=task.get("task_id", ""),
        task_type=task.get("task_type", ""),
        status=task.get("status", ""),
        done_list=to_display_node_list(task.get("done_nodes", [])),
        running_list=to_display_node_list(task.get("running_nodes", [])),
        document_id=task.get("document_id", ""),
        dataset_id=task.get("dataset_id", ""),
        failed_node=task.get("failed_node", ""),
        error_message=task.get("error_message", ""),
        created_at=task.get("created_at", ""),
        updated_at=task.get("updated_at", ""),
    )


def _document_status_from_record(document: dict, code: int = 200) -> DocumentStatusSchema:
    return DocumentStatusSchema(
        code=code,
        document_id=document.get("document_id", ""),
        dataset_id=document.get("dataset_id", ""),
        latest_task_id=document.get("latest_task_id", ""),
        file_name=document.get("file_name", ""),
        file_path=document.get("file_path", ""),
        local_dir=document.get("local_dir", ""),
        index_version=document.get("index_version", 0),
        status=document.get("status", ""),
        parse_status=document.get("parse_status", ""),
        index_status=document.get("index_status", ""),
        chunk_count=document.get("chunk_count", 0),
        subject_id=document.get("subject_id", ""),
        standard_subject_name=document.get("standard_subject_name", ""),
        md_path=document.get("md_path", ""),
        image_prefix=document.get("image_prefix", ""),
        parse_result_zip_path=document.get("parse_result_zip_path", ""),
        parse_result_dir=document.get("parse_result_dir", ""),
        deleted_at=document.get("deleted_at", ""),
        failed_node=document.get("failed_node", ""),
        error_message=document.get("error_message", ""),
        created_at=document.get("created_at", ""),
        updated_at=document.get("updated_at", ""),
    )


def _memory_task_status(task_id: str, owner_user_id: str) -> TaskStatusSchema:
    metadata = get_persistent_task_metadata(task_id)
    if metadata and metadata.get("owner_user_id") != owner_user_id:
        raise HTTPException(status_code=404, detail=f"task_id={task_id} 不存在")

    return TaskStatusSchema(
        code=200,
        task_id=task_id,
        status=get_task_status(task_id),
        done_list=get_done_task_list(task_id),
        running_list=get_running_task_list(task_id)
    )


# 返回task_id对应的任务状态列表
@app.get("/status/{task_id}")
def status(request: Request, task_id: str) -> TaskStatusSchema:
    owner_user_id = get_current_user_id(request)
    try:
        task = get_import_metadata_repository().get_task(task_id, owner_user_id)
        if task:
            return _task_status_from_record(task)
    except Exception as e:
        logger.warning(f"查询 Mongo 任务状态失败，回退内存状态，task_id={task_id}, error={e}")
        return _memory_task_status(task_id, owner_user_id)

    raise HTTPException(status_code=404, detail=f"task_id={task_id} 不存在")


@app.get("/documents")
def list_documents(
        request: Request,
        dataset_id: str = DEFAULT_DATASET_ID,
        status: str | None = None,
        keyword: str | None = None,
        limit: int = Query(default=20, ge=1, le=100),
) -> DocumentListSchema:
    owner_user_id = get_current_user_id(request)
    documents = get_import_metadata_repository().list_documents(
        owner_user_id=owner_user_id,
        dataset_id=dataset_id,
        status=status,
        keyword=keyword,
        limit=limit,
    )
    return DocumentListSchema(
        items=[_document_status_from_record(document) for document in documents],
    )


@app.get("/documents/{document_id}/tasks")
def list_document_tasks(
        request: Request,
        document_id: str,
        limit: int = Query(default=20, ge=1, le=100),
) -> TaskHistorySchema:
    owner_user_id = get_current_user_id(request)
    tasks = get_import_metadata_repository().list_tasks(
        document_id=document_id,
        owner_user_id=owner_user_id,
        limit=limit,
    )
    return TaskHistorySchema(
        document_id=document_id,
        items=[_task_status_from_record(task) for task in tasks],
    )


@app.get("/documents/{document_id}")
def document_status(request: Request, document_id: str) -> DocumentStatusSchema:
    owner_user_id = get_current_user_id(request)
    document = get_import_metadata_repository().get_document(document_id, owner_user_id)
    if not document:
        raise HTTPException(status_code=404, detail=f"document_id={document_id} 不存在")
    return _document_status_from_record(document)


def invoke_graph(
        task_id: str,
        dataset_id: str,
        document_id: str,
        index_version: int,
        owner_user_id: str,
        local_file_path_obj: Path,
        local_dir_path_obj: Path,
        tenant_id: str = DEFAULT_TENANT_ID,
        visibility: str = DEFAULT_VISIBILITY,
):
    state = create_default_state(
        task_id=task_id,
        dataset_id=dataset_id,
        document_id=document_id,
        index_version=index_version,
        owner_user_id=owner_user_id,
        tenant_id=tenant_id,
        visibility=visibility,
        local_file_path=str(local_file_path_obj),
        local_dir=str(local_dir_path_obj)
    )

    try:
        logger.info(f"{task_id}对应的文件解析任务开始!参数：{state}")
        update_task_status(task_id, TASK_STATUS_PROCESSING)

        final_state = kb_import_app.invoke(state)

        logger.info(
            f"{task_id} 对应的文件解析任务完成! "
            f"chunks数量={len(final_state.get('chunks', []))}, "
            f"subject_id={final_state.get('subject_id')}, "
            f"file_title={final_state.get('file_title')}, "
            f"md_content长度={len(final_state.get('md_content', ''))}"
        )
        update_task_status(task_id, TASK_STATUS_COMPLETED)
        safe_mark_import_completed(task_id)
    except Exception as e:
        failed_node = get_last_running_task_node_name(task_id)
        update_task_status(task_id, TASK_STATUS_FAILED)
        safe_mark_import_failed(task_id, failed_node, str(e))
        logger.exception(f"===== 全流程测试运行失败 =====,错误信息：{e}")


@app.delete("/documents/{document_id}")
def delete_document(
        request: Request,
        document_id: str,
) -> DeleteDocumentSchema:
    owner_user_id = get_current_user_id(request)
    try:
        document = delete_document_service(document_id, owner_user_id)
    except DocumentNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except DocumentStateError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        logger.exception(
            f"删除document失败，document_id={document_id}, owner_user_id={owner_user_id}, error={e}"
        )
        raise HTTPException(status_code=500, detail="删除文档失败") from e

    return DeleteDocumentSchema(
        message="文档删除成功",
        document_id=document_id,
        status=document.get("status", "deleted"),
        deleted_at=document.get("deleted_at", ""),
    )


@app.post("/documents/{document_id}/rebuild")
def rebuild_document(
        request: Request,
        background_tasks: BackgroundTasks,
        document_id: str,
) -> RebuildDocumentSchema:
    owner_user_id = get_current_user_id(request)
    try:
        preparation = prepare_document_rebuild(document_id, owner_user_id)
    except DocumentNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except DocumentStateError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        logger.exception(
            f"创建重建索引任务失败，document_id={document_id}, owner_user_id={owner_user_id}, error={e}"
        )
        raise HTTPException(status_code=500, detail="创建重建索引任务失败") from e

    task_id = preparation["task_id"]
    register_persistent_task(
        task_id,
        preparation["document_id"],
        preparation["dataset_id"],
        preparation["owner_user_id"],
    )
    background_tasks.add_task(
        invoke_graph,
        task_id=task_id,
        dataset_id=preparation["dataset_id"],
        document_id=preparation["document_id"],
        index_version=preparation["index_version"],
        owner_user_id=preparation["owner_user_id"],
        tenant_id=preparation["tenant_id"],
        visibility=preparation["visibility"],
        local_file_path_obj=preparation["source_file_path"],
        local_dir_path_obj=preparation["local_dir"],
    )

    return RebuildDocumentSchema(
        message="重建索引任务已创建",
        task_id=task_id,
        document_id=preparation["document_id"],
        dataset_id=preparation["dataset_id"],
        index_version=preparation["index_version"],
    )



# 上传文件
@app.post("/upload")
def upload(
        request: Request,
        background_tasks: BackgroundTasks,
        files: UploadFile = File(...),
        dataset_id: str = Form(DEFAULT_DATASET_ID),
):
    # 1.相关参数
    owner_user_id = get_current_user_id(request)
    task_id = str(uuid.uuid4())
    document_id = f"doc_{uuid.uuid4().hex}"
    local_dir_path_obj = PROJECT_ROOT / "output" / datetime.now().strftime("%Y%m%d") / task_id
    file_path_obj = local_dir_path_obj / files.filename
    repo = get_import_metadata_repository()
    try:
        document, _task = repo.create_import_metadata(
            dataset_id=dataset_id,
            document_id=document_id,
            task_id=task_id,
            owner_user_id=owner_user_id,
            file_name=files.filename,
            file_path=str(file_path_obj),
            local_dir=str(local_dir_path_obj),
        )
        dataset_id = document["dataset_id"]
        index_version = document.get("index_version", 0)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    register_persistent_task(task_id, document_id, dataset_id, owner_user_id)
    add_running_task(task_id, "upload_file")

    # 2.保存文件
    try:
        local_dir_path_obj.mkdir(parents=True, exist_ok=True)
        with file_path_obj.open("wb") as buffer:
            shutil.copyfileobj(files.file, buffer)
        add_done_task(task_id, "upload_file")
    except Exception as e:
        update_task_status(task_id, TASK_STATUS_FAILED)
        safe_mark_import_failed(task_id, "upload_file", str(e))
        logger.exception(f"{task_id}上传文件保存失败，错误信息：{e}")
        raise HTTPException(status_code=500, detail=f"上传文件保存失败：{e}") from e

    # 3.异步调用
    background_tasks.add_task(
        invoke_graph,
        task_id=task_id,
        dataset_id=dataset_id,
        document_id=document_id,
        index_version=index_version,
        owner_user_id=owner_user_id,
        local_file_path_obj=file_path_obj,
        local_dir_path_obj=local_dir_path_obj
    )

    # 4.主线程返回结果
    return UploadSchema(
        code=200,
        message="上传成功，正在处理中...",
        task_ids=[task_id],
        document_ids=[document_id],
        dataset_id=dataset_id,
        owner_user_id=owner_user_id,
        index_version=index_version,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.app_host, port=settings.import_app_port)
