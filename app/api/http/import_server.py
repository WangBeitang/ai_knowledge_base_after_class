import shutil
import uuid
from contextlib import asynccontextmanager
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
from app.api.schema.dataset_schema import (
    DatasetCreateRequest,
    DatasetListSchema,
    DatasetMemberDeleteSchema,
    DatasetMemberListSchema,
    DatasetMemberSchema,
    DatasetMemberUpsertRequest,
    DatasetSchema,
    DatasetUpdateRequest,
)
from app.api.schema.document_management_schema import (
    DocumentAccessUpdateRequest,
    DocumentAccessUpdateResponse,
    FailedRecordCleanupRequest,
    FailedRecordCleanupResponse,
)
from app.api.schema.chunk_schema import (
    ChunkBatchStatusChangeRequest,
    ChunkBatchStatusChangeResponse,
    ChunkDetailSchema,
    ChunkEnabledFilter,
    ChunkEventListSchema,
    ChunkListSchema,
    ChunkStatusChangeRequest,
    ChunkStatusChangeResponse,
)
from app.api.http.request_context import get_current_user_id
from app.infra.persistence.import_metadata_repository import (
    DEFAULT_DATASET_ID,
    DEFAULT_TENANT_ID,
    DEFAULT_VISIBILITY,
    get_import_metadata_repository,
    safe_reconcile_interrupted_tasks,
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
from app.rag.import_.chunk_management_service import (
    ChunkManagementError,
    ChunkNotFoundError,
    ChunkPermissionError,
    ChunkStateError,
    ChunkVersionConflictError,
    get_chunk_management_service,
)
from app.rag.management.access_control_service import (
    AccessControlService,
    AccessControlError,
    PermissionDeniedError,
)
from app.rag.management.dataset_management_service import DatasetManagementService
from app.rag.management.document_management_service import (
    DocumentManagementError,
    DocumentStateManagementError,
    DocumentVersionConflictError,
    DocumentManagementService,
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


@asynccontextmanager
async def import_service_lifespan(_app: FastAPI):
    """
    导入服务生命周期边界。

    lifespan 的中文含义是“应用生命周期”。这里在开始接收 HTTP 请求前
    收口旧进程遗留的 pending/processing 导入任务。收口函数内部已捕获
    Mongo 异常，所以可观测数据库暂时不可用不会阻止 API 进程启动。

    当前规则依赖“单进程、单实例”部署假设；未来切换多 worker 或独立
    任务队列时，必须升级为 worker_instance_id + heartbeat/lease，不能继续
    在启动时收口全部非终态任务。
    """
    summary = safe_reconcile_interrupted_tasks()
    logger.info(f"导入服务启动收口结果: {summary}")
    yield


app = FastAPI(
    title=settings.import_app_name,
    description="企业化 RAG 导入服务，负责文件上传、导入执行与状态查询。",
    version="0.3.0",
    lifespan=import_service_lifespan,
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
        owner_user_id=task.get("owner_user_id", ""),
        tenant_id=task.get("tenant_id", ""),
        failed_node=task.get("failed_node", ""),
        error_code=task.get("error_code", ""),
        error_message=task.get("error_message", ""),
        created_at=task.get("created_at", ""),
        updated_at=task.get("updated_at", ""),
    )


def _document_status_from_record(document: dict, code: int = 200) -> DocumentStatusSchema:
    return DocumentStatusSchema(
        code=code,
        document_id=document.get("document_id", ""),
        dataset_id=document.get("dataset_id", ""),
        owner_user_id=document.get("owner_user_id", ""),
        tenant_id=document.get("tenant_id", ""),
        visibility=document.get("visibility", ""),
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
        hidden_at=document.get("hidden_at", ""),
        record_kind=document.get("record_kind", "active"),
        history_group_key=document.get("history_group_key", ""),
        history_record_count=document.get("history_record_count", 0),
        superseded_by_document_id=document.get("superseded_by_document_id", ""),
        access_sync_status=document.get("access_sync_status", "none"),
        access_sync_task_id=document.get("access_sync_task_id", ""),
        failed_node=document.get("failed_node", ""),
        error_code=document.get("error_code", ""),
        error_message=document.get("error_message", ""),
        created_at=document.get("created_at", ""),
        updated_at=document.get("updated_at", ""),
    )


def _memory_task_status(task_id: str, owner_user_id: str) -> TaskStatusSchema:
    metadata = get_persistent_task_metadata(task_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"task_id={task_id} 不存在")
    if metadata.get("owner_user_id") != owner_user_id:
        raise HTTPException(status_code=404, detail=f"task_id={task_id} 不存在")

    return TaskStatusSchema(
        code=200,
        task_id=task_id,
        status=get_task_status(task_id),
        done_list=get_done_task_list(task_id),
        running_list=get_running_task_list(task_id),
        document_id=metadata.get("document_id", ""),
        dataset_id=metadata.get("dataset_id", ""),
        owner_user_id=metadata.get("owner_user_id", ""),
    )


def _raise_management_http_exception(error: Exception) -> None:
    if isinstance(error, PermissionDeniedError):
        raise HTTPException(status_code=403, detail=str(error)) from error
    if isinstance(error, AccessControlError):
        raise HTTPException(status_code=404, detail=str(error)) from error
    if isinstance(error, DocumentVersionConflictError):
        raise HTTPException(status_code=409, detail=str(error)) from error
    if isinstance(error, DocumentStateManagementError):
        raise HTTPException(status_code=409, detail=str(error)) from error
    if isinstance(error, DocumentManagementError):
        raise HTTPException(status_code=400, detail=str(error)) from error
    raise HTTPException(status_code=400, detail=str(error)) from error


def _access_control_service() -> AccessControlService:
    repo = get_import_metadata_repository()
    return AccessControlService(metadata_repository=repo)


def _dataset_management_service() -> DatasetManagementService:
    repo = get_import_metadata_repository()
    access = AccessControlService(metadata_repository=repo)
    return DatasetManagementService(metadata_repository=repo, access_control_service=access)


def _document_management_service() -> DocumentManagementService:
    repo = get_import_metadata_repository()
    access = AccessControlService(metadata_repository=repo)
    return DocumentManagementService(metadata_repository=repo, access_control_service=access)


def _resolve_document_owner_for_write(document_id: str, user_id: str) -> str:
    try:
        document, _role = _access_control_service().require_document_write(
            document_id=document_id,
            user_id=user_id,
            tenant_id=DEFAULT_TENANT_ID,
        )
        return str(document.get("owner_user_id") or user_id)
    except Exception as error:
        logger.warning(
            f"document 写权限预校验不可用，回退旧 owner 调用，document_id={document_id}, "
            f"owner_user_id={user_id}, error={error}"
        )
        return user_id


@app.post("/datasets", response_model=DatasetSchema)
def create_dataset(request: Request, payload: DatasetCreateRequest):
    owner_user_id = get_current_user_id(request)
    return DatasetSchema(**_dataset_management_service().create_dataset(
        user_id=owner_user_id,
        dataset_id=payload.dataset_id,
        name=payload.name,
        description=payload.description,
        visibility=payload.visibility.value,
    ))


@app.get("/datasets", response_model=DatasetListSchema)
def list_datasets(
        request: Request,
        visibility: str | None = None,
        limit: int = Query(default=50, ge=1, le=100),
):
    owner_user_id = get_current_user_id(request)
    return DatasetListSchema(**_dataset_management_service().list_datasets(
        user_id=owner_user_id,
        visibility=visibility,
        limit=limit,
    ))


@app.get("/datasets/{dataset_id}", response_model=DatasetSchema)
def dataset_detail(request: Request, dataset_id: str):
    owner_user_id = get_current_user_id(request)
    try:
        return DatasetSchema(**_dataset_management_service().get_dataset(
            dataset_id=dataset_id,
            user_id=owner_user_id,
        ))
    except Exception as error:
        _raise_management_http_exception(error)


@app.patch("/datasets/{dataset_id}", response_model=DatasetSchema)
def update_dataset(request: Request, dataset_id: str, payload: DatasetUpdateRequest):
    owner_user_id = get_current_user_id(request)
    try:
        return DatasetSchema(**_dataset_management_service().update_dataset(
            dataset_id=dataset_id,
            user_id=owner_user_id,
            fields=payload.model_dump(exclude_none=True),
        ))
    except Exception as error:
        _raise_management_http_exception(error)


@app.get("/datasets/{dataset_id}/members", response_model=DatasetMemberListSchema)
def list_dataset_members(
        request: Request,
        dataset_id: str,
        limit: int = Query(default=100, ge=1, le=200),
):
    owner_user_id = get_current_user_id(request)
    try:
        return DatasetMemberListSchema(**_dataset_management_service().list_members(
            dataset_id=dataset_id,
            user_id=owner_user_id,
            limit=limit,
        ))
    except Exception as error:
        _raise_management_http_exception(error)


@app.post("/datasets/{dataset_id}/members", response_model=DatasetMemberSchema)
def add_dataset_member(request: Request, dataset_id: str, payload: DatasetMemberUpsertRequest):
    owner_user_id = get_current_user_id(request)
    try:
        return DatasetMemberSchema(**_dataset_management_service().upsert_member(
            dataset_id=dataset_id,
            operator_user_id=owner_user_id,
            member_user_id=payload.user_id,
            role=payload.role.value,
        ))
    except Exception as error:
        _raise_management_http_exception(error)


@app.patch("/datasets/{dataset_id}/members/{member_user_id}", response_model=DatasetMemberSchema)
def update_dataset_member(
        request: Request,
        dataset_id: str,
        member_user_id: str,
        payload: DatasetMemberUpsertRequest,
):
    owner_user_id = get_current_user_id(request)
    try:
        return DatasetMemberSchema(**_dataset_management_service().upsert_member(
            dataset_id=dataset_id,
            operator_user_id=owner_user_id,
            member_user_id=member_user_id,
            role=payload.role.value,
        ))
    except Exception as error:
        _raise_management_http_exception(error)


@app.delete("/datasets/{dataset_id}/members/{member_user_id}", response_model=DatasetMemberDeleteSchema)
def delete_dataset_member(request: Request, dataset_id: str, member_user_id: str):
    owner_user_id = get_current_user_id(request)
    try:
        return DatasetMemberDeleteSchema(**_dataset_management_service().remove_member(
            dataset_id=dataset_id,
            operator_user_id=owner_user_id,
            member_user_id=member_user_id,
        ))
    except Exception as error:
        _raise_management_http_exception(error)


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
        fold_history: bool = True,
        limit: int = Query(default=20, ge=1, le=100),
) -> DocumentListSchema:
    owner_user_id = get_current_user_id(request)
    try:
        documents = _document_management_service().list_documents(
            user_id=owner_user_id,
            dataset_id=dataset_id,
            status=status,
            keyword=keyword,
            fold_history=fold_history,
            limit=limit,
        )
    except Exception as error:
        _raise_management_http_exception(error)
    return DocumentListSchema(
        items=[_document_status_from_record(document) for document in documents],
    )


@app.post("/documents/failed-records/cleanup", response_model=FailedRecordCleanupResponse)
def cleanup_failed_records(request: Request, payload: FailedRecordCleanupRequest):
    owner_user_id = get_current_user_id(request)
    try:
        return FailedRecordCleanupResponse(**_document_management_service().cleanup_failed_records(
            user_id=owner_user_id,
            dataset_id=payload.dataset_id,
            document_ids=payload.document_ids,
            only_superseded=payload.only_superseded,
            dry_run=payload.dry_run,
        ))
    except Exception as error:
        _raise_management_http_exception(error)


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
    try:
        document, _role = _access_control_service().require_document_read(
            document_id=document_id,
            user_id=owner_user_id,
            tenant_id=DEFAULT_TENANT_ID,
        )
    except Exception as error:
        _raise_management_http_exception(error)
    return _document_status_from_record(document)


@app.patch("/documents/{document_id}/access", response_model=DocumentAccessUpdateResponse)
def update_document_access(
        request: Request,
        background_tasks: BackgroundTasks,
        document_id: str,
        payload: DocumentAccessUpdateRequest,
):
    owner_user_id = get_current_user_id(request)
    try:
        result = _document_management_service().update_document_access(
            document_id=document_id,
            user_id=owner_user_id,
            expected_index_version=payload.expected_index_version,
            owner_user_id=payload.owner_user_id,
            visibility=payload.visibility.value if payload.visibility else None,
        )
    except Exception as error:
        _raise_management_http_exception(error)

    if result.get("requires_rebuild"):
        task_id = str(result["task_id"])
        register_persistent_task(
            task_id,
            str(result["document_id"]),
            str(result["dataset_id"]),
            str(result["owner_user_id"]),
        )
        background_tasks.add_task(
            invoke_graph,
            task_id=task_id,
            dataset_id=str(result["dataset_id"]),
            document_id=str(result["document_id"]),
            index_version=int(result["index_version"]),
            owner_user_id=str(result["owner_user_id"]),
            tenant_id=str(result.get("tenant_id") or DEFAULT_TENANT_ID),
            visibility=str(result["visibility"]),
            local_file_path_obj=result["source_file_path"],
            local_dir_path_obj=result["local_dir"],
        )

    public_result = {
        key: value
        for key, value in result.items()
        if key not in {"source_file_path", "local_dir", "dataset_id", "tenant_id"}
    }
    return DocumentAccessUpdateResponse(**public_result)


def _raise_chunk_management_http_exception(
        error: ChunkManagementError,
        *,
        document_id: str,
        owner_user_id: str,
) -> None:
    """把 chunk service 的业务异常稳定映射为 HTTP 状态码。"""
    if isinstance(error, ChunkNotFoundError):
        raise HTTPException(status_code=404, detail=str(error)) from error
    if isinstance(error, ChunkPermissionError):
        raise HTTPException(status_code=403, detail=str(error)) from error
    if isinstance(error, (ChunkVersionConflictError, ChunkStateError)):
        raise HTTPException(status_code=409, detail=str(error)) from error
    logger.warning(
        f"chunk 管理请求参数错误，document_id={document_id}, owner_user_id={owner_user_id}, error={error}"
    )
    raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/documents/{document_id}/chunks")
def list_document_chunks(
        request: Request,
        document_id: str,
        enabled: ChunkEnabledFilter = Query(default=ChunkEnabledFilter.ALL),
        limit: int = Query(default=100, ge=1, le=100),
) -> ChunkListSchema:
    owner_user_id = get_current_user_id(request)
    try:
        result = get_chunk_management_service().list_document_chunks(
            document_id=document_id,
            user_id=owner_user_id,
            tenant_id=DEFAULT_TENANT_ID,
            enabled=enabled.to_bool(),
            limit=limit,
        )
        return ChunkListSchema(**result)
    except ChunkManagementError as e:
        _raise_chunk_management_http_exception(
            e,
            document_id=document_id,
            owner_user_id=owner_user_id,
        )
    except Exception as e:
        logger.exception(
            f"查询 document chunk 列表失败，document_id={document_id}, owner_user_id={owner_user_id}, error={e}"
        )
        raise HTTPException(status_code=500, detail="查询 chunk 列表失败") from e


@app.get("/chunks", response_model=ChunkListSchema)
def list_chunks(
        request: Request,
        dataset_id: str = DEFAULT_DATASET_ID,
        document_id: str | None = None,
        enabled: ChunkEnabledFilter = Query(default=ChunkEnabledFilter.ALL),
        limit: int = Query(default=100, ge=1, le=100),
) -> ChunkListSchema:
    owner_user_id = get_current_user_id(request)
    try:
        _access_control_service().require_dataset_read(
            dataset_id=dataset_id,
            user_id=owner_user_id,
            tenant_id=DEFAULT_TENANT_ID,
        )
        return ChunkListSchema(**get_chunk_management_service().list_chunks(
            dataset_id=dataset_id,
            document_id=document_id,
            user_id=owner_user_id,
            tenant_id=DEFAULT_TENANT_ID,
            enabled=enabled.to_bool(),
            limit=limit,
        ))
    except ChunkManagementError as e:
        _raise_chunk_management_http_exception(
            e,
            document_id=document_id or "",
            owner_user_id=owner_user_id,
        )
    except Exception as error:
        _raise_management_http_exception(error)


@app.patch("/chunks/enabled", response_model=ChunkBatchStatusChangeResponse)
def change_chunks_enabled(request: Request, payload: ChunkBatchStatusChangeRequest):
    owner_user_id = get_current_user_id(request)
    try:
        return ChunkBatchStatusChangeResponse(**get_chunk_management_service().change_chunks_enabled(
            items=[item.model_dump(mode="json") for item in payload.items],
            user_id=owner_user_id,
            tenant_id=DEFAULT_TENANT_ID,
            enabled=payload.enabled,
            reason_type=payload.reason_type,
            reason_detail=payload.reason_detail,
        ))
    except Exception as error:
        _raise_management_http_exception(error)


@app.get("/documents/{document_id}/chunks/{chunk_id}")
def chunk_detail(
        request: Request,
        document_id: str,
        chunk_id: str,
) -> ChunkDetailSchema:
    owner_user_id = get_current_user_id(request)
    try:
        result = get_chunk_management_service().get_chunk_detail(
            document_id=document_id,
            chunk_id=chunk_id,
            user_id=owner_user_id,
            tenant_id=DEFAULT_TENANT_ID,
        )
        return ChunkDetailSchema(**result)
    except ChunkManagementError as e:
        _raise_chunk_management_http_exception(
            e,
            document_id=document_id,
            owner_user_id=owner_user_id,
        )
    except Exception as e:
        logger.exception(
            f"查询 chunk 详情失败，document_id={document_id}, chunk_id={chunk_id}, "
            f"owner_user_id={owner_user_id}, error={e}"
        )
        raise HTTPException(status_code=500, detail="查询 chunk 详情失败") from e


@app.patch("/documents/{document_id}/chunks/{chunk_id}/enabled")
def change_chunk_enabled(
        request: Request,
        document_id: str,
        chunk_id: str,
        payload: ChunkStatusChangeRequest,
) -> ChunkStatusChangeResponse:
    owner_user_id = get_current_user_id(request)
    try:
        result = get_chunk_management_service().change_chunk_enabled(
            document_id=document_id,
            chunk_id=chunk_id,
            user_id=owner_user_id,
            tenant_id=DEFAULT_TENANT_ID,
            enabled=payload.enabled,
            expected_index_version=payload.expected_index_version,
            reason_type=payload.reason_type,
            reason_detail=payload.reason_detail,
            trace_id=payload.trace_id,
        )
        return ChunkStatusChangeResponse(**result)
    except ChunkManagementError as e:
        _raise_chunk_management_http_exception(
            e,
            document_id=document_id,
            owner_user_id=owner_user_id,
        )
    except Exception as e:
        logger.exception(
            f"修改 chunk 启停状态失败，document_id={document_id}, chunk_id={chunk_id}, "
            f"owner_user_id={owner_user_id}, error={e}"
        )
        raise HTTPException(status_code=500, detail="修改 chunk 启停状态失败") from e


@app.get("/documents/{document_id}/chunks/{chunk_id}/events")
def list_chunk_events(
        request: Request,
        document_id: str,
        chunk_id: str,
        limit: int = Query(default=20, ge=1, le=100),
) -> ChunkEventListSchema:
    owner_user_id = get_current_user_id(request)
    try:
        result = get_chunk_management_service().list_chunk_events(
            document_id=document_id,
            chunk_id=chunk_id,
            user_id=owner_user_id,
            tenant_id=DEFAULT_TENANT_ID,
            limit=limit,
        )
        return ChunkEventListSchema(**result)
    except ChunkManagementError as e:
        _raise_chunk_management_http_exception(
            e,
            document_id=document_id,
            owner_user_id=owner_user_id,
        )
    except Exception as e:
        logger.exception(
            f"查询 chunk 启停历史失败，document_id={document_id}, chunk_id={chunk_id}, "
            f"owner_user_id={owner_user_id}, error={e}"
        )
        raise HTTPException(status_code=500, detail="查询 chunk 启停历史失败") from e


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
        document = delete_document_service(
            document_id,
            _resolve_document_owner_for_write(document_id, owner_user_id),
        )
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
        preparation = prepare_document_rebuild(
            document_id,
            _resolve_document_owner_for_write(document_id, owner_user_id),
        )
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
        visibility: str = Form(DEFAULT_VISIBILITY),
):
    # 1.相关参数
    owner_user_id = get_current_user_id(request)
    try:
        _access_control_service().require_dataset_write(
            dataset_id=dataset_id,
            user_id=owner_user_id,
            tenant_id=DEFAULT_TENANT_ID,
            allow_default_upload=True,
        )
    except Exception as error:
        _raise_management_http_exception(error)
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
            visibility=visibility,
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
