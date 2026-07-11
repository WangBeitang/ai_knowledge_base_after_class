import uuid
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from app.infra.object_storage.minio_gateway import minio_gateway
from app.infra.persistence.import_metadata_repository import (
    DEFAULT_TENANT_ID,
    DEFAULT_VISIBILITY,
    STATUS_COMPLETED,
    STATUS_DELETED,
    STATUS_FAILED,
    get_import_metadata_repository,
)
from app.rag.import_.index_service import remove_old_chunks
from app.shared.utils.path_util import PROJECT_ROOT


class DocumentNotFoundError(ValueError):
    """document 不存在，或不属于当前用户。API 将其映射为 404。"""


class DocumentStateError(ValueError):
    """document 存在，但当前状态不允许执行生命周期操作。"""


class RebuildPreparation(TypedDict):
    task_id: str
    document_id: str
    dataset_id: str
    owner_user_id: str
    tenant_id: str
    visibility: str
    index_version: int
    source_file_path: Path
    local_dir: Path


def _require_operable_document(document: dict, document_id: str, operation: str) -> None:
    """
    删除和重建只允许作用于已经结束当前导入任务的 document。

    uploaded/processing document 的旧后台任务仍可能继续写入 chunk。如果此时执行
    删除或重建，会出现“刚清理完又被旧任务写回”的竞态，因此统一拒绝。
    """
    if document.get("status") not in {STATUS_COMPLETED, STATUS_FAILED}:
        raise DocumentStateError(f"document_id={document_id} 当前正在处理，不能{operation}")


def delete_document(document_id: str, owner_user_id: str) -> dict:
    """
    清理当前用户 document 的检索产物，并将 Mongo document 标记为软删除。

    外部资源先清理，Mongo deleted 状态最后写入。任何清理异常都会继续抛出，
    document 保持非 deleted，调用方可以利用删除操作的幂等性重试。
    """
    repo = get_import_metadata_repository()
    document = repo.get_document(document_id, owner_user_id)
    if not document:
        # 不存在和 owner 不匹配统一表现为 not found，避免泄露其他用户资源。
        raise DocumentNotFoundError(f"document_id={document_id} 不存在")
    if document.get("status") == STATUS_DELETED:
        return document

    _require_operable_document(document, document_id, "删除")

    # 单个 document 只拥有自己的 chunk 和图片。标准主题/别名是全局知识体系，
    # 可能被其他 document 复用，不能在这里联动删除。
    remove_old_chunks(document_id)
    minio_gateway.delete_image_prefix(document.get("image_prefix", ""))

    return repo.mark_document_deleted(
        document_id=document_id,
        owner_user_id=owner_user_id,
    )


def prepare_document_rebuild(
    document_id: str,
    owner_user_id: str,
) -> RebuildPreparation:
    """
    校验重建条件、创建 rebuild task，并返回后台导入图所需参数。

    本方法不依赖 FastAPI，也不直接执行导入图；API 负责注册 persistent task，
    再通过 BackgroundTasks 调用已有 invoke_graph。
    """
    repo = get_import_metadata_repository()
    document = repo.get_document(document_id, owner_user_id)
    if not document:
        raise DocumentNotFoundError(f"document_id={document_id} 不存在")
    if document.get("status") == STATUS_DELETED:
        raise DocumentStateError(f"document_id={document_id} 已删除，不能重建索引")

    _require_operable_document(document, document_id, "重建索引")

    source_file_path = Path(document.get("file_path", ""))
    if not source_file_path.is_file():
        # 文件检查必须早于 task 创建和 index_version 递增，避免留下无法执行的任务。
        raise DocumentStateError(f"document_id={document_id} 的原始文件不存在，无法重建索引")

    task_id = str(uuid.uuid4())
    local_dir = PROJECT_ROOT / "output" / datetime.now().strftime("%Y%m%d") / task_id
    local_dir.mkdir(parents=True, exist_ok=True)

    updated_document, _task = repo.create_rebuild_task_metadata(
        document_id=document_id,
        task_id=task_id,
        owner_user_id=owner_user_id,
        local_dir=str(local_dir),
    )

    return RebuildPreparation(
        task_id=task_id,
        document_id=document_id,
        dataset_id=updated_document["dataset_id"],
        owner_user_id=owner_user_id,
        tenant_id=updated_document.get("tenant_id", DEFAULT_TENANT_ID),
        visibility=updated_document.get("visibility", DEFAULT_VISIBILITY),
        index_version=int(updated_document["index_version"]),
        source_file_path=source_file_path,
        local_dir=local_dir,
    )
