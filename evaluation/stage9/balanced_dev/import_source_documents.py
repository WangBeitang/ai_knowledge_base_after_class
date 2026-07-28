"""把 9.3.13 来源 PDF 通过生产导入图写入 Mongo/Milvus。

本脚本不从 PDF 页码直接伪造 ``chunk_id``。它复用线上上传入口背后的同一条
LangGraph 导入图，依次执行 PDF 解析、Markdown 图片处理、生产切分、主题识别、
BGE embedding（向量化）和 Milvus 索引。成功后才从 Milvus 回读真实的
``document_id + chunk_id + index_version``，并冻结不含正文的 chunk 清单。

安全边界：

- 固定 ``document_id`` 与来源 SHA256；同 ID 内容漂移时拒绝覆盖。
- 已完成且 SHA256 一致的文档只复用，不重复解析或重建索引。
- 已存在但未完成的文档不自动删除，避免脚本掩盖导入失败或误删外部状态。
- 清单只保存 chunk 内容哈希、长度和标题，不复制整份受版权保护的手册正文。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from app.api.http.import_server import invoke_graph  # noqa: E402
from app.infra.persistence.import_metadata_repository import (  # noqa: E402
    STATUS_COMPLETED,
    ImportMetadataRepository,
)
from app.infra.vectorstore.milvus_gateway import milvus_gateway  # noqa: E402
from app.rag.query.chunk_retrieval_utils import CHUNK_OUTPUT_FIELDS  # noqa: E402
from app.shared.utils.escape_milvus_string_utils import escape_milvus_string  # noqa: E402
from app.shared.utils.task_utils import (  # noqa: E402
    add_done_task,
    add_running_task,
    register_persistent_task,
)


IMPORT_VERSION = "stage9-balanced-dev-source-import-v1"
DEFAULT_SOURCE_MANIFEST = (
    PROJECT_ROOT / "evaluation/stage9/configs/balanced_dev_source_manifest_v1.json"
)
DEFAULT_OUTPUT_MANIFEST = (
    PROJECT_ROOT / "evaluation/stage9/artifacts/balanced_dev/source_import_manifest.json"
)
DEFAULT_IMPORT_ROOT = PROJECT_ROOT / "output/stage9_balanced_dev"


class ImportModel(BaseModel):
    """导入清单公共 schema（数据结构）；拒绝静默增加未知字段。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SourceSpec(ImportModel):
    """一份待导入来源文档及其不可变身份。"""

    source_id: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    source_role: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    local_path: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64)


class SourceManifest(ImportModel):
    """任务 9.3.13 的输入来源清单。"""

    manifest_version: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    owner_user_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    visibility: str = Field(min_length=1)
    sources: list[SourceSpec] = Field(min_length=1)


class ChunkIdentity(ImportModel):
    """生产切分后回读的真实 chunk 身份，不保存正文。"""

    document_id: str
    chunk_id: int | str
    index_version: int = Field(ge=1)
    chunk_index: int = Field(ge=0)
    title: str
    parent_title: str
    content_chars: int = Field(ge=1)
    content_sha256: str = Field(min_length=64, max_length=64)


class DocumentImportRecord(ImportModel):
    """一份文档的导入结果和 chunk 冻结摘要。"""

    source_id: str
    publisher: str
    title: str
    source_version: str
    source_url: str
    source_sha256: str = Field(min_length=64, max_length=64)
    document_id: str
    task_id: str
    dataset_id: str
    owner_user_id: str
    tenant_id: str
    visibility: str
    index_version: int = Field(ge=1)
    subject_id: str = Field(min_length=1)
    standard_subject_name: str
    chunk_count: int = Field(ge=1)
    import_action: str
    chunks: list[ChunkIdentity] = Field(min_length=1)


class SourceImportManifest(ImportModel):
    """生产导入后的机器可读冻结清单。"""

    import_version: str = IMPORT_VERSION
    source_manifest_version: str
    imported_at: str
    source_manifest_path: str
    source_manifest_sha256: str = Field(min_length=64, max_length=64)
    documents: list[DocumentImportRecord] = Field(min_length=1)


def sha256_file(path: Path) -> str:
    """流式计算文件 SHA256，避免把大型 PDF 一次性读入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source_manifest(path: Path) -> SourceManifest:
    """读取并验证来源清单与本地 PDF 内容身份。"""

    manifest = SourceManifest.model_validate_json(path.read_text(encoding="utf-8"))
    document_ids = [source.document_id for source in manifest.sources]
    source_ids = [source.source_id for source in manifest.sources]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("来源清单包含重复 document_id")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("来源清单包含重复 source_id")

    for source in manifest.sources:
        local_path = PROJECT_ROOT / source.local_path
        if not local_path.is_file():
            raise FileNotFoundError(f"来源文件不存在：{source.local_path}")
        actual_sha256 = sha256_file(local_path)
        if actual_sha256 != source.sha256:
            raise ValueError(
                f"来源文件 SHA256 漂移：source_id={source.source_id}, "
                f"expected={source.sha256}, actual={actual_sha256}"
            )
    return manifest


def _query_chunks(
    document: dict[str, Any],
    *,
    max_attempts: int = 8,
    retry_seconds: float = 1.0,
) -> list[dict[str, Any]]:
    """按当前文档版本回读全部生产 chunk，并等待 Milvus 新写入变为可见。

    导入图完成时 Mongo 状态和 Milvus insert 已成功，但 Milvus 默认一致性下新实体可能
    尚未立刻对 query 可见。这里以 Mongo ``chunk_count`` 为完成条件做有限重试；超过
    门限仍不一致才报错，不能把“暂时查到 0 条”冻结成正式清单。
    """

    document_id = str(document["document_id"])
    index_version = int(document["index_version"])
    filter_expr = (
        f'document_id == "{escape_milvus_string(document_id)}" '
        f"AND index_version == {index_version}"
    )
    expected_count = max(1, int(document.get("chunk_count") or 1))
    chunks: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        chunks = milvus_gateway.query_entities(
            collection_name=milvus_gateway.chunk_collection_name,
            filter_expr=filter_expr,
            output_fields=CHUNK_OUTPUT_FIELDS,
            limit=expected_count,
        )
        if len(chunks) == expected_count:
            break
        if attempt < max_attempts:
            print(
                f"[balanced_dev_import] document_id={document_id} "
                f"milvus_visible={len(chunks)}/{expected_count} "
                f"retry={attempt}/{max_attempts - 1}",
                flush=True,
            )
            time.sleep(retry_seconds)
    return sorted(
        chunks,
        key=lambda row: (int(row.get("chunk_index") or 0), str(row.get("chunk_id") or "")),
    )


def _freeze_chunks(document: dict[str, Any]) -> list[ChunkIdentity]:
    """把 Milvus 内容转换成可审计但不复制正文的身份记录。"""

    chunks = _query_chunks(document)
    expected_count = int(document.get("chunk_count") or 0)
    if len(chunks) != expected_count:
        raise ValueError(
            f"document_id={document['document_id']} 的 Mongo chunk_count={expected_count}，"
            f"但 Milvus 回读到 {len(chunks)} 条"
        )
    frozen: list[ChunkIdentity] = []
    for chunk in chunks:
        content = str(chunk.get("content") or "").strip()
        if not content:
            raise ValueError(f"document_id={document['document_id']} 存在空正文 chunk")
        frozen.append(
            ChunkIdentity(
                document_id=str(document["document_id"]),
                chunk_id=chunk["chunk_id"],
                index_version=int(document["index_version"]),
                chunk_index=int(chunk.get("chunk_index") or 0),
                title=str(chunk.get("title") or ""),
                parent_title=str(chunk.get("parent_title") or ""),
                content_chars=len(content),
                content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            )
        )
    return frozen


def _verify_reusable_document(
    repo: ImportMetadataRepository,
    source: SourceSpec,
    *,
    owner_user_id: str,
) -> dict[str, Any] | None:
    """只复用已完成且来源哈希完全一致的固定文档。"""

    document = repo.get_document(source.document_id, owner_user_id)
    if not document:
        document = repo.get_document_by_id(source.document_id)
    if not document:
        return None
    if document.get("owner_user_id") != owner_user_id:
        raise ValueError(f"document_id={source.document_id} 已被其他 owner 占用")
    if document.get("status") != STATUS_COMPLETED:
        raise ValueError(
            f"document_id={source.document_id} 已存在但 status={document.get('status')}，"
            "请先人工排查，脚本不会自动删除或覆盖"
        )
    if document.get("source_sha256") != source.sha256:
        raise ValueError(
            f"document_id={source.document_id} 已存在但 source_sha256 不一致，拒绝覆盖"
        )
    if int(document.get("chunk_count") or 0) <= 0:
        raise ValueError(f"document_id={source.document_id} 已完成但没有 chunk")
    return document


def import_source(
    repo: ImportMetadataRepository,
    manifest: SourceManifest,
    source: SourceSpec,
    *,
    import_root: Path,
) -> tuple[dict[str, Any], str]:
    """幂等导入单份来源，返回完成态 document 与 inserted/reused。"""

    reusable = _verify_reusable_document(
        repo,
        source,
        owner_user_id=manifest.owner_user_id,
    )
    if reusable is not None:
        return reusable, "reused"

    task_id = f"stage9-balanced-dev-{uuid.uuid4().hex}"
    local_dir = import_root / source.document_id / task_id
    target_file = local_dir / Path(source.local_path).name
    source_file = PROJECT_ROOT / source.local_path

    document, _task = repo.create_import_metadata(
        dataset_id=manifest.dataset_id,
        document_id=source.document_id,
        task_id=task_id,
        owner_user_id=manifest.owner_user_id,
        file_name=target_file.name,
        file_path=str(target_file),
        local_dir=str(local_dir),
        tenant_id=manifest.tenant_id,
        visibility=manifest.visibility,
    )
    register_persistent_task(
        task_id,
        source.document_id,
        str(document["dataset_id"]),
        manifest.owner_user_id,
    )
    add_running_task(task_id, "upload_file")
    local_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source_file, target_file)
    if sha256_file(target_file) != source.sha256:
        raise ValueError(f"复制后的来源文件 SHA256 不一致：source_id={source.source_id}")
    add_done_task(task_id, "upload_file")

    repo.update_document(
        source.document_id,
        source_id=source.source_id,
        source_publisher=source.publisher,
        source_title=source.title,
        source_url=source.source_url,
        source_version=source.source_version,
        source_role=source.source_role,
        source_sha256=source.sha256,
        source_manifest_version=manifest.manifest_version,
    )
    invoke_graph(
        task_id=task_id,
        dataset_id=str(document["dataset_id"]),
        document_id=source.document_id,
        index_version=int(document["index_version"]),
        owner_user_id=manifest.owner_user_id,
        tenant_id=manifest.tenant_id,
        visibility=manifest.visibility,
        local_file_path_obj=target_file,
        local_dir_path_obj=local_dir,
    )
    completed = repo.get_document(source.document_id, manifest.owner_user_id)
    if completed.get("status") != STATUS_COMPLETED:
        raise RuntimeError(
            f"来源导入失败：source_id={source.source_id}, "
            f"status={completed.get('status')}, failed_node={completed.get('failed_node')}, "
            f"error={completed.get('error_message')}"
        )
    return completed, "inserted"


def run_import(
    *,
    source_manifest_path: Path = DEFAULT_SOURCE_MANIFEST,
    output_manifest_path: Path = DEFAULT_OUTPUT_MANIFEST,
    import_root: Path = DEFAULT_IMPORT_ROOT,
    selected_source_ids: set[str] | None = None,
) -> SourceImportManifest:
    """导入指定或全部来源并写出生产 chunk 身份清单。"""

    manifest = load_source_manifest(source_manifest_path)
    sources = [
        source
        for source in manifest.sources
        if selected_source_ids is None or source.source_id in selected_source_ids
    ]
    if not sources:
        raise ValueError("没有选中任何来源")
    if selected_source_ids:
        missing = sorted(selected_source_ids - {source.source_id for source in sources})
        if missing:
            raise ValueError(f"未知 source_id：{missing}")

    repo = ImportMetadataRepository()
    records: list[DocumentImportRecord] = []
    for position, source in enumerate(sources, start=1):
        print(
            f"[balanced_dev_import] source={position}/{len(sources)} "
            f"source_id={source.source_id} status=running",
            flush=True,
        )
        document, import_action = import_source(
            repo,
            manifest,
            source,
            import_root=import_root,
        )
        chunks = _freeze_chunks(document)
        records.append(
            DocumentImportRecord(
                source_id=source.source_id,
                publisher=source.publisher,
                title=source.title,
                source_version=source.source_version,
                source_url=source.source_url,
                source_sha256=source.sha256,
                document_id=source.document_id,
                task_id=str(document["latest_task_id"]),
                dataset_id=str(document["dataset_id"]),
                owner_user_id=str(document["owner_user_id"]),
                tenant_id=str(document["tenant_id"]),
                visibility=str(document["visibility"]),
                index_version=int(document["index_version"]),
                subject_id=str(document.get("subject_id") or ""),
                standard_subject_name=str(document.get("standard_subject_name") or ""),
                chunk_count=len(chunks),
                import_action=import_action,
                chunks=chunks,
            )
        )
        print(
            f"[balanced_dev_import] source={position}/{len(sources)} "
            f"source_id={source.source_id} status=completed chunks={len(chunks)} "
            f"action={import_action}",
            flush=True,
        )

    result = SourceImportManifest(
        source_manifest_version=manifest.manifest_version,
        imported_at=datetime.now(UTC).isoformat(),
        source_manifest_path=str(source_manifest_path.relative_to(PROJECT_ROOT)),
        source_manifest_sha256=sha256_file(source_manifest_path),
        documents=records,
    )
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    """解析 CLI 参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--import-root", type=Path, default=DEFAULT_IMPORT_ROOT)
    parser.add_argument(
        "--source-id",
        action="append",
        default=[],
        help="只导入指定 source_id；可重复传入。省略时导入全部。",
    )
    return parser.parse_args()


def main() -> None:
    """CLI 入口。"""

    args = parse_args()
    result = run_import(
        source_manifest_path=args.source_manifest,
        output_manifest_path=args.output_manifest,
        import_root=args.import_root,
        selected_source_ids=set(args.source_id) or None,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output_manifest),
                "document_count": len(result.documents),
                "chunk_count": sum(document.chunk_count for document in result.documents),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
