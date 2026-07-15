"""
阶段 8 环境快照构建脚本。

Environment snapshot（环境快照）的作用是把一次离线评测依赖的 dataset、document、
chunk 启停状态、检索配置和 Planner 配置固定成 JSON 文件。后续 OfflineRagEnvironment
必须读取这个文件，而不是再次读取当前 Mongo/Milvus 的漂移状态，否则阶段 9 的 SFT/GRPO
对比结果无法复现。

脚本分两层：
- build_environment_snapshot：纯构建逻辑，可用 fake reader 单元测试，不连接外部服务。
- main：命令行入口，负责初始化真实 Mongo/Milvus reader、读取当前配置并原子写文件。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.evaluation.case_schema import (
    EnvironmentSnapshot,
    PlannerEvalCase,
    SnapshotChunkIdentity,
    SnapshotDocument,
    SplitManifest,
    load_planner_cases,
)
from app.rag.query.config import (
    POLICY_VERSION,
    RETRIEVAL_CONFIG_VERSION,
    build_retrieval_config_snapshot,
)
from app.shared.config.knowledge_base_config import DEFAULT_DATASET_ID


DEFAULT_CASE_FILES = (
    PROJECT_ROOT / "evaluation/stage8/cases/planner_cases.jsonl",
    PROJECT_ROOT / "evaluation/stage8/cases/demo_regression_cases.jsonl",
)
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "evaluation/stage8/snapshots/environment_snapshot.json"
DEFAULT_SPLIT_MANIFEST_PATH = PROJECT_ROOT / "evaluation/stage8/cases/split_manifest.json"
DEFAULT_CHUNK_QUERY_LIMIT = 10_000


class SnapshotBuildError(RuntimeError):
    """环境快照构建失败；错误信息面向评测执行者，必须能直接定位数据或配置问题。"""


class MetadataSnapshotReader(Protocol):
    """
    读取 dataset/document 元数据的边界。

    Metadata 的中文含义是“管理元数据”，这里来自 Mongo 的 datasets/documents 集合。
    该协议让单元测试可以注入 fake reader，避免测试阶段连接真实 Mongo。
    """

    def load_datasets(self, dataset_ids: list[str]) -> list[dict[str, Any]]:
        """按 dataset_id 读取未删除 dataset。"""

    def load_documents(self, dataset_ids: list[str]) -> list[dict[str, Any]]:
        """读取可进入评测快照的 completed document。"""


class ChunkSnapshotReader(Protocol):
    """
    读取 Milvus chunk 状态的边界。

    Chunk snapshot 只记录当前 document/index_version 下 Milvus 原始 enabled=true 的 chunk；
    人工禁用覆盖由 ChunkOverrideSnapshotReader 单独读取。
    """

    def load_enabled_chunks(self, document: dict[str, Any]) -> list[dict[str, Any]]:
        """读取某个 document 当前 index_version 下 enabled=true 的 chunk 行。"""


class ChunkOverrideSnapshotReader(Protocol):
    """
    读取 chunk_status_overrides 的边界。

    overrides 是“人工覆盖层”，不是 Milvus 原始 enabled 字段。阶段 8 必须把它冻结进
    snapshot，后续 replay 才不会受当前人工启停状态变化影响。
    """

    def load_disabled_overrides(self, document: dict[str, Any]) -> list[dict[str, Any]]:
        """读取某个 document 当前 index_version 下 manual_status=disabled 的覆盖记录。"""


@dataclass(frozen=True)
class EnvironmentSnapshotBuildResult:
    """
    构建结果。

    snapshot 是可持久化的环境契约；summary 是命令行展示用摘要，帮助快速确认本次快照
    包含多少 dataset、document、enabled chunk、disabled override 和 case。
    """

    snapshot: EnvironmentSnapshot
    summary: dict[str, Any]


class MongoMetadataSnapshotReader:
    """真实 Mongo metadata reader，包装 ImportMetadataRepository 的集合读取。"""

    def __init__(self, repository: Any) -> None:
        self.repository = repository

    def load_datasets(self, dataset_ids: list[str]) -> list[dict[str, Any]]:
        datasets: list[dict[str, Any]] = []
        for dataset_id in dataset_ids:
            dataset = self.repository.get_dataset(dataset_id)
            if dataset:
                datasets.append(_without_mongo_id(dataset))
        return datasets

    def load_documents(self, dataset_ids: list[str]) -> list[dict[str, Any]]:
        from app.infra.persistence.import_metadata_repository import STATUS_COMPLETED

        query = {
            "dataset_id": {"$in": dataset_ids},
            "status": STATUS_COMPLETED,
            "index_status": STATUS_COMPLETED,
            "deleted_at": {"$in": ["", None]},
        }
        cursor = self.repository.documents.find(query).sort([
            ("dataset_id", 1),
            ("document_id", 1),
        ])
        return [_without_mongo_id(document) for document in cursor]


class MilvusChunkSnapshotReader:
    """真实 Milvus chunk reader，读取当前 document/index_version 的 enabled chunk 身份。"""

    def __init__(self, gateway: Any, *, query_limit: int = DEFAULT_CHUNK_QUERY_LIMIT) -> None:
        if query_limit <= 0:
            raise ValueError("query_limit 必须大于 0")
        self.gateway = gateway
        self.query_limit = query_limit

    def load_enabled_chunks(self, document: dict[str, Any]) -> list[dict[str, Any]]:
        from app.rag.query.chunk_retrieval_utils import build_chunk_management_filter

        filter_expr = build_chunk_management_filter(
            dataset_ids=[document["dataset_id"]],
            owner_user_id=str(document.get("owner_user_id") or ""),
            tenant_id=str(document.get("tenant_id") or ""),
            document_id=str(document.get("document_id") or ""),
            index_version=int(document.get("index_version") or 0),
            enabled=True,
        )
        return list(self.gateway.query_entities(
            collection_name=self.gateway.chunk_collection_name,
            filter_expr=filter_expr,
            output_fields=[
                "chunk_id",
                "document_id",
                "index_version",
                "enabled",
                "chunk_index",
            ],
            limit=self.query_limit,
        ))


class MongoChunkOverrideSnapshotReader:
    """真实 Mongo override reader，包装 ChunkStatusRepository.get_overrides。"""

    def __init__(self, repository: Any) -> None:
        self.repository = repository

    def load_disabled_overrides(self, document: dict[str, Any]) -> list[dict[str, Any]]:
        from app.infra.persistence.chunk_status_repository import MANUAL_STATUS_DISABLED

        overrides = self.repository.get_overrides(
            document_id=str(document.get("document_id") or ""),
            index_version=int(document.get("index_version") or 0),
        )
        return [
            _without_mongo_id(override)
            for override in overrides
            if override.get("manual_status") == MANUAL_STATUS_DISABLED
        ]


def build_environment_snapshot(
        *,
        metadata_reader: MetadataSnapshotReader,
        chunk_reader: ChunkSnapshotReader,
        override_reader: ChunkOverrideSnapshotReader,
        cases: list[PlannerEvalCase],
        dataset_ids: list[str] | None = None,
        test_user_ids: list[str] | None = None,
        snapshot_id: str | None = None,
        created_by: str = "stage8_snapshot_builder",
        created_at: str | None = None,
        retrieval_config_version: str = RETRIEVAL_CONFIG_VERSION,
        retrieval_config_snapshot: dict[str, Any] | None = None,
        policy_version: str = POLICY_VERSION,
        reranker: dict[str, Any] | None = None,
        answer_model: dict[str, Any] | None = None,
        planner_registry: list[dict[str, Any]] | None = None,
        source_hashes: dict[str, str] | None = None,
) -> EnvironmentSnapshotBuildResult:
    """
    构建并校验环境快照。

    该函数不写文件，也不连接真实外部服务。调用方必须先把 case、配置和 reader 准备好。
    这样设计是为了保证“校验失败时不生成半截文件”：只有 snapshot 完整通过 Pydantic
    schema 和 case 引用校验后，外层才会执行原子写入。
    """
    normalized_dataset_ids = _normalize_unique_text_list(
        dataset_ids or _collect_referenced_dataset_ids_from_cases(cases) or [DEFAULT_DATASET_ID],
        field_name="dataset_ids",
    )
    normalized_test_user_ids = _normalize_unique_text_list(
        test_user_ids or _collect_test_user_ids_from_cases(cases),
        field_name="test_user_ids",
    )
    normalized_created_by = _require_text(created_by, field_name="created_by")
    normalized_snapshot_id = _require_text(
        snapshot_id or f"stage8-env-{_utc_now_compact_date()}-v1",
        field_name="snapshot_id",
    )

    datasets = metadata_reader.load_datasets(normalized_dataset_ids)
    dataset_ids_found = {str(dataset.get("dataset_id") or "").strip() for dataset in datasets}
    missing_datasets = [
        dataset_id for dataset_id in normalized_dataset_ids
        if dataset_id not in dataset_ids_found
    ]
    if missing_datasets:
        raise SnapshotBuildError(
            "dataset_id 不存在或已删除，无法构建环境快照："
            + ", ".join(missing_datasets)
        )

    raw_documents = metadata_reader.load_documents(normalized_dataset_ids)
    if not raw_documents:
        raise SnapshotBuildError(
            "没有可进入快照的 completed document，请先完成导入或缩小 dataset_ids"
        )

    documents = [_build_snapshot_document(document) for document in raw_documents]
    enabled_chunks: dict[str, list[int | str]] = {}
    chunk_identity_by_document: dict[str, set[tuple[str, int | str, int]]] = {}
    disabled_chunks: list[SnapshotChunkIdentity] = []
    disabled_identity_set: set[tuple[str, int | str, int]] = set()

    for document in raw_documents:
        snapshot_document = _build_snapshot_document(document)
        raw_enabled_chunks = chunk_reader.load_enabled_chunks(document)
        chunk_ids = _extract_enabled_chunk_ids(
            raw_enabled_chunks,
            document_id=snapshot_document.document_id,
            index_version=snapshot_document.index_version,
        )
        enabled_chunks[snapshot_document.document_id] = chunk_ids
        chunk_identity_by_document[snapshot_document.document_id] = {
            (snapshot_document.document_id, chunk_id, snapshot_document.index_version)
            for chunk_id in chunk_ids
        }

        for override in override_reader.load_disabled_overrides(document):
            disabled_chunk = SnapshotChunkIdentity(
                document_id=str(override.get("document_id") or snapshot_document.document_id),
                chunk_id=override.get("chunk_id"),
                index_version=int(override.get("index_version") or snapshot_document.index_version),
            )
            disabled_chunks.append(disabled_chunk)
            disabled_identity_set.add((
                disabled_chunk.document_id,
                disabled_chunk.chunk_id,
                disabled_chunk.index_version,
            ))

    snapshot = EnvironmentSnapshot(
        snapshot_id=normalized_snapshot_id,
        created_at=created_at or _utc_now_iso(),
        created_by=normalized_created_by,
        dataset_ids=normalized_dataset_ids,
        test_user_ids=normalized_test_user_ids,
        documents=documents,
        enabled_chunks=enabled_chunks,
        disabled_chunks=disabled_chunks,
        retrieval_config_version=_require_text(
            retrieval_config_version,
            field_name="retrieval_config_version",
        ),
        retrieval_config_snapshot=retrieval_config_snapshot or build_retrieval_config_snapshot(),
        policy_version=_require_text(policy_version, field_name="policy_version"),
        reranker=reranker or {},
        answer_model=answer_model or {},
        planner_registry=planner_registry or [],
        source_hashes=source_hashes or {},
    )
    validation_summary = validate_cases_against_snapshot(
        cases,
        snapshot=snapshot,
        chunk_identity_by_document=chunk_identity_by_document,
        disabled_identity_set=disabled_identity_set,
    )
    summary = {
        "snapshot_id": snapshot.snapshot_id,
        "dataset_count": len(snapshot.dataset_ids),
        "document_count": len(snapshot.documents),
        "enabled_chunk_count": sum(len(chunk_ids) for chunk_ids in snapshot.enabled_chunks.values()),
        "disabled_chunk_count": len(snapshot.disabled_chunks),
        "case_count": len(cases),
        **validation_summary,
    }
    return EnvironmentSnapshotBuildResult(snapshot=snapshot, summary=summary)


def build_and_write_environment_snapshot(
        *,
        output_path: str | Path,
        metadata_reader: MetadataSnapshotReader,
        chunk_reader: ChunkSnapshotReader,
        override_reader: ChunkOverrideSnapshotReader,
        cases: list[PlannerEvalCase],
        dataset_ids: list[str] | None = None,
        test_user_ids: list[str] | None = None,
        snapshot_id: str | None = None,
        created_by: str = "stage8_snapshot_builder",
        created_at: str | None = None,
        retrieval_config_version: str = RETRIEVAL_CONFIG_VERSION,
        retrieval_config_snapshot: dict[str, Any] | None = None,
        policy_version: str = POLICY_VERSION,
        reranker: dict[str, Any] | None = None,
        answer_model: dict[str, Any] | None = None,
        planner_registry: list[dict[str, Any]] | None = None,
        source_hashes: dict[str, str] | None = None,
) -> EnvironmentSnapshotBuildResult:
    """构建快照并原子写入 JSON 文件。构建失败时不会创建或替换目标文件。"""
    result = build_environment_snapshot(
        metadata_reader=metadata_reader,
        chunk_reader=chunk_reader,
        override_reader=override_reader,
        cases=cases,
        dataset_ids=dataset_ids,
        test_user_ids=test_user_ids,
        snapshot_id=snapshot_id,
        created_by=created_by,
        created_at=created_at,
        retrieval_config_version=retrieval_config_version,
        retrieval_config_snapshot=retrieval_config_snapshot,
        policy_version=policy_version,
        reranker=reranker,
        answer_model=answer_model,
        planner_registry=planner_registry,
        source_hashes=source_hashes,
    )
    write_environment_snapshot(output_path, result.snapshot)
    return result


def validate_cases_against_snapshot(
        cases: list[PlannerEvalCase],
        *,
        snapshot: EnvironmentSnapshot,
        chunk_identity_by_document: dict[str, set[tuple[str, int | str, int]]] | None = None,
        disabled_identity_set: set[tuple[str, int | str, int]] | None = None,
) -> dict[str, Any]:
    """
    校验 case 引用的 document/chunk/index_version 是否属于当前快照。

    这里故意把错误集中后一次性抛出，便于一次修完多个样本；旧 index_version 会给出当前
    snapshot 中的版本号，避免只看到“chunk 不存在”而不知道是重建索引造成的版本漂移。
    """
    document_version_by_id = {
        document.document_id: document.index_version
        for document in snapshot.documents
    }
    if chunk_identity_by_document is None:
        chunk_identity_by_document = {
            document_id: {
                (document_id, chunk_id, document_version_by_id[document_id])
                for chunk_id in chunk_ids
            }
            for document_id, chunk_ids in snapshot.enabled_chunks.items()
            if document_id in document_version_by_id
        }
    disabled_identity_set = disabled_identity_set or {
        (chunk.document_id, chunk.chunk_id, chunk.index_version)
        for chunk in snapshot.disabled_chunks
    }

    errors: list[str] = []
    missing_case_dataset_ids: set[str] = set()
    for case in cases:
        for dataset_id in case.dataset_ids:
            if dataset_id not in snapshot.dataset_ids:
                missing_case_dataset_ids.add(dataset_id)

        for document_id in case.source_document_ids:
            _validate_case_document_version(
                errors,
                case_id=case.case_id,
                document_id=document_id,
                expected_index_version=case.source_index_versions.get(document_id),
                document_version_by_id=document_version_by_id,
            )
        for document_id, index_version in case.source_index_versions.items():
            _validate_case_document_version(
                errors,
                case_id=case.case_id,
                document_id=document_id,
                expected_index_version=index_version,
                document_version_by_id=document_version_by_id,
            )
        for expected_chunk in case.expected_chunks:
            _validate_case_document_version(
                errors,
                case_id=case.case_id,
                document_id=expected_chunk.document_id,
                expected_index_version=expected_chunk.index_version,
                document_version_by_id=document_version_by_id,
            )
            identity = (
                expected_chunk.document_id,
                expected_chunk.chunk_id,
                expected_chunk.index_version,
            )
            if identity in disabled_identity_set:
                errors.append(
                    f"case_id={case.case_id} 引用的 chunk 当前被人工禁用："
                    f"document_id={expected_chunk.document_id}, "
                    f"chunk_id={expected_chunk.chunk_id}, "
                    f"index_version={expected_chunk.index_version}"
                )
            elif identity not in chunk_identity_by_document.get(expected_chunk.document_id, set()):
                errors.append(
                    f"case_id={case.case_id} 引用的 chunk 不在 snapshot 的 enabled_chunks 中："
                    f"document_id={expected_chunk.document_id}, "
                    f"chunk_id={expected_chunk.chunk_id}, "
                    f"index_version={expected_chunk.index_version}"
                )

    if errors:
        raise SnapshotBuildError("case 引用校验失败：\n- " + "\n- ".join(errors))

    return {
        "case_dataset_ids_missing_from_snapshot": sorted(missing_case_dataset_ids),
        "case_reference_check": "passed",
    }


def write_environment_snapshot(path: str | Path, snapshot: EnvironmentSnapshot) -> None:
    """
    原子写入快照文件。

    先写同目录临时文件，再 replace 到目标路径。这样即使进程在写入中途失败，也不会留下
    半截 environment_snapshot.json。
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        temp_path.write_text(
            snapshot.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def read_environment_snapshot(path: str | Path) -> EnvironmentSnapshot:
    """重复读取已生成快照，用于命令行自检和后续 baseline/replay。"""
    return EnvironmentSnapshot.model_validate_json(Path(path).read_text(encoding="utf-8"))


def update_split_manifest_snapshot_id(path: str | Path, snapshot_id: str) -> SplitManifest:
    """把 split_manifest.json 回填到当前 snapshot_id，并原子写回。"""
    manifest_path = Path(path)
    manifest = SplitManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    updated = manifest.model_copy(update={"snapshot_id": snapshot_id})
    temp_path = manifest_path.with_name(f".{manifest_path.name}.tmp")
    try:
        temp_path.write_text(updated.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temp_path.replace(manifest_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return updated


def load_cases_from_files(case_files: list[Path]) -> list[PlannerEvalCase]:
    """读取多个 JSONL case 文件，并执行跨文件 case_id/leakage_group 校验。"""
    cases: list[PlannerEvalCase] = []
    for case_file in case_files:
        cases.extend(load_planner_cases(case_file))
    from app.rag.evaluation.case_schema import validate_case_collection

    validate_case_collection(cases)
    return cases


def build_source_hashes(paths: list[Path]) -> dict[str, str]:
    """记录输入文件 SHA256，便于判断 case 或 manifest 是否和快照生成时一致。"""
    hashes: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        hashes[_display_path(path)] = (
            hashlib.sha256(path.read_bytes()).hexdigest()
        )
    return hashes


def build_default_runtime_metadata() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """
    读取可安全落盘的模型/Planner 摘要。

    这里明确不保存 API key、token 等密钥，只保存模型名、设备、base_url 和注册状态。
    base_url 只用于解释推理供应商/代理环境，如未来认为敏感可以改成 hash。
    """
    from app.infra.config.providers import infra_config
    from app.rag.management.planner_management_service import get_planner_status

    reranker = {
        "model": infra_config.reranker.bge_reranker_large,
        "device": infra_config.reranker.bge_reranker_device,
        "fp16": infra_config.reranker.bge_reranker_fp16,
    }
    answer_model = {
        "base_url": infra_config.llm.base_url,
        "model": infra_config.llm.llm_model,
        "temperature": infra_config.llm.llm_temperature,
    }
    planner_status = get_planner_status()
    planner_registry = list(planner_status.get("registered_planners") or [])
    return reranker, answer_model, planner_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建阶段 8 离线环境快照")
    parser.add_argument(
        "--dataset-id",
        action="append",
        dest="dataset_ids",
        help="要冻结的 dataset_id，可重复传入；默认从 case 文件收集，收集不到时使用默认知识库",
    )
    parser.add_argument(
        "--test-user-id",
        action="append",
        dest="test_user_ids",
        help="固定测试用户 ID，可重复传入；默认从 case.owner_user_id 收集",
    )
    parser.add_argument(
        "--case-file",
        action="append",
        type=Path,
        dest="case_files",
        help="阶段 8 JSONL case 文件，可重复传入",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="environment_snapshot.json 输出路径",
    )
    parser.add_argument(
        "--snapshot-id",
        help="快照 ID；默认 stage8-env-<UTC日期>-v1",
    )
    parser.add_argument(
        "--created-by",
        default="stage8_snapshot_builder",
        help="写入快照的创建者标识",
    )
    parser.add_argument(
        "--chunk-query-limit",
        type=int,
        default=DEFAULT_CHUNK_QUERY_LIMIT,
        help="单个 document 从 Milvus query 读取 enabled chunk 的最大行数",
    )
    parser.add_argument(
        "--update-split-manifest",
        action="store_true",
        help="生成快照后把 split_manifest.json 的 snapshot_id 回填为本次快照 ID",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST_PATH,
        help="需要回填 snapshot_id 的 split_manifest.json 路径",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case_files = [path.resolve() for path in (args.case_files or list(DEFAULT_CASE_FILES))]
    cases = load_cases_from_files(case_files)
    source_hashes = build_source_hashes([*case_files, args.split_manifest.resolve()])
    reranker, answer_model, planner_registry = build_default_runtime_metadata()

    from app.infra.persistence.chunk_status_repository import ChunkStatusRepository
    from app.infra.persistence.import_metadata_repository import ImportMetadataRepository
    from app.infra.vectorstore.milvus_gateway import milvus_gateway

    metadata_reader = MongoMetadataSnapshotReader(ImportMetadataRepository())
    chunk_reader = MilvusChunkSnapshotReader(milvus_gateway, query_limit=args.chunk_query_limit)
    override_reader = MongoChunkOverrideSnapshotReader(ChunkStatusRepository())

    result = build_and_write_environment_snapshot(
        output_path=args.output,
        metadata_reader=metadata_reader,
        chunk_reader=chunk_reader,
        override_reader=override_reader,
        cases=cases,
        dataset_ids=args.dataset_ids,
        test_user_ids=args.test_user_ids,
        snapshot_id=args.snapshot_id,
        created_by=args.created_by,
        retrieval_config_snapshot=build_retrieval_config_snapshot(),
        reranker=reranker,
        answer_model=answer_model,
        planner_registry=planner_registry,
        source_hashes=source_hashes,
    )
    read_environment_snapshot(args.output)
    if args.update_split_manifest:
        update_split_manifest_snapshot_id(args.split_manifest, result.snapshot.snapshot_id)
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    return 0


def _build_snapshot_document(document: dict[str, Any]) -> SnapshotDocument:
    return SnapshotDocument(
        document_id=_require_text(document.get("document_id"), field_name="document_id"),
        dataset_id=_require_text(document.get("dataset_id"), field_name="dataset_id"),
        index_version=_require_int(document.get("index_version"), field_name="index_version"),
        visibility=_require_text(document.get("visibility"), field_name="visibility"),
        chunk_count=max(0, _optional_int(document.get("chunk_count"), default=0)),
    )


def _extract_enabled_chunk_ids(
        raw_chunks: list[dict[str, Any]],
        *,
        document_id: str,
        index_version: int,
) -> list[int | str]:
    chunk_ids: list[int | str] = []
    seen: set[tuple[str, int | str]] = set()
    for chunk in sorted(raw_chunks, key=_chunk_sort_key):
        if str(chunk.get("document_id") or document_id) != document_id:
            raise SnapshotBuildError(
                f"Milvus 返回了不属于 document_id={document_id} 的 chunk：{chunk}"
            )
        chunk_index_version = _require_int(
            chunk.get("index_version", index_version),
            field_name="chunk.index_version",
        )
        if chunk_index_version != index_version:
            raise SnapshotBuildError(
                f"Milvus chunk index_version 与 document 不一致："
                f"document_id={document_id}, chunk_id={chunk.get('chunk_id')}, "
                f"document_index_version={index_version}, chunk_index_version={chunk_index_version}"
            )
        if chunk.get("enabled") is not True:
            raise SnapshotBuildError(
                f"Milvus enabled chunk 查询返回了非 enabled 记录："
                f"document_id={document_id}, chunk_id={chunk.get('chunk_id')}"
            )
        chunk_id = _normalize_chunk_id(chunk.get("chunk_id"))
        identity = (type(chunk_id).__name__, chunk_id)
        if identity in seen:
            continue
        seen.add(identity)
        chunk_ids.append(chunk_id)
    return chunk_ids


def _validate_case_document_version(
        errors: list[str],
        *,
        case_id: str,
        document_id: str,
        expected_index_version: int | None,
        document_version_by_id: dict[str, int],
) -> None:
    current_index_version = document_version_by_id.get(document_id)
    if current_index_version is None:
        errors.append(
            f"case_id={case_id} 引用的 document_id 不在 snapshot.documents 中：{document_id}"
        )
        return
    if expected_index_version is not None and expected_index_version != current_index_version:
        errors.append(
            f"case_id={case_id} 引用旧 index_version："
            f"document_id={document_id}, case_index_version={expected_index_version}, "
            f"snapshot_index_version={current_index_version}"
        )


def _collect_referenced_dataset_ids_from_cases(cases: list[PlannerEvalCase]) -> list[str]:
    """
    从真实引用了语料的 case 中收集 dataset_id。

    阶段 8.2 允许存在 pending 的行为边界候选，例如“私有文档隔离”或“实时信息必须 Web”，
    这些样本可能暂时没有真实 source_document/expected_chunk。默认构建 snapshot 时不能让
    这类候选 dataset 阻塞环境冻结；它们会在摘要的 missing_case_dataset_ids 中暴露出来，
    等绑定真实语料后再纳入 snapshot。显式传 --dataset-id 时仍按用户指定范围严格校验。
    """
    values: list[str] = []
    for case in cases:
        if case.source_document_ids or case.source_index_versions or case.expected_chunks:
            values.extend(case.dataset_ids)
    return _normalize_unique_text_list(
        values,
        field_name="case.dataset_ids",
        allow_empty=True,
    )


def _collect_test_user_ids_from_cases(cases: list[PlannerEvalCase]) -> list[str]:
    return _normalize_unique_text_list(
        [case.owner_user_id for case in cases if case.owner_user_id],
        field_name="case.owner_user_id",
        allow_empty=False,
    )


def _normalize_unique_text_list(
        values: list[str],
        *,
        field_name: str,
        allow_empty: bool = False,
) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        normalized.append(text)
        seen.add(text)
    if not normalized and not allow_empty:
        raise SnapshotBuildError(f"{field_name} 不能为空")
    return normalized


def _normalize_chunk_id(value: Any) -> int | str:
    if isinstance(value, bool):
        raise SnapshotBuildError("chunk_id 必须是字符串或整数，不能是 bool")
    if isinstance(value, int):
        return value
    text = str(value or "").strip()
    if not text:
        raise SnapshotBuildError("chunk_id 不能为空")
    return text


def _require_text(value: Any, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise SnapshotBuildError(f"{field_name} 不能为空")
    return text


def _require_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise SnapshotBuildError(f"{field_name} 必须是整数")
    try:
        int_value = int(value)
    except (TypeError, ValueError) as exc:
        raise SnapshotBuildError(f"{field_name} 必须是整数") from exc
    if int_value < 0:
        raise SnapshotBuildError(f"{field_name} 必须大于等于 0")
    return int_value


def _optional_int(value: Any, *, default: int) -> int:
    if value in (None, ""):
        return default
    return _require_int(value, field_name="optional_int")


def _without_mongo_id(document: dict[str, Any]) -> dict[str, Any]:
    result = dict(document)
    result.pop("_id", None)
    return result


def _display_path(path: Path) -> str:
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _chunk_sort_key(chunk: dict[str, Any]) -> tuple[int, str]:
    raw_index = chunk.get("chunk_index")
    chunk_index = raw_index if isinstance(raw_index, int) and not isinstance(raw_index, bool) else 0
    return chunk_index, str(chunk.get("chunk_id") or "")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_now_compact_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


if __name__ == "__main__":
    raise SystemExit(main())
