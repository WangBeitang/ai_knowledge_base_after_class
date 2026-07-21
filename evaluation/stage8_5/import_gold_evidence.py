"""导入阶段 8.5 gold 证据，并生成绑定真实 Milvus ID 的环境快照。

为什么不能直接把 ``gold_cases.jsonl`` 交给快照构建器：

- 作者态 gold 使用 ``chunk_ai4i_twf_rule`` 这类逻辑证据键，方便重写和人工审核。
- 当前 Milvus collection 的 ``chunk_id`` 是 ``auto_id=True`` 的整数主键，只有插入完成后
  才能拿到真实值。
- EnvironmentSnapshot（环境快照）必须冻结真实 ``document_id + chunk_id + index_version``，
  不能把逻辑键伪装成数据库主键。

因此本脚本按以下顺序执行：

1. 校验 Claude 二审对 20 条 case 全部给出 high-confidence gold；
2. 保留人工策划的 10 个 chunk 边界，生成 BGE-M3 向量并写入现有 Milvus collection；
3. 在 Mongo 创建两个 completed document 和对应 import task；
4. 记录逻辑 evidence_key 到真实整数 chunk_id 的绑定；
5. 生成 ``gold_cases_indexed.jsonl``，只在这份运行态文件中替换真实 chunk_id；
6. 从 Mongo/Milvus 当前状态构建独立的阶段 8.5 EnvironmentSnapshot。

脚本只管理两个固定 gold document，不会删除或重建知识库中的其他 document。若固定
document_id 已存在但版本/hash 不匹配，脚本拒绝覆盖，避免误删已有索引。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Milvus/Mongo/embedding 配置对象在 import 时读取环境变量，因此必须先加载项目 .env。
load_dotenv(PROJECT_ROOT / ".env")

from app.infra.persistence.import_metadata_repository import (  # noqa: E402
    STATUS_COMPLETED,
    STATUS_PROCESSING,
    ImportMetadataRepository,
)
from app.infra.persistence.chunk_status_repository import ChunkStatusRepository  # noqa: E402
from app.infra.vectorstore.milvus_gateway import milvus_gateway  # noqa: E402
from app.rag.evaluation.case_schema import (  # noqa: E402
    PlannerEvalCase,
    SplitManifest,
    load_planner_cases,
)
from app.rag.import_.embedding_service import generate_embeddings  # noqa: E402
from app.rag.import_.index_service import index_chunks, remove_old_chunks  # noqa: E402
from app.shared.config.knowledge_base_config import (  # noqa: E402
    DEFAULT_DATASET_ID,
    DEFAULT_TENANT_ID,
)
from app.shared.config.milvus_config import milvus_config  # noqa: E402
from app.shared.utils.escape_milvus_string_utils import escape_milvus_string  # noqa: E402
from evaluation.stage8.build_environment_snapshot import (  # noqa: E402
    EnvironmentSnapshotBuildResult,
    MilvusChunkSnapshotReader,
    MongoChunkOverrideSnapshotReader,
    MongoMetadataSnapshotReader,
    build_and_write_environment_snapshot,
    build_default_runtime_metadata,
    build_source_hashes,
    read_environment_snapshot,
    validate_cases_against_snapshot,
)
from evaluation.stage8_5.build_source_grounded_gold import (  # noqa: E402
    GOLD_VERSION,
    GoldCaseAudit,
    GoldEvidenceChunk,
)
from evaluation.stage8_5.stage85_schema import (  # noqa: E402
    read_jsonl,
    write_json,
    write_jsonl,
)


IMPORT_VERSION = "stage85-gold-import-v1"
DEFAULT_SNAPSHOT_ID = "stage85-env-20260721-v1"
DEFAULT_OWNER_USER_ID = "eval_demo_user"
DEFAULT_VISIBILITY = "public"


class ImportArtifactModel(BaseModel):
    """导入审计产物公共基类；拒绝未知字段，避免运行时身份字段悄悄丢失。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SecondReviewDecision(ImportArtifactModel):
    """独立二审的一条结论；只有无问题的 high-confidence gold 才允许入库。"""

    case_id: str = Field(min_length=1, description="二审对应的 gold case_id。")
    decision: str = Field(min_length=1, description="审核结论；本次导入要求为 gold。")
    confidence: str = Field(min_length=1, description="审核置信度；本次导入要求为 high。")
    source_support_summary: str = Field(default="", description="二审对官方来源支撑关系的摘要。")
    unsupported_answer_points: list[str] = Field(default_factory=list, description="来源不支持的答案点；必须为空。")
    incorrect_fact_ids: list[str] = Field(default_factory=list, description="二审发现错误的 fact_id；必须为空。")
    suggested_fix: str = Field(default="", description="建议修复；通过时应为空。")
    reviewer_notes: str = Field(default="", description="二审备注，不进入模型输入。")


class GoldChunkRuntimeBinding(ImportArtifactModel):
    """逻辑证据键到真实 Milvus chunk 身份的绑定。"""

    evidence_key: str = Field(min_length=1, description="作者态 gold 使用的逻辑 chunk 键。")
    chunk_id: int = Field(description="Milvus auto_id 生成的真实整数主键。")
    document_id: str = Field(min_length=1, description="所属 gold document ID。")
    index_version: int = Field(ge=1, description="document 索引版本；与 indexed case 和快照一致。")
    source_id: str = Field(min_length=1, description="UCI 来源 ID。")
    content_sha256: str = Field(min_length=64, max_length=64, description="入库正文 SHA256，用于检测内容漂移。")


class GoldDocumentImportRecord(ImportArtifactModel):
    """一个 gold document 的实际导入结果。"""

    document_id: str
    task_id: str
    source_id: str
    source_title: str
    dataset_id: str
    owner_user_id: str
    visibility: str
    index_version: int
    chunk_count: int
    source_hash: str
    import_action: str = Field(description="inserted 表示本次新建；reused 表示复用已核实同版本文档。")


class GoldImportManifest(ImportArtifactModel):
    """本次 gold 证据入库和快照绑定的机器可读清单。"""

    import_version: str = IMPORT_VERSION
    gold_version: str = GOLD_VERSION
    imported_at: str
    dataset_id: str
    snapshot_id: str
    snapshot_path: str
    source_evidence_path: str
    source_evidence_sha256: str
    second_review_path: str
    second_review_sha256: str
    second_review_count: int = Field(ge=1, description="通过导入门禁的独立二审记录数。")
    indexed_cases_path: str
    indexed_case_count: int = Field(ge=1, description="已绑定真实 Milvus chunk_id 的可运行 case 数。")
    documents: list[GoldDocumentImportRecord]
    chunk_bindings: list[GoldChunkRuntimeBinding]
    snapshot_summary: dict[str, Any]


class GoldCaseRuntimeBinding(ImportArtifactModel):
    """单条 case 的逻辑证据键与运行时 chunk_id 对照，便于人工抽查。"""

    case_id: str
    evidence_key: str
    chunk_id: int
    document_id: str
    index_version: int


def validate_second_review(
        cases: list[PlannerEvalCase],
        decisions: list[SecondReviewDecision],
) -> None:
    """二审必须覆盖全部且仅覆盖当前 20 条 gold，并且没有待修复项。"""

    case_ids = {case.case_id for case in cases}
    decision_ids = [decision.case_id for decision in decisions]
    if len(decision_ids) != len(set(decision_ids)):
        raise ValueError("gold 二审结果包含重复 case_id")
    missing = sorted(case_ids - set(decision_ids))
    extra = sorted(set(decision_ids) - case_ids)
    if missing or extra:
        raise ValueError(f"gold 二审覆盖不完整：missing={missing}, extra={extra}")

    failed: list[str] = []
    for decision in decisions:
        if decision.decision != "gold" or decision.confidence != "high":
            failed.append(f"{decision.case_id}: decision={decision.decision}, confidence={decision.confidence}")
        if decision.unsupported_answer_points or decision.incorrect_fact_ids or decision.suggested_fix:
            failed.append(f"{decision.case_id}: 二审仍包含 unsupported/incorrect/suggested_fix")
    if failed:
        raise ValueError("gold 二审未通过入库门禁：\n- " + "\n- ".join(failed))


def build_prepared_documents(
        evidence_chunks: list[GoldEvidenceChunk],
) -> dict[str, dict[str, Any]]:
    """保留 10 个策划 chunk 边界，构造 index_service 可接受的文档和 chunk。"""

    grouped: dict[str, list[GoldEvidenceChunk]] = defaultdict(list)
    for evidence in evidence_chunks:
        grouped[evidence.document_id].append(evidence)

    prepared: dict[str, dict[str, Any]] = {}
    for document_id, document_chunks in sorted(grouped.items()):
        source_ids = {chunk.source_id for chunk in document_chunks}
        source_titles = {chunk.source_title for chunk in document_chunks}
        source_urls = {chunk.source_url for chunk in document_chunks}
        if len(source_ids) != 1 or len(source_titles) != 1 or len(source_urls) != 1:
            raise ValueError(f"document_id={document_id} 混入多个来源，禁止导入")

        source_id = next(iter(source_ids))
        source_title = next(iter(source_titles))
        source_url = next(iter(source_urls))
        chunks: list[dict[str, Any]] = []
        for part, evidence in enumerate(document_chunks, start=1):
            content = _render_chunk_content(evidence)
            chunks.append({
                # evidence_key 是作者态稳定键；Milvus 的真实 chunk_id 由 auto_id 另行生成。
                "evidence_key": evidence.chunk_id,
                "source_id": evidence.source_id,
                "source_url": evidence.source_url,
                "source_locator": evidence.source_locator,
                "gold_version": GOLD_VERSION,
                "content": content,
                "title": evidence.topic,
                "parent_title": source_title,
                "part": part,
                "file_title": source_title,
                "subject_id": "",
                "standard_subject_name": source_title,
                "equipment_model": "AI4I 2020" if source_id == "uci-ai4i-2020" else "Hydraulic test rig",
                "part_name": evidence.topic,
                "alarm_code": "",
                "sop_type": "",
                "safety_level": "",
                "maintenance_stage": "",
            })

        prepared[document_id] = {
            "document_id": document_id,
            "source_id": source_id,
            "source_title": source_title,
            "source_url": source_url,
            "index_version": int(document_chunks[0].index_version),
            "source_hash": _document_source_hash(document_chunks),
            "chunks": chunks,
        }
    return prepared


def build_runtime_artifacts(
        *,
        cases: list[PlannerEvalCase],
        audits: list[GoldCaseAudit],
        bindings: list[GoldChunkRuntimeBinding],
        second_review_path: Path,
        snapshot_id: str,
        split_created_at: str | None = None,
) -> tuple[list[PlannerEvalCase], list[GoldCaseAudit], list[GoldCaseRuntimeBinding], SplitManifest]:
    """
    用真实 Milvus ID 生成运行态 case、二审后审计和 train-only split 清单。

    ``split_created_at`` 在首次生成时为空；幂等重跑时传入已有清单的创建时间。创建时间
    属于产物身份而不是“最后运行时间”，保留它才能让内容 hash 稳定并安全复用快照。
    """

    binding_by_key = {binding.evidence_key: binding for binding in bindings}
    indexed_cases: list[PlannerEvalCase] = []
    runtime_bindings: list[GoldCaseRuntimeBinding] = []
    for case in cases:
        payload = case.model_dump(mode="json")
        for expected_chunk in payload["expected_chunks"]:
            evidence_key = str(expected_chunk["chunk_id"])
            binding = binding_by_key.get(evidence_key)
            if binding is None:
                raise ValueError(f"case_id={case.case_id} 缺少 evidence_key={evidence_key} 的运行时绑定")
            if expected_chunk["document_id"] != binding.document_id:
                raise ValueError(f"case_id={case.case_id} 的 document_id 与运行时绑定不一致")
            expected_chunk["chunk_id"] = binding.chunk_id
            expected_chunk["index_version"] = binding.index_version
            payload["source_index_versions"][binding.document_id] = binding.index_version
            runtime_bindings.append(GoldCaseRuntimeBinding(
                case_id=case.case_id,
                evidence_key=evidence_key,
                chunk_id=binding.chunk_id,
                document_id=binding.document_id,
                index_version=binding.index_version,
            ))
        payload["notes"] = (
            f"{payload.get('notes', '')}; runtime_binding={IMPORT_VERSION}; snapshot_id={snapshot_id}"
        ).strip("; ")
        indexed_cases.append(PlannerEvalCase.model_validate(payload))

    audit_by_case = {audit.case_id: audit for audit in audits}
    if set(audit_by_case) != {case.case_id for case in cases}:
        raise ValueError("gold_case_audit.jsonl 与 gold_cases.jsonl 的 case_id 集合不一致")
    verified_audits = [
        audit.model_copy(update={
            "second_review_status": "passed",
            "second_reviewer_type": "independent_agent",
            "second_review_artifact": _display_path(second_review_path),
        })
        for audit in audits
    ]

    split_manifest = SplitManifest(
        manifest_id=f"{IMPORT_VERSION}-split-manifest",
        created_at=split_created_at or _utc_now_iso(),
        snapshot_id=snapshot_id,
        train_case_ids=[case.case_id for case in indexed_cases],
        dev_case_ids=[],
        test_case_ids=[],
        demo_regression_case_ids=[],
        leakage_group_to_split={case.leakage_group_id: case.split for case in indexed_cases},
        notes="20 条 source-grounded gold 全部为 train；held-out dev/test 必须使用独立来源文档。",
    )
    return indexed_cases, verified_audits, runtime_bindings, split_manifest


def _render_chunk_content(evidence: GoldEvidenceChunk) -> str:
    facts = "\n".join(f"- {fact.statement_zh}" for fact in evidence.facts)
    return (
        f"# {evidence.topic}\n\n"
        f"来源：{evidence.source_title}\n"
        f"来源链接：{evidence.source_url}\n"
        f"来源位置：{evidence.source_locator}\n"
        f"许可证：{evidence.license_name}\n\n"
        f"## 已核实事实\n\n{facts}"
    )


def _document_source_hash(chunks: list[GoldEvidenceChunk]) -> str:
    payload = [chunk.model_dump(mode="json") for chunk in chunks]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _materialize_markdown_documents(
        prepared_documents: dict[str, dict[str, Any]],
        output_dir: Path,
) -> dict[str, Path]:
    """生成可人工阅读的 Markdown 文件；实际 chunk 边界仍由 evidence_key 固定。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for document_id, document in prepared_documents.items():
        path = output_dir / f"{document_id}.md"
        body = "\n\n".join(chunk["content"] for chunk in document["chunks"]) + "\n"
        path.write_text(body, encoding="utf-8")
        paths[document_id] = path
    return paths


def _embed_prepared_documents(prepared_documents: dict[str, dict[str, Any]]) -> None:
    """在任何 Mongo/Milvus 写入前完成全部向量化，降低半成品导入概率。"""

    all_chunks = [
        chunk
        for document in prepared_documents.values()
        for chunk in document["chunks"]
    ]
    generate_embeddings(all_chunks)


def _query_document_chunks(document_id: str) -> list[dict[str, Any]]:
    client = milvus_gateway.client
    return list(client.query(
        collection_name=milvus_config.chunks_collection,
        filter=f"document_id=='{escape_milvus_string(document_id)}'",
        output_fields=[
            "chunk_id",
            "document_id",
            "index_version",
            "enabled",
            "evidence_key",
            "source_id",
            "content",
        ],
        limit=100,
    ))


def _wait_for_document_chunks(
        document_id: str,
        *,
        expected_count: int,
        timeout_seconds: float = 15.0,
) -> list[dict[str, Any]]:
    """等待 Milvus insert 对 query 可见，避免把一致性延迟误判为插入失败。"""

    client = milvus_gateway.client
    # flush 负责把已提交数据持久化；query 可见性仍可能稍晚，因此下面继续有界轮询。
    client.flush(collection_name=milvus_config.chunks_collection)
    deadline = time.monotonic() + timeout_seconds
    last_rows: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        last_rows = _query_document_chunks(document_id)
        if len(last_rows) == expected_count:
            return last_rows
        time.sleep(0.25)
    raise RuntimeError(
        f"document_id={document_id} 在 {timeout_seconds:.1f}s 内未达到预期 chunk 数："
        f"expected={expected_count}, actual={len(last_rows)}"
    )


def _cleanup_failed_gold_document(
        repository: ImportMetadataRepository,
        *,
        document: dict[str, Any],
) -> None:
    """清理本脚本自己创建且无 chunk 残留的 failed 元数据，允许安全重试。"""

    document_id = str(document.get("document_id") or "")
    if (
        not document_id
        or document.get("status") != "failed"
        or document.get("gold_import_version") != IMPORT_VERSION
    ):
        raise RuntimeError(f"document_id={document_id} 不是可自动清理的阶段 8.5 failed document")
    rows = _query_document_chunks(document_id)
    if rows:
        raise RuntimeError(f"document_id={document_id} 仍有 {len(rows)} 个 chunk，拒绝自动清理元数据")

    task_id = str(document.get("latest_task_id") or "")
    # 删除条件同时带上版本、状态和任务身份，避免并发重试时误删已被其他任务接管的记录。
    result = repository.documents.delete_one({
        "document_id": document_id,
        "status": "failed",
        "gold_import_version": IMPORT_VERSION,
        "latest_task_id": task_id,
    })
    if result.deleted_count != 1:
        raise RuntimeError(f"document_id={document_id} failed 元数据状态已变化，停止自动清理")
    if task_id:
        repository.tasks.delete_one({"task_id": task_id, "document_id": document_id})


def _bindings_from_rows(
        document: dict[str, Any],
        rows: list[dict[str, Any]],
) -> list[GoldChunkRuntimeBinding]:
    expected_keys = {str(chunk["evidence_key"]) for chunk in document["chunks"]}
    row_by_key = {str(row.get("evidence_key") or ""): row for row in rows}
    if set(row_by_key) != expected_keys:
        raise RuntimeError(
            f"document_id={document['document_id']} 的 Milvus evidence_key 不匹配："
            f"expected={sorted(expected_keys)}, actual={sorted(row_by_key)}"
        )
    bindings: list[GoldChunkRuntimeBinding] = []
    for chunk in document["chunks"]:
        evidence_key = str(chunk["evidence_key"])
        row = row_by_key[evidence_key]
        if row.get("enabled") is not True:
            raise RuntimeError(f"evidence_key={evidence_key} 入库后不是 enabled=true")
        bindings.append(GoldChunkRuntimeBinding(
            evidence_key=evidence_key,
            chunk_id=int(row["chunk_id"]),
            document_id=document["document_id"],
            index_version=int(row["index_version"]),
            source_id=document["source_id"],
            content_sha256=hashlib.sha256(str(row.get("content") or "").encode("utf-8")).hexdigest(),
        ))
    return bindings


def _import_documents(
        *,
        prepared_documents: dict[str, dict[str, Any]],
        markdown_paths: dict[str, Path],
        repository: ImportMetadataRepository,
        dataset_id: str,
        owner_user_id: str,
        visibility: str,
) -> tuple[list[GoldDocumentImportRecord], list[GoldChunkRuntimeBinding]]:
    """新建或严格复用两个 gold document；不覆盖 hash/版本不匹配的既有文档。"""

    document_records: list[GoldDocumentImportRecord] = []
    all_bindings: list[GoldChunkRuntimeBinding] = []
    for document_id, document in prepared_documents.items():
        existing = repository.get_document_by_id(document_id)
        if existing:
            if (
                existing.get("status") != STATUS_COMPLETED
                or existing.get("index_status") != STATUS_COMPLETED
                or existing.get("gold_source_hash") != document["source_hash"]
                or int(existing.get("index_version") or 0) != int(document["index_version"])
            ):
                raise RuntimeError(
                    f"document_id={document_id} 已存在但状态、版本或 gold_source_hash 不匹配；"
                    "为避免误删索引，本脚本拒绝覆盖"
                )
            rows = _query_document_chunks(document_id)
            bindings = _bindings_from_rows(document, rows)
            all_bindings.extend(bindings)
            document_records.append(GoldDocumentImportRecord(
                document_id=document_id,
                task_id=str(existing.get("latest_task_id") or ""),
                source_id=document["source_id"],
                source_title=document["source_title"],
                dataset_id=dataset_id,
                owner_user_id=owner_user_id,
                visibility=visibility,
                index_version=int(existing["index_version"]),
                chunk_count=len(bindings),
                source_hash=document["source_hash"],
                import_action="reused",
            ))
            continue

        task_id = f"task_stage85_gold_{uuid.uuid4().hex}"
        markdown_path = markdown_paths[document_id]
        repository.create_import_metadata(
            dataset_id=dataset_id,
            document_id=document_id,
            task_id=task_id,
            owner_user_id=owner_user_id,
            file_name=markdown_path.name,
            file_path=str(markdown_path),
            local_dir=str(markdown_path.parent),
            tenant_id=DEFAULT_TENANT_ID,
            visibility=visibility,
            index_version=int(document["index_version"]),
        )
        repository.update_task_status(task_id, STATUS_PROCESSING)
        repository.update_document(
            document_id,
            status=STATUS_PROCESSING,
            parse_status=STATUS_COMPLETED,
            index_status=STATUS_PROCESSING,
            md_path=str(markdown_path),
            gold_source_hash=document["source_hash"],
            gold_import_version=IMPORT_VERSION,
            source_id=document["source_id"],
            source_url=document["source_url"],
            license_name="CC BY 4.0",
        )
        state = {
            "task_id": task_id,
            "dataset_id": dataset_id,
            "document_id": document_id,
            "index_version": int(document["index_version"]),
            "owner_user_id": owner_user_id,
            "tenant_id": DEFAULT_TENANT_ID,
            "visibility": visibility,
            "file_title": document["source_title"],
            "chunks": document["chunks"],
        }
        try:
            index_chunks(state)
            rows = _wait_for_document_chunks(
                document_id,
                expected_count=len(document["chunks"]),
            )
            bindings = _bindings_from_rows(document, rows)
            repository.update_document(
                document_id,
                status=STATUS_COMPLETED,
                parse_status=STATUS_COMPLETED,
                index_status=STATUS_COMPLETED,
                chunk_count=len(bindings),
                standard_subject_name=document["source_title"],
                index_version=int(document["index_version"]),
                failed_node="",
                error_code="",
                error_message="",
            )
            repository.mark_import_completed(task_id)
        except Exception as error:
            # 当前 document 是本脚本刚创建的专用 ID，失败时清除其部分 chunk，避免孤儿数据进入召回。
            remove_old_chunks(document_id)
            repository.mark_import_failed(task_id, "stage85_gold_import", str(error))
            raise

        all_bindings.extend(bindings)
        document_records.append(GoldDocumentImportRecord(
            document_id=document_id,
            task_id=task_id,
            source_id=document["source_id"],
            source_title=document["source_title"],
            dataset_id=dataset_id,
            owner_user_id=owner_user_id,
            visibility=visibility,
            index_version=int(document["index_version"]),
            chunk_count=len(bindings),
            source_hash=document["source_hash"],
            import_action="inserted",
        ))
    return document_records, all_bindings


def _write_import_report(path: Path, manifest: GoldImportManifest) -> None:
    lines = [
        "# 阶段 8.5 gold 证据导入与环境快照报告",
        "",
        "## 结果",
        "",
        f"- 导入版本：`{manifest.import_version}`。",
        f"- 环境快照：`{manifest.snapshot_id}`。",
        f"- gold document：{len(manifest.documents)} 个。",
        f"- gold chunk：{len(manifest.chunk_bindings)} 个。",
        f"- independent second review：{manifest.second_review_count} 条。",
        f"- indexed gold case：{manifest.indexed_case_count} 条，全部通过二审并绑定真实 Milvus 整数 chunk_id。",
        "",
        "## 文档",
        "",
        "| document_id | action | index_version | chunks |",
        "|---|---|---:|---:|",
    ]
    for document in manifest.documents:
        lines.append(
            f"| `{document.document_id}` | `{document.import_action}` | {document.index_version} | {document.chunk_count} |"
        )
    lines.extend([
        "",
        "## 快照校验",
        "",
        f"- case_reference_check：`{manifest.snapshot_summary.get('case_reference_check')}`。",
        f"- snapshot document_count：{manifest.snapshot_summary.get('document_count')}。",
        f"- snapshot enabled_chunk_count：{manifest.snapshot_summary.get('enabled_chunk_count')}。",
        "- 快照包含默认知识库当前全部 completed document；20 条 gold 只引用本次两个 gold document。",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reuse_frozen_snapshot_if_compatible(
        snapshot_path: Path,
        *,
        snapshot_id: str,
        cases: list[PlannerEvalCase],
        source_hashes: dict[str, str],
) -> EnvironmentSnapshotBuildResult | None:
    """
    复用已经冻结且输入完全一致的快照；文件不存在时返回 ``None`` 让调用方首次构建。

    Snapshot（环境快照）的核心价值是不可变。相同 ``snapshot_id`` 若被当前数据库状态
    静默覆盖，之前的评测结果就无法重放。因此已有文件只允许只读复用，并同时检查：

    1. 文件内 snapshot_id 与本次参数一致；
    2. case、证据、二审和绑定文件的 SHA256 全部一致；
    3. 当前 indexed case 引用的 document/chunk/index_version 仍存在于该冻结快照。

    任一项变化都要求调用者换新的 snapshot ID 和输出路径，不能复用旧身份。
    """

    if not snapshot_path.exists():
        return None

    snapshot = read_environment_snapshot(snapshot_path)
    if snapshot.snapshot_id != snapshot_id:
        raise RuntimeError(
            f"快照文件已存在且 snapshot_id={snapshot.snapshot_id!r}，"
            f"不能用 {snapshot_id!r} 覆盖；请使用新的输出路径"
        )
    if snapshot.source_hashes != source_hashes:
        raise RuntimeError(
            f"snapshot_id={snapshot_id} 的输入文件 hash 已变化，拒绝覆盖冻结快照；"
            "请使用新的 snapshot ID 和输出路径"
        )

    validation_summary = validate_cases_against_snapshot(cases, snapshot=snapshot)
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


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    base_dir = args.base_dir.resolve()
    evidence_path = base_dir / "processed/gold_evidence_chunks.jsonl"
    source_cases_path = base_dir / "candidates/gold_cases.jsonl"
    audit_path = base_dir / "reviews/gold_case_audit.jsonl"
    second_review_path = base_dir / "reviews/gold_case_second_review.jsonl"
    indexed_cases_path = base_dir / "candidates/gold_cases_indexed.jsonl"
    runtime_bindings_path = base_dir / "processed/gold_case_runtime_bindings.jsonl"
    import_manifest_path = base_dir / "processed/gold_import_manifest.json"
    split_manifest_path = base_dir / "candidates/gold_split_manifest.json"
    snapshot_path = base_dir / "results/environment_snapshot_stage85.json"
    report_path = base_dir / "reports/阶段8.5Gold证据导入与快照报告.md"

    evidence_chunks = read_jsonl(evidence_path, GoldEvidenceChunk)
    cases = load_planner_cases(source_cases_path)
    audits = read_jsonl(audit_path, GoldCaseAudit)
    second_reviews = read_jsonl(second_review_path, SecondReviewDecision)
    validate_second_review(cases, second_reviews)

    prepared_documents = build_prepared_documents(evidence_chunks)
    markdown_paths = _materialize_markdown_documents(
        prepared_documents,
        base_dir / "processed/gold_documents",
    )

    # 先检查数据库固定 ID 冲突，再加载 BGE-M3；冲突时无需等待模型初始化。
    repository = ImportMetadataRepository()
    for document_id, document in prepared_documents.items():
        existing = repository.get_document_by_id(document_id)
        if not existing:
            continue
        if existing.get("gold_source_hash") != document["source_hash"]:
            raise RuntimeError(f"document_id={document_id} 已存在且 source hash 不同，拒绝覆盖")
        if existing.get("status") == "failed":
            _cleanup_failed_gold_document(repository, document=existing)

    _embed_prepared_documents(prepared_documents)
    document_records, bindings = _import_documents(
        prepared_documents=prepared_documents,
        markdown_paths=markdown_paths,
        repository=repository,
        dataset_id=args.dataset_id,
        owner_user_id=args.owner_user_id,
        visibility=args.visibility,
    )
    split_created_at: str | None = None
    if split_manifest_path.exists():
        existing_split_manifest = SplitManifest.model_validate_json(
            split_manifest_path.read_text(encoding="utf-8")
        )
        if existing_split_manifest.snapshot_id != args.snapshot_id:
            raise RuntimeError(
                f"split manifest 已绑定 snapshot_id={existing_split_manifest.snapshot_id!r}，"
                f"不能改写为 {args.snapshot_id!r}；请使用新的阶段目录或输出文件"
            )
        split_created_at = existing_split_manifest.created_at

    indexed_cases, verified_audits, runtime_case_bindings, split_manifest = build_runtime_artifacts(
        cases=cases,
        audits=audits,
        bindings=bindings,
        second_review_path=second_review_path,
        snapshot_id=args.snapshot_id,
        split_created_at=split_created_at,
    )
    write_jsonl(indexed_cases_path, indexed_cases)
    write_jsonl(audit_path, verified_audits)
    write_jsonl(runtime_bindings_path, runtime_case_bindings)
    write_json(split_manifest_path, split_manifest)

    reranker, answer_model, planner_registry = build_default_runtime_metadata()
    source_hashes = build_source_hashes([
        evidence_path,
        source_cases_path,
        indexed_cases_path,
        audit_path,
        second_review_path,
        runtime_bindings_path,
        split_manifest_path,
        *markdown_paths.values(),
    ])
    snapshot_result = reuse_frozen_snapshot_if_compatible(
        snapshot_path,
        snapshot_id=args.snapshot_id,
        cases=indexed_cases,
        source_hashes=source_hashes,
    )
    if snapshot_result is None:
        snapshot_result = build_and_write_environment_snapshot(
            output_path=snapshot_path,
            metadata_reader=MongoMetadataSnapshotReader(repository),
            chunk_reader=MilvusChunkSnapshotReader(milvus_gateway),
            override_reader=MongoChunkOverrideSnapshotReader(ChunkStatusRepository()),
            cases=indexed_cases,
            dataset_ids=[args.dataset_id],
            test_user_ids=[args.owner_user_id],
            snapshot_id=args.snapshot_id,
            created_by="stage85_gold_evidence_importer",
            reranker=reranker,
            answer_model=answer_model,
            planner_registry=planner_registry,
            source_hashes=source_hashes,
        )
        read_environment_snapshot(snapshot_path)

    manifest = GoldImportManifest(
        imported_at=_utc_now_iso(),
        dataset_id=args.dataset_id,
        snapshot_id=args.snapshot_id,
        snapshot_path=_display_path(snapshot_path),
        source_evidence_path=_display_path(evidence_path),
        source_evidence_sha256=_sha256(evidence_path),
        second_review_path=_display_path(second_review_path),
        second_review_sha256=_sha256(second_review_path),
        second_review_count=len(second_reviews),
        indexed_cases_path=_display_path(indexed_cases_path),
        indexed_case_count=len(indexed_cases),
        documents=document_records,
        chunk_bindings=bindings,
        snapshot_summary=snapshot_result.summary,
    )
    write_json(import_manifest_path, manifest)
    _write_import_report(report_path, manifest)
    print(manifest.model_dump_json(indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="导入阶段 8.5 gold 证据并生成新环境快照。")
    parser.add_argument("--base-dir", type=Path, default=PROJECT_ROOT / "evaluation/stage8_5")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--owner-user-id", default=DEFAULT_OWNER_USER_ID)
    parser.add_argument("--visibility", choices=["private", "shared", "public"], default=DEFAULT_VISIBILITY)
    parser.add_argument("--snapshot-id", default=DEFAULT_SNAPSHOT_ID)
    return parser


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
