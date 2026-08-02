"""恢复阶段 8.5 UCI Gold chunk 被 AutoID upsert 改写的原主键。

默认只做 dry-run（只读演练）。显式 ``--apply`` 后先备份完整当前记录，再把相同完整
记录以原 chunk_id 插入；只有原 ID、正文 SHA256 和主体全部验证通过，才删除临时新 ID。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cloud_grpo.backfill_stage85_gold_subjects import (  # noqa: E402
    DEFAULT_ENV_FILE,
    DEFAULT_MANIFEST,
    _collection_property,
    _load_targets,
    _read_manifest,
    _restore_auto_id_property,
    _sha256_bytes,
    _write_backup,
)


FUNCTION_OUTPUT_FIELDS = {"bm25_sparse_vector"}


def _query_document_rows(client: Any, collection: str) -> list[dict[str, Any]]:
    return list(client.query(
        collection_name=collection,
        filter=(
            'document_id in ['
            '"doc_stage85_uci_ai4i_official_description_v1",'
            '"doc_stage85_uci_hydraulic_official_description_v1"'
            '] AND index_version == 1'
        ),
        output_fields=["*", "dense_vector", "learned_sparse_vector"],
        limit=100,
        consistency_level="Strong",
    ))


def _query_ids(client: Any, collection: str, chunk_ids: list[int]) -> list[dict[str, Any]]:
    return list(client.query(
        collection_name=collection,
        ids=chunk_ids,
        output_fields=["*", "dense_vector", "learned_sparse_vector"],
        consistency_level="Strong",
    ))


def _build_recovery_payloads(
    targets: Mapping[int, Mapping[str, Any]],
    current_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[int], list[dict[str, Any]]]:
    target_by_sha = {
        str(target["content_sha256"]): (old_chunk_id, target)
        for old_chunk_id, target in targets.items()
    }
    if len(target_by_sha) != len(targets):
        raise ValueError("目标 manifest 存在重复正文 SHA256")

    payloads: list[dict[str, Any]] = []
    current_ids: list[int] = []
    mappings: list[dict[str, Any]] = []
    seen_old_ids: set[int] = set()
    for row in current_rows:
        content_sha256 = _sha256_bytes(str(row.get("content") or "").encode("utf-8"))
        matched = target_by_sha.get(content_sha256)
        if matched is None:
            raise ValueError(
                f"当前 UCI document 出现 manifest 外正文：chunk_id={row.get('chunk_id')}"
            )
        old_chunk_id, target = matched
        current_chunk_id = int(row["chunk_id"])
        if old_chunk_id in seen_old_ids:
            raise ValueError(f"原 chunk_id={old_chunk_id} 映射到多条当前记录")
        if current_chunk_id == old_chunk_id:
            raise ValueError(f"原 chunk_id={old_chunk_id} 已存在，不应再执行 AutoID 恢复")
        if str(row.get("document_id")) != str(target["document_id"]):
            raise ValueError(f"chunk_id={current_chunk_id} document_id 与 manifest 不一致")
        if int(row.get("index_version")) != int(target["index_version"]):
            raise ValueError(f"chunk_id={current_chunk_id} index_version 与 manifest 不一致")
        if str(row.get("subject_id") or "") != str(target["subject_id"]):
            raise ValueError(f"chunk_id={current_chunk_id} subject_id 尚未正确回填")
        if str(row.get("standard_subject_name") or "") != str(target["standard_subject_name"]):
            raise ValueError(f"chunk_id={current_chunk_id} standard_subject_name 尚未正确回填")

        payload = copy.deepcopy(row)
        payload["chunk_id"] = old_chunk_id
        for field_name in FUNCTION_OUTPUT_FIELDS:
            payload.pop(field_name, None)
        if len(payload.get("dense_vector") or []) != 1024:
            raise ValueError(f"chunk_id={current_chunk_id} dense_vector 维度不是 1024")
        if not payload.get("learned_sparse_vector"):
            raise ValueError(f"chunk_id={current_chunk_id} learned_sparse_vector 为空")
        if not str(payload.get("lexical_text") or ""):
            raise ValueError(f"chunk_id={current_chunk_id} lexical_text 为空")

        payloads.append(payload)
        current_ids.append(current_chunk_id)
        mappings.append({
            "old_chunk_id": old_chunk_id,
            "current_chunk_id": current_chunk_id,
            "document_id": target["document_id"],
            "content_sha256": content_sha256,
        })
        seen_old_ids.add(old_chunk_id)

    if seen_old_ids != set(targets):
        missing = sorted(set(targets) - seen_old_ids)
        raise ValueError(f"当前记录未覆盖全部原 ID：missing={missing}")
    if len(current_ids) != len(targets) or len(set(current_ids)) != len(targets):
        raise ValueError(
            f"当前新 ID 数量/唯一性与目标不一致：expected={len(targets)}, actual={current_ids}"
        )
    return payloads, current_ids, sorted(mappings, key=lambda item: item["old_chunk_id"])


def _validate_restored_rows(
    targets: Mapping[int, Mapping[str, Any]],
    restored_rows: list[dict[str, Any]],
) -> None:
    rows_by_id = {int(row["chunk_id"]): row for row in restored_rows}
    if set(rows_by_id) != set(targets):
        raise RuntimeError(
            "恢复后原 ID 不完整："
            f"missing={sorted(set(targets) - set(rows_by_id))}, "
            f"extra={sorted(set(rows_by_id) - set(targets))}"
        )
    for chunk_id, target in targets.items():
        row = rows_by_id[chunk_id]
        content_sha256 = _sha256_bytes(str(row.get("content") or "").encode("utf-8"))
        if content_sha256 != target["content_sha256"]:
            raise RuntimeError(f"恢复后 chunk_id={chunk_id} 正文 SHA256 不一致")
        if str(row.get("subject_id") or "") != target["subject_id"]:
            raise RuntimeError(f"恢复后 chunk_id={chunk_id} subject_id 不一致")
        if str(row.get("standard_subject_name") or "") != target["standard_subject_name"]:
            raise RuntimeError(f"恢复后 chunk_id={chunk_id} standard_subject_name 不一致")


def _wait_auto_id_property(client: Any, collection: str, expected: str) -> None:
    for _ in range(20):
        description = client.describe_collection(collection_name=collection)
        actual = _collection_property(description, "allow_insert_auto_id")
        if str(actual or "").lower() == expected.lower():
            return
        time.sleep(0.25)
    raise RuntimeError(f"allow_insert_auto_id 未变为 {expected}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if not args.env_file.is_file():
        raise FileNotFoundError(f"环境文件不存在：{args.env_file}")
    load_dotenv(args.env_file, override=True)

    from app.shared.clients.milvus_utils import get_milvus_client
    from app.shared.config.milvus_config import milvus_config

    manifest, manifest_sha256 = _read_manifest(args.manifest)
    targets = _load_targets(manifest)
    client = get_milvus_client()
    if client is None:
        raise RuntimeError("Milvus 客户端不可用")
    collection = milvus_config.chunks_collection
    description = client.describe_collection(collection_name=collection)
    original_property = _collection_property(description, "allow_insert_auto_id")

    old_rows_before = _query_ids(client, collection, sorted(targets))
    if old_rows_before:
        raise RuntimeError("原 ID 已有可查询记录，拒绝执行恢复以避免重复主键")
    current_rows = _query_document_rows(client, collection)
    payloads, current_ids, mappings = _build_recovery_payloads(targets, current_rows)
    summary = {
        "mode": "apply" if args.apply else "dry_run",
        "collection": collection,
        "manifest_sha256": manifest_sha256,
        "target_count": len(targets),
        "current_count": len(current_ids),
        "mappings": mappings,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if not args.apply:
        print("STAGE85_GOLD_AUTO_ID_RECOVERY_DRY_RUN=PASS")
        return 0
    if args.backup is None:
        raise ValueError("--apply 必须同时提供全新的 --backup 路径")

    _write_backup(args.backup, {
        "backup_version": "stage85-gold-auto-id-recovery-v1",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "collection": collection,
        "manifest_path": str(args.manifest),
        "manifest_sha256": manifest_sha256,
        "mappings": mappings,
        "current_records": current_rows,
    })

    insert_result: dict[str, Any] | None = None
    try:
        client.alter_collection_properties(
            collection_name=collection,
            properties={"allow_insert_auto_id": "true"},
        )
        _wait_auto_id_property(client, collection, "true")
        insert_result = client.insert(collection_name=collection, data=payloads)
        client.flush(collection_name=collection)

        inserted_ids = {int(item) for item in insert_result.get("ids") or []}
        expected_old_ids = set(targets)
        if inserted_ids != expected_old_ids:
            unexpected_ids = sorted(inserted_ids - expected_old_ids)
            if unexpected_ids:
                client.delete(collection_name=collection, ids=unexpected_ids)
                client.flush(collection_name=collection)
            raise RuntimeError(
                "显式 insert 未保留原 ID；已删除本次意外生成的新记录，"
                f"expected={sorted(expected_old_ids)}, actual={sorted(inserted_ids)}"
            )

        restored_rows = _query_ids(client, collection, sorted(targets))
        _validate_restored_rows(targets, restored_rows)

        client.delete(collection_name=collection, ids=current_ids)
        client.flush(collection_name=collection)
        remaining_current_rows = _query_ids(client, collection, current_ids)
        if remaining_current_rows:
            raise RuntimeError("恢复原 ID 后，临时新 ID 未全部删除")
        final_rows = _query_document_rows(client, collection)
        _validate_restored_rows(targets, final_rows)
    finally:
        _restore_auto_id_property(client, collection, original_property)

    print(json.dumps({
        "insert_result": insert_result,
        "deleted_temporary_ids": current_ids,
        "backup": str(args.backup),
    }, ensure_ascii=False, sort_keys=True))
    print("STAGE85_GOLD_AUTO_ID_RECOVERY_APPLY=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
