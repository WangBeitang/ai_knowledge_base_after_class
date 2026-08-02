"""为阶段 8.5 的 10 个 UCI Gold chunk 回填稳定主体身份。

脚本默认只做 dry-run（只读演练）。只有显式传入 ``--apply`` 才会修改 Milvus；写入前
会核对原导入 manifest（清单）、document/chunk/index 身份和正文 SHA256，并保存可回滚
的旧主体字段。它不修改冻结 case、正文、向量、权限或 chunk_id。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / "deploy/cloud_grpo/env.local"
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "evaluation/stage8_5/artifacts/intermediate/curated_gold/gold_import_manifest.json"
)
SUBJECT_BY_DOCUMENT = {
    "doc_stage85_uci_ai4i_official_description_v1": (
        "subject_uci_ai4i_2020",
        "AI4I 2020 Predictive Maintenance Dataset",
    ),
    "doc_stage85_uci_hydraulic_official_description_v1": (
        "subject_uci_hydraulic_condition",
        "Condition Monitoring of Hydraulic Systems",
    ),
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_manifest(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return json.loads(raw), _sha256_bytes(raw)


def _load_targets(manifest: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    targets: dict[int, dict[str, Any]] = {}
    for binding in manifest.get("chunk_bindings") or []:
        document_id = str(binding.get("document_id") or "")
        subject = SUBJECT_BY_DOCUMENT.get(document_id)
        if subject is None:
            continue
        chunk_id = int(binding["chunk_id"])
        if chunk_id in targets:
            raise ValueError(f"manifest 包含重复 chunk_id={chunk_id}")
        targets[chunk_id] = {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "index_version": int(binding["index_version"]),
            "content_sha256": str(binding["content_sha256"]),
            "subject_id": subject[0],
            "standard_subject_name": subject[1],
        }
    if len(targets) != 10:
        raise ValueError(f"只允许处理固定 10 个 UCI Gold chunk，实际={len(targets)}")
    return targets


def _query_target_rows(client: Any, collection: str, chunk_ids: list[int]) -> list[dict[str, Any]]:
    return list(client.query(
        collection_name=collection,
        ids=chunk_ids,
        output_fields=[
            "chunk_id",
            "document_id",
            "index_version",
            "subject_id",
            "standard_subject_name",
            "content",
        ],
        consistency_level="Strong",
    ))


def _validate_and_plan(
    targets: Mapping[int, Mapping[str, Any]],
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows_by_id = {int(row["chunk_id"]): row for row in rows}
    if set(rows_by_id) != set(targets):
        missing = sorted(set(targets) - set(rows_by_id))
        extra = sorted(set(rows_by_id) - set(targets))
        raise ValueError(f"目标 chunk 身份不完整：missing={missing}, extra={extra}")

    changes: list[dict[str, Any]] = []
    backups: list[dict[str, Any]] = []
    for chunk_id, target in sorted(targets.items()):
        row = rows_by_id[chunk_id]
        actual_identity = (str(row["document_id"]), int(row["index_version"]))
        expected_identity = (str(target["document_id"]), int(target["index_version"]))
        if actual_identity != expected_identity:
            raise ValueError(
                f"chunk_id={chunk_id} document/index 漂移："
                f"expected={expected_identity}, actual={actual_identity}"
            )
        actual_content_sha256 = _sha256_bytes(str(row["content"]).encode("utf-8"))
        if actual_content_sha256 != target["content_sha256"]:
            raise ValueError(
                f"chunk_id={chunk_id} 正文 SHA256 漂移，拒绝回填主体"
            )

        old_subject_id = str(row.get("subject_id") or "")
        old_subject_name = str(row.get("standard_subject_name") or "")
        new_subject_id = str(target["subject_id"])
        new_subject_name = str(target["standard_subject_name"])
        if old_subject_id not in {"", new_subject_id}:
            raise ValueError(
                f"chunk_id={chunk_id} 已有其他 subject_id={old_subject_id!r}，拒绝覆盖"
            )
        backups.append({
            "chunk_id": chunk_id,
            "document_id": target["document_id"],
            "index_version": target["index_version"],
            "content_sha256": actual_content_sha256,
            "subject_id": old_subject_id,
            "standard_subject_name": old_subject_name,
        })
        if old_subject_id != new_subject_id or old_subject_name != new_subject_name:
            changes.append({
                "chunk_id": chunk_id,
                "subject_id": new_subject_id,
                "standard_subject_name": new_subject_name,
            })
    return changes, backups


def _collection_property(description: Mapping[str, Any], key: str) -> str | None:
    properties = description.get("properties")
    if not isinstance(properties, Mapping):
        return None
    value = properties.get(key)
    return None if value is None else str(value)


def _restore_auto_id_property(client: Any, collection: str, original: str | None) -> None:
    if original is None:
        client.drop_collection_properties(
            collection_name=collection,
            property_keys=["allow_insert_auto_id"],
        )
    else:
        client.alter_collection_properties(
            collection_name=collection,
            properties={"allow_insert_auto_id": original},
        )


def _write_backup(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if not args.env_file.is_file():
        raise FileNotFoundError(f"环境文件不存在：{args.env_file}")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"导入 manifest 不存在：{args.manifest}")
    load_dotenv(args.env_file, override=True)

    # 配置对象在 import 时读取环境变量，必须放在 load_dotenv 之后。
    from app.shared.clients.milvus_utils import get_milvus_client
    from app.shared.config.milvus_config import milvus_config

    manifest, manifest_sha256 = _read_manifest(args.manifest)
    targets = _load_targets(manifest)
    client = get_milvus_client()
    if client is None:
        raise RuntimeError("Milvus 客户端不可用")
    collection = milvus_config.chunks_collection
    if not collection:
        raise RuntimeError("CHUNKS_COLLECTION 为空")

    description = client.describe_collection(collection_name=collection)
    if description.get("auto_id") is not True:
        raise RuntimeError("目标 collection 不是预期的 auto_id=True")
    if description.get("primary_field") not in {None, "chunk_id"}:
        raise RuntimeError(f"目标 collection 主键不是 chunk_id：{description.get('primary_field')}")

    rows_before = _query_target_rows(client, collection, sorted(targets))
    changes, backups = _validate_and_plan(targets, rows_before)
    summary = {
        "mode": "apply" if args.apply else "dry_run",
        "collection": collection,
        "manifest_sha256": manifest_sha256,
        "target_count": len(targets),
        "change_count": len(changes),
        "unchanged_count": len(targets) - len(changes),
        "changes": changes,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if not args.apply:
        print("STAGE85_GOLD_SUBJECT_BACKFILL_DRY_RUN=PASS")
        return 0
    if not changes:
        print("STAGE85_GOLD_SUBJECT_BACKFILL_ALREADY_APPLIED=PASS")
        return 0
    if args.backup is None:
        raise ValueError("--apply 必须同时提供全新的 --backup 路径")

    backup_payload = {
        "backup_version": "stage85-gold-subject-backfill-v1",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "collection": collection,
        "manifest_path": str(args.manifest),
        "manifest_sha256": manifest_sha256,
        "records": backups,
    }
    _write_backup(args.backup, backup_payload)

    original_property = _collection_property(description, "allow_insert_auto_id")
    try:
        client.alter_collection_properties(
            collection_name=collection,
            properties={"allow_insert_auto_id": "true"},
        )
        result = client.upsert(
            collection_name=collection,
            data=changes,
            partial_update=True,
        )
        client.flush(collection_name=collection)
    finally:
        _restore_auto_id_property(client, collection, original_property)

    rows_after = _query_target_rows(client, collection, sorted(targets))
    remaining, _ = _validate_and_plan(targets, rows_after)
    if remaining:
        raise RuntimeError(f"回填后仍有未生效记录：{remaining}")
    if {int(row["chunk_id"]) for row in rows_after} != set(targets):
        raise RuntimeError("回填后 chunk_id 集合发生变化")
    print(json.dumps({"upsert_result": result, "backup": str(args.backup)}, ensure_ascii=False))
    print("STAGE85_GOLD_SUBJECT_BACKFILL_APPLY=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
