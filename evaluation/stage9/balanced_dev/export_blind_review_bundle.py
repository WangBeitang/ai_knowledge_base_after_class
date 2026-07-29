"""导出任务 9.3.13 的 clean blind review（干净盲审）输入包。

正式 case 台账和 planner case 注册表必须保留历史审核字段，不能为了盲审删除审计证据。
本脚本改用显式字段白名单，把当前 pending queue、冻结来源、路线规则和脱敏泄漏参考投影
到独立目录。输出不包含任何旧 decision（审核决定）、reviewer（审核者）或审核备注。

边界：

- 只生成审核输入，不生成或合并审核决定。
- 不访问模型、GPU、Mongo、Milvus 或 Web；审核者后续可按包内来源身份做只读回查。
- 输出目录已存在时默认拒绝覆盖，避免审核开始后输入被静默替换。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.evaluation.case_schema import PlannerEvalCase  # noqa: E402
from evaluation.stage9.balanced_dev.build_balanced_dev_cases import (  # noqa: E402
    CASE_SPECS,
    HYDE_PROBES,
    _case_spec_fingerprint,
)


BUNDLE_VERSION = "stage9-balanced-dev-blind-review-bundle-v1"
SANITIZATION_POLICY_VERSION = "stage9-blind-review-sanitization-v1"
DEFAULT_QUEUE = (
    PROJECT_ROOT
    / "evaluation/stage9/artifacts/balanced_dev/second_review_queue.jsonl"
)
DEFAULT_PLANNER_CASES = (
    PROJECT_ROOT / "evaluation/stage8/cases/planner_cases.jsonl"
)
DEFAULT_CURATED_CASES = (
    PROJECT_ROOT
    / "evaluation/stage8_5/artifacts/intermediate/sft_seed/"
    "curated_seed_train_cases.jsonl"
)
DEFAULT_ROUTE_CASES = (
    PROJECT_ROOT / "evaluation/stage9/artifacts/route_seed/route_seed_cases.jsonl"
)
DEFAULT_LOCAL_SOURCE_MANIFEST = (
    PROJECT_ROOT / "evaluation/stage9/configs/balanced_dev_source_manifest_v1.json"
)
DEFAULT_LOCAL_EVIDENCE_MANIFEST = (
    PROJECT_ROOT
    / "evaluation/stage9/artifacts/balanced_dev/source_import_manifest.json"
)
DEFAULT_WEB_SOURCE_MANIFEST = (
    PROJECT_ROOT
    / "evaluation/stage9/configs/balanced_dev_web_source_manifest_v1.json"
)
DEFAULT_WEB_EVIDENCE_MANIFEST = (
    PROJECT_ROOT
    / "evaluation/stage9/artifacts/balanced_dev/web_evidence_manifest.json"
)
DEFAULT_ROUTE_MATRIX = (
    PROJECT_ROOT / "evaluation/stage9/configs/planner_eval_route_matrix_v1.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation/stage9/artifacts/balanced_dev/blind_review_bundle_v1"
)

DATA_FILE_NAMES = {
    "review_cases.jsonl",
    "leakage_reference.jsonl",
    "local_source_manifest.json",
    "local_evidence_manifest.json",
    "web_source_manifest.json",
    "web_evidence_manifest.json",
    "route_policy.json",
}
OUTPUT_FILE_NAMES = {
    *DATA_FILE_NAMES,
    "REVIEW_INSTRUCTIONS.md",
    "bundle_manifest.json",
}

# 这些 key 只属于审核生命周期，不能进入 blind review 输入。使用精确 key 而不是
# 模糊字符串匹配，避免把 expected_behavior 之类正常业务字段误判为污染。
BANNED_REVIEW_KEYS = {
    "decision",
    "evidence_check",
    "human_review_status",
    "independent_second_review",
    "leakage_check",
    "notes",
    "primary_source_review",
    "review_status",
    "reviewed_at",
    "reviewer_id",
    "reviewer_role",
    "route_check",
}
BANNED_HISTORY_MARKERS = {
    "independent-agent-round2",
    "independent-agent-round3",
    "second_review=passed",
    "second_review=pending",
}

SAFE_EVAL_CASE_FIELDS = (
    "case_group",
    "split",
    "leakage_group_id",
    "query",
    "query_variants",
    "dataset_ids",
    "privacy_scope",
    "source_document_ids",
    "source_index_versions",
    "expected_subject_ids",
    "expected_subject_names",
    "expected_chunks",
    "expected_web_evidence",
    "expected_answer_points",
    "expected_behavior",
    "acceptable_action_paths",
    "expected_identifiers",
    "label_source",
    "gold_origin",
)
LEAKAGE_REFERENCE_FIELDS = (
    "case_id",
    "split",
    "leakage_group_id",
    "query",
    "query_variants",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是 object：{path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL 每行必须是 object：{path}")
    return rows


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _logical_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _unique_by_id(
    rows: list[dict[str, Any]],
    *,
    field_name: str,
    source_name: str,
) -> dict[str, dict[str, Any]]:
    result = {str(row[field_name]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{source_name} 存在重复 {field_name}")
    return result


def _build_review_cases(
    *,
    queue_rows: list[dict[str, Any]],
    planner_rows: list[dict[str, Any]],
    case_specs: Iterable[Any] = CASE_SPECS,
    hyde_probes: dict[str, dict[str, Any]] = HYDE_PROBES,
    expected_route_counts: dict[str, int] | None = None,
    required_review_status: str | None = "pending",
) -> list[dict[str, Any]]:
    """把冻结 queue 和 case 契约合成可重算 fingerprint 的无审核结论输入。

    正常送审必须要求 ``pending``。审核结束后的历史包若需要从已冻结 queue 重建，可由
    调用方显式传入 ``None``；输出仍只投影 ``SAFE_EVAL_CASE_FIELDS``，不会写入审核状态。
    """

    queue_by_id = _unique_by_id(
        queue_rows,
        field_name="case_id",
        source_name="second_review_queue",
    )
    planner_by_id = _unique_by_id(
        planner_rows,
        field_name="case_id",
        source_name="planner_cases",
    )
    case_specs = tuple(case_specs)
    spec_by_id = {spec.case_id: spec for spec in case_specs}
    if len(spec_by_id) != len(case_specs):
        raise ValueError("CASE_SPECS 存在重复 case_id")

    unknown = sorted(set(queue_by_id) - set(spec_by_id))
    if unknown:
        raise ValueError(f"审核队列包含未知 CaseSpec：{unknown}")

    review_cases: list[dict[str, Any]] = []
    for queue_row in queue_rows:
        case_id = str(queue_row["case_id"])
        spec = spec_by_id[case_id]
        expected_fingerprint = _case_spec_fingerprint(spec)
        if queue_row.get("case_fingerprint") != expected_fingerprint:
            raise ValueError(f"审核队列 fingerprint 已漂移：{case_id}")
        if case_id not in planner_by_id:
            raise ValueError(f"planner_cases 缺少审核 case：{case_id}")

        case = PlannerEvalCase.model_validate(planner_by_id[case_id])
        case_payload = case.model_dump(mode="json")
        if case.query != spec.query:
            raise ValueError(f"planner case 与 CaseSpec query 漂移：{case_id}")
        if (
            required_review_status is not None
            and case.human_review_status.value != required_review_status
        ):
            raise ValueError(
                "clean bundle case 审核状态不符合导出阶段："
                f"{case_id}={case.human_review_status.value}, "
                f"required={required_review_status}"
            )

        fingerprint_payload = asdict(spec)
        if _sha256_bytes(_canonical_json(fingerprint_payload)) != expected_fingerprint:
            raise ValueError(f"CaseSpec canonical fingerprint 无法复算：{case_id}")

        review_cases.append(
            {
                "case_id": case_id,
                "case_fingerprint": expected_fingerprint,
                # fingerprint_payload 是 fingerprint 的唯一计算输入，审核者无需读取构建
                # 脚本或历史台账即可独立重算，避免扩大白名单。
                "fingerprint_payload": fingerprint_payload,
                # eval_contract 只保留会影响评测 State、证据和 Action 的业务字段，
                # 明确删除 human_review_status 与 notes。
                "eval_contract": {
                    field: case_payload[field] for field in SAFE_EVAL_CASE_FIELDS
                },
                "route_bucket": spec.route_bucket,
                "route_rationale": spec.route_rationale,
                "evidence_refs": queue_row.get("evidence_refs", []),
                "web_evidence_refs": queue_row.get("web_evidence_refs", []),
                "hyde_probe": hyde_probes.get(spec.hyde_probe_id),
            }
        )

    route_counts = Counter(row["route_bucket"] for row in review_cases)
    expected_route_counts = expected_route_counts or {
        "hyde_fallback": 3,
        "web_required": 5,
        "ask_clarification": 2,
    }
    if route_counts != expected_route_counts:
        raise ValueError(f"当前盲审队列路线分布漂移：{dict(route_counts)}")
    return review_cases


def _build_leakage_reference(
    *,
    planner_rows: list[dict[str, Any]],
    curated_rows: list[dict[str, Any]],
    route_rows: list[dict[str, Any]],
    target_case_ids: set[str],
) -> list[dict[str, Any]]:
    """只投影语义泄漏需要的字段，禁止把来源里的审核状态和 notes 带入审核包。"""

    sources = (
        ("planner_case_registry", planner_rows),
        ("curated_seed_source", curated_rows),
        ("route_seed_source", route_rows),
    )
    result: list[dict[str, Any]] = []
    for source_dataset, rows in sources:
        for row in rows:
            case_id = str(row.get("case_id") or "").strip()
            if not case_id:
                raise ValueError(f"{source_dataset} 存在空 case_id")
            if case_id in target_case_ids:
                continue
            projected = {
                "source_dataset": source_dataset,
                **{
                    field: row.get(field, [] if field == "query_variants" else "")
                    for field in LEAKAGE_REFERENCE_FIELDS
                },
            }
            if not projected["query"] or not projected["leakage_group_id"]:
                raise ValueError(
                    f"{source_dataset} 泄漏参考缺少 query/leakage_group_id：{case_id}"
                )
            result.append(projected)
    return result


def _subset_local_sources(
    *,
    review_cases: list[dict[str, Any]],
    source_manifest: dict[str, Any],
    evidence_manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    required_chunks = {
        (
            str(ref["source_id"]),
            str(ref["document_id"]),
            str(ref["chunk_id"]),
            int(ref["index_version"]),
        )
        for case in review_cases
        for ref in case["evidence_refs"]
    }
    required_source_ids = {key[0] for key in required_chunks}
    source_specs = {
        str(source["source_id"]): source
        for source in source_manifest.get("sources", [])
    }
    documents = {
        str(document["source_id"]): document
        for document in evidence_manifest.get("documents", [])
    }
    if not required_source_ids <= set(source_specs):
        raise ValueError("本地来源配置缺少审核 case 引用的 source_id")
    if not required_source_ids <= set(documents):
        raise ValueError("本地证据清单缺少审核 case 引用的 source_id")

    selected_documents = []
    found_chunk_keys: set[tuple[str, str, str, int]] = set()
    for source_id in sorted(required_source_ids):
        document = documents[source_id]
        selected_chunks = []
        for chunk in document["chunks"]:
            key = (
                source_id,
                str(chunk["document_id"]),
                str(chunk["chunk_id"]),
                int(chunk["index_version"]),
            )
            if key in required_chunks:
                selected_chunks.append(chunk)
                found_chunk_keys.add(key)
        selected_documents.append(
            {
                "source_id": document["source_id"],
                "publisher": document["publisher"],
                "title": document["title"],
                "source_version": document["source_version"],
                "source_url": document["source_url"],
                "source_sha256": document["source_sha256"],
                "document_id": document["document_id"],
                "dataset_id": document["dataset_id"],
                "owner_user_id": document["owner_user_id"],
                "tenant_id": document["tenant_id"],
                "visibility": document["visibility"],
                "index_version": document["index_version"],
                "subject_id": document["subject_id"],
                "standard_subject_name": document["standard_subject_name"],
                "source_chunk_count": document["chunk_count"],
                "included_chunk_count": len(selected_chunks),
                "chunks": selected_chunks,
            }
        )
    if found_chunk_keys != required_chunks:
        missing = sorted(required_chunks - found_chunk_keys)
        raise ValueError(f"本地证据清单缺少引用 chunk：{missing}")

    # 扩展引用与来源清单做第二次逐字段核验，防止 queue 自带 hash 漂移。
    chunks_by_key = {
        (
            str(document["source_id"]),
            str(chunk["document_id"]),
            str(chunk["chunk_id"]),
            int(chunk["index_version"]),
        ): chunk
        for document in selected_documents
        for chunk in document["chunks"]
    }
    for case in review_cases:
        for ref in case["evidence_refs"]:
            key = (
                str(ref["source_id"]),
                str(ref["document_id"]),
                str(ref["chunk_id"]),
                int(ref["index_version"]),
            )
            chunk = chunks_by_key[key]
            if (
                int(ref["chunk_index"]) != int(chunk["chunk_index"])
                or ref["content_sha256"] != chunk["content_sha256"]
            ):
                raise ValueError(f"queue 本地证据身份漂移：{case['case_id']} {key}")

    clean_source_manifest = {
        key: value
        for key, value in source_manifest.items()
        if key != "sources"
    }
    clean_source_manifest["sources"] = [
        source_specs[source_id] for source_id in sorted(required_source_ids)
    ]
    clean_evidence_manifest = {
        "import_version": evidence_manifest["import_version"],
        "source_manifest_version": evidence_manifest["source_manifest_version"],
        "imported_at": evidence_manifest["imported_at"],
        "source_manifest_sha256": evidence_manifest["source_manifest_sha256"],
        "documents": selected_documents,
    }
    return clean_source_manifest, clean_evidence_manifest


def _subset_web_sources(
    *,
    review_cases: list[dict[str, Any]],
    source_manifest: dict[str, Any],
    evidence_manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    required_fact_ids = {
        str(ref["source_id"]): {str(fact["fact_id"]) for fact in ref["facts"]}
        for case in review_cases
        for ref in case["web_evidence_refs"]
    }
    source_specs = {
        str(source["source_id"]): source
        for source in source_manifest.get("sources", [])
    }
    evidence_sources = {
        str(source["source_id"]): source
        for source in evidence_manifest.get("sources", [])
    }
    required_source_ids = set(required_fact_ids)
    if not required_source_ids <= set(source_specs):
        raise ValueError("Web 来源配置缺少审核 case 引用的 source_id")
    if not required_source_ids <= set(evidence_sources):
        raise ValueError("Web 证据清单缺少审核 case 引用的 source_id")

    selected_evidence_sources = []
    for source_id in sorted(required_source_ids):
        source = evidence_sources[source_id]
        facts_by_id = {str(fact["fact_id"]): fact for fact in source["facts"]}
        if not required_fact_ids[source_id] <= set(facts_by_id):
            raise ValueError(f"Web 证据缺少 fact_id：{source_id}")
        selected_evidence_sources.append(
            {
                **{key: value for key, value in source.items() if key != "facts"},
                "facts": [
                    facts_by_id[fact_id]
                    for fact_id in sorted(required_fact_ids[source_id])
                ],
            }
        )

    selected_by_id = {
        str(source["source_id"]): source for source in selected_evidence_sources
    }
    for case in review_cases:
        for ref in case["web_evidence_refs"]:
            source = selected_by_id[str(ref["source_id"])]
            if any(
                ref[field] != source[field]
                for field in (
                    "canonical_url",
                    "captured_at",
                    "response_sha256",
                    "extracted_text_sha256",
                    "evidence_content_sha256",
                )
            ):
                raise ValueError(
                    f"queue Web 证据身份漂移：{case['case_id']} {ref['source_id']}"
                )

    clean_source_manifest = {
        key: value
        for key, value in source_manifest.items()
        if key != "sources"
    }
    clean_source_manifest["sources"] = [
        source_specs[source_id] for source_id in sorted(required_source_ids)
    ]
    clean_evidence_manifest = {
        key: value
        for key, value in evidence_manifest.items()
        if key != "sources"
    }
    clean_evidence_manifest["source_count"] = len(selected_evidence_sources)
    clean_evidence_manifest["sources"] = selected_evidence_sources
    return clean_source_manifest, clean_evidence_manifest


def _build_route_policy(
    matrix: dict[str, Any],
    *,
    route_buckets: set[str],
) -> dict[str, Any]:
    """只保留路线定义，不带 observed coverage（历史覆盖结果）或审核数量。"""

    selected = [
        row
        for row in matrix.get("route_buckets", [])
        if str(row["route_bucket"]) in route_buckets
    ]
    if {str(row["route_bucket"]) for row in selected} != route_buckets:
        raise ValueError("路线矩阵缺少盲审队列需要的 route bucket")
    return {
        "matrix_version": matrix["matrix_version"],
        "route_buckets": selected,
        "blind_review_rule": (
            "逐 case 判断 evidence、route、leakage 与表达；"
            "不能根据每桶数量倒推审核决定。"
        ),
    }


def _find_banned_keys(value: Any, *, prefix: str = "") -> list[str]:
    findings: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in BANNED_REVIEW_KEYS:
                findings.append(path)
            findings.extend(_find_banned_keys(child, prefix=path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(_find_banned_keys(child, prefix=f"{prefix}[{index}]"))
    return findings


def _validate_clean_json_file(path: Path) -> None:
    values = _read_jsonl(path) if path.suffix == ".jsonl" else [_read_json(path)]
    findings = [
        finding
        for value in values
        for finding in _find_banned_keys(value)
    ]
    if findings:
        raise ValueError(f"盲审包包含禁止审核字段：{path.name} {findings[:5]}")
    text = path.read_text(encoding="utf-8")
    markers = sorted(marker for marker in BANNED_HISTORY_MARKERS if marker in text)
    if markers:
        raise ValueError(f"盲审包包含历史审核标记：{path.name} {markers}")


def _review_instructions(
    bundle_id: str,
    *,
    task_label: str = "balanced dev",
    bundle_version: str = BUNDLE_VERSION,
    case_count: int = 10,
) -> str:
    return f"""# Stage 9 {task_label} clean blind review inputs

- Bundle ID：`{bundle_id}`
- Bundle version：`{bundle_version}`
- 该目录是本轮唯一允许读取的项目审核数据目录。
- 只允许额外读取 `local_source_manifest.json` 指向的本地 PDF，以及包内 URL 指向的官方网页。
- 禁止读取仓库其他 case 台账、审核决定、审核报告、Git 历史或先前 Agent 对话。
- 如果意外看到历史逐 case 结论，立即停止并报告 `blind_review_contaminated`。

## 文件用途

- `review_cases.jsonl`：{case_count} 条待审 case、可复算 fingerprint 的 payload、评测契约和冻结引用。
- `local_source_manifest.json`：本轮引用的本地 PDF 身份和路径。
- `local_evidence_manifest.json`：本轮实际引用的生产 chunk 身份子集，不含正文和审核状态。
- `web_source_manifest.json`：本轮引用的官方网页及必需短语。
- `web_evidence_manifest.json`：冻结 URL、抓取时间和响应/事实 hash。
- `route_policy.json`：仅含本轮路线定义，不含历史覆盖或审核数量。
- `leakage_reference.jsonl`：只含泄漏检查所需 query、variants、split 和 leakage group。
- `bundle_manifest.json`：输入来源 hash、输出文件 hash 和污染扫描结果。

## Fingerprint 复算

对每条 `fingerprint_payload` 执行：

```python
hashlib.sha256(
    json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
```

结果必须等于同行 `case_fingerprint`。本目录不包含 decision 模板；审核输出 schema
由任务提示词单独提供，防止输入包混入任何预填决定。
"""


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"盲审包目录已存在，拒绝静默覆盖：{output_dir}"
            )
        unexpected = {
            path.name for path in output_dir.iterdir()
        } - OUTPUT_FILE_NAMES
        if unexpected:
            raise ValueError(
                f"盲审包目录包含未知文件，拒绝覆盖：{sorted(unexpected)}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)


def validate_blind_review_bundle(output_dir: Path) -> dict[str, Any]:
    """验证已落盘审核包的文件集合、hash、字段白名单和 case 身份。"""

    missing = sorted(name for name in OUTPUT_FILE_NAMES if not (output_dir / name).is_file())
    if missing:
        raise ValueError(f"盲审包缺少文件：{missing}")
    unexpected = sorted(
        path.name for path in output_dir.iterdir() if path.name not in OUTPUT_FILE_NAMES
    )
    if unexpected:
        raise ValueError(f"盲审包包含未知文件：{unexpected}")

    for name in DATA_FILE_NAMES:
        _validate_clean_json_file(output_dir / name)

    manifest = _read_json(output_dir / "bundle_manifest.json")
    for record in manifest["output_files"]:
        path = output_dir / record["file"]
        if _sha256_file(path) != record["sha256"]:
            raise ValueError(f"盲审包输出 hash 漂移：{record['file']}")

    review_cases = _read_jsonl(output_dir / "review_cases.jsonl")
    if int(manifest["case_count"]) != len(review_cases):
        raise ValueError("bundle manifest.case_count 与 review_cases 不一致")
    for row in review_cases:
        actual = _sha256_bytes(_canonical_json(row["fingerprint_payload"]))
        if actual != row["case_fingerprint"]:
            raise ValueError(f"盲审包 case fingerprint 不可复算：{row['case_id']}")

    return {
        "ok": True,
        "bundle_id": manifest["bundle_id"],
        "case_count": len(review_cases),
        "route_counts": dict(
            Counter(row["route_bucket"] for row in review_cases)
        ),
        "contamination_scan": "passed",
    }


def export_blind_review_bundle(
    *,
    queue_path: Path = DEFAULT_QUEUE,
    planner_cases_path: Path = DEFAULT_PLANNER_CASES,
    curated_cases_path: Path = DEFAULT_CURATED_CASES,
    route_cases_path: Path = DEFAULT_ROUTE_CASES,
    local_source_manifest_path: Path = DEFAULT_LOCAL_SOURCE_MANIFEST,
    local_evidence_manifest_path: Path = DEFAULT_LOCAL_EVIDENCE_MANIFEST,
    web_source_manifest_path: Path = DEFAULT_WEB_SOURCE_MANIFEST,
    web_evidence_manifest_path: Path = DEFAULT_WEB_EVIDENCE_MANIFEST,
    route_matrix_path: Path = DEFAULT_ROUTE_MATRIX,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    generated_at: str | None = None,
    overwrite: bool = False,
    case_specs: Iterable[Any] = CASE_SPECS,
    hyde_probes: dict[str, dict[str, Any]] = HYDE_PROBES,
    expected_route_counts: dict[str, int] | None = None,
    bundle_version: str = BUNDLE_VERSION,
    bundle_id_prefix: str = "stage9-balanced-dev-clean",
    task_label: str = "balanced dev",
    required_review_status: str | None = "pending",
) -> dict[str, Any]:
    """构建审核包并在返回前执行完整污染与 hash 校验。"""

    input_paths = {
        "queue": queue_path,
        "planner_cases": planner_cases_path,
        "curated_cases": curated_cases_path,
        "route_cases": route_cases_path,
        "local_source_manifest": local_source_manifest_path,
        "local_evidence_manifest": local_evidence_manifest_path,
        "web_source_manifest": web_source_manifest_path,
        "web_evidence_manifest": web_evidence_manifest_path,
        "route_matrix": route_matrix_path,
    }
    for name, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"盲审包输入不存在：{name}={path}")

    queue_rows = _read_jsonl(queue_path)
    planner_rows = _read_jsonl(planner_cases_path)
    curated_rows = _read_jsonl(curated_cases_path)
    route_rows = _read_jsonl(route_cases_path)
    review_cases = _build_review_cases(
        queue_rows=queue_rows,
        planner_rows=planner_rows,
        case_specs=case_specs,
        hyde_probes=hyde_probes,
        expected_route_counts=expected_route_counts,
        required_review_status=required_review_status,
    )
    target_case_ids = {row["case_id"] for row in review_cases}
    leakage_reference = _build_leakage_reference(
        planner_rows=planner_rows,
        curated_rows=curated_rows,
        route_rows=route_rows,
        target_case_ids=target_case_ids,
    )
    local_source_manifest, local_evidence_manifest = _subset_local_sources(
        review_cases=review_cases,
        source_manifest=_read_json(local_source_manifest_path),
        evidence_manifest=_read_json(local_evidence_manifest_path),
    )
    web_source_manifest, web_evidence_manifest = _subset_web_sources(
        review_cases=review_cases,
        source_manifest=_read_json(web_source_manifest_path),
        evidence_manifest=_read_json(web_evidence_manifest_path),
    )
    route_policy = _build_route_policy(
        _read_json(route_matrix_path),
        route_buckets={row["route_bucket"] for row in review_cases},
    )

    _prepare_output_dir(output_dir, overwrite=overwrite)
    _write_jsonl(output_dir / "review_cases.jsonl", review_cases)
    _write_jsonl(output_dir / "leakage_reference.jsonl", leakage_reference)
    _write_json(output_dir / "local_source_manifest.json", local_source_manifest)
    _write_json(output_dir / "local_evidence_manifest.json", local_evidence_manifest)
    _write_json(output_dir / "web_source_manifest.json", web_source_manifest)
    _write_json(output_dir / "web_evidence_manifest.json", web_evidence_manifest)
    _write_json(output_dir / "route_policy.json", route_policy)

    queue_sha256 = _sha256_file(queue_path)
    bundle_id = f"{bundle_id_prefix}-{queue_sha256[:16]}"
    (output_dir / "REVIEW_INSTRUCTIONS.md").write_text(
        _review_instructions(
            bundle_id,
            task_label=task_label,
            bundle_version=bundle_version,
            case_count=len(review_cases),
        ),
        encoding="utf-8",
    )

    generated_at = generated_at or datetime.now(UTC).replace(microsecond=0).isoformat()
    files_before_manifest = sorted(OUTPUT_FILE_NAMES - {"bundle_manifest.json"})
    manifest = {
        "bundle_version": bundle_version,
        "sanitization_policy_version": SANITIZATION_POLICY_VERSION,
        "bundle_id": bundle_id,
        "generated_at": generated_at,
        "case_count": len(review_cases),
        "route_counts": dict(
            sorted(Counter(row["route_bucket"] for row in review_cases).items())
        ),
        "queue_sha256": queue_sha256,
        "input_files": [
            {
                "name": name,
                "logical_path": _logical_path(path),
                "sha256": _sha256_file(path),
            }
            for name, path in input_paths.items()
        ],
        "output_files": [
            {
                "file": name,
                "sha256": _sha256_file(output_dir / name),
                "bytes": (output_dir / name).stat().st_size,
            }
            for name in files_before_manifest
        ],
        "contamination_scan": "passed",
        "historical_decision_count": 0,
    }
    _write_json(output_dir / "bundle_manifest.json", manifest)

    validation = validate_blind_review_bundle(output_dir)
    return {
        **validation,
        "output_dir": _logical_path(output_dir),
        "queue_sha256": queue_sha256,
        "bundle_manifest_sha256": _sha256_file(
            output_dir / "bundle_manifest.json"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--planner-cases", type=Path, default=DEFAULT_PLANNER_CASES)
    parser.add_argument("--curated-cases", type=Path, default=DEFAULT_CURATED_CASES)
    parser.add_argument("--route-cases", type=Path, default=DEFAULT_ROUTE_CASES)
    parser.add_argument(
        "--local-source-manifest",
        type=Path,
        default=DEFAULT_LOCAL_SOURCE_MANIFEST,
    )
    parser.add_argument(
        "--local-evidence-manifest",
        type=Path,
        default=DEFAULT_LOCAL_EVIDENCE_MANIFEST,
    )
    parser.add_argument(
        "--web-source-manifest",
        type=Path,
        default=DEFAULT_WEB_SOURCE_MANIFEST,
    )
    parser.add_argument(
        "--web-evidence-manifest",
        type=Path,
        default=DEFAULT_WEB_EVIDENCE_MANIFEST,
    )
    parser.add_argument("--route-matrix", type=Path, default=DEFAULT_ROUTE_MATRIX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--generated-at")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_blind_review_bundle(
        queue_path=args.queue,
        planner_cases_path=args.planner_cases,
        curated_cases_path=args.curated_cases,
        route_cases_path=args.route_cases,
        local_source_manifest_path=args.local_source_manifest,
        local_evidence_manifest_path=args.local_evidence_manifest,
        web_source_manifest_path=args.web_source_manifest,
        web_evidence_manifest_path=args.web_evidence_manifest,
        route_matrix_path=args.route_matrix,
        output_dir=args.output_dir,
        generated_at=args.generated_at,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
