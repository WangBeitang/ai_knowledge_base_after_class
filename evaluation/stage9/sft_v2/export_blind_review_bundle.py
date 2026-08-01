"""导出任务 9.3.22 的 SFT V2（监督微调第二版）干净盲审包。

该入口只做一件事：把 9.3.21 的 125 条新候选投影为不含生成自评、备用身份和
历史审核状态的公共审核输入。二审、三审读取同一目录，但必须把决定写入不同目录。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = PROJECT_ROOT / "evaluation/stage9/artifacts/sft_v2"
DEFAULT_CASES = ARTIFACT_DIR / "sft_v2_new_candidate_cases.jsonl"
DEFAULT_TRAJECTORIES = ARTIFACT_DIR / "sft_v2_new_candidate_trajectories.jsonl"
DEFAULT_PROVIDER_RECORDS = ARTIFACT_DIR / "sft_v2_provider_observations.jsonl"
DEFAULT_WEB_EVIDENCE = ARTIFACT_DIR / "sft_v2_web_evidence_manifest.json"
DEFAULT_SNAPSHOT = ARTIFACT_DIR / "sft_v2_environment_snapshot.json"
DEFAULT_OUTPUT_DIR = ARTIFACT_DIR / "blind_review_bundle_v1"

BUNDLE_VERSION = "sft-v2-clean-blind-review-bundle-v1"
EXPECTED_CASE_COUNT = 125
EXPECTED_ROUTE_COUNTS = {
    "ask_clarification": 7,
    "local_search -> answer": 12,
    "local_search -> ask_clarification": 9,
    "local_search -> hyde_search -> answer": 9,
    "local_search -> hyde_search -> ask_clarification": 6,
    "local_search -> hyde_search -> refuse": 4,
    "local_search -> hyde_search -> web_search -> answer": 12,
    "local_search -> hyde_search -> web_search -> ask_clarification": 5,
    "local_search -> hyde_search -> web_search -> refuse": 4,
    "local_search -> refuse": 5,
    "local_search -> web_search -> answer": 15,
    "local_search -> web_search -> ask_clarification": 5,
    "local_search -> web_search -> refuse": 3,
    "refuse": 3,
    "web_search -> answer": 16,
    "web_search -> ask_clarification": 6,
    "web_search -> refuse": 4,
}

OUTPUT_FILE_NAMES = {
    "REVIEW_INSTRUCTIONS.md",
    "review_cases.jsonl",
    "provider_observations.jsonl",
    "web_evidence_manifest.json",
    "environment_snapshot.json",
    "leakage_reference.jsonl",
    "bundle_manifest.json",
}
JSON_DATA_FILE_NAMES = OUTPUT_FILE_NAMES - {"REVIEW_INSTRUCTIONS.md"}

# 这些字段会暴露生成者结论、备用身份或审核生命周期，不能进入公共盲审输入。
BANNED_KEYS = {
    "artifact_status",
    "formal_gap_count",
    "generation_gate",
    "human_review_status",
    "reserve",
    "reserve_count",
    "review_notes",
    "review_status",
    "reviewed_at",
    "reviewer_id",
    "reward_summary",
}

SAFE_CASE_FIELDS = (
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


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _logical_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _unique_by(rows: list[dict[str, Any]], field: str, source: str) -> dict[str, dict[str, Any]]:
    result = {str(row[field]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{source} 存在重复 {field}")
    return result


def _find_banned_keys(value: Any, prefix: str = "") -> list[str]:
    findings: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in BANNED_KEYS:
                findings.append(path)
            findings.extend(_find_banned_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(_find_banned_keys(child, f"{prefix}[{index}]"))
    return findings


def _original_fingerprint(row: dict[str, Any]) -> str:
    payload = {
        "case": row["case_contract"],
        "route": row["route"],
        "source_evidence": row["source_evidence"],
        "trace_steps": [
            {**step, "duration_ms": 0} for step in row["trace_steps"]
        ],
        "generation_batch": row["generation_batch"],
    }
    return _sha256_bytes(_canonical_json(payload))


def _build_review_cases(
    case_rows: list[dict[str, Any]],
    trajectory_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(case_rows) != EXPECTED_CASE_COUNT or len(trajectory_rows) != EXPECTED_CASE_COUNT:
        raise ValueError("SFT V2 新候选必须恰好为 125 条")
    cases = _unique_by(case_rows, "case_id", "candidate cases")
    trajectories = _unique_by(trajectory_rows, "candidate_id", "candidate trajectories")
    if set(cases) != set(trajectories):
        raise ValueError("candidate cases 与 trajectories 的 case_id 集合不一致")

    review_rows: list[dict[str, Any]] = []
    for candidate_id in sorted(trajectories):
        row = trajectories[candidate_id]
        case = cases[candidate_id]
        if row.get("review_status") != "pending" or row.get("artifact_status") != "candidate":
            raise ValueError(f"候选不是 pending/candidate：{candidate_id}")
        if row.get("case_contract") != case:
            raise ValueError(f"case contract 漂移：{candidate_id}")
        if _original_fingerprint(row) != row.get("content_fingerprint"):
            raise ValueError(f"content_fingerprint 无法复算：{candidate_id}")

        blind_payload = {
            "candidate_id": candidate_id,
            "source_trace_id": row["source_trace_id"],
            "generation_batch": row["generation_batch"],
            "build_version": row["build_version"],
            "query": row["query"],
            "route": row["route"],
            "expected_terminal": row["expected_terminal"],
            "device_family": row["device_family"],
            "question_family": row["question_family"],
            "leakage_group_id": row["leakage_group_id"],
            "eval_contract": {field: case[field] for field in SAFE_CASE_FIELDS},
            "source_evidence": row["source_evidence"],
            "trace_steps": row["trace_steps"],
            "provider_record_ids": row["provider_record_ids"],
        }
        review_rows.append({
            "candidate_id": candidate_id,
            # 原 fingerprint 作为审核决定绑定身份；盲审 payload 单独提供可复算指纹。
            "content_fingerprint": row["content_fingerprint"],
            "blind_case_fingerprint": _sha256_bytes(_canonical_json(blind_payload)),
            "blind_fingerprint_payload": blind_payload,
        })

    route_counts = Counter(
        " -> ".join(row["blind_fingerprint_payload"]["route"])
        for row in review_rows
    )
    if dict(sorted(route_counts.items())) != EXPECTED_ROUTE_COUNTS:
        raise ValueError(f"17 条完整路线数量漂移：{dict(route_counts)}")
    if len({row["content_fingerprint"] for row in review_rows}) != EXPECTED_CASE_COUNT:
        raise ValueError("content_fingerprint 不唯一")
    if len({row["blind_case_fingerprint"] for row in review_rows}) != EXPECTED_CASE_COUNT:
        raise ValueError("blind_case_fingerprint 不唯一")
    return review_rows


def _select_provider_records(
    review_rows: list[dict[str, Any]],
    provider_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    provider_by_id = _unique_by(provider_rows, "record_id", "provider observations")
    required_ids = {
        record_id
        for row in review_rows
        for record_id in row["blind_fingerprint_payload"]["provider_record_ids"]
    }
    if required_ids != set(provider_by_id):
        missing = sorted(required_ids - set(provider_by_id))
        extra = sorted(set(provider_by_id) - required_ids)
        raise ValueError(f"Provider 记录集合不闭环：missing={missing[:3]}, extra={extra[:3]}")

    selected = sorted(
        provider_rows,
        key=lambda row: (str(row["case_id"]), int(row["step"]), str(row["record_id"])),
    )
    review_by_id = {row["candidate_id"]: row for row in review_rows}
    for record in selected:
        case_id = str(record["case_id"])
        if case_id not in review_by_id:
            raise ValueError(f"Provider 记录引用未知 case：{case_id}")
        payload = review_by_id[case_id]["blind_fingerprint_payload"]
        if record["run_id"] != payload["source_trace_id"]:
            raise ValueError(f"Provider run_id 漂移：{record['record_id']}")
        step_index = int(record["step"]) - 1
        trace_step = payload["trace_steps"][step_index]
        if record["action"] != trace_step["decision"]["action"]:
            raise ValueError(f"Provider action 与 trace 不一致：{record['record_id']}")
    return selected


def _leakage_source_paths() -> list[Path]:
    paths = {
        *PROJECT_ROOT.glob("evaluation/stage8/cases/*.jsonl"),
        *PROJECT_ROOT.glob("evaluation/stage9/artifacts/heldout_route_test/*cases*.jsonl"),
    }
    return sorted(path for path in paths if path.is_file())


def _build_leakage_reference(paths: list[Path], target_ids: set[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in paths:
        for row in _read_jsonl(path):
            if str(row.get("split") or "") not in {"dev", "test", "demo_regression"}:
                continue
            case_id = str(row.get("case_id") or "").strip()
            if not case_id or case_id in target_ids:
                continue
            result.append({
                "source_dataset": _logical_path(path),
                "case_id": case_id,
                "split": row.get("split"),
                "leakage_group_id": row.get("leakage_group_id", ""),
                "query": row.get("query", ""),
                "query_variants": row.get("query_variants") or [],
                "expected_chunks": [
                    {
                        "document_id": chunk.get("document_id"),
                        "chunk_id": chunk.get("chunk_id"),
                        "index_version": chunk.get("index_version"),
                    }
                    for chunk in row.get("expected_chunks") or []
                ],
            })
    return result


def _instructions(bundle_id: str) -> str:
    return f"""# SFT V2 clean blind review bundle

- Bundle ID：`{bundle_id}`
- 本目录是二审、三审共同使用的只读输入包。
- 两位审核者必须使用不同输出目录，且不得读取对方结果。
- `review_cases.jsonl` 含 125 条脱敏候选；已删除备用身份、生成自评和审核状态。
- `provider_observations.jsonl` 含真实 Provider（动作执行器/环境结果提供器）记录。
- `web_evidence_manifest.json`、`environment_snapshot.json` 用于来源与索引身份核对。
- `leakage_reference.jsonl` 仅用于 dev/test 的问题和 chunk（文本块）泄漏检查。
- `content_fingerprint` 绑定原候选；`blind_case_fingerprint` 可由同条 `blind_fingerprint_payload` 复算。
- 不允许修改本目录、候选源文件或 37 条旧轨迹；不得在本轮冻结正式集或运行训练。
"""


def validate_blind_review_bundle(output_dir: Path) -> dict[str, Any]:
    missing = sorted(name for name in OUTPUT_FILE_NAMES if not (output_dir / name).is_file())
    unexpected = sorted(path.name for path in output_dir.iterdir() if path.name not in OUTPUT_FILE_NAMES)
    if missing or unexpected:
        raise ValueError(f"盲审包文件集合错误：missing={missing}, unexpected={unexpected}")

    for name in JSON_DATA_FILE_NAMES:
        path = output_dir / name
        values = _read_jsonl(path) if path.suffix == ".jsonl" else [_read_json(path)]
        findings = [item for value in values for item in _find_banned_keys(value)]
        if findings:
            raise ValueError(f"盲审包包含禁止字段：{name} {findings[:5]}")

    manifest = _read_json(output_dir / "bundle_manifest.json")
    for record in manifest["output_files"]:
        path = output_dir / record["file"]
        if _sha256_file(path) != record["sha256"]:
            raise ValueError(f"盲审包文件 hash 漂移：{record['file']}")
    if _sha256_bytes(_canonical_json(manifest["output_files"])) != manifest["bundle_content_sha256"]:
        raise ValueError("盲审包整体内容 hash 漂移")

    review_rows = _read_jsonl(output_dir / "review_cases.jsonl")
    if len(review_rows) != EXPECTED_CASE_COUNT:
        raise ValueError("review_cases 不是 125 条")
    for row in review_rows:
        actual = _sha256_bytes(_canonical_json(row["blind_fingerprint_payload"]))
        if actual != row["blind_case_fingerprint"]:
            raise ValueError(f"blind_case_fingerprint 无法复算：{row['candidate_id']}")
    route_counts = dict(sorted(Counter(
        " -> ".join(row["blind_fingerprint_payload"]["route"])
        for row in review_rows
    ).items()))
    if route_counts != EXPECTED_ROUTE_COUNTS:
        raise ValueError("review_cases 路线数量漂移")

    return {
        "ok": True,
        "bundle_id": manifest["bundle_id"],
        "bundle_content_sha256": manifest["bundle_content_sha256"],
        "case_count": len(review_rows),
        "provider_record_count": len(_read_jsonl(output_dir / "provider_observations.jsonl")),
        "leakage_reference_count": len(_read_jsonl(output_dir / "leakage_reference.jsonl")),
        "route_counts": route_counts,
        "contamination_scan": "passed",
    }


def export_blind_review_bundle(
    *,
    cases_path: Path = DEFAULT_CASES,
    trajectories_path: Path = DEFAULT_TRAJECTORIES,
    provider_records_path: Path = DEFAULT_PROVIDER_RECORDS,
    web_evidence_path: Path = DEFAULT_WEB_EVIDENCE,
    snapshot_path: Path = DEFAULT_SNAPSHOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    generated_at: str | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"盲审包目录已存在，拒绝静默覆盖：{output_dir}")
    source_paths = {
        "candidate_cases": cases_path,
        "candidate_trajectories": trajectories_path,
        "provider_observations": provider_records_path,
        "web_evidence": web_evidence_path,
        "environment_snapshot": snapshot_path,
    }
    for name, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"盲审输入不存在：{name}={path}")

    review_rows = _build_review_cases(_read_jsonl(cases_path), _read_jsonl(trajectories_path))
    provider_rows = _select_provider_records(review_rows, _read_jsonl(provider_records_path))
    web_evidence = _read_json(web_evidence_path)
    snapshot = _read_json(snapshot_path)
    leakage_paths = _leakage_source_paths()
    leakage_rows = _build_leakage_reference(
        leakage_paths,
        {row["candidate_id"] for row in review_rows},
    )

    input_identity = [
        {"name": name, "logical_path": _logical_path(path), "sha256": _sha256_file(path)}
        for name, path in sorted(source_paths.items())
    ] + [
        {"name": "leakage_source", "logical_path": _logical_path(path), "sha256": _sha256_file(path)}
        for path in leakage_paths
    ]
    bundle_id = f"sft-v2-clean-{_sha256_bytes(_canonical_json(input_identity))[:16]}"

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".blind_review_bundle_v1-", dir=output_dir.parent) as temp:
        temp_dir = Path(temp)
        _write_jsonl(temp_dir / "review_cases.jsonl", review_rows)
        _write_jsonl(temp_dir / "provider_observations.jsonl", provider_rows)
        _write_json(temp_dir / "web_evidence_manifest.json", web_evidence)
        _write_json(temp_dir / "environment_snapshot.json", snapshot)
        _write_jsonl(temp_dir / "leakage_reference.jsonl", leakage_rows)
        (temp_dir / "REVIEW_INSTRUCTIONS.md").write_text(
            _instructions(bundle_id), encoding="utf-8"
        )

        output_records = [
            {
                "file": name,
                "sha256": _sha256_file(temp_dir / name),
                "bytes": (temp_dir / name).stat().st_size,
            }
            for name in sorted(OUTPUT_FILE_NAMES - {"bundle_manifest.json"})
        ]
        manifest = {
            "bundle_version": BUNDLE_VERSION,
            "bundle_id": bundle_id,
            "generated_at": generated_at or datetime.now(UTC).replace(microsecond=0).isoformat(),
            "case_count": len(review_rows),
            "provider_record_count": len(provider_rows),
            "leakage_reference_count": len(leakage_rows),
            "route_counts": EXPECTED_ROUTE_COUNTS,
            "input_files": input_identity,
            "output_files": output_records,
            "bundle_content_sha256": _sha256_bytes(_canonical_json(output_records)),
            "contamination_scan": "passed",
            "historical_review_decision_count": 0,
            "reviewer_outputs_must_be_separate": True,
            "formal_dataset_frozen": False,
            "training_performed": False,
        }
        _write_json(temp_dir / "bundle_manifest.json", manifest)
        validation = validate_blind_review_bundle(temp_dir)
        temp_dir.replace(output_dir)

    return {
        **validation,
        "output_dir": _logical_path(output_dir),
        "bundle_manifest_sha256": _sha256_file(output_dir / "bundle_manifest.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = (
        validate_blind_review_bundle(args.output_dir)
        if args.validate_only
        else export_blind_review_bundle(output_dir=args.output_dir)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
