"""冻结并校验阶段 9 SFT（监督微调）产物。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FREEZE_VERSION = "stage9-sft-artifact-freeze-v2"
REQUIRED_CHECKPOINT_FILES = (
    Path("checkpoint_manifest.json"),
    Path("train_metrics.json"),
    Path("training_config.json"),
    Path("model/adapter/adapter_config.json"),
    Path("model/adapter/adapter_model.safetensors"),
)
LEGACY_DEV_RUN_FILES = (
    Path("sft_eval_dev.json"),
    Path("cloud_run_report.json"),
    Path("dev_eval.log"),
    Path("command.txt"),
)
EXPANDED_DEV_RUN_FILES = (
    Path("sft_expanded_dev_eval.json"),
    Path("sft_9_4_admission_decision.json"),
    Path("阶段9-SFT-9.4准入报告.md"),
    Path("cloud_run_report.json"),
    Path("expanded_dev_gate.log"),
    Path("command.txt"),
)


def freeze_sft_artifacts(
        *,
        project_root: Path,
        checkpoint_dir: Path,
        train_run_dir: Path,
        dev_run_dir: Path,
        vllm_freeze: Path,
        source_training_config: Path,
        dev_cases: Path,
        dev_snapshot: Path,
        output_dir: Path,
        label: str = "sft-v1",
        overwrite: bool = False,
) -> dict[str, Any]:
    """生成带逐文件 SHA256（文件哈希）的 SFT 归档，并校验关键身份一致。"""

    project_root = project_root.resolve()
    checkpoint_abs = _resolve_project_path(project_root, checkpoint_dir)
    train_run_abs = _resolve_project_path(project_root, train_run_dir)
    dev_run_abs = _resolve_project_path(project_root, dev_run_dir)
    vllm_freeze_abs = _resolve_project_path(project_root, vllm_freeze)
    source_training_config_abs = _resolve_project_path(project_root, source_training_config)
    dev_cases_abs = _resolve_project_path(project_root, dev_cases)
    dev_snapshot_abs = _resolve_project_path(project_root, dev_snapshot)
    output_dir = output_dir.resolve()

    _require_directory(checkpoint_abs, "checkpoint 根目录")
    _require_directory(train_run_abs, "正式训练 run_dir")
    _require_directory(dev_run_abs, "dev eval run_dir")
    for relative_path in REQUIRED_CHECKPOINT_FILES:
        _require_file(checkpoint_abs / relative_path, f"checkpoint 文件 {relative_path}")
    _require_file(train_run_abs / "cloud_run_report.json", "正式训练 cloud run report")
    dev_run_kind, dev_eval_relative, required_dev_files = _dev_run_layout(
        dev_run_abs
    )
    for relative_path in required_dev_files:
        _require_file(dev_run_abs / relative_path, f"dev eval 文件 {relative_path}")
    for required_path, description in (
        (vllm_freeze_abs, "vLLM 环境冻结清单"),
        (source_training_config_abs, "源训练配置"),
        (dev_cases_abs, "dev case 数据"),
        (dev_snapshot_abs, "dev 环境快照"),
    ):
        _require_file(required_path, description)

    checkpoint_manifest = _read_json(checkpoint_abs / "checkpoint_manifest.json")
    run_id = str(checkpoint_manifest.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("checkpoint_manifest.json 缺少 run_id（运行身份）。")
    if checkpoint_abs.name != run_id:
        raise ValueError(
            "checkpoint 目录名与 checkpoint_manifest.run_id 不一致："
            f"directory={checkpoint_abs.name!r}, run_id={run_id!r}"
        )

    train_report = _read_json(train_run_abs / "cloud_run_report.json")
    _require_report_run_id(train_report, run_id, "正式训练 cloud run report")
    dev_report = _read_json(dev_run_abs / "cloud_run_report.json")
    _require_report_run_id(dev_report, run_id, "dev eval cloud run report")
    dev_eval = _read_json(dev_run_abs / dev_eval_relative)
    dev_checkpoint = _dev_eval_checkpoint(dev_eval)
    if run_id not in dev_checkpoint:
        raise ValueError(
            "dev eval 的 checkpoint 与正式训练 run_id 不一致："
            f"checkpoint={dev_checkpoint!r}, run_id={run_id!r}"
        )

    training_config = _read_json(checkpoint_abs / "training_config.json")
    supporting_paths = _supporting_paths(project_root, training_config)
    for supporting_path in supporting_paths:
        _require_file(supporting_path, "训练复现输入")

    selected_files = _collect_files(
        project_root,
        roots=(
            checkpoint_abs,
            train_run_abs,
            dev_run_abs,
            vllm_freeze_abs,
            source_training_config_abs,
            dev_cases_abs,
            dev_snapshot_abs,
            *supporting_paths,
        ),
    )
    records = [_file_record(project_root, path) for path in selected_files]
    freeze_manifest = {
        "freeze_version": FREEZE_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "label": label,
        "run_id": run_id,
        "checkpoint_dir": _relative_text(project_root, checkpoint_abs),
        "train_run_dir": _relative_text(project_root, train_run_abs),
        "dev_run_dir": _relative_text(project_root, dev_run_abs),
        "dev_run_kind": dev_run_kind,
        "vllm_freeze": _relative_text(project_root, vllm_freeze_abs),
        "code_version": checkpoint_manifest.get("code_version"),
        "base_model_id": checkpoint_manifest.get("base_model_id"),
        "model_profile_id": checkpoint_manifest.get("model_profile_id"),
        "snapshot_id": checkpoint_manifest.get("snapshot_id"),
        "reward_profile": checkpoint_manifest.get("reward_profile"),
        "sample_count": checkpoint_manifest.get("sample_count"),
        "file_count": len(records),
        "files": records,
    }

    safe_label = _safe_filename(label)
    archive_name = f"stage9_{safe_label}_{run_id}.tar.gz"
    archive_path = output_dir / archive_name
    manifest_path = output_dir / f"{archive_name}.manifest.json"
    archive_sha256_path = output_dir / f"{archive_name}.sha256"
    for output_path in (archive_path, manifest_path, archive_sha256_path):
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"输出已存在；如需覆盖请传 --overwrite：{output_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_text = json.dumps(freeze_manifest, ensure_ascii=False, indent=2) + "\n"
    sums_text = "".join(f"{record['sha256']}  {record['path']}\n" for record in records)
    manifest_path.write_text(manifest_text, encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="stage9-sft-freeze-") as temporary_dir:
        temporary_root = Path(temporary_dir)
        internal_manifest = temporary_root / "freeze_manifest.json"
        internal_sums = temporary_root / "SHA256SUMS.txt"
        internal_manifest.write_text(manifest_text, encoding="utf-8")
        internal_sums.write_text(sums_text, encoding="utf-8")
        with tarfile.open(archive_path, "w:gz") as archive:
            for path in selected_files:
                archive.add(path, arcname=_relative_text(project_root, path), recursive=False)
            archive.add(internal_manifest, arcname="_freeze/freeze_manifest.json", recursive=False)
            archive.add(internal_sums, arcname="_freeze/SHA256SUMS.txt", recursive=False)

    _verify_archive(archive_path, expected_paths=[record["path"] for record in records])
    archive_sha256 = _sha256(archive_path)
    archive_sha256_path.write_text(f"{archive_sha256}  {archive_name}\n", encoding="utf-8")

    return {
        "ok": True,
        "freeze_version": FREEZE_VERSION,
        "run_id": run_id,
        "file_count": len(records),
        "archive": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_sha256,
        "archive_sha256_file": str(archive_sha256_path),
        "manifest": str(manifest_path),
    }


def _dev_run_layout(dev_run_dir: Path) -> tuple[str, Path, tuple[Path, ...]]:
    """识别历史 7 条 dev run 或 9.3.16 expanded dev run，二者都可冻结但不能混装。"""

    expanded_exists = all(
        (dev_run_dir / path).is_file() for path in EXPANDED_DEV_RUN_FILES
    )
    legacy_exists = all(
        (dev_run_dir / path).is_file() for path in LEGACY_DEV_RUN_FILES
    )
    if expanded_exists:
        return (
            "stage9_3_16_expanded_dev_gate",
            Path("sft_expanded_dev_eval.json"),
            EXPANDED_DEV_RUN_FILES,
        )
    if legacy_exists:
        return (
            "legacy_dev_eval",
            Path("sft_eval_dev.json"),
            LEGACY_DEV_RUN_FILES,
        )
    raise FileNotFoundError(
        "dev run_dir 既不满足历史 dev eval，也不满足 9.3.16 expanded dev 产物契约"
    )


def _supporting_paths(project_root: Path, training_config: dict[str, Any]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for key in ("train_data", "train_manifest", "reward_profile", "model_profile_path"):
        raw_path = str(training_config.get(key) or "").strip()
        if raw_path:
            paths.append(_resolve_project_path(project_root, Path(raw_path)))
    return tuple(paths)


def _collect_files(project_root: Path, *, roots: Iterable[Path]) -> list[Path]:
    selected: dict[str, Path] = {}
    for root in roots:
        candidates = [root] if root.is_file() else sorted(root.rglob("*"))
        for candidate in candidates:
            if candidate.is_symlink():
                raise ValueError(f"冻结范围不允许符号链接：{candidate}")
            if not candidate.is_file():
                continue
            relative = _relative_text(project_root, candidate)
            selected[relative] = candidate
    return [selected[key] for key in sorted(selected)]


def _file_record(project_root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": _relative_text(project_root, path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _require_report_run_id(report: dict[str, Any], expected_run_id: str, description: str) -> None:
    reported_run_id = str((report.get("checkpoint_manifest") or {}).get("run_id") or "").strip()
    if reported_run_id != expected_run_id:
        raise ValueError(
            f"{description} 的 run_id 不一致："
            f"reported={reported_run_id!r}, expected={expected_run_id!r}"
        )


def _dev_eval_checkpoint(dev_eval: dict[str, Any]) -> str:
    """读取 dev eval 实际 Schema 中的 checkpoint；兼容早期顶层字段。"""

    top_level = str(dev_eval.get("checkpoint") or "").strip()
    if top_level:
        return top_level
    summaries = dev_eval.get("planner_summaries") or []
    if not isinstance(summaries, list) or not summaries:
        return ""
    first_summary = summaries[0]
    if not isinstance(first_summary, dict):
        return ""
    config = first_summary.get("config") or {}
    if not isinstance(config, dict):
        return ""
    return str(config.get("checkpoint") or "").strip()


def _verify_archive(archive_path: Path, *, expected_paths: list[str]) -> None:
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {member.name for member in archive.getmembers() if member.isfile()}
    expected = set(expected_paths) | {
        "_freeze/freeze_manifest.json",
        "_freeze/SHA256SUMS.txt",
    }
    missing = sorted(expected - members)
    if missing:
        raise RuntimeError(f"归档缺少文件：{missing}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 文件格式错误：{path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是 object（对象）：{path}")
    return payload


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description}不存在：{path}")


def _require_directory(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{description}不存在：{path}")


def _resolve_project_path(project_root: Path, path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"冻结输入必须位于项目目录内：{path}") from exc
    return resolved


def _relative_text(project_root: Path, path: Path) -> str:
    return path.resolve().relative_to(project_root).as_posix()


def _safe_filename(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    cleaned = cleaned.strip("-_")
    if not cleaned:
        raise ValueError("label（标签）不能是空字符串。")
    return cleaned


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="冻结并校验阶段 9 SFT（监督微调）产物。")
    parser.add_argument("--checkpoint-dir", required=True, type=Path, help="正式 checkpoint 根目录。")
    parser.add_argument("--train-run-dir", required=True, type=Path, help="正式训练 cloud run 目录。")
    parser.add_argument("--dev-run-dir", required=True, type=Path, help="dev eval cloud run 目录。")
    parser.add_argument(
        "--vllm-freeze",
        default=Path("evaluation/stage9/artifacts/cloud_runs/vllm_environment_freeze.txt"),
        type=Path,
        help="vLLM 环境冻结清单。",
    )
    parser.add_argument(
        "--source-training-config",
        default=Path("evaluation/stage9/configs/planner_sft_qwen3_5_4b_lora.json"),
        type=Path,
        help="训练时使用的源配置。",
    )
    parser.add_argument(
        "--dev-cases",
        default=Path("evaluation/stage8/cases/planner_cases.jsonl"),
        type=Path,
        help="dev eval 使用的 case 数据。",
    )
    parser.add_argument(
        "--dev-snapshot",
        default=Path("evaluation/stage8/snapshots/environment_snapshot.json"),
        type=Path,
        help="dev eval 使用的环境快照。",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("/root/autodl-tmp/stage9_backups"),
        type=Path,
        help="归档输出目录；AutoDL 默认放数据盘便于下载。",
    )
    parser.add_argument("--label", default="sft-v1", help="归档标签。")
    parser.add_argument("--overwrite", action="store_true", help="显式覆盖同名归档。")
    args = parser.parse_args(argv)

    result = freeze_sft_artifacts(
        project_root=PROJECT_ROOT,
        checkpoint_dir=args.checkpoint_dir,
        train_run_dir=args.train_run_dir,
        dev_run_dir=args.dev_run_dir,
        vllm_freeze=args.vllm_freeze,
        source_training_config=args.source_training_config,
        dev_cases=args.dev_cases,
        dev_snapshot=args.dev_snapshot,
        output_dir=args.output_dir,
        label=args.label,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
