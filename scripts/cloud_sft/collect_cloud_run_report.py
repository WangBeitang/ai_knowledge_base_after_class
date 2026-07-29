"""收集 Cloud SFT（云端监督微调）运行报告。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_VERSION = "stage9-cloud-sft-run-report-v2"


def build_report(
        *,
        training_config: Path,
        output: Path,
        train_manifest: Path | None = None,
        reward_profile: Path | None = None,
        model_profile: Path | None = None,
        checkpoint_dir: Path | None = None,
        dev_eval_output: Path | None = None,
        admission_decision_output: Path | None = None,
        admission_report: Path | None = None,
        commands: list[str] | None = None,
        notes: str = "",
) -> dict[str, Any]:
    """构造 cloud run report（云端运行报告），只记录可审计元数据，不记录密钥。"""

    training_payload = _read_required_json(training_config)
    inferred_model_profile = model_profile or _infer_model_profile_path(training_payload)
    inferred_train_manifest = train_manifest or _optional_path(training_payload.get("train_manifest"))
    inferred_reward_profile = reward_profile or _optional_path(training_payload.get("reward_profile"))
    checkpoint_manifest = _checkpoint_manifest_path(checkpoint_dir)

    files = {
        "training_config": _file_record(training_config),
        "model_profile": _file_record(inferred_model_profile) if inferred_model_profile else None,
        "train_manifest": _file_record(inferred_train_manifest) if inferred_train_manifest else None,
        "reward_profile": _file_record(inferred_reward_profile) if inferred_reward_profile else None,
        "checkpoint_manifest": _file_record(checkpoint_manifest) if checkpoint_manifest else None,
        "dev_eval_output": _file_record(dev_eval_output) if dev_eval_output else None,
        "admission_decision_output": (
            _file_record(admission_decision_output)
            if admission_decision_output
            else None
        ),
        "admission_report": _file_record(admission_report) if admission_report else None,
    }

    train_manifest_payload = _read_json(inferred_train_manifest) if inferred_train_manifest else None
    reward_profile_payload = _read_json(inferred_reward_profile) if inferred_reward_profile else None
    model_profile_payload = _read_json(inferred_model_profile) if inferred_model_profile else None
    checkpoint_payload = _read_json(checkpoint_manifest) if checkpoint_manifest else None
    dev_eval_payload = _read_json(dev_eval_output) if dev_eval_output else None
    admission_payload = (
        _read_json(admission_decision_output)
        if admission_decision_output
        else None
    )

    report = {
        "report_version": REPORT_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_version": _code_version(),
        "commands": commands or [],
        "notes": notes,
        "training_config": _summarize_training_config(training_payload),
        "model_profile": _summarize_model_profile(model_profile_payload),
        "train_manifest": _summarize_train_manifest(train_manifest_payload),
        "reward_profile": _summarize_reward_profile(reward_profile_payload),
        "checkpoint_manifest": _summarize_checkpoint(checkpoint_payload),
        "dev_eval": _summarize_dev_eval(dev_eval_payload),
        "admission_decision": _summarize_admission(admission_payload),
        "files": {name: record for name, record in files.items() if record is not None},
    }
    _write_json(output, report)
    return report


def _summarize_training_config(payload: dict[str, Any]) -> dict[str, Any]:
    """抽取训练配置中对复现实验最关键的字段。"""

    keys = (
        "run_name",
        "training_backend",
        "base_model_id",
        "model_profile_id",
        "model_profile_path",
        "train_data",
        "train_manifest",
        "reward_profile",
        "snapshot_id",
        "output_root",
        "max_input_tokens",
        "max_target_tokens",
        "max_train_samples",
        "max_steps",
        "num_epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "seed",
        "tuning_method",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
        "load_in_4bit",
        "bnb_compute_dtype",
        "gradient_checkpointing",
    )
    return {key: payload.get(key) for key in keys if key in payload}


def _summarize_model_profile(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """抽取 model profile（模型配置档案）的模型身份和推理边界。"""

    if payload is None:
        return None
    keys = (
        "profile_id",
        "base_model_id",
        "training_model_id",
        "serving_model_id",
        "role",
        "auto_train_enabled",
        "chat_template",
        "enable_thinking",
        "max_context_tokens",
        "max_target_tokens",
        "recommended_backend",
    )
    return {key: payload.get(key) for key in keys if key in payload}


def _summarize_train_manifest(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """抽取 train manifest（训练清单）的样本、split（数据切分）和来源边界。"""

    if payload is None:
        return None
    keys = (
        "manifest_id",
        "created_at",
        "snapshot_id",
        "reward_version",
        "split",
        "sample_count",
        "case_count",
        "trajectory_count",
        "source_case_count",
        "gold_origins",
        "review_status_counts",
        "action_counts",
        "route_family_counts",
    )
    return {key: payload.get(key) for key in keys if key in payload}


def _summarize_reward_profile(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """抽取 Reward profile（奖励函数配置）的版本和权重。"""

    if payload is None:
        return None
    keys = (
        "profile_name",
        "reward_version",
        "created_at",
        "snapshot_id",
        "case_split",
        "component_weights",
        "weights",
        "thresholds",
    )
    return {key: payload.get(key) for key in keys if key in payload}


def _summarize_checkpoint(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """抽取 checkpoint manifest（检查点清单）的模型、数据和训练结果身份。"""

    if payload is None:
        return None
    keys = (
        "run_id",
        "run_name",
        "policy_version",
        "training_backend",
        "base_model_id",
        "model_profile_id",
        "tuning_method",
        "adapter_id",
        "adapter_path",
        "quantization",
        "train_data",
        "train_manifest",
        "reward_profile",
        "snapshot_id",
        "code_version",
        "created_at",
        "seed",
        "framework_versions",
        "prompt_builder_version",
        "decision_codec_version",
        "sample_count",
        "source_case_count",
        "action_counts",
    )
    return {key: payload.get(key) for key in keys if key in payload}


def _summarize_dev_eval(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """抽取 dev eval（开发集评测）的样本数量、Reward（奖励）和路线分布。"""

    if payload is None:
        return None
    summaries = payload.get("planner_summaries") or []
    first_summary = summaries[0] if summaries else {}
    return {
        "run_id": payload.get("run_id"),
        "runner_version": payload.get("runner_version"),
        "split": payload.get("split"),
        "snapshot_id": payload.get("snapshot_id"),
        "reward_version": payload.get("reward_version"),
        "case_count": payload.get("case_count"),
        "planner_mode": first_summary.get("planner_mode"),
        "reward": first_summary.get("reward"),
        "usage": first_summary.get("usage"),
        "path_counts": (first_summary.get("config") or {}).get("path_counts"),
    }


def _summarize_admission(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """抽取 9.3.16 准入决定；逐 case 证据仍保留在原始 decision JSON。"""

    if payload is None:
        return None
    summary = payload.get("summary") or {}
    checkpoint = payload.get("checkpoint") or {}
    return {
        "admission_version": payload.get("admission_version"),
        "eval_run_id": payload.get("eval_run_id"),
        "checkpoint_run_id": checkpoint.get("run_id"),
        "snapshot_id": payload.get("snapshot_id"),
        "reward_version": payload.get("reward_version"),
        "decision": summary.get("decision"),
        "eligible_for_stage9_4": summary.get("eligible_for_stage9_4"),
        "case_count": summary.get("case_count"),
        "route_macro_accuracy": summary.get("route_macro_accuracy"),
        "failed_case_ids": summary.get("failed_case_ids"),
        "heldout_inference_result_count": payload.get(
            "heldout_inference_result_count"
        ),
    }


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = _resolve_path(path)
    if not resolved.exists():
        return None
    return json.loads(resolved.read_text(encoding="utf-8"))


def _read_required_json(path: Path) -> dict[str, Any]:
    resolved = _resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"必填 JSON 文件不存在：{path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    resolved = _resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _file_record(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = _resolve_path(path)
    record: dict[str, Any] = {
        "path": str(path),
        "exists": resolved.exists(),
    }
    if resolved.exists() and resolved.is_file():
        record["size_bytes"] = resolved.stat().st_size
        record["sha256"] = _sha256(resolved)
    return record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_version() -> dict[str, Any]:
    """返回 code version（代码版本），dirty 为 true 表示工作树存在未提交改动。"""

    revision = _run_git(["rev-parse", "HEAD"])
    short_revision = _run_git(["rev-parse", "--short", "HEAD"])
    status = _run_git(["status", "--short"])
    return {
        "git_revision": revision or "unknown",
        "git_short_revision": short_revision or "unknown",
        "dirty": bool(status),
        "status_short": status.splitlines() if status else [],
    }


def _run_git(args: list[str]) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return ""


def _infer_model_profile_path(training_payload: dict[str, Any] | None) -> Path | None:
    if not training_payload:
        return None
    raw_path = training_payload.get("model_profile_path")
    if raw_path:
        return Path(raw_path)
    profile_id = training_payload.get("model_profile_id")
    if profile_id:
        return Path("configs/planner_model_profiles") / f"{profile_id}.json"
    return None


def _checkpoint_manifest_path(checkpoint_dir: Path | None) -> Path | None:
    if checkpoint_dir is None:
        return None
    return checkpoint_dir / "checkpoint_manifest.json"


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    raw = str(value).strip()
    return Path(raw) if raw else None


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="收集阶段 9 云端 SFT（监督微调）运行报告。")
    parser.add_argument("--output", required=True, type=Path, help="报告输出 JSON 路径。")
    parser.add_argument(
        "--training-config",
        default=Path("evaluation/stage9/configs/planner_sft_qwen3_5_4b_lora.json"),
        type=Path,
        help="SFT（监督微调）训练配置 JSON。",
    )
    parser.add_argument("--train-manifest", type=Path, default=None, help="可选 train manifest（训练清单）路径。")
    parser.add_argument("--reward-profile", type=Path, default=None, help="可选 Reward profile（奖励函数配置）路径。")
    parser.add_argument("--model-profile", type=Path, default=None, help="可选 model profile（模型配置档案）路径。")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="可选 checkpoint（检查点）目录。")
    parser.add_argument("--dev-eval-output", type=Path, default=None, help="可选 dev eval（开发集评测）输出 JSON。")
    parser.add_argument(
        "--admission-decision-output",
        type=Path,
        default=None,
        help="可选 9.3.16 机器可读准入决定 JSON。",
    )
    parser.add_argument(
        "--admission-report",
        type=Path,
        default=None,
        help="可选 9.3.16 人工准入报告 Markdown。",
    )
    parser.add_argument("--command", action="append", default=[], help="本次运行命令，可重复传入。")
    parser.add_argument("--notes", default="", help="人工备注，不要写入密钥。")
    args = parser.parse_args(argv)

    report = build_report(
        training_config=args.training_config,
        output=args.output,
        train_manifest=args.train_manifest,
        reward_profile=args.reward_profile,
        model_profile=args.model_profile,
        checkpoint_dir=args.checkpoint_dir,
        dev_eval_output=args.dev_eval_output,
        admission_decision_output=args.admission_decision_output,
        admission_report=args.admission_report,
        commands=args.command,
        notes=args.notes,
    )
    print(json.dumps({
        "ok": True,
        "output": str(args.output),
        "report_version": report["report_version"],
        "training_config_sha256": report["files"]["training_config"].get("sha256"),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
