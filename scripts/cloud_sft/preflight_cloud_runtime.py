"""阶段 9 云端 SFT/vLLM runtime（运行时）结构化前置检查。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT_VERSION = "stage9-cloud-runtime-preflight-v1"
TRUTHY_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class PreflightCheck:
    """
    单项 preflight（前置检查）结果。

    layer（层级）用于快速区分 environment（环境变量）、storage（磁盘）、artifact（产物）、
    dependency（依赖）、model_cache（模型缓存）、cuda（显卡）和 network（端口）故障。
    status 只允许 passed/failed/skipped，避免用自然语言猜测是否通过。
    """

    name: str
    layer: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def build_preflight_report(
        *,
        project_root: Path,
        mode: str,
        checkpoint_dir: Path,
        adapter_path: Path,
        model_path: str,
        vllm_python: Path,
        vllm_freeze: Path,
        expected_vllm_version: str,
        expected_torch_backend: str,
        host: str,
        port: int,
        system_path: Path,
        data_path: Path,
        min_system_free_gb: float,
        min_data_free_gb: float,
        require_port_free: bool = True,
        environ: dict[str, str] | None = None,
        runtime_probe: Callable[[Path, dict[str, str]], dict[str, Any]] | None = None,
        model_cache_probe: Callable[[Path, str, dict[str, str]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """执行所有只读门禁并返回结构化报告；本函数不安装依赖、不下载模型、不启动服务。"""

    if mode not in {"no-card", "gpu"}:
        raise ValueError("mode 必须是 no-card 或 gpu")
    active_env = dict(os.environ if environ is None else environ)
    resolved_project_root = project_root.resolve()
    checkpoint_abs = _resolve_project_path(resolved_project_root, checkpoint_dir)
    adapter_abs = _resolve_project_path(resolved_project_root, adapter_path)
    vllm_freeze_abs = _resolve_project_path(resolved_project_root, vllm_freeze)
    runtime_probe = runtime_probe or _probe_vllm_runtime
    model_cache_probe = model_cache_probe or _probe_model_cache

    checks: list[PreflightCheck] = []
    checks.extend(_environment_checks(mode, active_env))
    checks.extend(_storage_checks(
        system_path=system_path,
        data_path=data_path,
        min_system_free_gb=min_system_free_gb,
        min_data_free_gb=min_data_free_gb,
    ))
    artifact_checks, checkpoint_manifest = _artifact_checks(
        project_root=resolved_project_root,
        checkpoint_dir=checkpoint_abs,
        adapter_path=adapter_abs,
        vllm_freeze=vllm_freeze_abs,
    )
    checks.extend(artifact_checks)

    runtime_metadata: dict[str, Any] = {}
    if not vllm_python.is_file():
        checks.append(_failed(
            "vllm_python",
            "dependency",
            "vLLM Python 解释器不存在",
            {"path": str(vllm_python)},
        ))
    else:
        try:
            runtime_metadata = runtime_probe(vllm_python, active_env)
            checks.extend(_runtime_checks(
                mode=mode,
                metadata=runtime_metadata,
                expected_vllm_version=expected_vllm_version,
                expected_torch_backend=expected_torch_backend,
            ))
        except Exception as exc:
            checks.append(_failed(
                "vllm_runtime",
                "dependency",
                "无法读取 vLLM/PyTorch runtime 元数据",
                {"error": str(exc), "python": str(vllm_python)},
            ))

    if vllm_python.is_file():
        try:
            model_cache = model_cache_probe(vllm_python, model_path, active_env)
            checks.append(_passed(
                "model_cache",
                "model_cache",
                "基础模型可以在 local_files_only（仅本地文件）模式解析",
                model_cache,
            ))
        except Exception as exc:
            checks.append(_failed(
                "model_cache",
                "model_cache",
                "基础模型离线缓存不可用；禁止在 GPU 计费模式临时下载",
                {"model_path": model_path, "error": str(exc)},
            ))

    checks.append(_port_check(host=host, port=port, require_port_free=require_port_free))
    if mode == "gpu":
        checks.append(_nvidia_smi_check())
    else:
        checks.append(PreflightCheck(
            name="nvidia_smi",
            layer="cuda",
            status="skipped",
            message="no-card 模式不要求 nvidia-smi",
        ))

    failed_checks = [check for check in checks if check.status == "failed"]
    report = {
        "preflight_version": PREFLIGHT_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": mode,
        "ok": not failed_checks,
        "failed_check_count": len(failed_checks),
        "check_count": len(checks),
        "runtime_identity": {
            "hostname": socket.gethostname(),
            "boot_id": _boot_id(),
            "git_revision": _git_revision(resolved_project_root),
            "project_root": str(resolved_project_root),
            "checkpoint_dir": _relative_or_absolute(resolved_project_root, checkpoint_abs),
            "adapter_path": _relative_or_absolute(resolved_project_root, adapter_abs),
            "checkpoint_run_id": checkpoint_manifest.get("run_id"),
            "model_path": model_path,
            "vllm_python": str(vllm_python),
            "vllm_freeze": _relative_or_absolute(resolved_project_root, vllm_freeze_abs),
            "host": host,
            "port": port,
        },
        "runtime_metadata": runtime_metadata,
        "checks": [asdict(check) for check in checks],
    }
    return report


def write_report(path: Path, report: dict[str, Any]) -> None:
    """写入 preflight JSON（前置检查报告），即使失败也保留结构化原因。"""

    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _environment_checks(mode: str, environ: dict[str, str]) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    omp_raw = str(environ.get("OMP_NUM_THREADS") or "").strip()
    try:
        omp_threads = int(omp_raw)
    except ValueError:
        omp_threads = 0
    if omp_threads > 0:
        checks.append(_passed(
            "omp_num_threads",
            "environment",
            "OMP_NUM_THREADS 是正整数",
            {"value": omp_threads},
        ))
    else:
        checks.append(_failed(
            "omp_num_threads",
            "environment",
            "OMP_NUM_THREADS 必须是正整数，不能为 0、空值或非法字符串",
            {"value": omp_raw or "<unset>"},
        ))

    expected_cuda_flag = "1" if mode == "gpu" else "0"
    actual_cuda_flag = str(environ.get("REQUIRE_CUDA") or "").strip()
    if actual_cuda_flag == expected_cuda_flag:
        checks.append(_passed(
            "require_cuda_mode",
            "environment",
            "REQUIRE_CUDA 与 preflight 模式一致",
            {"value": actual_cuda_flag, "mode": mode},
        ))
    else:
        checks.append(_failed(
            "require_cuda_mode",
            "environment",
            "REQUIRE_CUDA 与 preflight 模式不一致",
            {"value": actual_cuda_flag or "<unset>", "expected": expected_cuda_flag, "mode": mode},
        ))

    for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        raw_value = str(environ.get(variable) or "").strip().lower()
        if raw_value in TRUTHY_VALUES:
            checks.append(_passed(
                variable.lower(),
                "environment",
                f"{variable}=1，运行时禁止静默联网下载",
            ))
        else:
            checks.append(_failed(
                variable.lower(),
                "environment",
                f"{variable} 未开启；GPU 运行可能再次联网下载",
                {"value": raw_value or "<unset>"},
            ))
    return checks


def _storage_checks(
        *,
        system_path: Path,
        data_path: Path,
        min_system_free_gb: float,
        min_data_free_gb: float,
) -> list[PreflightCheck]:
    return [
        _disk_check("system_disk", system_path, min_system_free_gb),
        _disk_check("data_disk", data_path, min_data_free_gb),
    ]


def _disk_check(name: str, path: Path, minimum_free_gb: float) -> PreflightCheck:
    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        return _failed(name, "storage", "无法读取磁盘空间", {"path": str(path), "error": str(exc)})
    divisor = 1024 ** 3
    details = {
        "path": str(path),
        "free_bytes": usage.free,
        "free_gb": round(usage.free / divisor, 3),
        "minimum_free_gb": minimum_free_gb,
    }
    if usage.free >= minimum_free_gb * divisor:
        return _passed(name, "storage", "磁盘剩余空间达到门槛", details)
    return _failed(name, "storage", "磁盘剩余空间低于门槛", details)


def _artifact_checks(
        *,
        project_root: Path,
        checkpoint_dir: Path,
        adapter_path: Path,
        vllm_freeze: Path,
) -> tuple[list[PreflightCheck], dict[str, Any]]:
    checks: list[PreflightCheck] = []
    required_files = (
        checkpoint_dir / "checkpoint_manifest.json",
        checkpoint_dir / "train_metrics.json",
        checkpoint_dir / "training_config.json",
        adapter_path / "adapter_config.json",
        adapter_path / "adapter_model.safetensors",
        vllm_freeze,
    )
    missing = [
        _relative_or_absolute(project_root, path)
        for path in required_files
        if not path.is_file() or path.stat().st_size <= 0
    ]
    if missing:
        checks.append(_failed(
            "required_artifacts",
            "artifact",
            "checkpoint、adapter 或环境冻结清单缺失",
            {"missing": missing},
        ))
        return checks, {}
    checks.append(_passed(
        "required_artifacts",
        "artifact",
        "checkpoint、adapter 与环境冻结清单存在且非空",
        {"file_count": len(required_files)},
    ))

    try:
        manifest = json.loads((checkpoint_dir / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    except Exception as exc:
        checks.append(_failed(
            "checkpoint_identity",
            "artifact",
            "checkpoint manifest 无法读取",
            {"error": str(exc)},
        ))
        return checks, {}

    manifest_run_id = str(manifest.get("run_id") or "").strip()
    manifest_adapter = str(manifest.get("adapter_path") or "").strip()
    expected_adapter = _relative_or_absolute(project_root, adapter_path)
    identity_ok = (
        bool(manifest_run_id)
        and checkpoint_dir.name == manifest_run_id
        and manifest_adapter == expected_adapter
    )
    details = {
        "directory_run_id": checkpoint_dir.name,
        "manifest_run_id": manifest_run_id,
        "manifest_adapter_path": manifest_adapter,
        "expected_adapter_path": expected_adapter,
        "sample_count": manifest.get("sample_count"),
    }
    if identity_ok:
        checks.append(_passed(
            "checkpoint_identity",
            "artifact",
            "checkpoint 目录、run_id 和 adapter_path 一致",
            details,
        ))
    else:
        checks.append(_failed(
            "checkpoint_identity",
            "artifact",
            "checkpoint 目录、run_id 或 adapter_path 不一致",
            details,
        ))
    return checks, manifest


def _runtime_checks(
        *,
        mode: str,
        metadata: dict[str, Any],
        expected_vllm_version: str,
        expected_torch_backend: str,
) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    actual_vllm = str(metadata.get("vllm_version") or "")
    if actual_vllm == expected_vllm_version:
        checks.append(_passed(
            "vllm_version",
            "dependency",
            "vLLM 版本与冻结配置一致",
            {"actual": actual_vllm, "expected": expected_vllm_version},
        ))
    else:
        checks.append(_failed(
            "vllm_version",
            "dependency",
            "vLLM 版本与冻结配置不一致",
            {"actual": actual_vllm or "<missing>", "expected": expected_vllm_version},
        ))

    expected_cuda = _backend_cuda_version(expected_torch_backend)
    actual_cuda = str(metadata.get("torch_cuda") or "")
    if expected_cuda and actual_cuda == expected_cuda:
        checks.append(_passed(
            "torch_cuda_backend",
            "dependency",
            "PyTorch CUDA 版本与 VLLM_TORCH_BACKEND 一致",
            {"actual": actual_cuda, "expected": expected_cuda},
        ))
    else:
        checks.append(_failed(
            "torch_cuda_backend",
            "dependency",
            "PyTorch CUDA 版本与 VLLM_TORCH_BACKEND 不一致",
            {"actual": actual_cuda or "<missing>", "expected": expected_cuda or "<unknown>"},
        ))

    cuda_available = bool(metadata.get("cuda_available"))
    device_count = int(metadata.get("device_count") or 0)
    if mode == "gpu":
        if cuda_available and device_count > 0:
            checks.append(_passed(
                "torch_cuda_available",
                "cuda",
                "vLLM 环境可访问 CUDA GPU",
                {
                    "device_count": device_count,
                    "device_name": metadata.get("device_name"),
                },
            ))
        else:
            checks.append(_failed(
                "torch_cuda_available",
                "cuda",
                "GPU 模式下 vLLM 环境无法访问 CUDA",
                {"cuda_available": cuda_available, "device_count": device_count},
            ))
    else:
        checks.append(_passed(
            "torch_cuda_available",
            "cuda",
            "no-card 模式只校验 CUDA 构建版本，不要求实际 GPU",
            {"cuda_available": cuda_available, "device_count": device_count},
        ))
    return checks


def _probe_vllm_runtime(vllm_python: Path, environ: dict[str, str]) -> dict[str, Any]:
    code = """
import importlib.metadata
import json
import platform
import torch

print(json.dumps({
    "python_version": platform.python_version(),
    "vllm_version": importlib.metadata.version("vllm"),
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}, ensure_ascii=False))
"""
    completed = subprocess.run(
        [str(vllm_python), "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environ,
        timeout=60,
    )
    return _last_json_line(completed.stdout)


def _probe_model_cache(vllm_python: Path, model_path: str, environ: dict[str, str]) -> dict[str, Any]:
    local_path = Path(model_path)
    if local_path.exists():
        return {"model_path": model_path, "resolved_path": str(local_path.resolve()), "source": "local_path"}
    code = """
import json
import sys
from huggingface_hub import snapshot_download

resolved = snapshot_download(repo_id=sys.argv[1], local_files_only=True)
print(json.dumps({"model_path": sys.argv[1], "resolved_path": resolved, "source": "huggingface_cache"}))
"""
    completed = subprocess.run(
        [str(vllm_python), "-c", code, model_path],
        check=True,
        capture_output=True,
        text=True,
        env=environ,
        timeout=60,
    )
    return _last_json_line(completed.stdout)


def _port_check(*, host: str, port: int, require_port_free: bool) -> PreflightCheck:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    try:
        in_use = sock.connect_ex((host, port)) == 0
    finally:
        sock.close()
    details = {"host": host, "port": port, "in_use": in_use}
    if require_port_free and in_use:
        return _failed("planner_port", "network", "Planner 端口已被占用", details)
    if not require_port_free and not in_use:
        return _failed("planner_port", "network", "Planner 服务端口尚未监听", details)
    message = "Planner 端口空闲，可以启动服务" if require_port_free else "Planner 服务端口正在监听"
    return _passed("planner_port", "network", message, details)


def _nvidia_smi_check() -> PreflightCheck:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        return _failed(
            "nvidia_smi",
            "cuda",
            "GPU 模式下 nvidia-smi 不可用",
            {"error": str(exc)},
        )
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not rows:
        return _failed("nvidia_smi", "cuda", "nvidia-smi 未返回 GPU", {})
    return _passed(
        "nvidia_smi",
        "cuda",
        "nvidia-smi 返回可用 GPU",
        {"gpus": rows},
    )


def _backend_cuda_version(backend: str) -> str:
    normalized = str(backend or "").strip().lower()
    if not normalized.startswith("cu"):
        return ""
    digits = normalized[2:]
    if len(digits) < 2 or not digits.isdigit():
        return ""
    return f"{int(digits[:-1])}.{int(digits[-1])}"


def _last_json_line(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            break
        return payload
    raise ValueError(f"子进程未输出 JSON object：{output[-500:]}")


def _resolve_project_path(project_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _relative_or_absolute(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return str(path.resolve())


def _git_revision(project_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _boot_id() -> str:
    path = Path("/proc/sys/kernel/random/boot_id")
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return "unavailable"


def _passed(name: str, layer: str, message: str, details: dict[str, Any] | None = None) -> PreflightCheck:
    return PreflightCheck(name=name, layer=layer, status="passed", message=message, details=details or {})


def _failed(name: str, layer: str, message: str, details: dict[str, Any] | None = None) -> PreflightCheck:
    return PreflightCheck(name=name, layer=layer, status="failed", message=message, details=details or {})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行阶段 9 云端 SFT/vLLM 前置检查。")
    parser.add_argument("--mode", choices=["no-card", "gpu"], required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vllm-python", type=Path, required=True)
    parser.add_argument("--vllm-freeze", type=Path, required=True)
    parser.add_argument("--expected-vllm-version", required=True)
    parser.add_argument("--expected-torch-backend", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8019)
    parser.add_argument("--system-path", type=Path, default=Path("/"))
    parser.add_argument("--data-path", type=Path, default=Path("/root/autodl-tmp"))
    parser.add_argument("--min-system-free-gb", type=float, default=2.0)
    parser.add_argument("--min-data-free-gb", type=float, default=5.0)
    parser.add_argument("--allow-port-in-use", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_preflight_report(
        project_root=PROJECT_ROOT,
        mode=args.mode,
        checkpoint_dir=args.checkpoint_dir,
        adapter_path=args.adapter_path,
        model_path=args.model_path,
        vllm_python=args.vllm_python,
        vllm_freeze=args.vllm_freeze,
        expected_vllm_version=args.expected_vllm_version,
        expected_torch_backend=args.expected_torch_backend,
        host=args.host,
        port=args.port,
        system_path=args.system_path,
        data_path=args.data_path,
        min_system_free_gb=args.min_system_free_gb,
        min_data_free_gb=args.min_data_free_gb,
        require_port_free=not args.allow_port_in_use,
    )
    write_report(args.output, report)
    print(json.dumps({
        "ok": report["ok"],
        "mode": report["mode"],
        "failed_check_count": report["failed_check_count"],
        "output": str(args.output),
        "preflight_version": report["preflight_version"],
    }, ensure_ascii=False, sort_keys=True))
    if not report["ok"]:
        for check in report["checks"]:
            if check["status"] == "failed":
                print(json.dumps(check, ensure_ascii=False, sort_keys=True), file=sys.stderr)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
