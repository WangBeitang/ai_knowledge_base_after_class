import json
from pathlib import Path
from types import SimpleNamespace

from app.rag.query.contracts import PlannerDecision, PlannerReasonCode, QueryAction
from scripts.cloud_sft.preflight_cloud_runtime import build_preflight_report
from scripts.cloud_sft.probe_planner_actions import build_default_probes, run_action_probes


MODEL_ID = "qwen3_5_4b_sft_stage9"
ENDPOINT = "http://127.0.0.1:8019/v1/chat/completions"


def test_no_card_preflight_accepts_frozen_runtime_without_cuda(tmp_path):
    project_root, checkpoint, adapter, freeze, vllm_python = _preflight_files(tmp_path)

    report = build_preflight_report(
        project_root=project_root,
        mode="no-card",
        checkpoint_dir=checkpoint,
        adapter_path=adapter,
        model_path="Qwen/Qwen3.5-4B",
        vllm_python=vllm_python,
        vllm_freeze=freeze,
        expected_vllm_version="0.25.1",
        expected_torch_backend="cu130",
        host="127.0.0.1",
        port=0,
        system_path=tmp_path,
        data_path=tmp_path,
        min_system_free_gb=0,
        min_data_free_gb=0,
        environ=_runtime_environ(require_cuda="0"),
        runtime_probe=lambda _python, _environ: {
            "python_version": "3.12.3",
            "vllm_version": "0.25.1",
            "torch_version": "2.11.0+cu130",
            "torch_cuda": "13.0",
            "cuda_available": False,
            "device_count": 0,
            "device_name": None,
        },
        model_cache_probe=lambda _python, model, _environ: {
            "model_path": model,
            "resolved_path": "/cache/qwen",
            "source": "huggingface_cache",
        },
    )

    assert report["ok"] is True
    assert report["failed_check_count"] == 0
    assert report["runtime_identity"]["checkpoint_run_id"] == "run-1"
    assert {check["status"] for check in report["checks"]} <= {"passed", "skipped"}


def test_preflight_reports_environment_and_checkpoint_identity_failures(tmp_path):
    project_root, checkpoint, adapter, freeze, vllm_python = _preflight_files(
        tmp_path,
        manifest_adapter="wrong/adapter",
    )
    environ = _runtime_environ(require_cuda="1")
    environ.update({
        "OMP_NUM_THREADS": "0",
        "HF_HUB_OFFLINE": "0",
        "TRANSFORMERS_OFFLINE": "",
    })

    report = build_preflight_report(
        project_root=project_root,
        mode="no-card",
        checkpoint_dir=checkpoint,
        adapter_path=adapter,
        model_path="Qwen/Qwen3.5-4B",
        vllm_python=vllm_python,
        vllm_freeze=freeze,
        expected_vllm_version="0.25.1",
        expected_torch_backend="cu130",
        host="127.0.0.1",
        port=0,
        system_path=tmp_path,
        data_path=tmp_path,
        min_system_free_gb=0,
        min_data_free_gb=0,
        environ=environ,
        runtime_probe=lambda _python, _environ: {
            "vllm_version": "0.25.1",
            "torch_cuda": "13.0",
            "cuda_available": False,
            "device_count": 0,
        },
        model_cache_probe=lambda _python, _model, _environ: {"source": "test"},
    )

    failed = {check["name"]: check["layer"] for check in report["checks"] if check["status"] == "failed"}
    assert report["ok"] is False
    assert failed["omp_num_threads"] == "environment"
    assert failed["require_cuda_mode"] == "environment"
    assert failed["hf_hub_offline"] == "environment"
    assert failed["transformers_offline"] == "environment"
    assert failed["checkpoint_identity"] == "artifact"


def test_default_http_probes_cover_six_actions_and_web_disabled_boundary():
    probes = build_default_probes()
    action_probes = [probe for probe in probes if probe.probe_id.startswith("action-")]
    web_disabled = next(probe for probe in probes if probe.probe_id == "policy-web-disabled")

    assert len(probes) == 7
    assert {probe.expected_action for probe in action_probes} == set(QueryAction)
    assert web_disabled.context.web_search_allowed is False
    assert QueryAction.WEB_SEARCH not in web_disabled.context.allowed_actions


def test_http_probe_separates_engineering_gate_from_action_quality():
    report = run_action_probes(
        client=_FakePlannerClient(use_expected_actions=False),
        model_id=MODEL_ID,
        endpoint=ENDPOINT,
        strict_action_match=False,
    )

    assert report["ok"] is True
    assert report["summary"]["protocol_success_count"] == 7
    assert report["summary"]["engineering_ok_count"] == 7
    assert report["summary"]["expected_action_match_count"] < 7
    assert report["summary"]["web_disabled_policy_ok"] is True


def test_http_probe_strict_mode_requires_expected_actions():
    report = run_action_probes(
        client=_FakePlannerClient(use_expected_actions=True),
        model_id=MODEL_ID,
        endpoint=ENDPOINT,
        strict_action_match=True,
    )

    assert report["ok"] is True
    assert report["summary"]["expected_action_match_count"] == 7
    assert report["summary"]["target_action_coverage_ok"] is True


class _FakePlannerClient:
    def __init__(self, *, use_expected_actions: bool) -> None:
        self.probes = build_default_probes()
        self.use_expected_actions = use_expected_actions
        self.index = 0

    def request_decision(self, context):
        probe = self.probes[self.index]
        self.index += 1
        action = probe.expected_action if self.use_expected_actions else QueryAction.LOCAL_SEARCH
        decision = PlannerDecision(
            action=action,
            query=context.current_query,
            reason_code=PlannerReasonCode.SAFE_GUARD_TRIGGERED,
        )
        return SimpleNamespace(
            decision=decision,
            response_model_id=MODEL_ID,
            raw_output=json.dumps(decision.model_dump(mode="json"), ensure_ascii=False),
        )


def _preflight_files(
        tmp_path: Path,
        *,
        manifest_adapter: str = "checkpoints/run-1/model/adapter",
) -> tuple[Path, Path, Path, Path, Path]:
    project_root = tmp_path / "project"
    checkpoint = Path("checkpoints/run-1")
    adapter = checkpoint / "model/adapter"
    freeze = Path("artifacts/vllm_environment_freeze.txt")
    vllm_python = tmp_path / "vllm/bin/python"

    _write_json(
        project_root / checkpoint / "checkpoint_manifest.json",
        {
            "run_id": "run-1",
            "adapter_path": manifest_adapter,
            "sample_count": 155,
        },
    )
    _write_json(project_root / checkpoint / "train_metrics.json", {"train_loss": 0.28})
    _write_json(project_root / checkpoint / "training_config.json", {"run_name": "run-1"})
    _write_json(project_root / adapter / "adapter_config.json", {"r": 16})
    _write_text(project_root / adapter / "adapter_model.safetensors", "weights")
    _write_text(project_root / freeze, "vllm==0.25.1\n")
    _write_text(vllm_python, "# test interpreter placeholder\n")
    return project_root, checkpoint, adapter, freeze, vllm_python


def _runtime_environ(*, require_cuda: str) -> dict[str, str]:
    return {
        "OMP_NUM_THREADS": "1",
        "REQUIRE_CUDA": require_cuda,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def _write_json(path: Path, payload: dict) -> None:
    _write_text(path, json.dumps(payload, ensure_ascii=False) + "\n")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
