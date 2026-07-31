import json
import subprocess
from pathlib import Path

from scripts.cloud_sft.collect_cloud_run_report import REPORT_VERSION, build_report
from scripts.cloud_sft.freeze_sft_artifacts import (
    EXPANDED_DEV_RUN_FILES,
    FREEZE_VERSION,
    _dev_run_layout,
    freeze_sft_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DEPLOY_FILES = [
    Path("deploy/cloud_sft/README.md"),
    Path("deploy/cloud_sft/AUTODL_SFT_GUIDE.md"),
    Path("deploy/cloud_sft/env.example"),
    Path("deploy/cloud_sft/requirements-training.txt"),
    Path("deploy/cloud_sft/requirements-training.lock"),
    Path("deploy/cloud_sft/bootstrap_gpu_server.sh"),
    Path("deploy/cloud_sft/run_sft_smoke.sh"),
    Path("deploy/cloud_sft/run_sft_train.sh"),
    Path("deploy/cloud_sft/run_planner_server.sh"),
    Path("deploy/cloud_sft/run_runtime_preflight.sh"),
    Path("deploy/cloud_sft/run_planner_action_probe.sh"),
    Path("deploy/cloud_sft/run_provider_probe.sh"),
    Path("deploy/cloud_sft/run_gpu_acceptance_gate.sh"),
    Path("deploy/cloud_sft/run_dev_eval.sh"),
    Path("deploy/cloud_sft/run_expanded_dev_gate.sh"),
    Path("deploy/cloud_sft/run_sft_v1_corrected_replay_eval.sh"),
]

REQUIRED_SHELL_SCRIPTS = [
    Path("deploy/cloud_sft/bootstrap_gpu_server.sh"),
    Path("deploy/cloud_sft/run_sft_smoke.sh"),
    Path("deploy/cloud_sft/run_sft_train.sh"),
    Path("deploy/cloud_sft/run_planner_server.sh"),
    Path("deploy/cloud_sft/run_runtime_preflight.sh"),
    Path("deploy/cloud_sft/run_planner_action_probe.sh"),
    Path("deploy/cloud_sft/run_provider_probe.sh"),
    Path("deploy/cloud_sft/run_gpu_acceptance_gate.sh"),
    Path("deploy/cloud_sft/run_dev_eval.sh"),
    Path("deploy/cloud_sft/run_expanded_dev_gate.sh"),
    Path("deploy/cloud_sft/run_sft_v1_corrected_replay_eval.sh"),
]


def test_cloud_sft_package_files_exist():
    for path in REQUIRED_DEPLOY_FILES:
        assert (PROJECT_ROOT / path).exists(), f"缺少云端 SFT（监督微调）文件：{path}"
    assert (PROJECT_ROOT / "scripts/cloud_sft/collect_cloud_run_report.py").exists()
    assert (PROJECT_ROOT / "scripts/cloud_sft/freeze_sft_artifacts.py").exists()
    assert (PROJECT_ROOT / "scripts/cloud_sft/preflight_cloud_runtime.py").exists()
    assert (PROJECT_ROOT / "scripts/cloud_sft/probe_planner_actions.py").exists()


def test_cloud_sft_shell_scripts_have_valid_syntax():
    for path in REQUIRED_SHELL_SCRIPTS:
        subprocess.run(["bash", "-n", str(PROJECT_ROOT / path)], check=True)


def test_cloud_sft_env_example_has_no_real_secret():
    text = (PROJECT_ROOT / "deploy/cloud_sft/env.example").read_text(encoding="utf-8")

    assert "PLANNER_API_KEY=" in text
    assert "sk-" not in text
    assert "AKIA" not in text
    assert "BEGIN PRIVATE KEY" not in text
    assert "真实 api_key" in text


def test_cloud_sft_scripts_keep_bootstrap_and_training_separate():
    bootstrap = (PROJECT_ROOT / "deploy/cloud_sft/bootstrap_gpu_server.sh").read_text(encoding="utf-8")
    smoke = (PROJECT_ROOT / "deploy/cloud_sft/run_sft_smoke.sh").read_text(encoding="utf-8")
    train = (PROJECT_ROOT / "deploy/cloud_sft/run_sft_train.sh").read_text(encoding="utf-8")
    planner_server = (PROJECT_ROOT / "deploy/cloud_sft/run_planner_server.sh").read_text(encoding="utf-8")
    gpu_gate = (PROJECT_ROOT / "deploy/cloud_sft/run_gpu_acceptance_gate.sh").read_text(encoding="utf-8")
    runtime_preflight = (
        PROJECT_ROOT / "deploy/cloud_sft/run_runtime_preflight.sh"
    ).read_text(encoding="utf-8")

    assert "sft_train.py" not in bootstrap
    assert "STAGE9_SFT_SMOKE_MAX_SAMPLES" in smoke
    assert "STAGE9_SFT_SMOKE_MAX_STEPS" in smoke
    assert "evaluation/stage9/model_planner/sft_train.py" in train
    assert "deploy/planner_model_server/run_vllm_planner_server.sh" in planner_server
    assert "run_runtime_preflight.sh" in planner_server
    assert "run_planner_server.sh" in gpu_gate
    assert "run_planner_action_probe.sh" in gpu_gate
    assert "kill -TERM" in gpu_gate
    assert "--expected-sft-python" in runtime_preflight
    assert "--sft-requirements-lock" in runtime_preflight
    assert "sys.executable（实际解释器）" in runtime_preflight
    assert 'uv pip freeze --python "$PYTHON_BIN"' in runtime_preflight
    assert "sft_environment_freeze.txt" in runtime_preflight


def test_cloud_sft_uses_an_isolated_training_environment():
    requirements = (PROJECT_ROOT / "deploy/cloud_sft/requirements-training.txt").read_text(encoding="utf-8")
    requirements_lock = (PROJECT_ROOT / "deploy/cloud_sft/requirements-training.lock").read_text(encoding="utf-8")
    requirement_lines = {
        line.strip()
        for line in requirements.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    bootstrap = (PROJECT_ROOT / "deploy/cloud_sft/bootstrap_gpu_server.sh").read_text(encoding="utf-8")
    env_example = (PROJECT_ROOT / "deploy/cloud_sft/env.example").read_text(encoding="utf-8")

    assert "transformers==5.9.0" in requirement_lines
    assert "loguru==0.7.3" in requirement_lines
    assert "transformers==5.9.0" in requirements_lock
    assert "loguru==0.7.3" in requirements_lock
    assert all(not line.startswith("magic-pdf") for line in requirement_lines)
    assert "uv venv" in bootstrap
    assert "uv pip install" in bootstrap
    assert "uv sync" not in bootstrap
    assert "REQUIRE_CUDA=0：当前只做无卡环境准备" in bootstrap
    assert "SFT_VENV_PATH=" in env_example
    assert "SFT_REQUIREMENTS_LOCK=deploy/cloud_sft/requirements-training.lock" in env_example
    assert "UV_SYNC_ARGS=" not in env_example
    assert "TRANSFORMERS_CACHE=" not in env_example
    assert "OMP_NUM_THREADS=1" in env_example
    assert "HF_HUB_OFFLINE=1" in env_example
    assert "TRANSFORMERS_OFFLINE=1" in env_example
    assert "REQUIRE_GPU_PREFLIGHT=1" in env_example
    assert "VLLM_ENV_FREEZE=" in env_example
    assert "EXPANDED_DEV_MAX_CASES" not in env_example
    assert (
        "EXPANDED_DEV_SNAPSHOT="
        "evaluation/stage9/artifacts/heldout_route_test/environment_snapshot.json"
    ) in env_example


def test_collect_cloud_run_report_writes_auditable_json(tmp_path):
    output = tmp_path / "cloud_run_report.json"

    report = build_report(
        training_config=Path("evaluation/stage9/configs/planner_sft_qwen3_5_4b_lora.json"),
        output=output,
        commands=["uv run pytest tests/test_cloud_sft_package.py"],
        notes="unit test（单元测试）",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert report["report_version"] == REPORT_VERSION
    assert payload["report_version"] == REPORT_VERSION
    assert payload["code_version"]["git_short_revision"]
    assert payload["training_config"]["snapshot_id"] == "stage85-env-20260721-v2"
    assert payload["training_config"]["tuning_method"] == "lora"
    assert payload["model_profile"]["profile_id"] == "qwen3_5_4b"
    assert payload["train_manifest"]["sample_count"] == 155
    assert payload["reward_profile"]["reward_version"] == "reward-v1.1"
    assert payload["files"]["training_config"]["sha256"]
    assert payload["commands"] == ["uv run pytest tests/test_cloud_sft_package.py"]


def test_collect_cloud_run_report_binds_expanded_dev_admission(tmp_path):
    output = tmp_path / "cloud_run_report.json"
    admission_output = tmp_path / "sft_9_4_admission_decision.json"
    admission_report = tmp_path / "admission.md"
    _write_json(
        admission_output,
        {
            "admission_version": "stage9-sft-expanded-dev-admission-v1",
            "eval_run_id": "eval-1",
            "snapshot_id": "snapshot-1",
            "reward_version": "reward-v1.1",
            "heldout_inference_result_count": 0,
            "checkpoint": {"run_id": "checkpoint-1"},
            "summary": {
                "decision": "allow_stage9_4",
                "eligible_for_stage9_4": True,
                "case_count": 25,
                "route_macro_accuracy": 0.84,
                "failed_case_ids": [],
            },
        },
    )
    _write_text(admission_report, "# admission\n")

    build_report(
        training_config=Path(
            "evaluation/stage9/configs/planner_sft_qwen3_5_4b_lora.json"
        ),
        output=output,
        admission_decision_output=admission_output,
        admission_report=admission_report,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["admission_decision"]["decision"] == "allow_stage9_4"
    assert payload["admission_decision"]["eligible_for_stage9_4"] is True
    assert payload["admission_decision"]["heldout_inference_result_count"] == 0
    assert payload["files"]["admission_decision_output"]["sha256"]
    assert payload["files"]["admission_report"]["sha256"]


def test_freeze_sft_artifacts_recognizes_expanded_dev_run(tmp_path):
    for relative_path in EXPANDED_DEV_RUN_FILES:
        _write_text(tmp_path / relative_path, "artifact\n")

    kind, eval_path, required = _dev_run_layout(tmp_path)

    assert kind == "stage9_3_16_expanded_dev_gate"
    assert eval_path == Path("sft_expanded_dev_eval.json")
    assert required == EXPANDED_DEV_RUN_FILES


def test_freeze_sft_artifacts_writes_verified_archive(tmp_path):
    project_root = tmp_path / "project"
    checkpoint_dir = project_root / "evaluation/stage9/artifacts/sft/checkpoints/run-1"
    train_run_dir = project_root / "evaluation/stage9/artifacts/cloud_runs/sft_train_1"
    dev_run_dir = project_root / "evaluation/stage9/artifacts/cloud_runs/dev_eval_1"
    vllm_freeze = project_root / "evaluation/stage9/artifacts/cloud_runs/vllm_environment_freeze.txt"
    source_training_config = project_root / "evaluation/stage9/configs/planner_sft.json"
    dev_cases = project_root / "evaluation/stage8/cases/planner_cases.jsonl"
    dev_snapshot = project_root / "evaluation/stage8/snapshots/environment_snapshot.json"
    train_data = project_root / "evaluation/stage9/artifacts/sft/train.jsonl"
    train_manifest = project_root / "evaluation/stage9/artifacts/sft/train_manifest.json"
    reward_profile = project_root / "evaluation/stage9/configs/reward.json"
    model_profile = project_root / "configs/planner_model_profiles/model.json"

    checkpoint_manifest = {
        "run_id": "run-1",
        "base_model_id": "Qwen/Qwen3.5-4B",
        "model_profile_id": "model",
        "snapshot_id": "snapshot-1",
        "reward_profile": "evaluation/stage9/configs/reward.json",
        "sample_count": 155,
        "code_version": {"git_revision": "abc"},
    }
    training_config = {
        "train_data": "evaluation/stage9/artifacts/sft/train.jsonl",
        "train_manifest": "evaluation/stage9/artifacts/sft/train_manifest.json",
        "reward_profile": "evaluation/stage9/configs/reward.json",
        "model_profile_path": "configs/planner_model_profiles/model.json",
    }
    cloud_report = {"checkpoint_manifest": {"run_id": "run-1"}}

    _write_json(checkpoint_dir / "checkpoint_manifest.json", checkpoint_manifest)
    _write_json(checkpoint_dir / "train_metrics.json", {"train_loss": 0.28})
    _write_json(checkpoint_dir / "training_config.json", training_config)
    _write_json(checkpoint_dir / "model/adapter/adapter_config.json", {"r": 16})
    _write_text(checkpoint_dir / "model/adapter/adapter_model.safetensors", "weights")
    _write_json(train_run_dir / "cloud_run_report.json", cloud_report)
    _write_json(dev_run_dir / "cloud_run_report.json", cloud_report)
    _write_json(
        dev_run_dir / "sft_eval_dev.json",
        {"planner_summaries": [{"config": {"checkpoint": str(checkpoint_dir)}}]},
    )
    _write_text(dev_run_dir / "dev_eval.log", "completed")
    _write_text(dev_run_dir / "command.txt", "run dev eval")
    _write_text(vllm_freeze, "vllm==0.25.1")
    _write_json(source_training_config, training_config)
    _write_text(dev_cases, "{}\n")
    _write_json(dev_snapshot, {"snapshot_id": "snapshot-1"})
    _write_text(train_data, "{}\n")
    _write_json(train_manifest, {"sample_count": 155})
    _write_json(reward_profile, {"reward_version": "reward-v1.1"})
    _write_json(model_profile, {"profile_id": "model"})

    result = freeze_sft_artifacts(
        project_root=project_root,
        checkpoint_dir=checkpoint_dir.relative_to(project_root),
        train_run_dir=train_run_dir.relative_to(project_root),
        dev_run_dir=dev_run_dir.relative_to(project_root),
        vllm_freeze=vllm_freeze.relative_to(project_root),
        source_training_config=source_training_config.relative_to(project_root),
        dev_cases=dev_cases.relative_to(project_root),
        dev_snapshot=dev_snapshot.relative_to(project_root),
        output_dir=tmp_path / "backups",
    )

    archive = Path(result["archive"])
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert result["ok"] is True
    assert result["freeze_version"] == FREEZE_VERSION
    assert result["run_id"] == "run-1"
    assert archive.is_file()
    assert Path(result["archive_sha256_file"]).is_file()
    assert manifest["sample_count"] == 155
    assert manifest["file_count"] == result["file_count"]
    assert any(record["path"].endswith("adapter_model.safetensors") for record in manifest["files"])


def _write_json(path: Path, payload: dict):
    _write_text(path, json.dumps(payload, ensure_ascii=False) + "\n")


def _write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
