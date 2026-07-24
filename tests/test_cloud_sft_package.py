import json
import subprocess
from pathlib import Path

from scripts.cloud_sft.collect_cloud_run_report import REPORT_VERSION, build_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DEPLOY_FILES = [
    Path("deploy/cloud_sft/README.md"),
    Path("deploy/cloud_sft/env.example"),
    Path("deploy/cloud_sft/bootstrap_gpu_server.sh"),
    Path("deploy/cloud_sft/run_sft_smoke.sh"),
    Path("deploy/cloud_sft/run_sft_train.sh"),
    Path("deploy/cloud_sft/run_planner_server.sh"),
    Path("deploy/cloud_sft/run_dev_eval.sh"),
]

REQUIRED_SHELL_SCRIPTS = [
    Path("deploy/cloud_sft/bootstrap_gpu_server.sh"),
    Path("deploy/cloud_sft/run_sft_smoke.sh"),
    Path("deploy/cloud_sft/run_sft_train.sh"),
    Path("deploy/cloud_sft/run_planner_server.sh"),
    Path("deploy/cloud_sft/run_dev_eval.sh"),
]


def test_cloud_sft_package_files_exist():
    for path in REQUIRED_DEPLOY_FILES:
        assert (PROJECT_ROOT / path).exists(), f"缺少云端 SFT（监督微调）文件：{path}"
    assert (PROJECT_ROOT / "scripts/cloud_sft/collect_cloud_run_report.py").exists()


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

    assert "sft_train.py" not in bootstrap
    assert "STAGE9_SFT_SMOKE_MAX_SAMPLES" in smoke
    assert "STAGE9_SFT_SMOKE_MAX_STEPS" in smoke
    assert "evaluation/stage9/model_planner/sft_train.py" in train
    assert "deploy/planner_model_server/run_vllm_planner_server.sh" in planner_server


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
