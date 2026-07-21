from pathlib import Path

from app.rag.evaluation.baseline_runner import BaselineEvalOutput
from app.rag.evaluation.sft_exporter import SftExportManifest
from evaluation.stage8_5.pipelines.common.paths import stage85_layout
from evaluation.stage8_5.pipelines.sft_seed.generate_stage85_4_report import (
    build_markdown_report,
    load_sft_samples,
    validate_stage85_4_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAYOUT = stage85_layout(PROJECT_ROOT / "evaluation/stage8_5")


def test_stage85_sft_seed_artifacts_pass_training_boundary_validation():
    """实际落盘产物必须满足 20 case、40 决策样本和 train-only 审批边界。"""

    baseline = BaselineEvalOutput.model_validate_json(
        (LAYOUT.sft_intermediate / "reward_v1_1_baseline_train.json").read_text(encoding="utf-8")
    )
    manifest = SftExportManifest.model_validate_json(
        (LAYOUT.final / "sft_curated_seed_manifest.json").read_text(encoding="utf-8")
    )
    samples = load_sft_samples(LAYOUT.final / "sft_curated_seed_train.jsonl")

    stats = validate_stage85_4_artifacts(
        baseline=baseline,
        manifest=manifest,
        samples=samples,
    )

    assert stats["case_count"] == 20
    assert stats["sample_count"] == 40
    assert stats["average_total_reward"] == 0.85
    assert stats["action_counts"] == {"answer": 20, "local_search": 20}
    assert stats["component_scores"]["answer"] == 0.0
    assert "不表示 Gold 答案错误" in build_markdown_report(stats)
