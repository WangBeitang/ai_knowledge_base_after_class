import hashlib
import json
from pathlib import Path

from evaluation.stage8_5.pipelines.common.paths import stage85_layout


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE85_ROOT = PROJECT_ROOT / "evaluation/stage8_5"
LAYOUT = stage85_layout(STAGE85_ROOT)


def test_stage85_layout_separates_pipelines_reviews_and_final_training_inputs():
    """目录结构必须直接表达业务阶段，不能重新退化成 candidates/results 混放。"""

    assert {path.name for path in LAYOUT.final.iterdir()} == {
        "sft_curated_seed_manifest.json",
        "sft_curated_seed_train.jsonl",
    }
    assert (STAGE85_ROOT / "README.md").is_file()
    assert (STAGE85_ROOT / "pipelines/public_candidate").is_dir()
    assert (STAGE85_ROOT / "pipelines/curated_gold").is_dir()
    assert (STAGE85_ROOT / "pipelines/sft_seed").is_dir()

    for legacy_directory in ["candidates", "processed", "reviews", "results", "sources"]:
        assert not (STAGE85_ROOT / legacy_directory).exists()


def test_stage85_relocated_snapshot_hashes_point_to_existing_unchanged_inputs():
    """移动文件后，快照中的来源路径和 SHA256 必须仍能定位并验证真实输入。"""

    snapshot_paths = [
        LAYOUT.curated_intermediate / "environment_snapshot_import_v1.json",
        LAYOUT.sft_intermediate / "environment_snapshot_training_v2.json",
    ]
    for snapshot_path in snapshot_paths:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        for relative_path, expected_sha256 in snapshot["source_hashes"].items():
            source_path = PROJECT_ROOT / relative_path
            assert source_path.is_file(), relative_path
            assert hashlib.sha256(source_path.read_bytes()).hexdigest() == expected_sha256
