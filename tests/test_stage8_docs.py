from pathlib import Path

from evaluation.stage8.generate_eval_report import main as generate_eval_report


def test_stage8_eval_report_contains_required_acceptance_sections():
    report = Path("evaluation/stage8/reports/阶段8评测报告.md").read_text(encoding="utf-8")

    assert "重构前后召回和引用质量" in report
    assert "rule/api/local_base baseline 对比" in report
    assert "Demo 回归集结果" in report
    assert "Reward 分项" in report
    assert "阶段 9 前置结论" in report
    assert "snapshot_expected_chunks" in report
    assert "SFT 数据导出" in report


def test_stage8_data_dictionary_documents_file_contracts():
    dictionary = Path("docs/数据字典.md").read_text(encoding="utf-8")

    for term in [
        "阶段 8 离线评测与训练数据文件",
        "PlannerEvalCase",
        "ExpectedChunk",
        "ExpectedBehavior",
        "EnvironmentSnapshot",
        "PlannerEvalResult",
        "BaselineEvalOutput",
        "SftPlannerSample",
        "SftExportManifest",
        "完整 chunk 正文",
        "模型私有思维链",
    ]:
        assert term in dictionary


def test_stage8_eval_report_generator_can_write_markdown(tmp_path: Path):
    output_path = tmp_path / "stage8_report.md"

    exit_code = generate_eval_report(["--output", str(output_path)])

    report = output_path.read_text(encoding="utf-8")
    assert exit_code == 0
    assert "# 阶段 8 评测报告" in report
    assert "sample_count" in report
    assert "rule/api/local_base" in report
