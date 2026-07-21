from collections import Counter
from pathlib import Path

from app.rag.evaluation.case_schema import GoldOrigin, load_planner_cases
from evaluation.stage9.route_seed.build_route_seed_cases import (
    build_route_seed_cases,
    read_route_seed_paths,
    write_jsonl,
)
from evaluation.stage9.route_seed.export_route_sft_data import (
    export_and_merge_route_sft,
)
from evaluation.stage9.route_seed.generate_route_seed_report import (
    build_route_seed_report,
)
from evaluation.stage9.route_seed.run_route_seed_paths import (
    run_route_seed_paths,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CASES = PROJECT_ROOT / "evaluation/stage8_5/artifacts/intermediate/sft_seed/curated_seed_train_cases.jsonl"
SNAPSHOT = PROJECT_ROOT / "evaluation/stage8_5/artifacts/intermediate/sft_seed/environment_snapshot_training_v2.json"
BASE_SFT = PROJECT_ROOT / "evaluation/stage8_5/artifacts/final/sft_curated_seed_train.jsonl"
BASE_MANIFEST = PROJECT_ROOT / "evaluation/stage8_5/artifacts/final/sft_curated_seed_manifest.json"


def test_stage9_route_seed_builds_50_reviewed_train_cases():
    cases, paths, reviews = build_route_seed_cases(load_planner_cases(SOURCE_CASES))

    assert len(cases) == 50
    assert len(paths) == 50
    assert len(reviews) == 50
    assert {case.split.value for case in cases} == {"train"}
    assert {case.gold_origin for case in cases} == {GoldOrigin.ROUTE_SEED_GOLD}
    assert {case.human_review_status.value for case in cases} == {"reviewed"}
    assert Counter(path.route_family for path in paths) == {
        "ask_clarification": 10,
        "hyde_fallback": 10,
        "multi_step_fallback": 10,
        "refuse": 10,
        "web_search": 10,
    }


def test_stage9_route_seed_runs_exports_and_merges(tmp_path: Path):
    cases, paths, reviews = build_route_seed_cases(load_planner_cases(SOURCE_CASES))
    cases_path = tmp_path / "route_seed_cases.jsonl"
    paths_path = tmp_path / "route_seed_action_paths.jsonl"
    review_path = tmp_path / "route_seed_review.jsonl"
    baseline_path = tmp_path / "route_seed_baseline_train.json"
    route_sft_path = tmp_path / "sft_route_seed_train.jsonl"
    route_manifest_path = tmp_path / "sft_route_seed_manifest.json"
    merged_sft_path = tmp_path / "sft_planner_stage9_train.jsonl"
    merged_manifest_path = tmp_path / "sft_planner_stage9_manifest.json"
    report_path = tmp_path / "stage9_route_seed_report.md"

    write_jsonl(cases_path, cases)
    write_jsonl(paths_path, paths)
    write_jsonl(review_path, reviews)

    baseline = run_route_seed_paths(
        cases=load_planner_cases(cases_path),
        paths=read_route_seed_paths(paths_path),
        snapshot_path=SNAPSHOT,
        output_path=baseline_path,
        run_id="pytest_stage9_route_seed",
    )

    assert baseline.case_count == 50
    assert not any(result.errors for result in baseline.results)
    assert baseline.planner_summaries[0].config["route_family_counts"] == {
        "ask_clarification": 10,
        "hyde_fallback": 10,
        "multi_step_fallback": 10,
        "refuse": 10,
        "web_search": 10,
    }

    manifest = export_and_merge_route_sft(
        eval_result_path=baseline_path,
        cases_path=cases_path,
        paths_path=paths_path,
        route_output_path=route_sft_path,
        route_manifest_path=route_manifest_path,
        base_sft_path=BASE_SFT,
        base_manifest_path=BASE_MANIFEST,
        merged_output_path=merged_sft_path,
        merged_manifest_path=merged_manifest_path,
    )

    assert manifest.sample_count == 155
    assert manifest.source_case_count == 70
    assert manifest.action_counts == {
        "answer": 30,
        "ask_clarification": 10,
        "hyde_search": 20,
        "local_search": 55,
        "refuse": 30,
        "web_search": 10,
    }
    assert manifest.route_family_counts["stop_when_enough"] == 40
    assert manifest.route_family_counts["hyde_fallback"] == 30
    assert manifest.gold_origin_counts == {
        "curated_seed_gold": 40,
        "route_seed_gold": 115,
    }

    report = build_route_seed_report(
        cases_path=cases_path,
        paths_path=paths_path,
        baseline_path=baseline_path,
        route_sft_path=route_sft_path,
        route_manifest_path=route_manifest_path,
        merged_sft_path=merged_sft_path,
        merged_manifest_path=merged_manifest_path,
    )
    report_path.write_text(report, encoding="utf-8")
    assert "阶段 9 SFT 路线覆盖报告" in report
    assert "Web answer 需要真实 Web/replay provider" in report

