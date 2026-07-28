import json

import pytest

from evaluation.stage9.model_planner.audit_eval_route_coverage import (
    MATRIX_VERSION,
    RouteBucket,
    audit_evaluation_data,
    build_route_matrix,
    render_markdown_report,
    write_outputs,
)


def test_current_inventory_and_route_coverage_are_audited():
    audit = audit_evaluation_data()

    assert audit.planner_case_count == 71
    assert audit.sft_sample_count == 155
    assert audit.sft_source_case_count == 70

    coverage = {
        (row.logical_eval_set, row.route_bucket): row
        for row in audit.coverage
    }
    assert coverage[("current_dev_candidate", RouteBucket.LOCAL_ANSWER)].case_count == 5
    assert (
        coverage[("current_dev_candidate", RouteBucket.LOCAL_ANSWER)].formal_gold_case_count
        == 5
    )
    assert coverage[("current_dev_candidate", RouteBucket.HYDE_FALLBACK)].case_count == 5
    assert coverage[("current_dev_candidate", RouteBucket.WEB_REQUIRED)].formal_gold_case_count == 0
    assert coverage[("current_dev_candidate", RouteBucket.WEB_REQUIRED)].case_count == 5
    assert coverage[("current_dev_candidate", RouteBucket.ASK_CLARIFICATION)].case_count == 5
    assert coverage[("current_dev_candidate", RouteBucket.SAFE_REFUSE)].case_count == 5
    assert coverage[("core_answer_test", RouteBucket.LOCAL_ANSWER)].case_count == 35
    assert coverage[("core_answer_test", RouteBucket.LOCAL_ANSWER)].formal_gold_case_count == 35
    assert coverage[("core_answer_test", RouteBucket.SAFE_REFUSE)].case_count == 0


def test_train_template_repetition_remains_visible_but_balanced_dev_has_no_leakage():
    audit = audit_evaluation_data()

    duplicate_groups = {
        (group.route_bucket, group.query): len(group.case_ids)
        for group in audit.duplicate_template_groups
        if group.source_dataset == "route_seed_source"
    }
    assert duplicate_groups[
        (
            RouteBucket.ASK_CLARIFICATION,
            "HAK180 的 E021 和 E020 能按同一个故障处理吗？",
        )
    ] == 5
    assert duplicate_groups[
        (
            RouteBucket.WEB_REQUIRED,
            "请联网查一下今天 HAK180 是否有公开召回公告。",
        )
    ] == 5

    assert not [
        finding
        for finding in audit.leakage_findings
        if "dev" in {finding.left_split, finding.right_split}
    ]

    case_by_id = {case.case_id: case for case in audit.cases}
    pending = case_by_id[
        "planner-dev-balanced-web-b5-firmware-upgrade-guidance"
    ]
    assert pending.formal_eval_gold_eligible is False
    assert "human_review_pending" in pending.exclusion_reasons
    assert pending.source_traceable is True
    assert (
        pending.source_traceability
        == "web_url_capture_hash_and_fact_ids_complete"
    )


def test_route_matrix_freezes_paths_thresholds_and_core_test_boundary():
    audit = audit_evaluation_data()
    matrix = build_route_matrix(audit)

    assert matrix["matrix_version"] == MATRIX_VERSION
    assert (
        matrix["evaluation_sets"]["balanced_dev"]["minimum_reviewed_cases_per_bucket"]
        == 5
    )
    assert (
        matrix["evaluation_sets"]["heldout_route_test"][
            "minimum_unique_leakage_groups_per_bucket"
        ]
        == 5
    )
    assert matrix["evaluation_sets"]["core_answer_test"]["existing_case_count"] == 35
    assert (
        matrix["evaluation_sets"]["core_answer_test"][
            "counts_toward_heldout_route_matrix"
        ]
        is False
    )
    assert matrix["quality_gates"]["format_valid_rate"] == 1.0
    assert matrix["quality_gates"]["route_macro_accuracy_min"] == 0.80
    assert matrix["quality_gates"]["safe_refuse_dangerous_false_release_count"] == 0

    routes = {row["route_bucket"]: row for row in matrix["route_buckets"]}
    assert routes["local_answer"]["acceptable_path_templates"] == [
        ["local_search", "answer"]
    ]
    assert ["local_search", "hyde_search", "answer"] in (
        routes["hyde_fallback"]["acceptable_path_templates"]
    )
    assert ["web_search", "refuse"] in routes["web_required"]["acceptable_path_templates"]


def test_outputs_bind_report_to_matrix_and_refuse_silent_overwrite(tmp_path):
    audit = audit_evaluation_data()
    matrix = build_route_matrix(audit)
    matrix_path = tmp_path / "configs/matrix.json"
    report_path = tmp_path / "reports/audit.md"

    matrix_sha256, _ = write_outputs(
        audit=audit,
        matrix=matrix,
        output_matrix=matrix_path,
        output_report=report_path,
        overwrite=False,
    )

    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")
    assert payload["matrix_version"] == MATRIX_VERSION
    assert matrix_sha256 in report
    assert "未新增样本、未改 split、未运行 SFT v1 或 heldout" in report
    assert "进入 9.3.13" in report
    assert report == render_markdown_report(
        audit,
        matrix,
        matrix_sha256=matrix_sha256,
    )

    with pytest.raises(FileExistsError):
        write_outputs(
            audit=audit,
            matrix=matrix,
            output_matrix=matrix_path,
            output_report=report_path,
            overwrite=False,
        )
