import json
from pathlib import Path

import pytest

from evaluation.stage8_5.pipelines.public_candidate.build_fault_cards import main as build_fault_cards
from evaluation.stage8_5.pipelines.public_candidate.generate_candidate_cases import (
    main as generate_candidate_cases,
)
from evaluation.stage8_5.pipelines.public_candidate.generate_stage85_report import (
    main as generate_stage85_report,
)
from evaluation.stage8_5.pipelines.public_candidate.split_candidate_cases import (
    main as split_candidate_cases,
)
from evaluation.stage8_5.pipelines.public_candidate.validate_candidate_cases import (
    main as validate_candidate_cases,
)
from evaluation.stage8_5.pipelines.public_candidate.validate_sources import main as validate_sources


def test_stage85_source_validation_blocks_approved_source_without_training_permission(tmp_path: Path):
    sources_path = tmp_path / "source_manifest.jsonl"
    licenses_path = tmp_path / "license_manifest.jsonl"
    report_path = tmp_path / "data_quality_report.json"
    _write_jsonl(licenses_path, [_license_payload(training_allowed=False)])
    _write_jsonl(sources_path, [_source_payload()])

    exit_code = validate_sources([
        "--sources", str(sources_path),
        "--licenses", str(licenses_path),
        "--report", str(report_path),
    ])

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert report["source_counts"]["approved"] == 1
    assert any(issue["code"] == "approved_training_not_allowed" for issue in report["issues"])
    assert report["approved_source_ids"] == []


def test_stage85_entry_pipeline_generates_review_queue_and_report(tmp_path: Path):
    sources_path = tmp_path / "source_manifest.jsonl"
    licenses_path = tmp_path / "license_manifest.jsonl"
    source_report_path = tmp_path / "source_report.json"
    cards_input_path = tmp_path / "cards_input.jsonl"
    cards_output_path = tmp_path / "cards_output.jsonl"
    card_report_path = tmp_path / "card_report.json"
    candidates_path = tmp_path / "planner_case_candidates.jsonl"
    rejected_from_cards_path = tmp_path / "rejected_from_cards.jsonl"
    generated_report_path = tmp_path / "generated_report.json"
    approved_path = tmp_path / "schema_approved_cases.jsonl"
    review_path = tmp_path / "review_queue.jsonl"
    rejected_path = tmp_path / "rejected_cases.jsonl"
    validation_report_path = tmp_path / "validation_report.json"
    markdown_path = tmp_path / "stage85_report.md"

    _write_jsonl(licenses_path, [_license_payload()])
    _write_jsonl(sources_path, [_source_payload()])
    assert validate_sources([
        "--sources", str(sources_path),
        "--licenses", str(licenses_path),
        "--report", str(source_report_path),
    ]) == 0

    _write_jsonl(cards_input_path, [_fault_card_payload()])
    assert build_fault_cards([
        "--input", str(cards_input_path),
        "--output", str(cards_output_path),
        "--source-report", str(source_report_path),
        "--report", str(card_report_path),
    ]) == 0

    assert generate_candidate_cases([
        "--cards", str(cards_output_path),
        "--output", str(candidates_path),
        "--rejected", str(rejected_from_cards_path),
        "--report", str(generated_report_path),
        "--split", "train",
    ]) == 0

    assert validate_candidate_cases([
        "--input", str(candidates_path),
        "--approved", str(approved_path),
        "--review", str(review_path),
        "--rejected", str(rejected_path),
        "--report", str(validation_report_path),
    ]) == 0

    review_cases = _read_jsonl(review_path)
    validation_report = json.loads(validation_report_path.read_text(encoding="utf-8"))
    assert len(review_cases) == 1
    assert review_cases[0]["human_review_status"] == "pending"
    assert review_cases[0]["expected_chunks"][0]["relevance"] == "required"
    assert validation_report["case_counts"] == {
        "approved": 0,
        "rejected": 0,
        "review": 1,
        "total": 1,
        "valid_total": 1,
    }

    assert generate_stage85_report([
        "--report", str(validation_report_path),
        "--output", str(markdown_path),
    ]) == 0
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "阶段 8.5 数据处理报告" in markdown
    assert "候选 case 统计" in markdown


def test_stage85_candidate_validation_routes_reviewed_and_invalid_records(tmp_path: Path):
    candidates_path = tmp_path / "planner_case_candidates.jsonl"
    approved_path = tmp_path / "schema_approved_cases.jsonl"
    review_path = tmp_path / "review_queue.jsonl"
    rejected_path = tmp_path / "rejected_cases.jsonl"
    report_path = tmp_path / "validation_report.json"

    reviewed_payload = _planner_case_payload(human_review_status="reviewed")
    invalid_payload = {
        "case_id": "stage85-invalid-case",
        "query": "缺少大部分字段，应该进入 rejected。",
    }
    _write_jsonl(candidates_path, [reviewed_payload, invalid_payload])

    exit_code = validate_candidate_cases([
        "--input", str(candidates_path),
        "--approved", str(approved_path),
        "--review", str(review_path),
        "--rejected", str(rejected_path),
        "--report", str(report_path),
    ])

    assert exit_code == 1
    assert len(_read_jsonl(approved_path)) == 1
    assert _read_jsonl(review_path) == []
    rejected = _read_jsonl(rejected_path)
    assert len(rejected) == 1
    assert rejected[0]["issues"][0]["code"] == "planner_case_schema_error"


def test_stage85_split_manifest_reuses_stage8_leakage_validation(tmp_path: Path):
    cases_path = tmp_path / "schema_approved_cases.jsonl"
    manifest_path = tmp_path / "split_manifest.json"
    first = _planner_case_payload(case_id="stage85-train-bearing-001", leakage_group_id="bearing-same", split="train")
    second = _planner_case_payload(case_id="stage85-dev-bearing-001", leakage_group_id="bearing-same", split="dev")
    _write_jsonl(cases_path, [first, second])

    with pytest.raises(ValueError, match="leakage_group_id 不能跨 split"):
        split_candidate_cases([
            "--cases", str(cases_path),
            "--output", str(manifest_path),
        ])


def _license_payload(*, training_allowed: bool = True) -> dict:
    return {
        "license_id": "cc-by-4.0",
        "license_name": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "redistribution_allowed": True,
        "training_allowed": training_allowed,
        "commercial_use_allowed": True,
        "notes": "测试用许可证记录。",
    }


def _source_payload() -> dict:
    return {
        "source_id": "uci-metropt3",
        "source_type": "timeseries",
        "title": "MetroPT-3 Dataset",
        "publisher": "UCI Machine Learning Repository",
        "url_or_path": "https://archive.ics.uci.edu/dataset/791/metropt%2B3%2Bdataset",
        "collected_at": "2026-07-19T00:00:00+00:00",
        "source_hash": "",
        "license_name": "CC BY 4.0",
        "license_url": "",
        "redistribution_allowed": None,
        "training_allowed": None,
        "commercial_use_allowed": None,
        "approval_status": "approved",
        "reject_reason": "",
        "notes": "测试用 approved source，权限从 license_manifest 继承。",
    }


def _fault_card_payload() -> dict:
    return {
        "card_id": "metropt-air-leak-001",
        "source_id": "uci-metropt3",
        "source_document_id": "doc_metropt3_card",
        "source_section": "air_leak_failure",
        "equipment_model": "MetroPT-3 APU",
        "component_name": "air compressor",
        "alarm_code": "",
        "symptom": "空压机出现空气泄漏，压力恢复慢。",
        "possible_causes": ["阀门泄漏", "管路密封异常"],
        "diagnostic_steps": ["检查 DV_pressure 和 Reservoirs 信号趋势", "确认 COMP 是否持续异常运行"],
        "maintenance_actions": ["检查气路密封", "复核阀门状态"],
        "safety_notes": ["停机泄压后再检查管路。"],
        "evidence_text": "测试卡片证据摘要。",
        "evidence_chunk_ids": [
            {
                "document_id": "doc_metropt3_card",
                "chunk_id": "chunk_air_leak",
                "index_version": 1,
                "relevance": "required",
                "answer_point_ids": ["symptom", "diagnostic_step"],
            }
        ],
        "quality_flags": [],
        "candidate_queries": ["MetroPT-3 APU 空压机压力恢复慢可能是什么问题？"],
        "expected_answer_points": ["空气泄漏", "检查 DV_pressure", "停机泄压"],
    }


def _planner_case_payload(**overrides) -> dict:
    payload = {
        "case_id": "stage85-reviewed-air-leak-001",
        "case_group": "core",
        "split": "train",
        "leakage_group_id": "stage85-metropt-air-leak",
        "query": "MetroPT-3 APU 空压机压力恢复慢可能是什么问题？",
        "query_variants": [],
        "dataset_ids": ["dataset_default_equipment_ops"],
        "owner_user_id": "eval_demo_user",
        "tenant_id": "tenant_default",
        "privacy_scope": "public_demo",
        "source_document_ids": ["doc_metropt3_card"],
        "source_index_versions": {"doc_metropt3_card": 1},
        "expected_subject_ids": [],
        "expected_subject_names": ["MetroPT-3 APU"],
        "expected_chunks": [
            {
                "document_id": "doc_metropt3_card",
                "chunk_id": "chunk_air_leak",
                "index_version": 1,
                "relevance": "required",
                "answer_point_ids": ["symptom"],
            }
        ],
        "expected_answer_points": ["空气泄漏", "检查 DV_pressure", "停机泄压"],
        "expected_behavior": {
            "should_answer": True,
            "should_refuse": False,
            "should_ask_clarification": False,
            "should_call_web": False,
            "web_required_reason": "",
            "forbidden_actions": ["web_search"],
        },
        "acceptable_action_paths": [["local_search", "answer"]],
        "expected_identifiers": {"equipment_model": ["MetroPT-3 APU"]},
        "label_source": "synthetic",
        "human_review_status": "reviewed",
        "notes": "测试用 reviewed candidate。",
    }
    payload.update(overrides)
    return payload


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
