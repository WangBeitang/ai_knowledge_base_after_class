import json
from pathlib import Path

from app.rag.evaluation.case_schema import PlannerEvalCase
from evaluation.stage8_5.pipelines.common.paths import stage85_layout
from evaluation.stage8_5.pipelines.public_candidate.seed_public_candidate_pool import (
    main as seed_public_candidate_pool,
)


def test_stage85_seed_public_candidate_pool_materializes_auditable_first_batch(tmp_path: Path):
    base_dir = tmp_path / "stage8_5"

    assert seed_public_candidate_pool(["--base-dir", str(base_dir)]) == 0
    layout = stage85_layout(base_dir)

    sources = _read_jsonl(layout.public_intermediate / "sources/source_manifest.jsonl")
    cards = _read_jsonl(layout.public_intermediate / "fault_scenario_cards.jsonl")
    candidates = _read_jsonl(layout.public_intermediate / "planner_case_candidates.jsonl")
    approved_payloads = _read_jsonl(layout.public_review / "schema_approved_cases.jsonl")
    review_payloads = _read_jsonl(layout.public_review / "review_queue.jsonl")
    rejected_payloads = _read_jsonl(layout.public_review / "rejected_cases.jsonl")
    chunk_map = _read_jsonl(layout.public_intermediate / "chunk_source_map.jsonl")
    report = json.loads((layout.public_intermediate / "data_quality_report.json").read_text(encoding="utf-8"))
    split_manifest = json.loads((layout.public_review / "split_manifest.json").read_text(encoding="utf-8"))
    markdown_report = (layout.reports / "阶段8.5数据处理报告.md").read_text(encoding="utf-8")

    assert len(sources) == 3
    assert {source["approval_status"] for source in sources} == {"approved"}
    assert len(cards) == 26
    assert len(candidates) == 52
    assert len(approved_payloads) == 24
    assert len(review_payloads) == 28
    assert rejected_payloads == []
    assert report["source_counts"] == {"approved": 3, "pending": 0, "rejected": 0, "total": 3}
    assert report["case_counts"] == {"approved": 24, "rejected": 0, "review": 28, "total": 52}
    assert report["split_counts"] == {"dev": 6, "test": 6, "train": 12}
    assert report["issues"] == []

    approved_ids = {payload["case_id"] for payload in approved_payloads}
    review_ids = {payload["case_id"] for payload in review_payloads}
    assert not approved_ids.intersection(review_ids)
    assert len(split_manifest["train_case_ids"]) == 12
    assert len(split_manifest["dev_case_ids"]) == 6
    assert len(split_manifest["test_case_ids"]) == 6
    assert set(split_manifest["train_case_ids"]).issubset(approved_ids)
    assert set(split_manifest["dev_case_ids"]).issubset(approved_ids)
    assert set(split_manifest["test_case_ids"]).issubset(approved_ids)

    chunk_identities = {
        (record["document_id"], str(record["chunk_id"]), int(record["index_version"]))
        for record in chunk_map
    }
    for payload in approved_payloads:
        case = PlannerEvalCase.model_validate(payload)
        assert case.human_review_status.value == "reviewed"
        assert case.expected_behavior.should_answer is True
        assert case.source_document_ids
        assert case.source_index_versions
        assert "source_id=" in case.notes
        assert "card_id=" in case.notes
        assert "seed_version=stage85-public-seed-v1" in case.notes
        assert any(chunk.relevance.value == "required" for chunk in case.expected_chunks)
        for chunk in case.expected_chunks:
            identity = (chunk.document_id, str(chunk.chunk_id), int(chunk.index_version))
            assert identity in chunk_identities

    for payload in review_payloads:
        case = PlannerEvalCase.model_validate(payload)
        assert case.human_review_status.value == "pending"

    assert "阶段 8.5 数据处理报告" in markdown_report
    assert "| `approved` | 24 |" in markdown_report
    assert "| `review` | 28 |" in markdown_report


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
