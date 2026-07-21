import json
from collections import Counter
from pathlib import Path

from app.rag.evaluation.case_schema import PlannerEvalCase
from evaluation.stage8_5.pipelines.common.paths import stage85_layout
from evaluation.stage8_5.pipelines.curated_gold.build_source_grounded_gold import (
    GoldCaseAudit,
    GoldEvidenceChunk,
    main as build_source_grounded_gold,
)


def test_stage85_builds_twenty_source_grounded_gold_cases(tmp_path: Path):
    """20 条 gold 必须形成“case -> 答案点 -> 来源事实 -> UCI 页面”的完整证据闭包。"""

    base_dir = tmp_path / "stage8_5"
    assert build_source_grounded_gold(["--base-dir", str(base_dir)]) == 0
    layout = stage85_layout(base_dir)

    evidence_chunks = [
        GoldEvidenceChunk.model_validate(payload)
        for payload in _read_jsonl(layout.curated_intermediate / "gold_evidence_chunks.jsonl")
    ]
    documents = _read_jsonl(layout.curated_intermediate / "gold_evidence_documents.jsonl")
    cases = [
        PlannerEvalCase.model_validate(payload)
        for payload in _read_jsonl(layout.curated_intermediate / "gold_cases_authoring.jsonl")
    ]
    audits = [
        GoldCaseAudit.model_validate(payload)
        for payload in _read_jsonl(layout.curated_review / "gold_case_audit.jsonl")
    ]

    assert len(evidence_chunks) == 10
    assert len(documents) == 2
    assert len(cases) == 20
    assert len(audits) == 20
    assert Counter(audit.source_id for audit in audits) == {
        "uci-ai4i-2020": 10,
        "uci-hydraulic-condition": 10,
    }
    assert {document["processing_status"] for document in documents} == {"gold_evidence_ready_for_import"}

    chunk_by_id = {chunk.chunk_id: chunk for chunk in evidence_chunks}
    case_by_id = {case.case_id: case for case in cases}
    assert len(chunk_by_id) == len(evidence_chunks)
    assert len(case_by_id) == len(cases)
    assert len({audit.rewritten_from_case_id for audit in audits}) == 20

    for case in cases:
        assert case.split.value == "train"
        assert case.label_source.value == "api_assisted"
        assert case.human_review_status.value == "reviewed"
        assert case.acceptable_action_paths == [["local_search", "answer"]]
        assert {action.value for action in case.expected_behavior.forbidden_actions} == {
            "hyde_search",
            "web_search",
        }
        assert len(case.expected_chunks) == 1
        assert case.expected_chunks[0].chunk_id in chunk_by_id

    for audit in audits:
        case = case_by_id[audit.case_id]
        chunk = chunk_by_id[audit.chunk_id]
        fact_by_id = {fact.fact_id: fact.statement_zh for fact in chunk.facts}

        assert audit.gold_status == "source_verified"
        assert audit.reviewer_type == "primary_agent"
        assert audit.second_review_status == "pending"
        assert audit.document_id == chunk.document_id
        assert case.expected_chunks[0].document_id == chunk.document_id
        assert case.expected_chunks[0].chunk_id == chunk.chunk_id
        assert case.expected_answer_points == [item.answer_point for item in audit.answer_evidence]
        assert case.expected_chunks[0].answer_point_ids == [
            item.answer_point_id for item in audit.answer_evidence
        ]

        for answer_evidence in audit.answer_evidence:
            # 当前第一版坚持“一条答案点对应一个原子事实”；后续若合并事实，测试会显式暴露契约变化。
            assert len(answer_evidence.evidence_fact_ids) == 1
            fact_id = answer_evidence.evidence_fact_ids[0]
            assert answer_evidence.answer_point == fact_by_id[fact_id]

        rendered_case = " ".join([case.query, *case.expected_answer_points])
        for excluded_text in audit.excluded_content:
            assert excluded_text not in rendered_case


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
