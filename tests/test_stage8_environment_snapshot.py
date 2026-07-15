from pathlib import Path

import pytest

from app.rag.evaluation.case_schema import EnvironmentSnapshot, PlannerEvalCase
from evaluation.stage8.build_environment_snapshot import (
    SnapshotBuildError,
    build_and_write_environment_snapshot,
)


class FakeMetadataReader:
    def __init__(self, *, datasets: list[dict], documents: list[dict]) -> None:
        self.datasets = datasets
        self.documents = documents

    def load_datasets(self, dataset_ids: list[str]) -> list[dict]:
        return [
            dataset for dataset in self.datasets
            if dataset["dataset_id"] in dataset_ids
        ]

    def load_documents(self, dataset_ids: list[str]) -> list[dict]:
        return [
            document for document in self.documents
            if document["dataset_id"] in dataset_ids
        ]


class FakeChunkReader:
    def __init__(self, chunks_by_document: dict[str, list[dict]]) -> None:
        self.chunks_by_document = chunks_by_document

    def load_enabled_chunks(self, document: dict) -> list[dict]:
        return list(self.chunks_by_document.get(document["document_id"], []))


class FakeOverrideReader:
    def __init__(self, overrides_by_document: dict[str, list[dict]] | None = None) -> None:
        self.overrides_by_document = overrides_by_document or {}

    def load_disabled_overrides(self, document: dict) -> list[dict]:
        return list(self.overrides_by_document.get(document["document_id"], []))


def _document(index_version: int = 3) -> dict:
    return {
        "document_id": "doc_hak180_manual",
        "dataset_id": "dataset_default_equipment_ops",
        "owner_user_id": "eval_demo_user",
        "tenant_id": "tenant_default",
        "visibility": "public",
        "index_version": index_version,
        "chunk_count": 2,
        "status": "completed",
        "index_status": "completed",
    }


def _dataset() -> dict:
    return {
        "dataset_id": "dataset_default_equipment_ops",
        "status": "active",
    }


def _chunk(chunk_id: int = 12345, index_version: int = 3) -> dict:
    return {
        "document_id": "doc_hak180_manual",
        "chunk_id": chunk_id,
        "index_version": index_version,
        "enabled": True,
        "chunk_index": 0,
    }


def _case(index_version: int = 3, chunk_id: int = 12345) -> PlannerEvalCase:
    return PlannerEvalCase(
        case_id="dev-alarm-e020-001",
        case_group="core",
        split="dev",
        leakage_group_id="hak180-e020",
        query="HAK180 设备的 E020 是什么故障？",
        dataset_ids=["dataset_default_equipment_ops"],
        owner_user_id="eval_demo_user",
        tenant_id="tenant_default",
        privacy_scope="public_demo",
        source_document_ids=["doc_hak180_manual"],
        source_index_versions={"doc_hak180_manual": index_version},
        expected_subject_names=["HAK 180 烫金机"],
        expected_chunks=[
            {
                "document_id": "doc_hak180_manual",
                "chunk_id": chunk_id,
                "index_version": index_version,
                "relevance": "required",
                "answer_point_ids": ["alarm_meaning"],
            }
        ],
        expected_answer_points=["说明 E020 的故障含义"],
        expected_behavior={
            "should_answer": True,
            "should_refuse": False,
            "should_ask_clarification": False,
            "should_call_web": False,
            "forbidden_actions": ["web_search"],
        },
        acceptable_action_paths=[["local_search", "answer"]],
        expected_identifiers={"alarm_code": ["E020"]},
        label_source="manual",
        human_review_status="reviewed",
    )


def _pending_behavior_case_without_real_corpus() -> PlannerEvalCase:
    return PlannerEvalCase(
        case_id="planner-train-private-doc-isolation-maintenance-record",
        case_group="private_doc",
        split="train",
        leakage_group_id="synthetic-private-doc-isolation-maintenance-record",
        query="我私有维修记录里，P3500 上次更换的是哪个部件？",
        dataset_ids=["dataset_private_maintenance_candidate"],
        owner_user_id="eval_private_user",
        tenant_id="tenant_default",
        privacy_scope="private_user",
        expected_behavior={
            "should_answer": False,
            "should_refuse": True,
            "should_ask_clarification": False,
            "should_call_web": False,
            "forbidden_actions": ["web_search"],
        },
        acceptable_action_paths=[["refuse"]],
        label_source="synthetic",
        human_review_status="pending",
    )


def test_stage8_environment_snapshot_builder_writes_and_reads_snapshot(tmp_path: Path):
    output_path = tmp_path / "environment_snapshot.json"

    result = build_and_write_environment_snapshot(
        output_path=output_path,
        metadata_reader=FakeMetadataReader(datasets=[_dataset()], documents=[_document()]),
        chunk_reader=FakeChunkReader({"doc_hak180_manual": [_chunk(), _chunk(67890)]}),
        override_reader=FakeOverrideReader(),
        cases=[_case()],
        snapshot_id="stage8-env-test-v1",
        created_at="2026-07-15T00:00:00+00:00",
        retrieval_config_snapshot={"retrieval_mode": "dense_learned_sparse_bm25"},
        reranker={"model": "bge-reranker"},
        answer_model={"model": "qwen"},
        planner_registry=[{"planner_mode": "rule", "enabled_for_eval": True}],
        source_hashes={"cases.jsonl": "abc"},
    )

    snapshot = EnvironmentSnapshot.model_validate_json(output_path.read_text(encoding="utf-8"))

    assert result.summary["case_reference_check"] == "passed"
    assert result.summary["enabled_chunk_count"] == 2
    assert snapshot.snapshot_id == "stage8-env-test-v1"
    assert snapshot.enabled_chunks["doc_hak180_manual"] == [12345, 67890]
    assert snapshot.reranker["model"] == "bge-reranker"


def test_stage8_environment_snapshot_builder_reports_pending_case_dataset_without_blocking(
        tmp_path: Path,
):
    output_path = tmp_path / "environment_snapshot.json"

    result = build_and_write_environment_snapshot(
        output_path=output_path,
        metadata_reader=FakeMetadataReader(datasets=[_dataset()], documents=[_document()]),
        chunk_reader=FakeChunkReader({"doc_hak180_manual": [_chunk()]}),
        override_reader=FakeOverrideReader(),
        cases=[_case(), _pending_behavior_case_without_real_corpus()],
        snapshot_id="stage8-env-test-v1",
    )

    assert output_path.exists()
    assert result.summary["case_reference_check"] == "passed"
    assert result.summary["case_dataset_ids_missing_from_snapshot"] == [
        "dataset_private_maintenance_candidate"
    ]


def test_stage8_environment_snapshot_builder_fails_without_partial_file_when_chunk_missing(
        tmp_path: Path,
):
    output_path = tmp_path / "environment_snapshot.json"

    with pytest.raises(SnapshotBuildError, match="enabled_chunks"):
        build_and_write_environment_snapshot(
            output_path=output_path,
            metadata_reader=FakeMetadataReader(datasets=[_dataset()], documents=[_document()]),
            chunk_reader=FakeChunkReader({"doc_hak180_manual": [_chunk(99999)]}),
            override_reader=FakeOverrideReader(),
            cases=[_case()],
            snapshot_id="stage8-env-test-v1",
        )

    assert not output_path.exists()
    assert not (tmp_path / ".environment_snapshot.json.tmp").exists()


def test_stage8_environment_snapshot_builder_fails_without_partial_file_when_dataset_missing(
        tmp_path: Path,
):
    output_path = tmp_path / "environment_snapshot.json"

    with pytest.raises(SnapshotBuildError, match="dataset_id 不存在"):
        build_and_write_environment_snapshot(
            output_path=output_path,
            metadata_reader=FakeMetadataReader(datasets=[], documents=[_document()]),
            chunk_reader=FakeChunkReader({"doc_hak180_manual": [_chunk()]}),
            override_reader=FakeOverrideReader(),
            cases=[_case()],
            snapshot_id="stage8-env-test-v1",
        )

    assert not output_path.exists()
    assert not (tmp_path / ".environment_snapshot.json.tmp").exists()


def test_stage8_environment_snapshot_builder_reports_old_index_version(tmp_path: Path):
    output_path = tmp_path / "environment_snapshot.json"

    with pytest.raises(SnapshotBuildError, match="引用旧 index_version"):
        build_and_write_environment_snapshot(
            output_path=output_path,
            metadata_reader=FakeMetadataReader(datasets=[_dataset()], documents=[_document(index_version=4)]),
            chunk_reader=FakeChunkReader({"doc_hak180_manual": [_chunk(index_version=4)]}),
            override_reader=FakeOverrideReader(),
            cases=[_case(index_version=3)],
            snapshot_id="stage8-env-test-v1",
        )

    assert not output_path.exists()


def test_stage8_environment_snapshot_builder_rejects_disabled_expected_chunk(tmp_path: Path):
    output_path = tmp_path / "environment_snapshot.json"

    with pytest.raises(SnapshotBuildError, match="当前被人工禁用"):
        build_and_write_environment_snapshot(
            output_path=output_path,
            metadata_reader=FakeMetadataReader(datasets=[_dataset()], documents=[_document()]),
            chunk_reader=FakeChunkReader({"doc_hak180_manual": [_chunk()]}),
            override_reader=FakeOverrideReader({
                "doc_hak180_manual": [{
                    "document_id": "doc_hak180_manual",
                    "chunk_id": 12345,
                    "index_version": 3,
                    "manual_status": "disabled",
                }],
            }),
            cases=[_case()],
            snapshot_id="stage8-env-test-v1",
        )

    assert not output_path.exists()
