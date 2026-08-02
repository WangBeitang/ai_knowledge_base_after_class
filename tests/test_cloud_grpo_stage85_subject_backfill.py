import json
from pathlib import Path

import pytest

from scripts.cloud_grpo.backfill_stage85_gold_subjects import (
    DEFAULT_MANIFEST,
    SUBJECT_BY_DOCUMENT,
    _load_targets,
    _sha256_bytes,
    _validate_and_plan,
)
from scripts.cloud_grpo.recover_stage85_gold_auto_ids import (
    _build_recovery_payloads,
    _validate_restored_rows,
)


def test_stage85_subject_backfill_targets_only_fixed_ten_gold_chunks():
    manifest = json.loads(Path(DEFAULT_MANIFEST).read_text(encoding="utf-8"))

    targets = _load_targets(manifest)

    assert len(targets) == 10
    assert {target["document_id"] for target in targets.values()} == set(SUBJECT_BY_DOCUMENT)
    assert all(target["subject_id"] for target in targets.values())


def test_stage85_subject_backfill_plans_only_subject_fields():
    content = "frozen gold evidence"
    targets = {
        123: {
            "chunk_id": 123,
            "document_id": "doc_stage85_uci_ai4i_official_description_v1",
            "index_version": 1,
            "content_sha256": _sha256_bytes(content.encode("utf-8")),
            "subject_id": "subject_uci_ai4i_2020",
            "standard_subject_name": "AI4I 2020 Predictive Maintenance Dataset",
        }
    }
    rows = [{
        "chunk_id": 123,
        "document_id": "doc_stage85_uci_ai4i_official_description_v1",
        "index_version": 1,
        "subject_id": "",
        "standard_subject_name": "old title",
        "content": content,
    }]

    changes, backups = _validate_and_plan(targets, rows)

    assert changes == [{
        "chunk_id": 123,
        "subject_id": "subject_uci_ai4i_2020",
        "standard_subject_name": "AI4I 2020 Predictive Maintenance Dataset",
    }]
    assert backups[0]["chunk_id"] == 123
    assert backups[0]["subject_id"] == ""


@pytest.mark.parametrize(
    ("content", "subject_id", "error"),
    [
        ("drifted", "", "正文 SHA256 漂移"),
        ("frozen gold evidence", "subject_other", "已有其他 subject_id"),
    ],
)
def test_stage85_subject_backfill_rejects_drift_or_overwrite(content, subject_id, error):
    expected_content = "frozen gold evidence"
    targets = {
        123: {
            "chunk_id": 123,
            "document_id": "doc_stage85_uci_ai4i_official_description_v1",
            "index_version": 1,
            "content_sha256": _sha256_bytes(expected_content.encode("utf-8")),
            "subject_id": "subject_uci_ai4i_2020",
            "standard_subject_name": "AI4I 2020 Predictive Maintenance Dataset",
        }
    }
    rows = [{
        "chunk_id": 123,
        "document_id": "doc_stage85_uci_ai4i_official_description_v1",
        "index_version": 1,
        "subject_id": subject_id,
        "standard_subject_name": "old title",
        "content": content,
    }]

    with pytest.raises(ValueError, match=error):
        _validate_and_plan(targets, rows)


def test_stage85_auto_id_recovery_builds_full_old_id_payload_before_delete():
    content = "frozen gold evidence"
    targets = {
        123: {
            "chunk_id": 123,
            "document_id": "doc_stage85_uci_ai4i_official_description_v1",
            "index_version": 1,
            "content_sha256": _sha256_bytes(content.encode("utf-8")),
            "subject_id": "subject_uci_ai4i_2020",
            "standard_subject_name": "AI4I 2020 Predictive Maintenance Dataset",
        }
    }
    current_rows = [{
        "chunk_id": 999,
        "document_id": "doc_stage85_uci_ai4i_official_description_v1",
        "index_version": 1,
        "subject_id": "subject_uci_ai4i_2020",
        "standard_subject_name": "AI4I 2020 Predictive Maintenance Dataset",
        "content": content,
        "lexical_text": content,
        "dense_vector": [0.0] * 1024,
        "learned_sparse_vector": {1: 0.5},
        "bm25_sparse_vector": {2: 0.5},
        "source_id": "uci-ai4i-2020",
    }]

    payloads, current_ids, mappings = _build_recovery_payloads(targets, current_rows)

    assert current_ids == [999]
    assert payloads[0]["chunk_id"] == 123
    assert payloads[0]["source_id"] == "uci-ai4i-2020"
    assert "bm25_sparse_vector" not in payloads[0]
    assert mappings[0]["old_chunk_id"] == 123
    assert mappings[0]["current_chunk_id"] == 999
    _validate_restored_rows(targets, payloads)


def test_stage85_auto_id_recovery_rejects_missing_target_before_mutation():
    targets = {
        123: {
            "chunk_id": 123,
            "document_id": "doc_stage85_uci_ai4i_official_description_v1",
            "index_version": 1,
            "content_sha256": "0" * 64,
            "subject_id": "subject_uci_ai4i_2020",
            "standard_subject_name": "AI4I 2020 Predictive Maintenance Dataset",
        }
    }

    with pytest.raises(ValueError, match="未覆盖全部原 ID"):
        _build_recovery_payloads(targets, [])
