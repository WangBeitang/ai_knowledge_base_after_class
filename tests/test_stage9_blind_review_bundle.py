import json
from collections import Counter
from pathlib import Path

import pytest

from evaluation.stage9.balanced_dev.export_blind_review_bundle import (
    BANNED_HISTORY_MARKERS,
    BANNED_REVIEW_KEYS,
    DATA_FILE_NAMES,
    DEFAULT_OUTPUT_DIR,
    OUTPUT_FILE_NAMES,
    export_blind_review_bundle,
    validate_blind_review_bundle,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _all_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def test_blind_review_bundle_exports_only_sanitized_review_inputs(tmp_path):
    output_dir = tmp_path / "blind_review_bundle"

    result = export_blind_review_bundle(
        output_dir=output_dir,
        generated_at="2026-07-28T10:00:00+00:00",
    )

    assert result["ok"] is True
    assert result["case_count"] == 10
    assert result["route_counts"] == {
        "hyde_fallback": 3,
        "web_required": 5,
        "ask_clarification": 2,
    }
    assert {path.name for path in output_dir.iterdir()} == OUTPUT_FILE_NAMES

    for name in DATA_FILE_NAMES:
        path = output_dir / name
        values = (
            _read_jsonl(path)
            if path.suffix == ".jsonl"
            else [json.loads(path.read_text(encoding="utf-8"))]
        )
        assert not (set(_all_keys(values)) & BANNED_REVIEW_KEYS)
        text = path.read_text(encoding="utf-8")
        assert not any(marker in text for marker in BANNED_HISTORY_MARKERS)

    review_cases = _read_jsonl(output_dir / "review_cases.jsonl")
    assert Counter(row["route_bucket"] for row in review_cases) == {
        "hyde_fallback": 3,
        "web_required": 5,
        "ask_clarification": 2,
    }
    assert all("decision_schema" not in row for row in review_cases)
    assert all("required_checks" not in row for row in review_cases)
    ask_cases = [
        row for row in review_cases if row["route_bucket"] == "ask_clarification"
    ]
    assert all(
        row["eval_contract"]["expected_subject_ids"] == []
        and row["eval_contract"]["expected_subject_names"] == []
        for row in ask_cases
    )

    leakage_rows = _read_jsonl(output_dir / "leakage_reference.jsonl")
    target_ids = {row["case_id"] for row in review_cases}
    assert not (target_ids & {row["case_id"] for row in leakage_rows})
    assert set(leakage_rows[0]) == {
        "source_dataset",
        "case_id",
        "split",
        "leakage_group_id",
        "query",
        "query_variants",
    }

    local_evidence = json.loads(
        (output_dir / "local_evidence_manifest.json").read_text(encoding="utf-8")
    )
    assert len(local_evidence["documents"]) == 2
    assert sum(
        document["included_chunk_count"]
        for document in local_evidence["documents"]
    ) == 6
    web_evidence = json.loads(
        (output_dir / "web_evidence_manifest.json").read_text(encoding="utf-8")
    )
    assert web_evidence["source_count"] == 5


def test_blind_review_bundle_refuses_silent_overwrite(tmp_path):
    output_dir = tmp_path / "blind_review_bundle"
    export_blind_review_bundle(
        output_dir=output_dir,
        generated_at="2026-07-28T10:00:00+00:00",
    )

    with pytest.raises(FileExistsError, match="拒绝静默覆盖"):
        export_blind_review_bundle(
            output_dir=output_dir,
            generated_at="2026-07-28T10:00:00+00:00",
        )


def test_blind_review_bundle_validator_rejects_injected_review_field(tmp_path):
    output_dir = tmp_path / "blind_review_bundle"
    export_blind_review_bundle(
        output_dir=output_dir,
        generated_at="2026-07-28T10:00:00+00:00",
    )
    review_cases_path = output_dir / "review_cases.jsonl"
    rows = _read_jsonl(review_cases_path)
    rows[0]["human_review_status"] = "reviewed"
    review_cases_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="禁止审核字段"):
        validate_blind_review_bundle(output_dir)


def test_checked_in_blind_review_bundle_is_valid():
    result = validate_blind_review_bundle(DEFAULT_OUTPUT_DIR)

    assert result["ok"] is True
    assert result["case_count"] == 10
    assert result["contamination_scan"] == "passed"
