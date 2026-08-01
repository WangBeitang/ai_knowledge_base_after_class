import json
import shutil
from pathlib import Path

import pytest

from evaluation.stage9.sft_v2.export_blind_review_bundle import (
    BANNED_KEYS,
    DEFAULT_OUTPUT_DIR,
    EXPECTED_ROUTE_COUNTS,
    OUTPUT_FILE_NAMES,
    export_blind_review_bundle,
    validate_blind_review_bundle,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _all_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def test_sft_v2_blind_bundle_is_clean_and_complete(tmp_path):
    output_dir = tmp_path / "blind_review_bundle_v1"
    result = export_blind_review_bundle(
        output_dir=output_dir,
        generated_at="2026-08-01T00:00:00+00:00",
    )

    assert result["ok"] is True
    assert result["case_count"] == 125
    assert result["provider_record_count"] == 199
    assert result["route_counts"] == EXPECTED_ROUTE_COUNTS
    assert result["contamination_scan"] == "passed"
    assert {path.name for path in output_dir.iterdir()} == OUTPUT_FILE_NAMES

    review_rows = _read_jsonl(output_dir / "review_cases.jsonl")
    assert not (set(_all_keys(review_rows)) & BANNED_KEYS)
    assert all(row["content_fingerprint"] for row in review_rows)
    assert all(row["blind_case_fingerprint"] for row in review_rows)


def test_sft_v2_blind_bundle_refuses_overwrite_and_detects_pollution(tmp_path):
    output_dir = tmp_path / "blind_review_bundle_v1"
    export_blind_review_bundle(output_dir=output_dir)
    with pytest.raises(FileExistsError, match="拒绝静默覆盖"):
        export_blind_review_bundle(output_dir=output_dir)

    polluted_dir = tmp_path / "polluted"
    shutil.copytree(output_dir, polluted_dir)
    path = polluted_dir / "review_cases.jsonl"
    rows = _read_jsonl(path)
    rows[0]["reserve"] = True
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="禁止字段"):
        validate_blind_review_bundle(polluted_dir)


def test_checked_in_sft_v2_blind_bundle_is_valid():
    result = validate_blind_review_bundle(DEFAULT_OUTPUT_DIR)
    assert result["case_count"] == 125
    assert result["provider_record_count"] == 199
