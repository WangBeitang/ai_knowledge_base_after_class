import hashlib
import json
from pathlib import Path

import pytest

from app.rag.evaluation.action_providers import (
    ProviderObservationRecord,
    ReplayActionProvider,
    read_provider_observation_records,
)
from app.rag.evaluation.baseline_runner import load_environment_snapshot
from app.rag.evaluation.case_schema import PlannerEvalCase, load_planner_cases
from app.rag.query.contracts import (
    EvidenceSourceType,
    ObservationStatus,
    QueryAction,
    RetrievalCandidate,
    RetrievalChannel,
    RetrievalObservation,
)
from evaluation.stage9.providers.record_expanded_dev_observations import (
    DEFAULT_CASES,
    DEFAULT_SNAPSHOT,
    record_expanded_dev_observations,
)
from evaluation.stage9.providers.validate_expanded_dev_replay import (
    REPLAY_CONTRACT_VERSION,
    validate_expanded_dev_replay,
)
from evaluation.stage9.model_planner.eval_model_planner import (
    REPLAY_PROVIDER_NAME,
    _build_provider,
)


def test_expanded_dev_replay_contract_accepts_complete_real_records(tmp_path: Path):
    records = _write_complete_records(tmp_path)

    contract = validate_expanded_dev_replay(
        cases_path=DEFAULT_CASES,
        snapshot_path=DEFAULT_SNAPSHOT,
        records_path=records,
    )

    assert contract.contract_version == REPLAY_CONTRACT_VERSION
    assert contract.ok is True
    assert contract.case_count == 25
    assert contract.record_count == 55
    assert contract.required_record_count == 55
    assert contract.extra_record_count == 0
    assert len(contract.route_checks) == 10
    assert all(check.passed for check in contract.route_checks)
    assert all(not coverage.missing_actions for coverage in contract.case_coverage)


def test_expanded_dev_replay_contract_rejects_hyde_target_leaked_into_local(
    tmp_path: Path,
):
    cases = _dev_cases()
    hyde_case = next(case for case in cases if _is_strict_hyde(case))
    records = _write_complete_records(
        tmp_path,
        local_override={
            hyde_case.case_id: _local_candidate(hyde_case, hyde_case.expected_chunks[0])
        },
    )

    with pytest.raises(ValueError, match="路线 Observation 契约不成立"):
        validate_expanded_dev_replay(
            cases_path=DEFAULT_CASES,
            snapshot_path=DEFAULT_SNAPSHOT,
            records_path=records,
        )


def test_expanded_dev_replay_contract_rejects_missing_standard_action(
    tmp_path: Path,
):
    records = _write_complete_records(tmp_path)
    lines = records.read_text(encoding="utf-8").splitlines()
    records.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="缺少标准查询动作记录"):
        validate_expanded_dev_replay(
            cases_path=DEFAULT_CASES,
            snapshot_path=DEFAULT_SNAPSHOT,
            records_path=records,
        )


def test_expanded_dev_replay_contract_rejects_missing_safety_warning(
    tmp_path: Path,
):
    cases = _dev_cases()
    safety_case = next(
        case for case in cases if case.expected_behavior.should_refuse
    )
    records = _write_complete_records(
        tmp_path,
        local_override={
            safety_case.case_id: _generic_local_candidate(safety_case)
        },
    )

    with pytest.raises(ValueError, match="安全证据"):
        validate_expanded_dev_replay(
            cases_path=DEFAULT_CASES,
            snapshot_path=DEFAULT_SNAPSHOT,
            records_path=records,
        )


def test_expanded_dev_replay_contract_rejects_non_milvus_source(tmp_path: Path):
    records = _write_complete_records(tmp_path)
    payloads = [
        json.loads(line)
        for line in records.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    payloads[0]["wrapped_provider_name"] = "snapshot_expected_chunks"
    records.write_text(
        "\n".join(
            json.dumps(payload, ensure_ascii=False, sort_keys=True)
            for payload in payloads
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="不是由真实 Milvus Provider 产生"):
        validate_expanded_dev_replay(
            cases_path=DEFAULT_CASES,
            snapshot_path=DEFAULT_SNAPSHOT,
            records_path=records,
        )


def test_expanded_dev_replay_contract_rejects_non_recording_provider(
    tmp_path: Path,
):
    records = _write_complete_records(tmp_path)
    payloads = [
        json.loads(line)
        for line in records.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    payloads[0]["provider_name"] = "handwritten_fixture"
    records.write_text(
        "\n".join(
            json.dumps(payload, ensure_ascii=False, sort_keys=True)
            for payload in payloads
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="不是由 RecordingActionProvider 记录"):
        validate_expanded_dev_replay(
            cases_path=DEFAULT_CASES,
            snapshot_path=DEFAULT_SNAPSHOT,
            records_path=records,
        )


def test_expanded_dev_replay_contract_rejects_observation_score_drift(
    tmp_path: Path,
):
    records = _write_complete_records(tmp_path)
    payloads = [
        json.loads(line)
        for line in records.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    payloads[0]["observation"]["top_rerank_score"] = 0.123456
    records.write_text(
        "\n".join(
            json.dumps(payload, ensure_ascii=False, sort_keys=True)
            for payload in payloads
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="top_rerank_score 不一致"):
        validate_expanded_dev_replay(
            cases_path=DEFAULT_CASES,
            snapshot_path=DEFAULT_SNAPSHOT,
            records_path=records,
        )


def test_expanded_dev_replay_contract_rejects_truncated_candidates(
    tmp_path: Path,
):
    records = _write_complete_records(tmp_path)
    payloads = [
        json.loads(line)
        for line in records.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    payloads[0]["candidates"][0]["content_truncated"] = True
    records.write_text(
        "\n".join(
            json.dumps(payload, ensure_ascii=False, sort_keys=True)
            for payload in payloads
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="候选正文被截断"):
        validate_expanded_dev_replay(
            cases_path=DEFAULT_CASES,
            snapshot_path=DEFAULT_SNAPSHOT,
            records_path=records,
        )


def test_model_planner_eval_builds_strict_replay_provider(tmp_path: Path):
    records = _write_complete_records(tmp_path)
    cases = _dev_cases()
    snapshot = load_environment_snapshot(DEFAULT_SNAPSHOT)

    provider = _build_provider(
        REPLAY_PROVIDER_NAME,
        cases,
        snapshot,
        provider_records_path=records,
    )

    assert isinstance(provider, ReplayActionProvider)
    with pytest.raises(ValueError, match="provider_records_path"):
        _build_provider(REPLAY_PROVIDER_NAME, cases, snapshot)


def test_expanded_dev_recorder_covers_local_and_hyde_without_terminal_action(
    tmp_path: Path,
):
    output = tmp_path / "recorded.jsonl"

    counts = record_expanded_dev_observations(
        cases_path=DEFAULT_CASES,
        snapshot_path=DEFAULT_SNAPSHOT,
        output_path=output,
        chunk_status_filter_enabled=False,
        max_cases=1,
        action_provider=_FakeMilvusProvider(),
    )
    records = read_provider_observation_records(output)

    assert counts["case_count"] == 1
    assert counts["record_count"] == 2
    assert counts["error_count"] == 0
    assert [record.action for record in records] == [
        QueryAction.LOCAL_SEARCH,
        QueryAction.HYDE_SEARCH,
    ]
    assert all(
        record.wrapped_provider_name == "milvus_action_provider"
        for record in records
    )
    assert all(
        record.candidates[0]["content_truncated"] is False
        and len(record.candidates[0]["content"]) > 500
        for record in records
    )


def _write_complete_records(
    tmp_path: Path,
    *,
    local_override: dict[str, RetrievalCandidate] | None = None,
) -> Path:
    cases = _dev_cases()
    snapshot = load_environment_snapshot(DEFAULT_SNAPSHOT)
    config_hash = _stable_hash(snapshot.retrieval_config_snapshot)
    records: list[ProviderObservationRecord] = []
    for case in cases:
        generic = _generic_local_candidate(case)
        local_candidate = (
            (local_override or {}).get(case.case_id)
            or (
                generic
                if _is_strict_hyde(case)
                else (
                    _local_candidate(case, case.expected_chunks[0])
                    if case.expected_chunks
                    else generic
                )
            )
        )
        hyde_candidate = (
            _local_candidate(
                case,
                case.expected_chunks[0],
                channel=RetrievalChannel.HYDE,
            )
            if _is_strict_hyde(case)
            else generic.model_copy(deep=True)
        )
        records.append(
            _record(
                case,
                QueryAction.LOCAL_SEARCH,
                [local_candidate],
                snapshot_id=snapshot.snapshot_id,
                retrieval_config_version=snapshot.retrieval_config_version,
                retrieval_config_hash=config_hash,
            )
        )
        records.append(
            _record(
                case,
                QueryAction.HYDE_SEARCH,
                [hyde_candidate],
                snapshot_id=snapshot.snapshot_id,
                retrieval_config_version=snapshot.retrieval_config_version,
                retrieval_config_hash=config_hash,
            )
        )
        if case.expected_behavior.should_call_web:
            records.append(
                _record(
                    case,
                    QueryAction.WEB_SEARCH,
                    [_web_candidate(case)],
                    snapshot_id=snapshot.snapshot_id,
                    retrieval_config_version=snapshot.retrieval_config_version,
                    retrieval_config_hash=config_hash,
                )
            )

    output = tmp_path / "expanded_dev_provider_observations.jsonl"
    output.write_text(
        "\n".join(
            json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
            for record in records
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def _record(
    case: PlannerEvalCase,
    action: QueryAction,
    candidates: list[RetrievalCandidate],
    *,
    snapshot_id: str,
    retrieval_config_version: str,
    retrieval_config_hash: str,
) -> ProviderObservationRecord:
    observation = RetrievalObservation(
        action=action,
        status=ObservationStatus.SUCCESS if candidates else ObservationStatus.EMPTY,
        channel_counts={action.value: len(candidates)},
        candidate_count=len(candidates),
        reranked_count=len(candidates),
        top_rerank_score=max(
            (candidate.rerank_score or candidate.retrieval_score)
            for candidate in candidates
        ) if candidates else None,
    )
    return ProviderObservationRecord(
        record_id=f"record-{case.case_id}-{action.value}",
        recorded_at="2026-07-30T00:00:00+00:00",
        provider_name="recording_action_provider",
        wrapped_provider_name="milvus_action_provider",
        run_id=f"run-{case.case_id}",
        case_id=case.case_id,
        snapshot_id=snapshot_id,
        step={
            QueryAction.LOCAL_SEARCH: 1,
            QueryAction.HYDE_SEARCH: 2,
            QueryAction.WEB_SEARCH: 3,
        }[action],
        action=action,
        query=case.query,
        retrieval_config_version=retrieval_config_version,
        retrieval_config_hash=retrieval_config_hash,
        candidate_count=len(candidates),
        candidates=[
            {**candidate.model_dump(mode="json"), "content_truncated": False}
            for candidate in candidates
        ],
        observation=observation.model_dump(mode="json"),
        duration_ms=1,
    )


def _dev_cases() -> list[PlannerEvalCase]:
    return sorted(
        (
            case
            for case in load_planner_cases(DEFAULT_CASES)
            if case.split.value == "dev"
        ),
        key=lambda case: case.case_id,
    )


def _is_strict_hyde(case: PlannerEvalCase) -> bool:
    return bool(case.acceptable_action_paths) and all(
        QueryAction.HYDE_SEARCH in path
        for path in case.acceptable_action_paths
    )


def _generic_local_candidate(case: PlannerEvalCase) -> RetrievalCandidate:
    snapshot = load_environment_snapshot(DEFAULT_SNAPSHOT)
    expected = {
        (chunk.document_id, str(chunk.chunk_id))
        for chunk in case.expected_chunks
    }
    versions = {
        document.document_id: document.index_version
        for document in snapshot.documents
    }
    for document_id, chunk_ids in snapshot.enabled_chunks.items():
        for chunk_id in chunk_ids:
            if (document_id, str(chunk_id)) not in expected:
                return RetrievalCandidate(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    dataset_id=case.dataset_ids[0],
                    index_version=versions[document_id],
                    chunk_index=0,
                    enabled=True,
                    title="非目标真实候选",
                    content="与当前问题相关但不足以支持目标答案的候选内容。",
                    source_type=EvidenceSourceType.LOCAL,
                    retrieval_channels=[RetrievalChannel.ORIGINAL],
                    retrieval_rank=1,
                    retrieval_score=0.1,
                    rerank_score=0.1,
                )
    raise AssertionError("测试 snapshot 没有可用的非目标 chunk")


def _local_candidate(
    case: PlannerEvalCase,
    expected_chunk,
    *,
    channel: RetrievalChannel = RetrievalChannel.ORIGINAL,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        document_id=expected_chunk.document_id,
        chunk_id=expected_chunk.chunk_id,
        dataset_id=case.dataset_ids[0],
        index_version=expected_chunk.index_version,
        chunk_index=0,
        enabled=True,
        title="目标真实候选",
        content="来源手册中的目标证据内容。",
        source_type=EvidenceSourceType.LOCAL,
        retrieval_channels=[channel],
        retrieval_rank=1,
        retrieval_score=0.9,
        rerank_score=0.9,
    )


def _web_candidate(case: PlannerEvalCase) -> RetrievalCandidate:
    evidence = case.expected_web_evidence[0]
    return RetrievalCandidate(
        title=evidence.source_title,
        source_title=evidence.source_title,
        content="已冻结的真实网页候选摘要。",
        source_type=EvidenceSourceType.WEB,
        retrieval_channels=[RetrievalChannel.WEB],
        retrieval_rank=1,
        retrieval_score=0.9,
        rerank_score=0.9,
        url=evidence.url,
    )


def _stable_hash(payload: dict) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class _FakeMilvusProvider:
    provider_name = "milvus_action_provider"

    def local_search(self, state, decision):
        return [self._candidate(state, RetrievalChannel.ORIGINAL)]

    def hyde_search(self, state, decision):
        return [self._candidate(state, RetrievalChannel.HYDE)]

    def web_search(self, state, decision):
        raise AssertionError("max_cases=1 的首条 direct ask case 不应录制 Web")

    @staticmethod
    def _candidate(state, channel):
        document_id = next(iter(state.snapshot.enabled_chunks))
        chunk_id = state.snapshot.enabled_chunks[document_id][0]
        version = next(
            document.index_version
            for document in state.snapshot.documents
            if document.document_id == document_id
        )
        return RetrievalCandidate(
            document_id=document_id,
            chunk_id=chunk_id,
            dataset_id=state.dataset_ids[0],
            index_version=version,
            chunk_index=0,
            enabled=True,
            title="测试真实候选",
            content="用于验证录制动作覆盖，不代表真实召回质量。" * 40,
            source_type=EvidenceSourceType.LOCAL,
            retrieval_channels=[channel],
            retrieval_rank=1,
            retrieval_score=0.5,
            rerank_score=0.5,
        )
