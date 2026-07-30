"""校验 expanded dev（扩展开发集）的真实 Provider 回放契约。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.evaluation.action_providers import (  # noqa: E402
    ProviderObservationRecord,
    read_provider_observation_records,
)
from app.rag.evaluation.baseline_runner import load_environment_snapshot  # noqa: E402
from app.rag.evaluation.case_schema import (  # noqa: E402
    CaseSplit,
    ChunkRelevance,
    EnvironmentSnapshot,
    HumanReviewStatus,
    PlannerEvalCase,
    load_planner_cases,
)
from app.rag.query.contracts import (  # noqa: E402
    EvidenceSourceType,
    ObservationStatus,
    QueryAction,
    RetrievalCandidate,
    RetrievalObservation,
)
from evaluation.stage9.providers.record_expanded_dev_observations import (  # noqa: E402
    DEFAULT_CASES,
    DEFAULT_RECORDS,
    DEFAULT_SNAPSHOT,
    retrieval_actions_for_case,
)


REPLAY_CONTRACT_VERSION = "stage9-expanded-dev-replay-contract-v1"
EXPECTED_RECORDING_PROVIDER = "recording_action_provider"
EXPECTED_WRAPPED_PROVIDER = "milvus_action_provider"
DEFAULT_CONTRACT_OUTPUT = (
    PROJECT_ROOT
    / "evaluation/stage9/artifacts/provider_records/expanded_dev_replay_contract.json"
)
DEFAULT_REPORT_OUTPUT = (
    PROJECT_ROOT
    / "evaluation/stage9/artifacts/reports/阶段9-expanded-dev离线评测环境修复报告.md"
)


class ReplayContractModel(BaseModel):
    """9.3.18 回放契约公共 schema（数据结构），拒绝未知字段避免产物漂移。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CaseReplayCoverage(ReplayContractModel):
    """一条 case（评测样本）的标准查询动作覆盖情况。"""

    case_id: str = Field(min_length=1, description="评测样本 ID。")
    required_actions: list[QueryAction] = Field(
        description="本 case 的 acceptable_action_paths 实际需要冻结的检索动作。",
    )
    recorded_actions: list[QueryAction] = Field(
        description="使用 case.query 作为标准查询时已经找到的真实动作记录。",
    )
    missing_actions: list[QueryAction] = Field(
        description="缺失动作；完整 9.3.18 产物中必须为空。",
    )


class RouteObservationCheck(ReplayContractModel):
    """HyDE 或安全拒绝路线的真实 Observation（观察结果）检查。"""

    case_id: str = Field(min_length=1)
    route_type: str = Field(
        min_length=1,
        description="hyde_fallback 表示检索改善；safe_refuse 表示安全警告可见。",
    )
    local_target_rank: int | None = Field(
        default=None,
        ge=1,
        description="目标证据在 local_search 前五候选中的名次；未进入前五时为空。",
    )
    hyde_target_rank: int | None = Field(
        default=None,
        ge=1,
        description="目标证据在 hyde_search 前五候选中的名次；安全拒绝路线为空。",
    )
    passed: bool
    explanation: str = Field(min_length=1)


class ExpandedDevReplayContract(ReplayContractModel):
    """9.3.18 冻结的离线评测环境身份与逐路线校验结论。"""

    contract_version: str = REPLAY_CONTRACT_VERSION
    created_at: str
    ok: bool
    snapshot_id: str = Field(min_length=1, description="真实记录绑定的环境快照 ID。")
    records_path: str = Field(min_length=1, description="Provider（动作执行器）记录文件路径。")
    records_sha256: str = Field(
        min_length=64,
        max_length=64,
        description="记录文件 SHA256；模型复评必须绑定同一内容。",
    )
    wrapped_provider_name: str = Field(
        min_length=1,
        description="产生记录的真实内层执行器；9.3.18 固定为 milvus_action_provider。",
    )
    case_count: int = Field(ge=1)
    record_count: int = Field(ge=1)
    required_record_count: int = Field(
        ge=1,
        description="25 条标准 query 覆盖所有可接受检索路径所需的最少记录数。",
    )
    extra_record_count: int = Field(
        ge=0,
        description="标准 query 之外的显式 query 变体记录数；允许存在但仍须通过身份检查。",
    )
    case_coverage: list[CaseReplayCoverage]
    route_checks: list[RouteObservationCheck]


def validate_expanded_dev_replay(
        *,
        cases_path: str | Path,
        snapshot_path: str | Path,
        records_path: str | Path,
) -> ExpandedDevReplayContract:
    """
    验证真实记录是否足以重建可信的 expanded dev 离线评测环境。

    本函数只读 JSON/JSONL，不连接 Milvus、不加载 Planner 模型。它验证记录来源、快照、
    检索配置、候选身份、动作覆盖以及 HyDE/安全拒绝的关键前置条件。
    """

    records_file = Path(records_path)
    if not records_file.is_file():
        raise FileNotFoundError(f"expanded dev Provider 记录不存在：{records_file}")
    cases = _reviewed_dev_cases(cases_path)
    snapshot = load_environment_snapshot(snapshot_path)
    records = read_provider_observation_records(records_file)
    if not records:
        raise ValueError("expanded dev Provider 记录为空")

    case_by_id = {case.case_id: case for case in cases}
    expected_config_hash = _stable_hash(snapshot.retrieval_config_snapshot)
    indexed: dict[tuple[str, QueryAction, str], ProviderObservationRecord] = {}
    for record in records:
        case = case_by_id.get(record.case_id)
        if case is None:
            raise ValueError(f"Provider 记录包含非 reviewed dev case：{record.case_id}")
        if record.snapshot_id != snapshot.snapshot_id:
            raise ValueError(
                f"{record.case_id}/{record.action.value} snapshot_id 不一致："
                f"{record.snapshot_id} != {snapshot.snapshot_id}"
            )
        if record.provider_name != EXPECTED_RECORDING_PROVIDER:
            raise ValueError(
                f"{record.case_id}/{record.action.value} 不是由 RecordingActionProvider 记录："
                f"{record.provider_name}"
            )
        if record.wrapped_provider_name != EXPECTED_WRAPPED_PROVIDER:
            raise ValueError(
                f"{record.case_id}/{record.action.value} 不是由真实 Milvus Provider 产生："
                f"{record.wrapped_provider_name}"
            )
        if record.retrieval_config_version != snapshot.retrieval_config_version:
            raise ValueError(
                f"{record.case_id}/{record.action.value} retrieval_config_version 不一致"
            )
        if record.retrieval_config_hash != expected_config_hash:
            raise ValueError(
                f"{record.case_id}/{record.action.value} retrieval_config_hash 不一致"
            )
        _validate_record_payload(record, snapshot=snapshot)
        key = (record.case_id, record.action, record.query)
        if key in indexed:
            raise ValueError(
                f"Provider 记录存在重复 case/action/query："
                f"{record.case_id}/{record.action.value}/{record.query}"
            )
        indexed[key] = record

    coverage: list[CaseReplayCoverage] = []
    required_keys: set[tuple[str, QueryAction, str]] = set()
    for case in cases:
        required_actions = _required_actions(case)
        recorded_actions: list[QueryAction] = []
        missing_actions: list[QueryAction] = []
        for action in required_actions:
            key = (case.case_id, action, case.query)
            required_keys.add(key)
            if key in indexed:
                recorded_actions.append(action)
            else:
                missing_actions.append(action)
        coverage.append(
            CaseReplayCoverage(
                case_id=case.case_id,
                required_actions=required_actions,
                recorded_actions=recorded_actions,
                missing_actions=missing_actions,
            )
        )
    missing = [
        f"{item.case_id}:{action.value}"
        for item in coverage
        for action in item.missing_actions
    ]
    if missing:
        raise ValueError(f"expanded dev 回放缺少标准查询动作记录：{missing}")

    route_checks = [
        *_validate_hyde_routes(cases, indexed),
        *_validate_safe_refuse_routes(cases, indexed),
    ]
    route_contract_ok = all(check.passed for check in route_checks)

    return ExpandedDevReplayContract(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        ok=route_contract_ok,
        snapshot_id=snapshot.snapshot_id,
        records_path=_logical(records_file),
        records_sha256=_sha256(records_file),
        wrapped_provider_name=EXPECTED_WRAPPED_PROVIDER,
        case_count=len(cases),
        record_count=len(records),
        required_record_count=len(required_keys),
        extra_record_count=len(indexed) - len(required_keys),
        case_coverage=coverage,
        route_checks=route_checks,
    )


def write_replay_contract(
        contract: ExpandedDevReplayContract,
        *,
        output_path: str | Path,
        report_path: str | Path,
        overwrite: bool = False,
) -> None:
    """原子写出机器可读契约和中文报告；默认拒绝覆盖历史证据。"""

    output = Path(output_path)
    report = Path(report_path)
    for path in (output, report):
        if path.exists() and not overwrite:
            raise FileExistsError(f"9.3.18 产物已存在，拒绝静默覆盖：{path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(contract.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    report.write_text(render_replay_report(contract), encoding="utf-8")


def render_replay_report(contract: ExpandedDevReplayContract) -> str:
    """生成人工可读的 9.3.18 离线评测环境修复报告。"""

    hyde_checks = [
        check for check in contract.route_checks if check.route_type == "hyde_fallback"
    ]
    safety_checks = [
        check for check in contract.route_checks if check.route_type == "safe_refuse"
    ]
    route_rows = [
        "| "
        + " | ".join(
            [
                check.case_id,
                check.route_type,
                str(check.local_target_rank or "-"),
                str(check.hyde_target_rank or "-"),
                "通过" if check.passed else "失败",
                check.explanation,
            ]
        )
        + " |"
        for check in contract.route_checks
    ]
    return "\n".join(
        [
            "# 阶段 9 expanded dev 离线评测环境修复报告",
            "",
            "## 结论",
            "",
            f"- 契约版本：`{contract.contract_version}`。",
            f"- 校验状态：`{'通过' if contract.ok else '未通过'}`。",
            f"- 环境快照：`{contract.snapshot_id}`。",
            f"- 真实内层 Provider（动作执行器）：`{contract.wrapped_provider_name}`。",
            f"- reviewed dev（已审核开发集）：`{contract.case_count}` 条。",
            f"- 真实动作记录：`{contract.record_count}` 条；标准覆盖至少需要 `{contract.required_record_count}` 条。",
            f"- 记录 SHA256：`{contract.records_sha256}`。",
            "",
            "## 关键路线",
            "",
            f"- HyDE（假设式改写检索）：`{sum(check.passed for check in hyde_checks)}/{len(hyde_checks)}` "
            "满足“首次目标未进 Top 5、HyDE 后目标进入 Top 5”。",
            f"- safe_refuse（安全拒绝）：`{sum(check.passed for check in safety_checks)}/{len(safety_checks)}` "
            "能在 local_search 中看到来源手册警告证据。",
            "",
            "## 逐条结果",
            "",
            "| case_id | 路线 | local 目标名次 | HyDE 目标名次 | 结论 | 说明 |",
            "|---|---|---:|---:|---|---|",
            *route_rows,
            "",
            "## 边界",
            "",
            (
                "- 本报告证明真实检索记录满足关键路线契约，可以进入不可变 Replay（回放）。"
                if contract.ok
                else "- 真实检索记录本身完整，但关键路线契约未通过；不得用于模型准入复评。"
            ),
            "- 本报告不运行 SFT checkpoint，不代表模型已经掌握 HyDE 或安全拒绝。",
            "- 后续模型复评必须同时绑定本记录文件、环境 snapshot 和 SHA256。",
            "",
        ]
    )


def _reviewed_dev_cases(path: str | Path) -> list[PlannerEvalCase]:
    cases = sorted(
        (
            case
            for case in load_planner_cases(path)
            if case.split == CaseSplit.DEV
            and case.human_review_status == HumanReviewStatus.REVIEWED
        ),
        key=lambda case: case.case_id,
    )
    if len(cases) != 25:
        raise ValueError(f"expanded dev 必须恰好包含 25 条 reviewed dev，实际为 {len(cases)}")
    return cases


def _required_actions(case: PlannerEvalCase) -> list[QueryAction]:
    """复用录制器的路线动作选择，避免录制与校验两侧发生契约漂移。"""

    return retrieval_actions_for_case(case)


def _validate_record_payload(
        record: ProviderObservationRecord,
        *,
        snapshot: EnvironmentSnapshot,
) -> None:
    if record.error is not None:
        raise ValueError(
            f"{record.case_id}/{record.action.value} 真实 Provider 执行失败：{record.error}"
        )
    observation = RetrievalObservation.model_validate(record.observation)
    if observation.action != record.action:
        raise ValueError(f"{record.case_id}/{record.action.value} Observation.action 不一致")
    if observation.candidate_count != record.candidate_count:
        raise ValueError(f"{record.case_id}/{record.action.value} candidate_count 不一致")
    expected_status = (
        ObservationStatus.SUCCESS if record.candidate_count else ObservationStatus.EMPTY
    )
    if observation.status != expected_status:
        raise ValueError(
            f"{record.case_id}/{record.action.value} Observation.status 与候选数量不一致"
        )
    if any(bool(item.get("content_truncated")) for item in record.candidates):
        raise ValueError(
            f"{record.case_id}/{record.action.value} 候选正文被截断，不能作为正式 Replay"
        )
    candidates = [_candidate(item) for item in record.candidates]
    if len(candidates) != record.candidate_count:
        raise ValueError(f"{record.case_id}/{record.action.value} candidates 数量不一致")
    if observation.reranked_count != record.candidate_count:
        raise ValueError(f"{record.case_id}/{record.action.value} reranked_count 不一致")
    expected_top_score = (
        max(
            candidate.rerank_score
            if candidate.rerank_score is not None
            else min(1.0, float(candidate.retrieval_score))
            for candidate in candidates
        )
        if candidates
        else None
    )
    if observation.top_rerank_score != expected_top_score:
        raise ValueError(f"{record.case_id}/{record.action.value} top_rerank_score 不一致")

    enabled = {
        (document_id, str(chunk_id))
        for document_id, chunk_ids in snapshot.enabled_chunks.items()
        for chunk_id in chunk_ids
    }
    document_version = {
        document.document_id: document.index_version
        for document in snapshot.documents
    }
    for candidate in candidates:
        if record.action == QueryAction.WEB_SEARCH:
            if candidate.source_type != EvidenceSourceType.WEB:
                raise ValueError(f"{record.case_id}/web_search 包含非 Web 候选")
            continue
        if candidate.source_type != EvidenceSourceType.LOCAL:
            raise ValueError(
                f"{record.case_id}/{record.action.value} 包含非本地 Milvus 候选"
            )
        identity = (str(candidate.document_id), str(candidate.chunk_id))
        if identity not in enabled:
            raise ValueError(
                f"{record.case_id}/{record.action.value} 候选不属于冻结 enabled_chunks：{identity}"
            )
        if candidate.index_version != document_version.get(str(candidate.document_id)):
            raise ValueError(
                f"{record.case_id}/{record.action.value} 候选 index_version 不属于冻结文档版本"
            )


def _validate_hyde_routes(
        cases: list[PlannerEvalCase],
        records: dict[tuple[str, QueryAction, str], ProviderObservationRecord],
) -> list[RouteObservationCheck]:
    checks: list[RouteObservationCheck] = []
    for case in cases:
        if (
            not case.acceptable_action_paths
            or not all(
                QueryAction.HYDE_SEARCH in path
                for path in case.acceptable_action_paths
            )
        ):
            continue
        local = records[(case.case_id, QueryAction.LOCAL_SEARCH, case.query)]
        hyde = records[(case.case_id, QueryAction.HYDE_SEARCH, case.query)]
        local_rank = _target_rank(local, case)
        hyde_rank = _target_rank(hyde, case)
        passed = local_rank is None and hyde_rank is not None
        checks.append(
            RouteObservationCheck(
                case_id=case.case_id,
                route_type="hyde_fallback",
                local_target_rank=local_rank,
                hyde_target_rank=hyde_rank,
                passed=passed,
                explanation=(
                    "local_search 目标未进 Top 5，hyde_search 目标进入 Top 5"
                    if passed
                    else
                    f"期望 local=None 且 hyde<=5，实际 local={local_rank}, hyde={hyde_rank}"
                ),
            )
        )
    if len(checks) != 5:
        raise ValueError(f"expanded dev HyDE case 数量必须为 5，实际为 {len(checks)}")
    return checks


def _validate_safe_refuse_routes(
        cases: list[PlannerEvalCase],
        records: dict[tuple[str, QueryAction, str], ProviderObservationRecord],
) -> list[RouteObservationCheck]:
    checks: list[RouteObservationCheck] = []
    for case in cases:
        if not case.expected_behavior.should_refuse:
            continue
        local = records[(case.case_id, QueryAction.LOCAL_SEARCH, case.query)]
        local_rank = _target_rank(local, case)
        passed = local_rank is not None
        checks.append(
            RouteObservationCheck(
                case_id=case.case_id,
                route_type="safe_refuse",
                local_target_rank=local_rank,
                passed=passed,
                explanation=(
                    "local_search 前五候选包含来源手册安全证据"
                    if passed
                    else "local_search 前五候选没有来源手册安全证据"
                ),
            )
        )
    if len(checks) != 5:
        raise ValueError(f"expanded dev safe_refuse case 数量必须为 5，实际为 {len(checks)}")
    return checks


def _target_rank(record: ProviderObservationRecord, case: PlannerEvalCase) -> int | None:
    """
    返回 required chunk（必需证据）的前五名次。

    supporting chunk（辅助证据）可以帮助模型理解同义词，但不能代表路线目标已经被
    local_search 命中。例如 P5 的“注册申请码就是 S/N”只能解释用户在问什么，真正回答
    “怎么打印信息页”的操作 chunk 仍必须由后续 HyDE 找到。
    """

    expected = {
        (chunk.document_id, str(chunk.chunk_id), chunk.index_version)
        for chunk in case.expected_chunks
        if chunk.relevance == ChunkRelevance.REQUIRED
    }
    if not expected:
        raise ValueError(f"{case.case_id} 没有 required expected chunk，无法校验路线")
    for rank, payload in enumerate(record.candidates[:5], start=1):
        candidate = _candidate(payload)
        identity = (
            str(candidate.document_id),
            str(candidate.chunk_id),
            candidate.index_version,
        )
        if identity in expected:
            return rank
    return None


def _candidate(payload: dict[str, Any]) -> RetrievalCandidate:
    normalized = dict(payload)
    normalized.pop("content_truncated", None)
    return RetrievalCandidate.model_validate(normalized)


def _stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _logical(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="校验 expanded dev 的真实 Provider 回放契约，不运行模型。",
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_CONTRACT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    contract = validate_expanded_dev_replay(
        cases_path=args.cases,
        snapshot_path=args.snapshot,
        records_path=args.records,
    )
    write_replay_contract(
        contract,
        output_path=args.output,
        report_path=args.report,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "ok": contract.ok,
                "contract_version": contract.contract_version,
                "case_count": contract.case_count,
                "record_count": contract.record_count,
                "required_record_count": contract.required_record_count,
                "records_sha256": contract.records_sha256,
                "output": str(args.output),
                "report": str(args.report),
            },
            ensure_ascii=False,
        )
    )
    return 0 if contract.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
