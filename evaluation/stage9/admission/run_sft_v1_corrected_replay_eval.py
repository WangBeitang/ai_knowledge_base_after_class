"""任务 9.3.20：使用冻结 Replay（回放）校正 SFT v1（监督微调第一版）事实基线。

本入口复用 9.3.16 的 checkpoint（检查点）、25 条 reviewed dev（已审核开发集）、
Reward v1.1（奖励函数第一点一版）和阈值，只把旧的标签派生 Provider（动作执行器）
替换为 9.3.18 冻结的真实 Observation（观察结果）回放。

它不会覆盖 9.3.16 原始输出，不运行 heldout test（留出测试），也不会直接放行 9.4。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

from app.rag.evaluation.action_providers import (
    ProviderObservationRecord,
    read_provider_observation_records,
)
from app.rag.evaluation.baseline_runner import (
    SNAPSHOT_EXPECTED_PROVIDER_NAME,
    BaselineEvalOutput,
    load_environment_snapshot,
)
from app.rag.evaluation.case_schema import (
    CaseSplit,
    HumanReviewStatus,
    PlannerEvalCase,
    PlannerMode,
    SplitManifest,
    load_planner_cases,
)
from app.rag.evaluation.reward import REWARD_VERSION, RewardConfig
from app.rag.query.contracts import QueryAction
from app.rag.query.model_planner.checkpoint_runtime import (
    CheckpointManifest,
    load_checkpoint_manifest,
)
from evaluation.stage9.admission.run_sft_expanded_dev_gate import (
    DEFAULT_CASES,
    DEFAULT_REWARD_IMPLEMENTATION,
    DEFAULT_REWARD_PROFILE,
    DEFAULT_ROUTE_MATRIX,
    DEFAULT_SNAPSHOT,
    DEFAULT_SPLIT_MANIFEST,
    BucketAdmission,
    CaseAdmission,
    CheckpointIdentity,
    FrozenFile,
    GateCheck,
    _bucket_admissions,
    _canonical_cases_sha256,
    _case_admission,
    _checkpoint_reproduction_files,
    _file_record,
    _gate_checks,
    _logical,
    _read_json,
    _reward_config_from_profile,
    _route_bucket,
    _same_path,
    _sha256 as _admission_sha256,
    _thresholds_from_matrix,
    _validate_checkpoint,
)
from evaluation.stage9.model_planner.audit_eval_route_coverage import RouteBucket
from evaluation.stage9.model_planner.checkpoint_io import current_code_version
from evaluation.stage9.model_planner.eval_model_planner import (
    REPLAY_PROVIDER_NAME,
    run_model_planner_eval,
)
from evaluation.stage9.providers.record_expanded_dev_observations import (
    DEFAULT_RECORDS,
)
from evaluation.stage9.providers.validate_expanded_dev_replay import (
    DEFAULT_CONTRACT_OUTPUT,
    ExpandedDevReplayContract,
    validate_expanded_dev_replay,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CORRECTED_REPLAY_VERSION = "stage9-sft-v1-corrected-replay-eval-v1"
DEFAULT_OLD_EVAL = (
    PROJECT_ROOT / "evaluation/stage9/artifacts/sft/sft_expanded_dev_eval.json"
)
DEFAULT_OLD_DECISION = (
    PROJECT_ROOT
    / "evaluation/stage9/artifacts/sft/sft_9_4_admission_decision.json"
)
DEFAULT_REWARD_REGRESSION = (
    PROJECT_ROOT
    / "evaluation/stage9/artifacts/reward/reward_v1_1_replay_regression.json"
)
DEFAULT_CORRECTED_EVAL = (
    PROJECT_ROOT
    / "evaluation/stage9/artifacts/sft/sft_v1_corrected_replay_eval.json"
)
DEFAULT_COMPARISON = (
    PROJECT_ROOT
    / "evaluation/stage9/artifacts/sft/sft_v1_corrected_replay_comparison.json"
)
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "evaluation/stage9/artifacts/reports/阶段9-SFT-v1校正复评报告.md"
)
RETRIEVAL_ACTIONS = {
    QueryAction.LOCAL_SEARCH,
    QueryAction.HYDE_SEARCH,
    QueryAction.WEB_SEARCH,
}


class CorrectedReplayModel(BaseModel):
    """9.3.20 输出公共 schema（数据结构）；拒绝未知字段防止产物漂移。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class AttributionCategory(str, Enum):
    """新旧结果变化的责任分类及触发场景。"""

    UNCHANGED_PASS = "unchanged_pass"
    PROVIDER_FALSE_NEGATIVE_CORRECTED = "provider_false_negative_corrected"
    PERSISTENT_MODEL_FAILURE = "persistent_model_failure"
    REPLAY_EXPOSED_FAILURE = "replay_exposed_failure"
    REPLAY_EXECUTION_FAILURE = "replay_execution_failure"


class ObservationEvidence(CorrectedReplayModel):
    """一条实际进入新轨迹的结构化 Observation（观察结果）证据。"""

    step: int = Field(ge=1, description="轨迹步骤号，从 1 开始。")
    action: str = Field(min_length=1, description="本步执行的检索动作。")
    query: str = Field(min_length=1, description="模型本步实际提交给回放执行器的查询。")
    status: str = Field(min_length=1, description="观察结果状态，例如 success、empty 或 failed。")
    candidate_count: int = Field(ge=0, description="本步回放返回的候选数量。")
    top_rerank_score: float | None = Field(
        default=None,
        description="观察结果中的最高重排序分数；没有候选时为空。",
    )
    error_code: str | None = Field(
        default=None,
        description="执行失败错误码；正常执行时为空。",
    )
    record_id: str | None = Field(
        default=None,
        description="与本步 case、动作和查询完全匹配的 9.3.18 记录 ID。",
    )
    retrieved_chunk_ids: list[str] = Field(
        default_factory=list,
        description="本步真实回放候选中的本地文本块 ID。",
    )
    web_urls: list[str] = Field(
        default_factory=list,
        description="本步真实回放候选中的网页 URL。",
    )


class LegacyObservationProjection(CorrectedReplayModel):
    """9.3.16 旧结果能够恢复出的 Observation（观察结果）最小投影。"""

    provider_name: str = Field(min_length=1)
    full_observation_available: bool = False
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    citation_chunk_ids: list[str] = Field(default_factory=list)
    note: str = Field(
        default=(
            "9.3.16 原始结果没有保存完整逐步 Observation；这里只保留旧结果中的"
            " retrieved_chunk_ids 与 citation_chunk_ids，不伪造候选和分数。"
        )
    )


class CaseComparison(CorrectedReplayModel):
    """一条开发样本的新旧路线、观察结果和责任归因。"""

    case_id: str = Field(min_length=1)
    route_bucket: RouteBucket
    expected_action_paths: list[list[str]]
    old_action_path: list[str]
    new_action_path: list[str]
    old_terminal_action: str | None
    new_terminal_action: str | None
    old_route_correct: bool
    new_route_correct: bool
    old_case_gate_passed: bool
    new_case_gate_passed: bool
    old_failure_categories: list[str]
    new_failure_categories: list[str]
    old_observation: LegacyObservationProjection
    new_observations: list[ObservationEvidence]
    attribution: AttributionCategory
    attribution_explanation: str = Field(min_length=1)


class CorrectedReplaySummary(CorrectedReplayModel):
    """9.3.20 汇总结论；只校正事实基线，不承担 9.4 准入。"""

    decision: str = "corrected_baseline_only"
    eligible_for_stage9_4: bool = False
    case_count: int = Field(ge=1)
    old_route_correct_count: int = Field(ge=0)
    new_route_correct_count: int = Field(ge=0)
    route_correct_delta: int
    old_route_macro_accuracy: float = Field(ge=0, le=1)
    new_route_macro_accuracy: float = Field(ge=0, le=1)
    changed_action_path_count: int = Field(ge=0)
    new_execution_failure_count: int = Field(ge=0)
    new_failed_case_ids: list[str]
    attribution_counts: dict[str, int]
    provider_false_negative_corrected_case_ids: list[str]
    persistent_model_failure_case_ids: list[str]
    replay_exposed_failure_case_ids: list[str]
    replay_execution_failure_case_ids: list[str]
    next_step: str = (
        "人工确认逐 case 归因后，9.3.21 只能按修正后仍成立的失败补独立 "
        "train-only（仅训练）数据；本结果不得直接放行 9.4。"
    )


class CorrectedReplayEvaluation(CorrectedReplayModel):
    """9.3.20 的机器可读新旧对比与证据边界。"""

    evaluation_version: str = CORRECTED_REPLAY_VERSION
    evaluated_at: str
    old_eval_run_id: str = Field(min_length=1)
    new_eval_run_id: str = Field(min_length=1)
    selected_split: str = "dev"
    snapshot_id: str = Field(min_length=1)
    reward_version: str = Field(min_length=1)
    action_provider: str = "replay_action_provider"
    model_execution_performed: bool = True
    heldout_inference_result_count: int = 0
    balanced_dev_canonical_sha256: str = Field(min_length=64, max_length=64)
    checkpoint: CheckpointIdentity
    inputs: dict[str, FrozenFile]
    frozen_thresholds: dict[str, float | int]
    replay_gate_checks: list[GateCheck]
    replay_buckets: list[BucketAdmission]
    cases: list[CaseComparison]
    summary: CorrectedReplaySummary


class LegacyAdmissionEvidence(CorrectedReplayModel):
    """9.3.16 冻结决定中 9.3.20 需要的最小只读证据。"""

    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    eval_run_id: str = Field(min_length=1)
    action_provider: str = Field(min_length=1)
    heldout_inference_result_count: int = 0
    checkpoint: CheckpointIdentity
    cases: list[CaseAdmission] = Field(min_length=1)


@dataclass(frozen=True)
class CorrectedInputContract:
    """9.3.19 回归通过后可用于 9.3.20 的当前输入合同。"""

    all_case_count: int
    selected_cases: tuple[PlannerEvalCase, ...]
    balanced_dev_canonical_sha256: str
    snapshot_id: str
    reward_config: RewardConfig
    reward_profile_name: str
    thresholds: dict[str, float | int]
    checkpoint_manifest: CheckpointManifest
    checkpoint_dir: Path
    inputs: dict[str, FrozenFile]


@dataclass(frozen=True)
class CorrectedReplayPreflight:
    """模型加载前冻结的 9.3.20 输入；只存在于运行内存。"""

    contract: CorrectedInputContract
    old_eval: BaselineEvalOutput
    old_cases: tuple[CaseAdmission, ...]
    old_buckets: tuple[BucketAdmission, ...]
    replay_contract: ExpandedDevReplayContract
    records: tuple[ProviderObservationRecord, ...]


def load_corrected_replay_preflight(
    *,
    checkpoint_dir: Path,
    old_eval_path: Path = DEFAULT_OLD_EVAL,
    old_decision_path: Path = DEFAULT_OLD_DECISION,
    cases_path: Path = DEFAULT_CASES,
    split_manifest_path: Path = DEFAULT_SPLIT_MANIFEST,
    snapshot_path: Path = DEFAULT_SNAPSHOT,
    route_matrix_path: Path = DEFAULT_ROUTE_MATRIX,
    reward_profile_path: Path = DEFAULT_REWARD_PROFILE,
    reward_regression_path: Path = DEFAULT_REWARD_REGRESSION,
    reward_implementation_path: Path = DEFAULT_REWARD_IMPLEMENTATION,
    provider_records_path: Path = DEFAULT_RECORDS,
    replay_contract_path: Path = DEFAULT_CONTRACT_OUTPUT,
) -> CorrectedReplayPreflight:
    """只读校验旧评测、检查点、开发集和 9.3.18 回放身份。"""

    contract = load_corrected_input_contract(
        checkpoint_dir=checkpoint_dir,
        cases_path=cases_path,
        split_manifest_path=split_manifest_path,
        snapshot_path=snapshot_path,
        route_matrix_path=route_matrix_path,
        reward_profile_path=reward_profile_path,
        reward_regression_path=reward_regression_path,
        reward_implementation_path=reward_implementation_path,
        provider_records_path=provider_records_path,
        replay_contract_path=replay_contract_path,
    )
    if not old_eval_path.is_file():
        raise FileNotFoundError(f"9.3.16 原始评测不存在：{old_eval_path}")
    old_eval = BaselineEvalOutput.model_validate_json(
        old_eval_path.read_text(encoding="utf-8")
    )
    if not old_decision_path.is_file():
        raise FileNotFoundError(f"9.3.16 原始准入决定不存在：{old_decision_path}")
    old_admission = LegacyAdmissionEvidence.model_validate_json(
        old_decision_path.read_text(encoding="utf-8")
    )
    _validate_legacy_evidence(
        old_eval=old_eval,
        old_admission=old_admission,
        contract=contract,
    )

    if not replay_contract_path.is_file():
        raise FileNotFoundError(f"9.3.18 回放契约不存在：{replay_contract_path}")
    frozen_replay = ExpandedDevReplayContract.model_validate_json(
        replay_contract_path.read_text(encoding="utf-8")
    )
    current_replay = validate_expanded_dev_replay(
        cases_path=cases_path,
        snapshot_path=snapshot_path,
        records_path=provider_records_path,
    )
    _validate_replay_binding(
        frozen=frozen_replay,
        current=current_replay,
        provider_records_path=provider_records_path,
    )
    return CorrectedReplayPreflight(
        contract=contract,
        old_eval=old_eval,
        old_cases=tuple(old_admission.cases),
        old_buckets=tuple(
            _bucket_admissions(
                old_admission.cases,
                threshold=float(
                    contract.thresholds["per_route_bucket_accuracy_min"]
                ),
            )
        ),
        replay_contract=frozen_replay,
        records=tuple(read_provider_observation_records(provider_records_path)),
    )


def load_corrected_input_contract(
    *,
    checkpoint_dir: Path,
    cases_path: Path = DEFAULT_CASES,
    split_manifest_path: Path = DEFAULT_SPLIT_MANIFEST,
    snapshot_path: Path = DEFAULT_SNAPSHOT,
    route_matrix_path: Path = DEFAULT_ROUTE_MATRIX,
    reward_profile_path: Path = DEFAULT_REWARD_PROFILE,
    reward_regression_path: Path = DEFAULT_REWARD_REGRESSION,
    reward_implementation_path: Path = DEFAULT_REWARD_IMPLEMENTATION,
    provider_records_path: Path = DEFAULT_RECORDS,
    replay_contract_path: Path = DEFAULT_CONTRACT_OUTPUT,
) -> CorrectedInputContract:
    """使用 9.3.19 回归产物绑定当前 25 条开发样本和回放输入。"""

    checkpoint_dir = checkpoint_dir.resolve()
    paths = {
        "planner_cases": cases_path,
        "split_manifest": split_manifest_path,
        "environment_snapshot": snapshot_path,
        "route_matrix": route_matrix_path,
        "reward_profile_v1_1": reward_profile_path,
        "reward_regression_9_3_19": reward_regression_path,
        "reward_implementation": reward_implementation_path,
        "provider_records_9_3_18": provider_records_path,
        "replay_contract_9_3_18": replay_contract_path,
        "checkpoint_manifest": checkpoint_dir / "checkpoint_manifest.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"9.3.20 输入不存在：{name}={path}")

    regression = _read_json(reward_regression_path)
    summary = regression.get("summary") or {}
    if summary.get("decision") != "pass_keep_v1_1":
        raise ValueError("9.3.19 未通过，禁止执行 9.3.20")
    if regression.get("reward_mutation_performed") is not False:
        raise ValueError("9.3.19 修改了 Reward（奖励函数），不符合 9.3.20 边界")
    if regression.get("profile_mutation_performed") is not False:
        raise ValueError("9.3.19 修改了 Reward profile（奖励函数配置），不符合 9.3.20 边界")
    if regression.get("model_execution_performed") is not False:
        raise ValueError("9.3.19 回归产物意外包含模型执行")
    if regression.get("heldout_inference_result_count") != 0:
        raise ValueError("9.3.19 回归产物包含 heldout（留出集）推理")
    if regression.get("selected_split") != "dev":
        raise ValueError("9.3.19 回归不是 dev（开发集）")
    if regression.get("reward_version") != REWARD_VERSION:
        raise ValueError("9.3.19 Reward（奖励函数）版本不一致")
    if regression.get("action_provider") != "replay_action_provider":
        raise ValueError("9.3.19 没有绑定 Replay Provider（回放动作执行器）")

    all_cases = load_planner_cases(cases_path)
    selected_cases = sorted(
        (case for case in all_cases if case.split == CaseSplit.DEV),
        key=lambda case: case.case_id,
    )
    if len(selected_cases) != 25:
        raise ValueError(f"9.3.20 dev（开发集）必须为 25 条，实际为 {len(selected_cases)}")
    not_reviewed = [
        case.case_id
        for case in selected_cases
        if case.human_review_status != HumanReviewStatus.REVIEWED
    ]
    if not_reviewed:
        raise ValueError(f"9.3.20 包含未 reviewed（已审核）样本：{not_reviewed}")
    selected_ids = [case.case_id for case in selected_cases]
    if selected_ids != sorted(regression.get("balanced_dev_case_ids") or []):
        raise ValueError("9.3.20 样本 ID 与 9.3.19 不一致")
    canonical_hash = _canonical_cases_sha256(selected_cases)
    if canonical_hash != regression.get("balanced_dev_canonical_sha256"):
        raise ValueError("9.3.20 样本内容指纹与 9.3.19 不一致")
    route_counts = Counter(_route_bucket(case) for case in selected_cases)
    if route_counts != {bucket: 5 for bucket in RouteBucket}:
        raise ValueError(f"9.3.20 五路线分布错误：{dict(route_counts)}")

    split_manifest = SplitManifest.model_validate_json(
        split_manifest_path.read_text(encoding="utf-8")
    )
    if sorted(split_manifest.dev_case_ids) != selected_ids:
        raise ValueError("9.3.20 split manifest（划分清单）与当前开发集不一致")
    snapshot = load_environment_snapshot(snapshot_path)
    if split_manifest.snapshot_id != snapshot.snapshot_id:
        raise ValueError("9.3.20 split manifest（划分清单）与快照不一致")
    if regression.get("snapshot_id") != snapshot.snapshot_id:
        raise ValueError("9.3.20 快照与 9.3.19 不一致")

    profile = _read_json(reward_profile_path)
    reward_config = _reward_config_from_profile(profile)
    if reward_config.model_dump(mode="json") != regression.get("reward_config"):
        raise ValueError("9.3.20 Reward config（奖励函数配置）与 9.3.19 不一致")
    _validate_regression_input_hashes(regression=regression, paths=paths)

    matrix = _read_json(route_matrix_path)
    thresholds = _thresholds_from_matrix(matrix)
    manifest = load_checkpoint_manifest(checkpoint_dir)
    _validate_checkpoint(
        checkpoint_dir=checkpoint_dir,
        manifest=manifest,
        reward_profile_path=reward_profile_path,
    )
    paths.update(_checkpoint_reproduction_files(manifest))
    return CorrectedInputContract(
        all_case_count=len(all_cases),
        selected_cases=tuple(selected_cases),
        balanced_dev_canonical_sha256=canonical_hash,
        snapshot_id=snapshot.snapshot_id,
        reward_config=reward_config,
        reward_profile_name=str(profile["profile_name"]),
        thresholds=thresholds,
        checkpoint_manifest=manifest,
        checkpoint_dir=checkpoint_dir,
        inputs={name: _file_record(path) for name, path in paths.items()},
    )


def run_corrected_replay_eval(
    *,
    checkpoint_dir: Path,
    old_eval_path: Path,
    old_decision_path: Path,
    corrected_eval_path: Path,
    comparison_output_path: Path,
    report_path: Path,
    cases_path: Path = DEFAULT_CASES,
    split_manifest_path: Path = DEFAULT_SPLIT_MANIFEST,
    snapshot_path: Path = DEFAULT_SNAPSHOT,
    route_matrix_path: Path = DEFAULT_ROUTE_MATRIX,
    reward_profile_path: Path = DEFAULT_REWARD_PROFILE,
    reward_regression_path: Path = DEFAULT_REWARD_REGRESSION,
    reward_implementation_path: Path = DEFAULT_REWARD_IMPLEMENTATION,
    provider_records_path: Path = DEFAULT_RECORDS,
    replay_contract_path: Path = DEFAULT_CONTRACT_OUTPUT,
) -> CorrectedReplayEvaluation:
    """运行 25 条 SFT v1（监督微调第一版）回放复评并原子写新产物。"""

    outputs = (corrected_eval_path, comparison_output_path, report_path)
    _ensure_outputs_absent(outputs)
    preflight = load_corrected_replay_preflight(
        checkpoint_dir=checkpoint_dir,
        old_eval_path=old_eval_path,
        old_decision_path=old_decision_path,
        cases_path=cases_path,
        split_manifest_path=split_manifest_path,
        snapshot_path=snapshot_path,
        route_matrix_path=route_matrix_path,
        reward_profile_path=reward_profile_path,
        reward_regression_path=reward_regression_path,
        reward_implementation_path=reward_implementation_path,
        provider_records_path=provider_records_path,
        replay_contract_path=replay_contract_path,
    )

    temp_paths = tuple(path.with_name(path.name + ".tmp") for path in outputs)
    _ensure_outputs_absent(temp_paths)
    for path in temp_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        new_eval = run_model_planner_eval(
            checkpoint_dir=checkpoint_dir,
            cases=list(preflight.contract.selected_cases),
            snapshot_path=snapshot_path,
            split=CaseSplit.DEV,
            output_path=temp_paths[0],
            provider_name=REPLAY_PROVIDER_NAME,
            provider_records_path=provider_records_path,
            reward_config=preflight.contract.reward_config,
            include_trace_evidence=True,
        )
        comparison = build_corrected_replay_evaluation(
            preflight=preflight,
            new_eval=new_eval,
            old_eval_path=old_eval_path,
            corrected_eval_path=corrected_eval_path,
            corrected_eval_content_path=temp_paths[0],
            provider_records_path=provider_records_path,
            replay_contract_path=replay_contract_path,
        )
        temp_paths[1].write_text(
            json.dumps(
                comparison.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temp_paths[2].write_text(render_report(comparison), encoding="utf-8")
        for source, target in zip(temp_paths, outputs, strict=True):
            os.replace(source, target)
        return comparison
    finally:
        for path in temp_paths:
            path.unlink(missing_ok=True)


def build_corrected_replay_evaluation(
    *,
    preflight: CorrectedReplayPreflight,
    new_eval: BaselineEvalOutput,
    old_eval_path: Path,
    corrected_eval_path: Path,
    corrected_eval_content_path: Path,
    provider_records_path: Path,
    replay_contract_path: Path,
) -> CorrectedReplayEvaluation:
    """把已经完成的新评测投影成逐样本新旧对比，不再次运行模型。"""

    _validate_new_eval_identity(new_eval, contract=preflight.contract)
    case_by_id = {
        case.case_id: case for case in preflight.contract.selected_cases
    }
    new_result_by_id = {result.case_id: result for result in new_eval.results}
    new_cases = [
        _case_admission(case_by_id[case_id], new_result_by_id[case_id])
        for case_id in sorted(case_by_id)
    ]
    new_case_by_id = {case.case_id: case for case in new_cases}
    old_case_by_id = {case.case_id: case for case in preflight.old_cases}
    old_result_by_id = {
        result.case_id: result for result in preflight.old_eval.results
    }
    records_by_key = {
        (record.case_id, record.action.value, record.query): record
        for record in preflight.records
    }
    comparisons = [
        _compare_case(
            case=case_by_id[case_id],
            old_case=old_case_by_id[case_id],
            new_case=new_case_by_id[case_id],
            old_result=old_result_by_id[case_id],
            new_result=new_result_by_id[case_id],
            records_by_key=records_by_key,
            old_provider_name=preflight.old_eval.action_provider,
        )
        for case_id in sorted(case_by_id)
    ]
    new_buckets = _bucket_admissions(
        new_cases,
        threshold=float(
            preflight.contract.thresholds["per_route_bucket_accuracy_min"]
        ),
    )
    new_checks = _gate_checks(
        new_cases,
        buckets=new_buckets,
        thresholds=preflight.contract.thresholds,
    )
    old_macro = mean(bucket.route_accuracy for bucket in preflight.old_buckets)
    new_macro = mean(bucket.route_accuracy for bucket in new_buckets)
    attribution_counts = Counter(item.attribution.value for item in comparisons)
    inputs = dict(preflight.contract.inputs)
    inputs.update(
        {
            "old_9_3_16_eval": _file_record(old_eval_path),
            "provider_records_9_3_18": _file_record(provider_records_path),
            "replay_contract_9_3_18": _file_record(replay_contract_path),
            "corrected_replay_eval": _file_record_as(
                corrected_eval_path,
                content_path=corrected_eval_content_path,
            ),
        }
    )
    return CorrectedReplayEvaluation(
        evaluated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        old_eval_run_id=preflight.old_eval.run_id,
        new_eval_run_id=new_eval.run_id,
        snapshot_id=new_eval.snapshot_id,
        reward_version=new_eval.reward_version,
        balanced_dev_canonical_sha256=(
            preflight.contract.balanced_dev_canonical_sha256
        ),
        checkpoint=_checkpoint_identity(preflight.contract),
        inputs=inputs,
        frozen_thresholds=dict(preflight.contract.thresholds),
        replay_gate_checks=new_checks,
        replay_buckets=new_buckets,
        cases=comparisons,
        summary=CorrectedReplaySummary(
            case_count=len(comparisons),
            old_route_correct_count=sum(
                item.old_route_correct for item in comparisons
            ),
            new_route_correct_count=sum(
                item.new_route_correct for item in comparisons
            ),
            route_correct_delta=(
                sum(item.new_route_correct for item in comparisons)
                - sum(item.old_route_correct for item in comparisons)
            ),
            old_route_macro_accuracy=old_macro,
            new_route_macro_accuracy=new_macro,
            changed_action_path_count=sum(
                item.old_action_path != item.new_action_path
                for item in comparisons
            ),
            new_execution_failure_count=sum(
                case.execution_failed for case in new_cases
            ),
            new_failed_case_ids=sorted(
                case.case_id for case in new_cases
                if not case.case_gate_passed
            ),
            attribution_counts=dict(sorted(attribution_counts.items())),
            provider_false_negative_corrected_case_ids=_ids_for_attribution(
                comparisons,
                AttributionCategory.PROVIDER_FALSE_NEGATIVE_CORRECTED,
            ),
            persistent_model_failure_case_ids=_ids_for_attribution(
                comparisons,
                AttributionCategory.PERSISTENT_MODEL_FAILURE,
            ),
            replay_exposed_failure_case_ids=_ids_for_attribution(
                comparisons,
                AttributionCategory.REPLAY_EXPOSED_FAILURE,
            ),
            replay_execution_failure_case_ids=_ids_for_attribution(
                comparisons,
                AttributionCategory.REPLAY_EXECUTION_FAILURE,
            ),
        ),
    )


def _compare_case(
    *,
    case: PlannerEvalCase,
    old_case: CaseAdmission,
    new_case: CaseAdmission,
    old_result: Any,
    new_result: Any,
    records_by_key: dict[
        tuple[str, str, str],
        ProviderObservationRecord,
    ],
    old_provider_name: str,
) -> CaseComparison:
    attribution, explanation = _attribution(old_case, new_case)
    return CaseComparison(
        case_id=case.case_id,
        route_bucket=new_case.route_bucket,
        expected_action_paths=new_case.expected_action_paths,
        old_action_path=old_case.actual_action_path,
        new_action_path=new_case.actual_action_path,
        old_terminal_action=old_case.actual_terminal_action,
        new_terminal_action=new_case.actual_terminal_action,
        old_route_correct=old_case.route_correct,
        new_route_correct=new_case.route_correct,
        old_case_gate_passed=old_case.case_gate_passed,
        new_case_gate_passed=new_case.case_gate_passed,
        old_failure_categories=old_case.failure_categories,
        new_failure_categories=new_case.failure_categories,
        old_observation=LegacyObservationProjection(
            provider_name=old_provider_name,
            retrieved_chunk_ids=[
                str(value) for value in old_result.retrieved_chunk_ids
            ],
            citation_chunk_ids=[
                str(value) for value in old_result.citation_chunk_ids
            ],
        ),
        new_observations=_new_observation_evidence(
            case_id=case.case_id,
            new_result=new_result,
            records_by_key=records_by_key,
        ),
        attribution=attribution,
        attribution_explanation=explanation,
    )


def _new_observation_evidence(
    *,
    case_id: str,
    new_result: Any,
    records_by_key: dict[
        tuple[str, str, str],
        ProviderObservationRecord,
    ],
) -> list[ObservationEvidence]:
    evidence: list[ObservationEvidence] = []
    raw_steps = new_result.usage.get("trace_steps") or []
    for raw_step in raw_steps:
        decision = raw_step.get("decision") or {}
        action = str(decision.get("action") or "")
        if action not in {item.value for item in RETRIEVAL_ACTIONS}:
            continue
        query = str(decision.get("query") or "")
        observation = raw_step.get("output_observation") or {}
        record = records_by_key.get((case_id, action, query))
        candidates = record.candidates if record is not None else []
        evidence.append(
            ObservationEvidence(
                step=int(raw_step["step"]),
                action=action,
                query=query,
                status=str(observation.get("status") or "failed"),
                candidate_count=int(observation.get("candidate_count") or 0),
                top_rerank_score=observation.get("top_rerank_score"),
                error_code=observation.get("error_code"),
                record_id=record.record_id if record is not None else None,
                retrieved_chunk_ids=[
                    str(candidate["chunk_id"])
                    for candidate in candidates
                    if candidate.get("chunk_id") is not None
                ],
                web_urls=[
                    str(candidate["url"])
                    for candidate in candidates
                    if candidate.get("url")
                ],
            )
        )
    return evidence


def _attribution(
    old_case: CaseAdmission,
    new_case: CaseAdmission,
) -> tuple[AttributionCategory, str]:
    if new_case.execution_failed:
        return (
            AttributionCategory.REPLAY_EXECUTION_FAILURE,
            "新回放轨迹发生执行失败，当前不能归因给模型训练覆盖。",
        )
    if old_case.case_gate_passed and new_case.case_gate_passed:
        return AttributionCategory.UNCHANGED_PASS, "新旧环境下均通过。"
    if not old_case.case_gate_passed and new_case.case_gate_passed:
        return (
            AttributionCategory.PROVIDER_FALSE_NEGATIVE_CORRECTED,
            "旧环境失败而真实回放环境通过，旧失败属于 Provider 契约误判。",
        )
    if old_case.case_gate_passed and not new_case.case_gate_passed:
        return (
            AttributionCategory.REPLAY_EXPOSED_FAILURE,
            "旧标签派生环境通过而真实回放失败，真实环境暴露了新的模型路线问题。",
        )
    return (
        AttributionCategory.PERSISTENT_MODEL_FAILURE,
        "新旧环境下均失败；在回放执行正常的前提下保留为模型路线失败。",
    )


def _validate_new_eval_identity(
    output: BaselineEvalOutput,
    *,
    contract: CorrectedInputContract,
) -> None:
    expected_ids = {case.case_id for case in contract.selected_cases}
    actual_ids = [result.case_id for result in output.results]
    if output.split != CaseSplit.DEV or output.case_count != 25:
        raise ValueError("9.3.20 必须完整运行 25 条 dev（开发集）")
    if len(actual_ids) != 25 or len(set(actual_ids)) != 25:
        raise ValueError("9.3.20 结果数量或 case_id 唯一性错误")
    if set(actual_ids) != expected_ids:
        raise ValueError("9.3.20 混入非 balanced dev（平衡开发集）样本")
    if output.requested_planners != [PlannerMode.SFT]:
        raise ValueError("9.3.20 只允许 SFT Planner（监督微调规划器）")
    if output.snapshot_id != contract.snapshot_id:
        raise ValueError("9.3.20 snapshot_id（快照标识）不一致")
    if output.reward_version != contract.reward_config.reward_version:
        raise ValueError("9.3.20 Reward（奖励函数）版本不一致")
    if output.action_provider != "ReplayActionProvider":
        raise ValueError("9.3.20 顶层 Provider（动作执行器）不是严格回放")
    if len(output.planner_summaries) != 1:
        raise ValueError("9.3.20 必须恰好有一个 SFT（监督微调）摘要")
    summary = output.planner_summaries[0]
    if summary.config.get("action_provider") != REPLAY_PROVIDER_NAME:
        raise ValueError("9.3.20 摘要没有绑定 replay（回放）执行器")
    if not _same_path(
        Path(str(summary.config.get("checkpoint") or "")),
        contract.checkpoint_dir,
    ):
        raise ValueError("9.3.20 checkpoint（检查点）与 9.3.16 不一致")
    for result in output.results:
        if result.run_id != output.run_id:
            raise ValueError(f"case={result.case_id} run_id（运行标识）不一致")
        if result.split != CaseSplit.DEV or result.planner_mode != PlannerMode.SFT:
            raise ValueError(f"case={result.case_id} split/planner_mode（划分/规划器模式）不一致")
        if result.snapshot_id != output.snapshot_id:
            raise ValueError(f"case={result.case_id} snapshot_id（快照标识）不一致")
        if result.reward_version != output.reward_version:
            raise ValueError(f"case={result.case_id} Reward（奖励函数）版本不一致")
        if "trace_steps" not in result.usage:
            raise ValueError(f"case={result.case_id} 缺少新 Observation（观察结果）证据")


def _validate_replay_binding(
    *,
    frozen: ExpandedDevReplayContract,
    current: ExpandedDevReplayContract,
    provider_records_path: Path,
) -> None:
    if not frozen.ok or not current.ok:
        raise ValueError("9.3.18 Replay（回放）契约未通过")
    if frozen.contract_version != current.contract_version:
        raise ValueError("9.3.18 Replay（回放）契约版本不一致")
    fields = (
        "snapshot_id",
        "records_sha256",
        "wrapped_provider_name",
        "case_count",
        "record_count",
        "required_record_count",
        "extra_record_count",
        "case_coverage",
        "route_checks",
    )
    for field_name in fields:
        if getattr(frozen, field_name) != getattr(current, field_name):
            raise ValueError(f"9.3.18 Replay（回放）冻结字段漂移：{field_name}")
    if frozen.records_sha256 != _sha256(provider_records_path):
        raise ValueError("9.3.18 Provider records（动作执行记录）SHA256 不一致")
    frozen_records_path = Path(frozen.records_path)
    if not frozen_records_path.is_absolute():
        frozen_records_path = PROJECT_ROOT / frozen_records_path
    if frozen_records_path.resolve() != provider_records_path.resolve():
        raise ValueError("9.3.18 Replay（回放）记录路径不一致")


def _validate_regression_input_hashes(
    *,
    regression: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    bindings = {
        "planner_cases": "planner_cases",
        "environment_snapshot": "environment_snapshot",
        "route_matrix": "route_matrix",
        "reward_profile_v1_1": "reward_profile_v1_1",
        "reward_implementation": "reward_implementation",
        "provider_records_9_3_18": "provider_records_9_3_18",
        "replay_contract_9_3_18": "replay_contract_9_3_18",
    }
    regression_inputs = regression.get("inputs") or {}
    for path_name, input_name in bindings.items():
        expected_hash = str(
            (regression_inputs.get(input_name) or {}).get("sha256") or ""
        )
        if _admission_sha256(paths[path_name]) != expected_hash:
            raise ValueError(
                f"9.3.20 输入已偏离 9.3.19：{path_name}"
            )


def _validate_legacy_evidence(
    *,
    old_eval: BaselineEvalOutput,
    old_admission: LegacyAdmissionEvidence,
    contract: CorrectedInputContract,
) -> None:
    if len(old_eval.planner_summaries) != 1:
        raise ValueError(
            "9.3.16 原始评测必须且只能包含一个 Planner（规划器）汇总，"
            f"实际为 {len(old_eval.planner_summaries)}"
        )
    if old_eval.planner_summaries[0].planner_mode != PlannerMode.SFT:
        raise ValueError(
            "9.3.16 原始评测 Planner（规划器）必须是 sft（监督微调），"
            f"实际为 {old_eval.planner_summaries[0].planner_mode.value}"
        )
    expected_ids = {case.case_id for case in contract.selected_cases}
    eval_ids = [result.case_id for result in old_eval.results]
    admission_ids = [case.case_id for case in old_admission.cases]
    if old_eval.run_id != old_admission.eval_run_id:
        raise ValueError("9.3.16 原始评测与准入决定 run_id（运行标识）不一致")
    if old_eval.action_provider != "SnapshotExpectedChunkActionProvider":
        raise ValueError("9.3.16 原始评测 Provider（动作执行器）身份错误")
    if old_admission.action_provider != SNAPSHOT_EXPECTED_PROVIDER_NAME:
        raise ValueError("9.3.16 原始准入决定 Provider（动作执行器）身份错误")
    if old_admission.heldout_inference_result_count != 0:
        raise ValueError("9.3.16 原始准入决定包含 heldout（留出集）推理")
    if old_eval.split != CaseSplit.DEV or old_eval.case_count != 25:
        raise ValueError("9.3.16 原始评测不是完整 25 条 dev（开发集）")
    if len(eval_ids) != 25 or set(eval_ids) != expected_ids:
        raise ValueError("9.3.16 原始评测样本 ID 与当前开发集不一致")
    if len(admission_ids) != 25 or set(admission_ids) != expected_ids:
        raise ValueError("9.3.16 原始准入决定样本 ID 与当前开发集不一致")
    if old_eval.snapshot_id != contract.snapshot_id:
        raise ValueError("9.3.16 原始评测快照与 9.3.20 不一致")
    if old_eval.reward_version != contract.reward_config.reward_version:
        raise ValueError("9.3.16 原始评测 Reward（奖励函数）版本不一致")
    if old_admission.checkpoint.run_id != contract.checkpoint_manifest.run_id:
        raise ValueError("9.3.16 原始准入决定 checkpoint（检查点）身份不一致")
    if not _same_path(
        Path(old_admission.checkpoint.checkpoint_dir),
        contract.checkpoint_dir,
    ):
        raise ValueError("9.3.16 原始准入决定 checkpoint（检查点）路径不一致")
    summary = old_eval.planner_summaries[0]
    if not _same_path(
        Path(str(summary.config.get("checkpoint") or "")),
        contract.checkpoint_dir,
    ):
        raise ValueError("9.3.16 原始评测 checkpoint（检查点）路径不一致")


def _checkpoint_identity(
    contract: CorrectedInputContract,
) -> CheckpointIdentity:
    manifest = contract.checkpoint_manifest
    return CheckpointIdentity(
        run_id=manifest.run_id,
        checkpoint_dir=_logical(contract.checkpoint_dir),
        policy_version=manifest.policy_version,
        training_backend=manifest.training_backend.value,
        tuning_method=manifest.tuning_method.value,
        base_model_id=manifest.base_model_id,
        adapter_id=manifest.adapter_id,
        adapter_path=manifest.adapter_path,
        training_snapshot_id=manifest.snapshot_id,
        training_code_version=manifest.code_version,
        evaluation_code_version=current_code_version(),
        sample_count=manifest.sample_count,
    )


def _ids_for_attribution(
    cases: Iterable[CaseComparison],
    category: AttributionCategory,
) -> list[str]:
    return sorted(
        case.case_id for case in cases if case.attribution == category
    )


def _file_record_as(path: Path, *, content_path: Path) -> FrozenFile:
    return FrozenFile(
        path=_logical(path),
        size_bytes=content_path.stat().st_size,
        sha256=_sha256(content_path),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ensure_outputs_absent(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.exists():
            raise FileExistsError(f"9.3.20 输出已存在，拒绝静默覆盖：{path}")


def render_report(evaluation: CorrectedReplayEvaluation) -> str:
    """渲染 9.3.20 中文归因报告，不改变机器可读结论。"""

    summary = evaluation.summary
    lines = [
        "# 阶段 9 SFT v1 校正复评报告",
        "",
        "## 结论",
        "",
        f"- 任务结论：`{summary.decision}`（只校正事实基线，不直接放行 9.4）。",
        f"- 25 条 reviewed dev（已审核开发集）：旧路线正确 "
        f"`{summary.old_route_correct_count}/25`，新路线正确 "
        f"`{summary.new_route_correct_count}/25`，变化 "
        f"`{summary.route_correct_delta:+d}`。",
        f"- 旧/新路线宏平均准确率：`{summary.old_route_macro_accuracy:.4f}` / "
        f"`{summary.new_route_macro_accuracy:.4f}`。",
        f"- 新环境执行失败：`{summary.new_execution_failure_count}` 条。",
        f"- 修正后的失败：`{len(summary.new_failed_case_ids)}` 条。",
        f"- 下一步：{summary.next_step}",
        "",
        "## 证据边界",
        "",
        "- 本次只运行同一 SFT v1（监督微调第一版）checkpoint（检查点）和 25 条 dev（开发集）。",
        "- Provider（动作执行器）固定为 9.3.18 冻结的真实检索 Replay（回放）；"
        "没有连接 Milvus（向量数据库）或 Web（网页检索）重新录制。",
        "- 9.3.16 旧结果没有保存完整逐步 Observation（观察结果）；旧侧只展示可验证的"
        "检索文本块与引用投影，新侧保存本次结构化 Trace（执行轨迹）中的完整观察摘要。",
        "- heldout test（留出测试）推理结果数固定为 `0`。",
        f"- 是否允许进入 9.4：`{str(summary.eligible_for_stage9_4).lower()}`。",
        "",
        "## 归因汇总",
        "",
        "| attribution（归因） | count（数量） |",
        "|---|---:|",
    ]
    for name, count in summary.attribution_counts.items():
        lines.append(f"| `{name}` | {count} |")
    lines.extend(
        [
            "",
            "## 五路线新结果",
            "",
            "| route bucket（路线桶） | correct/case（正确/总数） | accuracy（准确率） |",
            "|---|---:|---:|",
        ]
    )
    for bucket in evaluation.replay_buckets:
        lines.append(
            f"| `{bucket.route_bucket.value}` | "
            f"{bucket.route_correct_count}/{bucket.case_count} | "
            f"{bucket.route_accuracy:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 逐 case 对比",
            "",
            "| case_id（样本标识） | bucket（路线桶） | old path（旧路径） | "
            "new path（新路径） | old/new pass（旧/新通过） | attribution（归因） |",
            "|---|---|---|---|---|---|",
        ]
    )
    for case in evaluation.cases:
        lines.append(
            f"| `{case.case_id}` | `{case.route_bucket.value}` | "
            f"`{' -> '.join(case.old_action_path)}` | "
            f"`{' -> '.join(case.new_action_path)}` | "
            f"`{str(case.old_case_gate_passed).lower()}/"
            f"{str(case.new_case_gate_passed).lower()}` | "
            f"`{case.attribution.value}` |"
        )
    lines.extend(
        [
            "",
            "## 输入身份",
            "",
            "| input（输入） | path（路径） | SHA256（文件内容哈希） |",
            "|---|---|---|",
        ]
    )
    for name, record in evaluation.inputs.items():
        lines.append(f"| `{name}` | `{record.path}` | `{record.sha256}` |")
    lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--old-eval", type=Path, default=DEFAULT_OLD_EVAL)
    parser.add_argument(
        "--old-decision",
        type=Path,
        default=DEFAULT_OLD_DECISION,
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--route-matrix", type=Path, default=DEFAULT_ROUTE_MATRIX)
    parser.add_argument("--reward-profile", type=Path, default=DEFAULT_REWARD_PROFILE)
    parser.add_argument(
        "--reward-regression",
        type=Path,
        default=DEFAULT_REWARD_REGRESSION,
    )
    parser.add_argument(
        "--reward-implementation",
        type=Path,
        default=DEFAULT_REWARD_IMPLEMENTATION,
    )
    parser.add_argument("--provider-records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument(
        "--replay-contract",
        type=Path,
        default=DEFAULT_CONTRACT_OUTPUT,
    )
    parser.add_argument(
        "--corrected-eval-output",
        type=Path,
        default=DEFAULT_CORRECTED_EVAL,
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=DEFAULT_COMPARISON,
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只校验旧评测、检查点和回放身份，不加载模型、不写输出。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    outputs = (
        args.corrected_eval_output,
        args.comparison_output,
        args.report,
    )
    _ensure_outputs_absent(outputs)
    if args.preflight_only:
        preflight = load_corrected_replay_preflight(
            checkpoint_dir=args.checkpoint,
            old_eval_path=args.old_eval,
            old_decision_path=args.old_decision,
            cases_path=args.cases,
            split_manifest_path=args.split_manifest,
            snapshot_path=args.snapshot,
            route_matrix_path=args.route_matrix,
            reward_profile_path=args.reward_profile,
            reward_regression_path=args.reward_regression,
            reward_implementation_path=args.reward_implementation,
            provider_records_path=args.provider_records,
            replay_contract_path=args.replay_contract,
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "preflight_only": True,
                    "model_execution_performed": False,
                    "heldout_inference_result_count": 0,
                    "checkpoint_run_id": (
                        preflight.contract.checkpoint_manifest.run_id
                    ),
                    "old_eval_run_id": preflight.old_eval.run_id,
                    "case_count": len(preflight.contract.selected_cases),
                    "snapshot_id": preflight.contract.snapshot_id,
                    "reward_version": (
                        preflight.contract.reward_config.reward_version
                    ),
                    "balanced_dev_canonical_sha256": (
                        preflight.contract.balanced_dev_canonical_sha256
                    ),
                    "provider_records_sha256": (
                        preflight.replay_contract.records_sha256
                    ),
                },
                ensure_ascii=False,
            )
        )
        return 0

    evaluation = run_corrected_replay_eval(
        checkpoint_dir=args.checkpoint,
        old_eval_path=args.old_eval,
        old_decision_path=args.old_decision,
        corrected_eval_path=args.corrected_eval_output,
        comparison_output_path=args.comparison_output,
        report_path=args.report,
        cases_path=args.cases,
        split_manifest_path=args.split_manifest,
        snapshot_path=args.snapshot,
        route_matrix_path=args.route_matrix,
        reward_profile_path=args.reward_profile,
        reward_regression_path=args.reward_regression,
        reward_implementation_path=args.reward_implementation,
        provider_records_path=args.provider_records,
        replay_contract_path=args.replay_contract,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "decision": evaluation.summary.decision,
                "eligible_for_stage9_4": False,
                "case_count": evaluation.summary.case_count,
                "old_route_correct_count": (
                    evaluation.summary.old_route_correct_count
                ),
                "new_route_correct_count": (
                    evaluation.summary.new_route_correct_count
                ),
                "new_failed_case_count": len(
                    evaluation.summary.new_failed_case_ids
                ),
                "comparison_output": _logical(args.comparison_output),
                "report": _logical(args.report),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
