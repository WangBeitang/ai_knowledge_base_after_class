"""任务 9.3.16：运行 balanced dev 并形成 SFT checkpoint 的 9.4 准入结论。

本模块只允许 25 条 reviewed dev（已审核开发集）进入模型推理。它会在加载模型前校验
9.3.12 路线矩阵、9.3.15A Reward v1.1 验证、checkpoint manifest（检查点清单）和
EnvironmentSnapshot（环境快照）的身份；test/heldout case 不进入运行列表。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

from app.rag.evaluation.baseline_runner import (
    SNAPSHOT_EXPECTED_PROVIDER_NAME,
    BaselineEvalOutput,
    load_environment_snapshot,
)
from app.rag.evaluation.case_schema import (
    CaseSplit,
    HumanReviewStatus,
    PlannerEvalCase,
    PlannerEvalResult,
    PlannerMode,
    SplitManifest,
    load_planner_cases,
)
from app.rag.evaluation.reward import REWARD_VERSION, RewardConfig, RewardWeights
from app.rag.query.contracts import QueryAction
from app.rag.query.model_planner.checkpoint_runtime import (
    CheckpointManifest,
    TrainingBackend,
    TuningMethod,
    load_checkpoint_manifest,
)
from evaluation.stage9.model_planner.audit_eval_route_coverage import RouteBucket
from evaluation.stage9.model_planner.checkpoint_io import current_code_version
from evaluation.stage9.model_planner.eval_model_planner import run_model_planner_eval


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ADMISSION_VERSION = "stage9-sft-expanded-dev-admission-v1"
DEFAULT_CASES = PROJECT_ROOT / "evaluation/stage8/cases/planner_cases.jsonl"
DEFAULT_SPLIT_MANIFEST = (
    PROJECT_ROOT / "evaluation/stage8/cases/split_manifest.json"
)
DEFAULT_SNAPSHOT = (
    PROJECT_ROOT
    / "evaluation/stage9/artifacts/heldout_route_test/environment_snapshot.json"
)
DEFAULT_ROUTE_MATRIX = (
    PROJECT_ROOT / "evaluation/stage9/configs/planner_eval_route_matrix_v1.json"
)
DEFAULT_REWARD_PROFILE = (
    PROJECT_ROOT / "evaluation/stage9/configs/reward_v1_1_training_profile.json"
)
DEFAULT_REWARD_VALIDATION = (
    PROJECT_ROOT
    / "evaluation/stage9/artifacts/reward/reward_v1_1_balanced_dev_validation.json"
)
DEFAULT_REWARD_IMPLEMENTATION = PROJECT_ROOT / "app/rag/evaluation/reward.py"
DEFAULT_EVAL_OUTPUT = (
    PROJECT_ROOT / "evaluation/stage9/artifacts/sft/sft_expanded_dev_eval.json"
)
DEFAULT_DECISION_OUTPUT = (
    PROJECT_ROOT
    / "evaluation/stage9/artifacts/sft/sft_9_4_admission_decision.json"
)
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "evaluation/stage9/artifacts/reports/阶段9-SFT-9.4准入报告.md"
)


class AdmissionDecision(str, Enum):
    """9.3.16 的两种模型选择结论。"""

    ALLOW_STAGE9_4 = "allow_stage9_4"  # 全部门禁通过，冻结当前组合进入 9.4。
    TRAIN_SFT_V2 = "reject_sft_v1_train_sft_v2"  # 当前候选失败，只能补 train-only 数据后重训。


class AdmissionModel(BaseModel):
    """9.3.16 冻结输出的公共 schema（数据结构）；拒绝未知字段避免报告漂移。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class FrozenFile(AdmissionModel):
    """一个输入或产物的逻辑路径、字节数和 SHA256 身份。"""

    path: str = Field(min_length=1, description="项目相对路径；项目外测试文件保留绝对路径。")
    size_bytes: int = Field(ge=0, description="哈希时的文件字节数。")
    sha256: str = Field(min_length=64, max_length=64, description="文件内容 SHA256。")


class CheckpointIdentity(AdmissionModel):
    """本次被选择或淘汰的 checkpoint 身份。"""

    run_id: str = Field(min_length=1, description="checkpoint manifest 中的正式运行 ID。")
    checkpoint_dir: str = Field(min_length=1, description="本次 direct runtime 实际加载的目录。")
    policy_version: str = Field(min_length=1, description="模型策略版本，写入每次 Planner 推理。")
    training_backend: str = Field(description="正式门禁只允许 transformers_causal_lm。")
    tuning_method: str = Field(description="阶段 9 正式候选只允许 LoRA 或 QLoRA。")
    base_model_id: str = Field(min_length=1)
    adapter_id: str = ""
    adapter_path: str = Field(min_length=1)
    training_snapshot_id: str = Field(
        min_length=1,
        description="训练数据绑定快照；它与本次 expanded dev 评测快照是两个生命周期。",
    )
    training_code_version: str = Field(min_length=1)
    evaluation_code_version: str = Field(min_length=1)
    sample_count: int = Field(ge=1)


class CaseAdmission(AdmissionModel):
    """一条 balanced dev 的期望路线、实际路线和门禁归因。"""

    case_id: str = Field(min_length=1)
    route_bucket: RouteBucket
    expected_action_paths: list[list[str]]
    expected_terminal_action: str
    should_call_web: bool
    forbidden_actions: list[str]
    actual_action_path: list[str]
    actual_terminal_action: str | None
    terminal_reason_code: str
    path_match: bool
    terminal_match: bool
    web_behavior_match: bool
    format_valid: bool
    execution_failed: bool
    forbidden_actions_used: list[str]
    route_correct: bool = Field(
        description="期望路径、终态和 Web 使用同时正确；用于路线 accuracy（准确率）。"
    )
    case_gate_passed: bool = Field(
        description="路线正确且没有格式、执行或禁止动作问题。"
    )
    safe_refuse_false_release: bool = Field(
        description="危险拒绝样本是否被错误放行为非 refuse 终态。"
    )
    total_reward: float = Field(ge=0, le=1)
    component_scores: dict[str, float]
    duration_ms: int = Field(ge=0)
    failure_categories: list[str]
    errors: list[dict[str, Any]]


class BucketAdmission(AdmissionModel):
    """一个 route bucket（路线桶）的独立准入结果。"""

    route_bucket: RouteBucket
    case_count: int = Field(ge=1)
    route_correct_count: int = Field(ge=0)
    route_accuracy: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    passed: bool
    failed_case_ids: list[str]


class GateCheck(AdmissionModel):
    """一项 9.3.12 预先冻结的工程阈值检查。"""

    name: str = Field(min_length=1)
    observed: float | int
    operator: str = Field(description="当前只使用 >= 或 <=，便于机器报告直接解释。")
    threshold: float | int
    passed: bool
    explanation: str


class AdmissionSummary(AdmissionModel):
    """9.3.16 总体指标和最终 9.4 准入决定。"""

    decision: AdmissionDecision
    eligible_for_stage9_4: bool
    case_count: int = Field(ge=1)
    route_correct_count: int = Field(ge=0)
    overall_route_accuracy: float = Field(ge=0, le=1)
    route_macro_accuracy: float = Field(ge=0, le=1)
    format_valid_rate: float = Field(ge=0, le=1)
    execution_failure_count: int = Field(ge=0)
    forbidden_action_count: int = Field(ge=0)
    safe_refuse_dangerous_false_release_count: int = Field(ge=0)
    actual_terminal_action_counts: dict[str, int] = Field(
        description="实际 answer/refuse/ask_clarification/none 终态数量。"
    )
    terminal_confusion_matrix: dict[str, dict[str, int]] = Field(
        description="key=期望终态，value=实际终态计数，用于定位终止动作混淆。"
    )
    failure_category_counts: dict[str, int] = Field(
        description="逐 case failure_categories 的聚合数量。"
    )
    failed_case_ids: list[str]
    failed_route_buckets: list[RouteBucket]
    next_step: str


class ExpandedDevAdmission(AdmissionModel):
    """9.3.16 完整冻结结论；只含 dev 推理，不含任何 heldout test 结果。"""

    admission_version: str = ADMISSION_VERSION
    evaluated_at: str
    eval_run_id: str
    selected_split: str = "dev"
    reward_version: str
    reward_profile_name: str
    snapshot_id: str
    action_provider: str
    model_execution_performed: bool = True
    heldout_inference_result_count: int = 0
    input_case_count: int = 25
    non_dev_case_count_ignored: int = Field(
        ge=0,
        description="同一 registry 中未进入模型运行的 train/test case 数。",
    )
    balanced_dev_canonical_sha256: str = Field(min_length=64, max_length=64)
    retrieval_quality_verified: bool = Field(
        description="snapshot_expected_chunks 下固定为 false，不包装成真实 Milvus/Web 质量。"
    )
    answer_quality_interpretable_as_model_quality: bool = Field(
        description="离线占位 answer executor 下固定为 false。"
    )
    checkpoint: CheckpointIdentity
    inputs: dict[str, FrozenFile]
    thresholds: dict[str, float | int]
    gate_checks: list[GateCheck]
    buckets: list[BucketAdmission]
    cases: list[CaseAdmission]
    summary: AdmissionSummary


@dataclass(frozen=True)
class AdmissionContract:
    """模型加载前已经验证过的冻结输入，避免 GPU 计费后才发现数据身份错误。"""

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


def run_expanded_dev_gate(
    *,
    checkpoint_dir: Path,
    cases_path: Path = DEFAULT_CASES,
    split_manifest_path: Path = DEFAULT_SPLIT_MANIFEST,
    snapshot_path: Path = DEFAULT_SNAPSHOT,
    route_matrix_path: Path = DEFAULT_ROUTE_MATRIX,
    reward_profile_path: Path = DEFAULT_REWARD_PROFILE,
    reward_validation_path: Path = DEFAULT_REWARD_VALIDATION,
    reward_implementation_path: Path = DEFAULT_REWARD_IMPLEMENTATION,
    eval_output_path: Path = DEFAULT_EVAL_OUTPUT,
    decision_output_path: Path = DEFAULT_DECISION_OUTPUT,
    report_path: Path = DEFAULT_REPORT,
    provider_name: str = SNAPSHOT_EXPECTED_PROVIDER_NAME,
    overwrite: bool = False,
) -> ExpandedDevAdmission:
    """先做无模型合同检查，再运行 25 条 dev，并写入准入结论。"""

    if provider_name != SNAPSHOT_EXPECTED_PROVIDER_NAME:
        raise ValueError(
            "9.3.16 首轮准入固定使用 snapshot_expected_chunks；"
            "真实 Milvus/Web 质量必须单独报告，不能混改冻结阈值。"
        )
    _ensure_outputs_available(
        (eval_output_path, decision_output_path, report_path),
        overwrite=overwrite,
    )
    contract = load_admission_contract(
        checkpoint_dir=checkpoint_dir,
        cases_path=cases_path,
        split_manifest_path=split_manifest_path,
        snapshot_path=snapshot_path,
        route_matrix_path=route_matrix_path,
        reward_profile_path=reward_profile_path,
        reward_validation_path=reward_validation_path,
        reward_implementation_path=reward_implementation_path,
    )
    eval_output = run_model_planner_eval(
        checkpoint_dir=checkpoint_dir,
        cases=list(contract.selected_cases),
        snapshot_path=snapshot_path,
        split=CaseSplit.DEV,
        output_path=eval_output_path,
        provider_name=provider_name,
        reward_config=contract.reward_config,
    )
    admission = build_admission(
        eval_output=eval_output,
        contract=contract,
        eval_output_path=eval_output_path,
    )
    write_admission_outputs(
        admission,
        decision_output_path=decision_output_path,
        report_path=report_path,
        overwrite=overwrite,
    )
    return admission


def load_admission_contract(
    *,
    checkpoint_dir: Path,
    cases_path: Path = DEFAULT_CASES,
    split_manifest_path: Path = DEFAULT_SPLIT_MANIFEST,
    snapshot_path: Path = DEFAULT_SNAPSHOT,
    route_matrix_path: Path = DEFAULT_ROUTE_MATRIX,
    reward_profile_path: Path = DEFAULT_REWARD_PROFILE,
    reward_validation_path: Path = DEFAULT_REWARD_VALIDATION,
    reward_implementation_path: Path = DEFAULT_REWARD_IMPLEMENTATION,
) -> AdmissionContract:
    """只读校验所有静态输入；本函数不加载模型、不生成推理结果。"""

    checkpoint_dir = checkpoint_dir.resolve()
    paths = {
        "planner_cases": cases_path,
        "split_manifest": split_manifest_path,
        "environment_snapshot": snapshot_path,
        "route_matrix": route_matrix_path,
        "reward_profile_v1_1": reward_profile_path,
        "reward_validation_9_3_15a": reward_validation_path,
        "reward_implementation": reward_implementation_path,
        "checkpoint_manifest": checkpoint_dir / "checkpoint_manifest.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"9.3.16 输入不存在：{name}={path}")

    matrix = _read_json(route_matrix_path)
    profile = _read_json(reward_profile_path)
    reward_validation = _read_json(reward_validation_path)
    cases = load_planner_cases(cases_path)
    split_manifest = SplitManifest.model_validate_json(
        split_manifest_path.read_text(encoding="utf-8")
    )
    selected_cases = _select_balanced_dev(
        cases,
        matrix=matrix,
        reward_validation=reward_validation,
    )
    canonical_hash = _canonical_cases_sha256(selected_cases)
    _validate_reward_boundary(
        reward_validation=reward_validation,
        paths=paths,
        canonical_hash=canonical_hash,
        selected_cases=selected_cases,
    )
    snapshot = load_environment_snapshot(snapshot_path)
    selected_ids = [case.case_id for case in selected_cases]
    if sorted(split_manifest.dev_case_ids) != selected_ids:
        raise ValueError("split manifest 的 dev_case_ids 与 25 条 balanced dev 不一致")
    if split_manifest.snapshot_id != snapshot.snapshot_id:
        raise ValueError("split manifest 与 9.3.16 snapshot_id 不一致")
    cases_hash = _sha256(cases_path)
    if snapshot.source_hashes.get(_logical(cases_path)) != cases_hash:
        raise ValueError("9.3.16 snapshot 未绑定当前 planner_cases SHA256")
    if snapshot.snapshot_id != reward_validation.get("snapshot_id"):
        raise ValueError("9.3.16 snapshot_id 与 9.3.15A 验证输入不一致")

    reward_config = _reward_config_from_profile(profile)
    thresholds = _thresholds_from_matrix(matrix)
    manifest = load_checkpoint_manifest(checkpoint_dir)
    _validate_checkpoint(
        checkpoint_dir=checkpoint_dir,
        manifest=manifest,
        reward_profile_path=reward_profile_path,
    )
    paths.update(_checkpoint_reproduction_files(manifest))
    return AdmissionContract(
        all_case_count=len(cases),
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


def build_admission(
    *,
    eval_output: BaselineEvalOutput,
    contract: AdmissionContract,
    eval_output_path: Path,
) -> ExpandedDevAdmission:
    """把真实 expanded dev 运行结果投影成冻结阈值和逐 case 准入结论。"""

    _validate_eval_identity(eval_output, contract=contract)
    case_by_id = {case.case_id: case for case in contract.selected_cases}
    result_by_id = {result.case_id: result for result in eval_output.results}
    case_results = [
        _case_admission(case_by_id[case_id], result_by_id[case_id])
        for case_id in sorted(case_by_id)
    ]
    buckets = _bucket_admissions(
        case_results,
        threshold=float(contract.thresholds["per_route_bucket_accuracy_min"]),
    )
    checks = _gate_checks(
        case_results,
        buckets=buckets,
        thresholds=contract.thresholds,
    )
    all_passed = all(check.passed for check in checks)
    decision = (
        AdmissionDecision.ALLOW_STAGE9_4
        if all_passed
        else AdmissionDecision.TRAIN_SFT_V2
    )
    failed_case_ids = sorted(
        case.case_id for case in case_results if not case.case_gate_passed
    )
    failed_buckets = [
        bucket.route_bucket for bucket in buckets if not bucket.passed
    ]
    manifest = contract.checkpoint_manifest
    inputs = dict(contract.inputs)
    inputs["expanded_dev_eval"] = _file_record(eval_output_path)
    return ExpandedDevAdmission(
        evaluated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        eval_run_id=eval_output.run_id,
        reward_version=eval_output.reward_version,
        reward_profile_name=contract.reward_profile_name,
        snapshot_id=eval_output.snapshot_id,
        action_provider=SNAPSHOT_EXPECTED_PROVIDER_NAME,
        non_dev_case_count_ignored=(
            contract.all_case_count - len(contract.selected_cases)
        ),
        balanced_dev_canonical_sha256=contract.balanced_dev_canonical_sha256,
        retrieval_quality_verified=False,
        answer_quality_interpretable_as_model_quality=False,
        checkpoint=CheckpointIdentity(
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
        ),
        inputs=inputs,
        thresholds=dict(contract.thresholds),
        gate_checks=checks,
        buckets=buckets,
        cases=case_results,
        summary=AdmissionSummary(
            decision=decision,
            eligible_for_stage9_4=all_passed,
            case_count=len(case_results),
            route_correct_count=sum(case.route_correct for case in case_results),
            overall_route_accuracy=mean(
                float(case.route_correct) for case in case_results
            ),
            route_macro_accuracy=mean(
                bucket.route_accuracy for bucket in buckets
            ),
            format_valid_rate=mean(
                float(case.format_valid) for case in case_results
            ),
            execution_failure_count=sum(
                case.execution_failed for case in case_results
            ),
            forbidden_action_count=sum(
                len(case.forbidden_actions_used) for case in case_results
            ),
            safe_refuse_dangerous_false_release_count=sum(
                case.safe_refuse_false_release for case in case_results
            ),
            actual_terminal_action_counts=dict(
                sorted(
                    Counter(
                        case.actual_terminal_action or "none"
                        for case in case_results
                    ).items()
                )
            ),
            terminal_confusion_matrix=_terminal_confusion_matrix(case_results),
            failure_category_counts=dict(
                sorted(
                    Counter(
                        failure
                        for case in case_results
                        for failure in case.failure_categories
                    ).items()
                )
            ),
            failed_case_ids=failed_case_ids,
            failed_route_buckets=failed_buckets,
            next_step=(
                "冻结 checkpoint、Reward profile、snapshot、dev manifest 和运行命令，"
                "完成异地备份后允许进入 9.4。"
                if all_passed
                else
                "按逐 case 失败归因补独立 train-only 路线样本，训练 SFT v2 后重跑 9.3.16；"
                "不得把 balanced dev 或 heldout test 写入训练集。"
            ),
        ),
    )


def load_and_build_admission(
    *,
    eval_output_path: Path,
    checkpoint_dir: Path,
    cases_path: Path = DEFAULT_CASES,
    split_manifest_path: Path = DEFAULT_SPLIT_MANIFEST,
    snapshot_path: Path = DEFAULT_SNAPSHOT,
    route_matrix_path: Path = DEFAULT_ROUTE_MATRIX,
    reward_profile_path: Path = DEFAULT_REWARD_PROFILE,
    reward_validation_path: Path = DEFAULT_REWARD_VALIDATION,
    reward_implementation_path: Path = DEFAULT_REWARD_IMPLEMENTATION,
) -> ExpandedDevAdmission:
    """测试和离线复核入口：读取已生成 eval，不再次运行模型。"""

    contract = load_admission_contract(
        checkpoint_dir=checkpoint_dir,
        cases_path=cases_path,
        split_manifest_path=split_manifest_path,
        snapshot_path=snapshot_path,
        route_matrix_path=route_matrix_path,
        reward_profile_path=reward_profile_path,
        reward_validation_path=reward_validation_path,
        reward_implementation_path=reward_implementation_path,
    )
    eval_output = BaselineEvalOutput.model_validate_json(
        eval_output_path.read_text(encoding="utf-8")
    )
    return build_admission(
        eval_output=eval_output,
        contract=contract,
        eval_output_path=eval_output_path,
    )


def _select_balanced_dev(
    cases: Iterable[PlannerEvalCase],
    *,
    matrix: dict[str, Any],
    reward_validation: dict[str, Any],
) -> list[PlannerEvalCase]:
    """只选择 9.3.15A 已绑定的 25 条 reviewed dev；test 不进入返回值。"""

    selected = sorted(
        (case for case in cases if case.split == CaseSplit.DEV),
        key=lambda case: case.case_id,
    )
    expected_ids = list(reward_validation.get("balanced_dev_case_ids") or [])
    actual_ids = [case.case_id for case in selected]
    if len(selected) != 25 or actual_ids != sorted(expected_ids):
        raise ValueError(
            "9.3.16 balanced dev 与 9.3.15A 冻结 ID 不一致："
            f"actual_count={len(selected)}"
        )
    not_reviewed = [
        case.case_id
        for case in selected
        if case.human_review_status != HumanReviewStatus.REVIEWED
    ]
    if not_reviewed:
        raise ValueError(f"9.3.16 dev 含未 reviewed case：{not_reviewed}")
    minimum = int(
        matrix["evaluation_sets"]["balanced_dev"][
            "minimum_reviewed_cases_per_bucket"
        ]
    )
    counts = Counter(_route_bucket(case) for case in selected)
    expected = {bucket: minimum for bucket in RouteBucket}
    if counts != expected:
        raise ValueError(
            "9.3.16 五路线分布不符合冻结矩阵："
            f"actual={dict(counts)}, expected={expected}"
        )
    return selected


def _validate_reward_boundary(
    *,
    reward_validation: dict[str, Any],
    paths: dict[str, Path],
    canonical_hash: str,
    selected_cases: list[PlannerEvalCase],
) -> None:
    """要求 9.3.15A 已通过，并复核它绑定的五个核心输入仍未变化。"""

    summary = reward_validation.get("summary") or {}
    if summary.get("decision") != "pass_keep_v1_1":
        raise ValueError("9.3.15A 未得出 pass_keep_v1_1，禁止进入 9.3.16")
    if reward_validation.get("reward_version") != REWARD_VERSION:
        raise ValueError("9.3.16 只允许 9.3.15A 验证通过的 Reward v1.1")
    if reward_validation.get("heldout_inference_result_count") != 0:
        raise ValueError("9.3.15A 产物含 heldout 推理结果，违反准入边界")
    if canonical_hash != reward_validation.get("balanced_dev_canonical_sha256"):
        raise ValueError("balanced dev 内容已在 9.3.15A 后发生变化")
    if [case.case_id for case in selected_cases] != sorted(
        reward_validation.get("balanced_dev_case_ids") or []
    ):
        raise ValueError("balanced dev case ID 与 9.3.15A 不一致")
    validation_inputs = reward_validation.get("inputs") or {}
    bindings = {
        "planner_cases": "planner_cases",
        "environment_snapshot": "environment_snapshot",
        "route_matrix": "route_matrix",
        "reward_profile_v1_1": "reward_profile_v1_1",
        "reward_implementation": "reward_implementation",
    }
    for path_name, validation_name in bindings.items():
        expected_hash = str(
            (validation_inputs.get(validation_name) or {}).get("sha256") or ""
        )
        if _sha256(paths[path_name]) != expected_hash:
            raise ValueError(
                f"9.3.16 输入已偏离 9.3.15A：{path_name}"
            )


def _validate_checkpoint(
    *,
    checkpoint_dir: Path,
    manifest: CheckpointManifest,
    reward_profile_path: Path,
) -> None:
    """拒绝 debug smoke、全量微调和 Reward 身份不一致的 checkpoint。"""

    if checkpoint_dir.name != manifest.run_id:
        raise ValueError("checkpoint 目录名与 manifest.run_id 不一致")
    if manifest.training_backend != TrainingBackend.TRANSFORMERS_CAUSAL_LM:
        raise ValueError("9.3.16 正式准入禁止使用 debug_memorized smoke checkpoint")
    if manifest.tuning_method not in {TuningMethod.LORA, TuningMethod.QLORA}:
        raise ValueError("9.3.16 当前只允许 LoRA/QLoRA checkpoint")
    if not manifest.adapter_path:
        raise ValueError("checkpoint manifest 缺少 adapter_path")
    if _resolve_project_path(manifest.reward_profile) != reward_profile_path.resolve():
        raise ValueError("checkpoint 训练时 Reward profile 与 9.3.16 不一致")
    adapter_path = _resolve_project_path(manifest.adapter_path)
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter_path / filename).is_file():
            raise FileNotFoundError(f"checkpoint adapter 缺少 {filename}")


def _checkpoint_reproduction_files(
    manifest: CheckpointManifest,
) -> dict[str, Path]:
    """返回会影响 direct runtime 推理或训练身份的 checkpoint 内部文件。"""

    if not manifest.tokenizer_path:
        raise ValueError("9.3.16 checkpoint manifest 缺少 tokenizer_path")
    adapter_path = _resolve_project_path(manifest.adapter_path)
    tokenizer_path = _resolve_project_path(manifest.tokenizer_path)
    files = {
        "checkpoint_training_config": _resolve_project_path(
            manifest.training_config_path
        ),
        "checkpoint_train_metrics": _resolve_project_path(
            manifest.train_metrics_path
        ),
        "adapter_config": adapter_path / "adapter_config.json",
        "adapter_weights": adapter_path / "adapter_model.safetensors",
        "tokenizer_json": tokenizer_path / "tokenizer.json",
        "tokenizer_config": tokenizer_path / "tokenizer_config.json",
        "chat_template": tokenizer_path / "chat_template.jinja",
    }
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint 复现文件不存在：{name}={path}")
    return files


def _validate_eval_identity(
    output: BaselineEvalOutput,
    *,
    contract: AdmissionContract,
) -> None:
    """确认原始 eval 只包含本次 25 条 dev 和指定 checkpoint。"""

    expected_ids = {case.case_id for case in contract.selected_cases}
    actual_ids = [result.case_id for result in output.results]
    if output.split != CaseSplit.DEV or output.case_count != 25:
        raise ValueError("9.3.16 eval 必须是完整 25 条 dev")
    if len(actual_ids) != 25 or len(set(actual_ids)) != 25:
        raise ValueError("9.3.16 eval 结果数量或 case_id 唯一性错误")
    if set(actual_ids) != expected_ids:
        raise ValueError("9.3.16 eval 混入非 balanced dev case")
    if output.requested_planners != [PlannerMode.SFT]:
        raise ValueError("9.3.16 只允许 SFT Planner")
    if output.snapshot_id != contract.snapshot_id:
        raise ValueError("9.3.16 eval snapshot_id 不一致")
    if output.reward_version != contract.reward_config.reward_version:
        raise ValueError("9.3.16 eval Reward 版本不一致")
    if output.action_provider != "SnapshotExpectedChunkActionProvider":
        raise ValueError("9.3.16 eval 顶层 ActionProvider 不一致")
    if len(output.planner_summaries) != 1:
        raise ValueError("9.3.16 eval 必须恰好有一个 SFT summary")
    summary = output.planner_summaries[0]
    if summary.planner_mode != PlannerMode.SFT or summary.status != "completed":
        raise ValueError("9.3.16 SFT summary 未完成")
    if summary.case_count != 25 or summary.completed_case_count + summary.failed_case_count != 25:
        raise ValueError("9.3.16 summary case 计数不一致")
    if summary.config.get("action_provider") != SNAPSHOT_EXPECTED_PROVIDER_NAME:
        raise ValueError("9.3.16 summary provider 不一致")
    if not _same_path(
        Path(str(summary.config.get("checkpoint") or "")),
        contract.checkpoint_dir,
    ):
        raise ValueError("9.3.16 eval checkpoint 与准入候选不一致")
    for result in output.results:
        if result.run_id != output.run_id:
            raise ValueError(f"case={result.case_id} run_id 与顶层不一致")
        if result.split != CaseSplit.DEV or result.planner_mode != PlannerMode.SFT:
            raise ValueError(f"case={result.case_id} split/planner_mode 不一致")
        if result.snapshot_id != output.snapshot_id:
            raise ValueError(f"case={result.case_id} snapshot_id 不一致")
        if result.reward_version != output.reward_version:
            raise ValueError(f"case={result.case_id} reward_version 不一致")


def _case_admission(
    case: PlannerEvalCase,
    result: PlannerEvalResult,
) -> CaseAdmission:
    """计算单 case 路线正确性；不使用总 Reward 高低替代人工接受路径。"""

    expected_paths = [
        [action.value for action in path] for path in case.acceptable_action_paths
    ]
    actual_path = [action.value for action in result.action_path]
    path_match = actual_path in expected_paths
    metric_path_match = result.metrics.get("path_match")
    if not isinstance(metric_path_match, bool) or metric_path_match != path_match:
        raise ValueError(f"case={case.case_id} path_match 与 Reward metrics 不一致")
    expected_terminal = _expected_terminal(case)
    actual_terminal = (
        result.terminal_action.value if result.terminal_action else None
    )
    terminal_match = actual_terminal == expected_terminal
    used_web = QueryAction.WEB_SEARCH.value in actual_path
    web_behavior_match = used_web == case.expected_behavior.should_call_web
    format_valid = result.metrics.get("format_valid")
    if not isinstance(format_valid, bool):
        raise ValueError(f"case={case.case_id} 缺少布尔 format_valid")
    trajectory_status = str(result.usage.get("trajectory_status") or "")
    execution_failed = bool(result.errors) or (
        bool(trajectory_status) and trajectory_status != "completed"
    )
    forbidden = {action.value for action in case.expected_behavior.forbidden_actions}
    forbidden_used = sorted(forbidden.intersection(actual_path))
    route_correct = path_match and terminal_match and web_behavior_match
    safe_false_release = (
        _route_bucket(case) == RouteBucket.SAFE_REFUSE
        and actual_terminal != QueryAction.REFUSE.value
    )
    case_passed = (
        route_correct
        and format_valid
        and not execution_failed
        and not forbidden_used
        and not safe_false_release
    )
    failures: list[str] = []
    if not path_match:
        failures.append("wrong_action_path")
    if not terminal_match:
        failures.append("wrong_terminal_action")
    if not web_behavior_match:
        failures.append("web_behavior_mismatch")
    if not format_valid:
        failures.append("invalid_format")
    if execution_failed:
        failures.append("execution_failure")
    if forbidden_used:
        failures.append("forbidden_action_used")
    if safe_false_release:
        failures.append("safe_refuse_false_release")
    reward_components = result.reward.get("components") or {}
    return CaseAdmission(
        case_id=case.case_id,
        route_bucket=_route_bucket(case),
        expected_action_paths=expected_paths,
        expected_terminal_action=expected_terminal,
        should_call_web=case.expected_behavior.should_call_web,
        forbidden_actions=sorted(forbidden),
        actual_action_path=actual_path,
        actual_terminal_action=actual_terminal,
        terminal_reason_code=result.terminal_reason_code,
        path_match=path_match,
        terminal_match=terminal_match,
        web_behavior_match=web_behavior_match,
        format_valid=format_valid,
        execution_failed=execution_failed,
        forbidden_actions_used=forbidden_used,
        route_correct=route_correct,
        case_gate_passed=case_passed,
        safe_refuse_false_release=safe_false_release,
        total_reward=float(result.reward["total_reward"]),
        component_scores={
            str(name): float(component["score"])
            for name, component in reward_components.items()
        },
        duration_ms=int(result.usage.get("duration_ms") or 0),
        failure_categories=failures,
        errors=list(result.errors),
    )


def _bucket_admissions(
    cases: list[CaseAdmission],
    *,
    threshold: float,
) -> list[BucketAdmission]:
    output: list[BucketAdmission] = []
    for bucket in RouteBucket:
        selected = [case for case in cases if case.route_bucket == bucket]
        correct = sum(case.route_correct for case in selected)
        accuracy = correct / len(selected)
        output.append(
            BucketAdmission(
                route_bucket=bucket,
                case_count=len(selected),
                route_correct_count=correct,
                route_accuracy=accuracy,
                threshold=threshold,
                passed=accuracy >= threshold,
                failed_case_ids=[
                    case.case_id for case in selected if not case.route_correct
                ],
            )
        )
    return output


def _terminal_confusion_matrix(
    cases: list[CaseAdmission],
) -> dict[str, dict[str, int]]:
    """汇总期望/实际终态；none 显式保留，避免无终态被忽略。"""

    terminals = (
        QueryAction.ANSWER.value,
        QueryAction.ASK_CLARIFICATION.value,
        QueryAction.REFUSE.value,
        "none",
    )
    matrix = {
        expected: {actual: 0 for actual in terminals}
        for expected in terminals[:-1]
    }
    for case in cases:
        matrix[case.expected_terminal_action][
            case.actual_terminal_action or "none"
        ] += 1
    return matrix


def _gate_checks(
    cases: list[CaseAdmission],
    *,
    buckets: list[BucketAdmission],
    thresholds: dict[str, float | int],
) -> list[GateCheck]:
    format_rate = mean(float(case.format_valid) for case in cases)
    failure_count = sum(case.execution_failed for case in cases)
    forbidden_count = sum(len(case.forbidden_actions_used) for case in cases)
    macro_accuracy = mean(bucket.route_accuracy for bucket in buckets)
    per_bucket_passed = sum(bucket.passed for bucket in buckets)
    false_release_count = sum(case.safe_refuse_false_release for case in cases)
    return [
        _check(
            "format_valid_rate",
            format_rate,
            ">=",
            thresholds["format_valid_rate"],
            "结构化 Planner 输出必须全部合法。",
        ),
        _check(
            "execution_failure_count",
            failure_count,
            "<=",
            thresholds["execution_failure_count"],
            "模型解析、Action 执行和终态运行不得失败。",
        ),
        _check(
            "forbidden_action_count",
            forbidden_count,
            "<=",
            thresholds["forbidden_action_count"],
            "case 明确禁止的 Action 不得出现。",
        ),
        _check(
            "route_macro_accuracy",
            macro_accuracy,
            ">=",
            thresholds["route_macro_accuracy_min"],
            "五路线等权平均，避免 local_answer 的数量掩盖其他路线。",
        ),
        _check(
            "per_route_bucket_passed_count",
            per_bucket_passed,
            ">=",
            len(RouteBucket),
            "五个路线桶都必须达到各桶最低准确率。",
        ),
        _check(
            "safe_refuse_dangerous_false_release_count",
            false_release_count,
            "<=",
            thresholds["safe_refuse_dangerous_false_release_count"],
            "危险请求不得被错误放行为回答或澄清。",
        ),
    ]


def _check(
    name: str,
    observed: float | int,
    operator: str,
    threshold: float | int,
    explanation: str,
) -> GateCheck:
    if operator == ">=":
        passed = observed >= threshold
    elif operator == "<=":
        passed = observed <= threshold
    else:
        raise ValueError(f"未知门禁运算符：{operator}")
    return GateCheck(
        name=name,
        observed=observed,
        operator=operator,
        threshold=threshold,
        passed=passed,
        explanation=explanation,
    )


def _thresholds_from_matrix(matrix: dict[str, Any]) -> dict[str, float | int]:
    gates = matrix.get("quality_gates") or {}
    if not (matrix.get("freeze_policy") or {}).get(
        "thresholds_must_not_change_after_results"
    ):
        raise ValueError("路线矩阵没有冻结结果后不可调阈值的门禁")
    return {
        "format_valid_rate": float(gates["format_valid_rate"]),
        "execution_failure_count": int(gates["execution_failure_count"]),
        "forbidden_action_count": int(gates["forbidden_action_count"]),
        "route_macro_accuracy_min": float(gates["route_macro_accuracy_min"]),
        "per_route_bucket_accuracy_min": float(
            gates["per_route_bucket_accuracy_min"]
        ),
        "safe_refuse_dangerous_false_release_count": int(
            gates["safe_refuse_dangerous_false_release_count"]
        ),
    }


def _reward_config_from_profile(profile: dict[str, Any]) -> RewardConfig:
    if profile.get("decision") != "frozen":
        raise ValueError("Reward profile 尚未 frozen")
    if profile.get("reward_version") != REWARD_VERSION:
        raise ValueError(f"9.3.16 只允许 {REWARD_VERSION}")
    return RewardConfig(
        reward_version=REWARD_VERSION,
        weights=RewardWeights.model_validate(profile.get("weights")),
    )


def _route_bucket(case: PlannerEvalCase) -> RouteBucket:
    behavior = case.expected_behavior
    if behavior.should_call_web:
        return RouteBucket.WEB_REQUIRED
    if behavior.should_ask_clarification:
        return RouteBucket.ASK_CLARIFICATION
    if behavior.should_refuse:
        return RouteBucket.SAFE_REFUSE
    if case.acceptable_action_paths and all(
        QueryAction.HYDE_SEARCH in path for path in case.acceptable_action_paths
    ):
        return RouteBucket.HYDE_FALLBACK
    return RouteBucket.LOCAL_ANSWER


def _expected_terminal(case: PlannerEvalCase) -> str:
    behavior = case.expected_behavior
    if behavior.should_answer:
        return QueryAction.ANSWER.value
    if behavior.should_refuse:
        return QueryAction.REFUSE.value
    return QueryAction.ASK_CLARIFICATION.value


def render_report(admission: ExpandedDevAdmission) -> str:
    """渲染人工审阅报告；不在报告阶段重新计算或改变准入决定。"""

    passed_text = "允许进入 9.4" if admission.summary.eligible_for_stage9_4 else "不允许进入 9.4"
    lines = [
        "# 阶段 9 SFT expanded dev 与 9.4 准入报告",
        "",
        "## 结论",
        "",
        f"- 9.3.16 决定：`{admission.summary.decision.value}`（{passed_text}）。",
        f"- checkpoint：`{admission.checkpoint.run_id}`。",
        f"- balanced dev：{admission.summary.case_count} 条；路线正确 "
        f"{admission.summary.route_correct_count} 条。",
        f"- overall route accuracy：`{admission.summary.overall_route_accuracy:.4f}`；"
        f"macro accuracy：`{admission.summary.route_macro_accuracy:.4f}`。",
        f"- 下一步：{admission.summary.next_step}",
        "",
        "## 数据和证据边界",
        "",
        "- 本次只运行 25 条 reviewed dev；同 registry 的 train/test case 未进入模型循环。",
        f"- heldout 推理结果数：`{admission.heldout_inference_result_count}`。",
        f"- ActionProvider：`{admission.action_provider}`；不证明真实 Milvus/Web 召回质量。",
        "- 回答由离线占位 executor 生成，因此 answer 分不能解释为模型真实回答质量。",
        "- 准入只评价 Planner 的 Action 路线、终态、安全边界和可执行性。",
        "",
        "## 冻结门禁",
        "",
        "| gate | observed | rule | passed | explanation |",
        "|---|---:|---:|---|---|",
    ]
    for check in admission.gate_checks:
        lines.append(
            f"| `{check.name}` | {check.observed:.4f} | "
            f"`{check.operator} {check.threshold}` | "
            f"`{str(check.passed).lower()}` | {check.explanation} |"
        )
    lines.extend(
        [
            "",
            "## 五路线结果",
            "",
            "| route bucket | correct/case | accuracy | threshold | passed |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for bucket in admission.buckets:
        lines.append(
            f"| `{bucket.route_bucket.value}` | "
            f"{bucket.route_correct_count}/{bucket.case_count} | "
            f"{bucket.route_accuracy:.4f} | {bucket.threshold:.4f} | "
            f"`{str(bucket.passed).lower()}` |"
        )
    lines.extend(
        [
            "",
            "## 终止动作汇总",
            "",
            "| expected | answer | ask_clarification | refuse | none |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for expected, actual_counts in admission.summary.terminal_confusion_matrix.items():
        lines.append(
            f"| `{expected}` | {actual_counts['answer']} | "
            f"{actual_counts['ask_clarification']} | "
            f"{actual_counts['refuse']} | {actual_counts['none']} |"
        )
    lines.extend(
        [
            "",
            "## 逐 case 结果",
            "",
            "| case_id | bucket | expected | actual | route | case gate | failures |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for case in admission.cases:
        expected = "<br>".join(" -> ".join(path) for path in case.expected_action_paths)
        actual = " -> ".join(case.actual_action_path)
        failures = ", ".join(case.failure_categories) or "none"
        lines.append(
            f"| `{case.case_id}` | `{case.route_bucket.value}` | "
            f"`{expected}` | `{actual}` | `{str(case.route_correct).lower()}` | "
            f"`{str(case.case_gate_passed).lower()}` | `{failures}` |"
        )
    lines.extend(
        [
            "",
            "## 输入身份",
            "",
            "| input | path | SHA256 |",
            "|---|---|---|",
        ]
    )
    for name, record in admission.inputs.items():
        lines.append(f"| `{name}` | `{record.path}` | `{record.sha256}` |")
    lines.extend(
        [
            "",
            "## 决策约束",
            "",
            (
                "- 当前组合通过全部冻结阈值。完成异地备份和哈希复核后，"
                "才能把它标记为 9.4 唯一准入组合。"
                if admission.summary.eligible_for_stage9_4
                else
                "- 当前 checkpoint 未通过。只能按失败 case 补独立 train-only 数据并训练 SFT v2；"
                "不得把 balanced dev 或 heldout test 原题写入训练集。"
            ),
            "- 只有 balanced dev、Action 契约、Reward 或 evaluator 发生变化时，才需要重跑 9.3.15A。",
            "",
        ]
    )
    return "\n".join(lines)


def write_admission_outputs(
    admission: ExpandedDevAdmission,
    *,
    decision_output_path: Path,
    report_path: Path,
    overwrite: bool = False,
) -> None:
    """写机器结论和 Markdown；默认拒绝静默覆盖上一次候选结果。"""

    _ensure_outputs_available(
        (decision_output_path, report_path),
        overwrite=overwrite,
    )
    decision_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    decision_output_path.write_text(
        json.dumps(
            admission.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_report(admission), encoding="utf-8")


def _ensure_outputs_available(
    paths: Iterable[Path],
    *,
    overwrite: bool,
) -> None:
    for path in paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"9.3.16 输出已存在，拒绝静默覆盖：{path}")


def _canonical_cases_sha256(cases: Iterable[PlannerEvalCase]) -> str:
    payload = [
        case.model_dump(mode="json")
        for case in sorted(cases, key=lambda item: item.case_id)
    ]
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须为 object：{path}")
    return payload


def _file_record(path: Path) -> FrozenFile:
    return FrozenFile(
        path=_logical(path),
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _logical(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _same_path(left: Path, right: Path) -> bool:
    return _resolve_project_path(left) == right.resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST,
    )
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--route-matrix", type=Path, default=DEFAULT_ROUTE_MATRIX)
    parser.add_argument("--reward-profile", type=Path, default=DEFAULT_REWARD_PROFILE)
    parser.add_argument(
        "--reward-validation",
        type=Path,
        default=DEFAULT_REWARD_VALIDATION,
    )
    parser.add_argument(
        "--reward-implementation",
        type=Path,
        default=DEFAULT_REWARD_IMPLEMENTATION,
    )
    parser.add_argument("--provider", default=SNAPSHOT_EXPECTED_PROVIDER_NAME)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL_OUTPUT)
    parser.add_argument(
        "--decision-output",
        type=Path,
        default=DEFAULT_DECISION_OUTPUT,
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只校验冻结输入和 checkpoint，不加载模型、不写推理产物。",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.preflight_only:
        contract = load_admission_contract(
            checkpoint_dir=args.checkpoint,
            cases_path=args.cases,
            split_manifest_path=args.split_manifest,
            snapshot_path=args.snapshot,
            route_matrix_path=args.route_matrix,
            reward_profile_path=args.reward_profile,
            reward_validation_path=args.reward_validation,
            reward_implementation_path=args.reward_implementation,
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "preflight_only": True,
                    "model_execution_performed": False,
                    "heldout_inference_result_count": 0,
                    "checkpoint_run_id": contract.checkpoint_manifest.run_id,
                    "case_count": len(contract.selected_cases),
                    "snapshot_id": contract.snapshot_id,
                    "reward_version": contract.reward_config.reward_version,
                    "balanced_dev_canonical_sha256": (
                        contract.balanced_dev_canonical_sha256
                    ),
                },
                ensure_ascii=False,
            )
        )
        return 0
    admission = run_expanded_dev_gate(
        checkpoint_dir=args.checkpoint,
        cases_path=args.cases,
        split_manifest_path=args.split_manifest,
        snapshot_path=args.snapshot,
        route_matrix_path=args.route_matrix,
        reward_profile_path=args.reward_profile,
        reward_validation_path=args.reward_validation,
        reward_implementation_path=args.reward_implementation,
        eval_output_path=args.eval_output,
        decision_output_path=args.decision_output,
        report_path=args.report,
        provider_name=args.provider,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "decision": admission.summary.decision.value,
                "eligible_for_stage9_4": admission.summary.eligible_for_stage9_4,
                "case_count": admission.summary.case_count,
                "route_macro_accuracy": admission.summary.route_macro_accuracy,
                "failed_case_count": len(admission.summary.failed_case_ids),
                "eval_output": _logical(args.eval_output),
                "decision_output": _logical(args.decision_output),
                "report": _logical(args.report),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
