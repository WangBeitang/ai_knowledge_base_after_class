"""
阶段 8.7 Planner SFT 数据导出。

SFT exporter 的中文含义是“监督微调数据导出器”。它读取 8.6 baseline runner 的
PlannerEvalResult，只把高质量、格式合法、未越过 split 边界的轨迹转换成训练样本。
这里不训练模型，也不计算 GRPO 组内 advantage；它只是阶段 9 SFT 训练前的数据闸门。

导出的每条样本对应一次 Planner 决策，而不是一整段最终答案。这样 SFT 学的是
“在当前 State/Observation 下输出哪个结构化 Action”，而不是背设备手册正文。
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.rag.evaluation.baseline_runner import BaselineEvalOutput
from app.rag.evaluation.case_schema import (
    CaseSplit,
    HumanReviewStatus,
    PlannerEvalCase,
    PlannerEvalResult,
    PlannerMode,
    PrivacyScope,
    load_planner_cases,
)
from app.rag.query.contracts import PlannerReasonCode, QueryAction


SFT_EXPORT_VERSION = "stage8-sft-export-v1"
DEFAULT_REWARD_THRESHOLD = 0.80
DEFAULT_ALLOWED_SPLITS = (CaseSplit.TRAIN, CaseSplit.DEV)
DISALLOWED_TRAINING_SPLITS = {CaseSplit.TEST, CaseSplit.DEMO_REGRESSION}
RETRIEVAL_ACTIONS = {QueryAction.LOCAL_SEARCH, QueryAction.HYDE_SEARCH, QueryAction.WEB_SEARCH}


# 第一部分：SFT 导出 schema。先固定训练样本和 manifest 形状，阶段 9 才能稳定读取。
class SftExportModel(BaseModel):
    """SFT 导出 schema 公共基类；拒绝未知字段，避免训练数据格式悄悄漂移。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


class SftExportConfig(SftExportModel):
    """
    SFT 导出过滤配置。

    reward_threshold 是自动筛选阈值；人工 reviewed 或与可接受路径一致的非 API 轨迹可作为
    额外进入条件。API teacher 轨迹更严格：低于阈值时一律不导出，避免强模型低质量轨迹
    污染 Planner SFT。
    """

    # 自动选择训练轨迹的最低 Reward。默认 0.80，阶段 9 前可以在 dev 上重新冻结。
    reward_threshold: float = Field(default=DEFAULT_REWARD_THRESHOLD, ge=0, le=1)
    # 允许导出的 split。默认 train/dev；test 和 demo_regression 会被硬拒绝。
    allowed_splits: tuple[CaseSplit, ...] = DEFAULT_ALLOWED_SPLITS
    # 私有文档样本是否必须人工复核。默认必须，避免未经确认的私有数据进入训练。
    require_private_review: bool = True

    @model_validator(mode="after")
    def reject_test_and_demo_splits(self) -> "SftExportConfig":
        """配置层禁止把 held-out test 或 demo 放入训练导出白名单。"""
        disallowed = DISALLOWED_TRAINING_SPLITS.intersection(self.allowed_splits)
        if disallowed:
            values = ", ".join(sorted(split.value for split in disallowed))
            raise ValueError(f"SFT 导出不能允许 test/demo split：{values}")
        if not self.allowed_splits:
            raise ValueError("allowed_splits 不能为空")
        return self


class SftPlannerSample(SftExportModel):
    """
    单条 Planner SFT 样本。

    它只保存裁剪后的输入上下文和目标 PlannerDecision，不保存完整 chunk 正文、不保存答案
    Prompt，也不保存模型私有思维链。
    """

    # 训练样本稳定 ID。由 trace_id、turn_index 和 action 组合生成，便于回溯。
    sample_id: str = Field(min_length=1)
    # 来源评测 case ID。
    source_case_id: str = Field(min_length=1)
    # 来源离线轨迹 ID，对应 PlannerEvalResult.trace_id。
    source_trace_id: str = Field(min_length=1)
    # 来源 split，只允许 train/dev。
    split: CaseSplit
    # 第几次 Planner 决策，从 1 开始。
    turn_index: int = Field(ge=1)
    # 裁剪后的 Planner 输入上下文。只保留结构化状态、Action 历史和有限 Observation 摘要。
    input_context: dict[str, Any]
    # 目标结构化 PlannerDecision，例如 {"action": "local_search", "query": "..."}。
    target_decision: dict[str, Any]
    # 来源轨迹 Reward 摘要。只保存总分、关键分项和进入原因，不保存完整 Trace。
    reward_summary: dict[str, Any]
    # 标签来源：rule、api_teacher、local_base 或 manual。
    label_source: str = Field(min_length=1)
    # 复核状态：reviewed 表示人工复核样本，auto_selected 表示自动按 Reward/路径筛选。
    review_status: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_training_boundaries(self) -> "SftPlannerSample":
        """样本层再次拒绝 test/demo，并检查不会携带明显的 chunk 正文字段。"""
        if self.split in DISALLOWED_TRAINING_SPLITS:
            raise ValueError("SFT 样本不能来自 test/demo split")
        if _contains_forbidden_payload_key(self.model_dump(mode="python")):
            raise ValueError("SFT 样本不能包含完整 chunk 正文、私有思维链或答案 Prompt 字段")
        return self


class SftExportManifest(SftExportModel):
    """
    SFT 导出清单。

    Manifest 的中文含义是“清单”。它记录数据从哪里来、哪些过滤规则生效、最终各类来源
    占比是多少，防止阶段 9 训练时只看到 JSONL 而不知道样本边界。
    """

    # manifest 稳定 ID。
    manifest_id: str = Field(min_length=1)
    # 导出脚本版本。
    export_version: str = SFT_EXPORT_VERSION
    # 创建时间，UTC ISO 字符串。
    created_at: str = Field(min_length=1)
    # 来源 baseline runner run_id。
    source_run_id: str = Field(min_length=1)
    # 来源环境快照 ID。
    snapshot_id: str = Field(min_length=1)
    # 来源 Reward 版本。
    reward_version: str = Field(min_length=1)
    # 自动选择阈值。
    reward_threshold: float = Field(ge=0, le=1)
    # 本次允许导出的 split。
    allowed_splits: list[CaseSplit]
    # 去重后的来源 case 数。
    exported_case_count: int = Field(ge=0)
    # 进入导出的轨迹数。
    exported_trajectory_count: int = Field(ge=0)
    # 最终 SFT 样本数；一条轨迹会拆成多条 Planner 决策样本。
    sample_count: int = Field(ge=0)
    # 按 label_source 统计的样本数。
    source_counts: dict[str, int] = Field(default_factory=dict)
    # 按 label_source 统计的样本占比。
    source_ratios: dict[str, float] = Field(default_factory=dict)
    # 按 split 统计的样本数。
    split_counts: dict[str, int] = Field(default_factory=dict)
    # 按 review_status 统计的样本数。
    review_status_counts: dict[str, int] = Field(default_factory=dict)
    # 被过滤轨迹的首要原因统计。
    filter_counts: dict[str, int] = Field(default_factory=dict)
    # 实际生效的过滤规则，写给人看，也方便报告复用。
    filter_rules: list[str] = Field(default_factory=list)
    # 明确不导出的字段边界。
    excluded_payloads: list[str] = Field(default_factory=list)


class SftExportResult(SftExportModel):
    """SFT 导出的内存结果，包含样本和 manifest。"""

    samples: list[SftPlannerSample] = Field(default_factory=list)
    manifest: SftExportManifest


# 第二部分：主入口。阶段 9 或命令行脚本应该从 export_sft_samples 进入。
def export_sft_samples(
        *,
        eval_output: BaselineEvalOutput,
        cases: list[PlannerEvalCase],
        config: SftExportConfig | None = None,
) -> SftExportResult:
    """
    从 baseline eval result 导出 Planner SFT 样本。

    数据流：先按 case_id 找到人工标注；再逐条评测结果做 split/Reward/格式/隐私过滤；
    通过的轨迹按 action_path 拆成多条单步 SFT 样本；最后生成 manifest。
    """
    active_config = config or SftExportConfig()
    case_by_id = _case_map(cases)
    samples: list[SftPlannerSample] = []
    accepted_results: list[PlannerEvalResult] = []
    filter_counts: Counter[str] = Counter()

    for result in eval_output.results:
        case = case_by_id.get(result.case_id)
        if case is None:
            raise ValueError(f"评测结果引用了 cases 文件中不存在的 case_id：{result.case_id}")
        decision = _selection_decision(result, case, active_config)
        if not decision.accepted:
            filter_counts[decision.primary_reason] += 1
            continue
        accepted_results.append(result)
        samples.extend(_samples_from_result(
            result=result,
            case=case,
            selection_reason=decision.primary_reason,
            config=active_config,
        ))

    manifest = _build_manifest(
        eval_output=eval_output,
        samples=samples,
        accepted_results=accepted_results,
        filter_counts=filter_counts,
        config=active_config,
    )
    return SftExportResult(samples=samples, manifest=manifest)


# 第三部分：文件读写。CLI 与单元测试共用，避免脚本里重复 JSON/JSONL 处理。
def load_baseline_eval_output(path: str | Path) -> BaselineEvalOutput:
    """读取 8.6 baseline runner 输出文件，并校验 BaselineEvalOutput schema。"""
    eval_path = Path(path)
    return BaselineEvalOutput.model_validate_json(eval_path.read_text(encoding="utf-8"))


def export_sft_samples_from_files(
        *,
        eval_result_path: str | Path,
        cases_path: str | Path,
        output_path: str | Path,
        manifest_path: str | Path,
        config: SftExportConfig | None = None,
) -> SftExportResult:
    """从文件读取评测结果和 cases，导出 JSONL 样本和 manifest。"""
    eval_output = load_baseline_eval_output(eval_result_path)
    cases = load_planner_cases(cases_path)
    result = export_sft_samples(eval_output=eval_output, cases=cases, config=config)
    write_sft_samples(result.samples, output_path)
    write_sft_manifest(result.manifest, manifest_path)
    return result


def write_sft_samples(samples: list[SftPlannerSample], path: str | Path) -> None:
    """把 SFT 样本写成 JSONL；没有样本时也生成空文件，便于流水线判断。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(sample.model_dump(mode="json"), ensure_ascii=False)
        for sample in samples
    ]
    output_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def write_sft_manifest(manifest: SftExportManifest, path: str | Path) -> None:
    """把导出 manifest 写成 UTF-8 JSON。"""
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# 第四部分：过滤逻辑。这里决定“哪些轨迹可以教模型”。
class _SelectionDecision(BaseModel):
    """内部选择结果。primary_reason 会进入 manifest.filter_counts 或 reward_summary.selected_by。"""

    accepted: bool
    primary_reason: str


def _selection_decision(
        result: PlannerEvalResult,
        case: PlannerEvalCase,
        config: SftExportConfig,
) -> _SelectionDecision:
    if result.split in DISALLOWED_TRAINING_SPLITS or result.split not in config.allowed_splits:
        return _reject("split_not_allowed")
    if case.human_review_status == HumanReviewStatus.REJECTED:
        return _reject("case_review_rejected")
    if config.require_private_review and case.privacy_scope == PrivacyScope.PRIVATE_USER:
        if case.human_review_status != HumanReviewStatus.REVIEWED:
            return _reject("private_case_not_reviewed")
    if result.errors:
        return _reject("trajectory_has_errors")
    if not _reward_format_valid(result):
        return _reject("format_invalid")
    if _invalid_citation_count(result) > 0:
        return _reject("invalid_citation")
    if _terminal_behavior_mismatch(result, case):
        return _reject("terminal_behavior_mismatch")

    total_reward = _total_reward(result)
    manual_reviewed = case.human_review_status == HumanReviewStatus.REVIEWED
    path_match = _path_match(result)

    if result.planner_mode == PlannerMode.API and total_reward < config.reward_threshold:
        return _reject("api_reward_below_threshold")
    if total_reward >= config.reward_threshold:
        return _accept("reward_threshold")
    if manual_reviewed:
        return _accept("human_reviewed")
    if path_match and result.planner_mode == PlannerMode.RULE:
        return _accept("rule_path_match")
    return _reject("reward_below_threshold")


def _accept(reason: str) -> _SelectionDecision:
    return _SelectionDecision(accepted=True, primary_reason=reason)


def _reject(reason: str) -> _SelectionDecision:
    return _SelectionDecision(accepted=False, primary_reason=reason)


# 第五部分：样本构造。把一条完整轨迹拆成多条“输入上下文 -> 下一步 Action”。
def _samples_from_result(
        *,
        result: PlannerEvalResult,
        case: PlannerEvalCase,
        selection_reason: str,
        config: SftExportConfig,
) -> list[SftPlannerSample]:
    samples: list[SftPlannerSample] = []
    actions = list(result.action_path)
    for index, action in enumerate(actions):
        turn_index = index + 1
        samples.append(SftPlannerSample(
            sample_id=_sample_id(result, turn_index, action),
            source_case_id=result.case_id,
            source_trace_id=result.trace_id or result.run_id,
            split=result.split,
            turn_index=turn_index,
            input_context=_input_context(case, result, actions[:index]),
            target_decision=_target_decision(case, result, action, is_terminal=(index == len(actions) - 1)),
            reward_summary=_reward_summary(result, selection_reason, config),
            label_source=_label_source(result, case),
            review_status=_review_status(case),
        ))
    return samples


def _input_context(
        case: PlannerEvalCase,
        result: PlannerEvalResult,
        previous_actions: list[QueryAction],
) -> dict[str, Any]:
    """
    构造裁剪后的 PlannerContext。

    这里故意不放完整 chunk 正文，也不放 expected_answer_points。训练模型只需要学会基于
    当前结构化状态和 Observation 摘要选择下一步 Action，不能直接看到人工答案要点。
    """
    return {
        "query": case.query,
        "current_query": case.query,
        "dataset_ids": list(case.dataset_ids),
        "subject_ids": list(case.expected_subject_ids),
        "standard_subject_names": list(case.expected_subject_names),
        "query_identifiers": dict(case.expected_identifiers),
        "web_search_allowed": bool(case.expected_behavior.should_call_web),
        "planner_step": len(previous_actions),
        "allowed_actions": _allowed_actions(case),
        "action_history": [
            {"step": step, "action": action.value}
            for step, action in enumerate(previous_actions, start=1)
        ],
        "latest_observation": _latest_observation_summary(result, previous_actions),
    }


def _target_decision(
        case: PlannerEvalCase,
        result: PlannerEvalResult,
        action: QueryAction,
        *,
        is_terminal: bool,
) -> dict[str, Any]:
    return {
        "action": action.value,
        "query": case.query,
        "reason_code": _reason_code_for_target(result, action, is_terminal=is_terminal),
    }


def _latest_observation_summary(
        result: PlannerEvalResult,
        previous_actions: list[QueryAction],
) -> dict[str, Any] | None:
    if not previous_actions:
        return None
    previous_action = previous_actions[-1]
    if previous_action not in RETRIEVAL_ACTIONS:
        return None
    retrieved_chunk_ids = [str(chunk_id) for chunk_id in result.retrieved_chunk_ids]
    return {
        "action": previous_action.value,
        "status": "success" if retrieved_chunk_ids else "empty",
        "candidate_count": len(retrieved_chunk_ids),
        "retrieved_chunk_ids": retrieved_chunk_ids[:10],
        "citation_chunk_ids": [],
        "contains_full_chunk_content": False,
    }


def _allowed_actions(case: PlannerEvalCase) -> list[str]:
    actions = [
        QueryAction.LOCAL_SEARCH,
        QueryAction.HYDE_SEARCH,
        QueryAction.ANSWER,
        QueryAction.ASK_CLARIFICATION,
        QueryAction.REFUSE,
    ]
    if case.expected_behavior.should_call_web:
        actions.insert(2, QueryAction.WEB_SEARCH)
    return [action.value for action in actions]


# 第六部分：manifest 与统计。这里回答“导出了多少、来自哪里、过滤掉什么”。
def _build_manifest(
        *,
        eval_output: BaselineEvalOutput,
        samples: list[SftPlannerSample],
        accepted_results: list[PlannerEvalResult],
        filter_counts: Counter[str],
        config: SftExportConfig,
) -> SftExportManifest:
    source_counts = Counter(sample.label_source for sample in samples)
    split_counts = Counter(sample.split.value for sample in samples)
    review_counts = Counter(sample.review_status for sample in samples)
    return SftExportManifest(
        manifest_id=f"sft_manifest_{eval_output.run_id}",
        created_at=datetime.now(UTC).isoformat(),
        source_run_id=eval_output.run_id,
        snapshot_id=eval_output.snapshot_id,
        reward_version=eval_output.reward_version,
        reward_threshold=config.reward_threshold,
        allowed_splits=list(config.allowed_splits),
        exported_case_count=len({result.case_id for result in accepted_results}),
        exported_trajectory_count=len(accepted_results),
        sample_count=len(samples),
        source_counts=dict(sorted(source_counts.items())),
        source_ratios=_ratios(source_counts),
        split_counts=dict(sorted(split_counts.items())),
        review_status_counts=dict(sorted(review_counts.items())),
        filter_counts=dict(sorted(filter_counts.items())),
        filter_rules=[
            "test/demo split 永不导出",
            "格式非法、执行错误、无效引用轨迹不导出",
            "API teacher 轨迹必须达到 Reward 阈值",
            "私有文档样本默认必须人工 reviewed",
            "导出样本不包含完整 chunk 正文、答案 Prompt 或模型私有思维链",
        ],
        excluded_payloads=[
            "full_chunk_content",
            "answer_prompt",
            "private_chain_of_thought",
            "model_reasoning_text",
        ],
    )


def _ratios(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {
        key: value / total
        for key, value in sorted(counter.items())
    }


# 第七部分：内部读取和判断工具。放在末尾，保持主流程阅读顺序清楚。
def _case_map(cases: list[PlannerEvalCase]) -> dict[str, PlannerEvalCase]:
    mapping: dict[str, PlannerEvalCase] = {}
    for case in cases:
        if case.case_id in mapping:
            raise ValueError(f"case_id 重复：{case.case_id}")
        mapping[case.case_id] = case
    return mapping


def _total_reward(result: PlannerEvalResult) -> float:
    return float(result.reward.get("total_reward", result.metrics.get("total_reward", 0.0)) or 0.0)


def _reward_format_valid(result: PlannerEvalResult) -> bool:
    return bool(result.reward.get("format_valid", result.metrics.get("format_valid", False)))


def _invalid_citation_count(result: PlannerEvalResult) -> int:
    try:
        return int(result.reward["components"]["citation"]["details"].get("invalid_citation_count", 0))
    except (KeyError, TypeError, AttributeError, ValueError):
        return 0


def _path_match(result: PlannerEvalResult) -> bool:
    metric_value = result.metrics.get("path_match")
    if metric_value is not None:
        return bool(metric_value)
    try:
        return bool(result.reward["components"]["behavior"]["details"].get("path_match", False))
    except (KeyError, TypeError, AttributeError):
        return False


def _terminal_behavior_mismatch(result: PlannerEvalResult, case: PlannerEvalCase) -> bool:
    if case.expected_behavior.should_answer:
        return result.terminal_action != QueryAction.ANSWER
    if case.expected_behavior.should_refuse:
        return result.terminal_action != QueryAction.REFUSE
    return result.terminal_action != QueryAction.ASK_CLARIFICATION


def _label_source(result: PlannerEvalResult, case: PlannerEvalCase) -> str:
    if result.planner_mode == PlannerMode.API:
        return "api_teacher"
    if result.planner_mode == PlannerMode.RULE:
        return "rule"
    if case.human_review_status == HumanReviewStatus.REVIEWED:
        return "manual"
    return result.planner_mode.value


def _review_status(case: PlannerEvalCase) -> str:
    return "reviewed" if case.human_review_status == HumanReviewStatus.REVIEWED else "auto_selected"


def _reward_summary(
        result: PlannerEvalResult,
        selection_reason: str,
        config: SftExportConfig,
) -> dict[str, Any]:
    components = result.reward.get("components", {})
    return {
        "reward_version": result.reward_version,
        "total_reward": _total_reward(result),
        "raw_total_reward": result.reward.get("raw_total_reward"),
        "format_valid": _reward_format_valid(result),
        "path_match": _path_match(result),
        "selected_by": selection_reason,
        "reward_threshold": config.reward_threshold,
        "component_scores": {
            name: value.get("score")
            for name, value in components.items()
            if isinstance(value, dict)
        },
    }


def _reason_code_for_target(
        result: PlannerEvalResult,
        action: QueryAction,
        *,
        is_terminal: bool,
) -> str:
    if is_terminal and result.terminal_reason_code:
        return result.terminal_reason_code
    defaults = {
        QueryAction.LOCAL_SEARCH: PlannerReasonCode.INITIAL_LOCAL_SEARCH,
        QueryAction.HYDE_SEARCH: PlannerReasonCode.LOCAL_LOW_SCORE,
        QueryAction.WEB_SEARCH: PlannerReasonCode.REALTIME_QUERY,
        QueryAction.ANSWER: PlannerReasonCode.LOCAL_EVIDENCE_SUFFICIENT,
        QueryAction.ASK_CLARIFICATION: PlannerReasonCode.SUBJECT_REQUIRED,
        QueryAction.REFUSE: PlannerReasonCode.SAFE_GUARD_TRIGGERED,
    }
    return defaults[action].value


def _sample_id(result: PlannerEvalResult, turn_index: int, action: QueryAction) -> str:
    source_trace_id = result.trace_id or result.run_id
    return f"sft_{source_trace_id}_{turn_index:02d}_{action.value}"


def _contains_forbidden_payload_key(payload: Any) -> bool:
    forbidden_keys = {
        "content",
        "content_excerpt",
        "full_chunk_content",
        "answer_prompt",
        "private_chain_of_thought",
        "model_reasoning_text",
    }
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in forbidden_keys:
                return True
            if _contains_forbidden_payload_key(value):
                return True
    if isinstance(payload, list):
        return any(_contains_forbidden_payload_key(value) for value in payload)
    return False


def parse_allowed_splits(value: str | Iterable[str | CaseSplit]) -> tuple[CaseSplit, ...]:
    """解析 allowed_splits 参数，支持 'train,dev' 或列表输入。"""
    raw_values: Iterable[str | CaseSplit]
    if isinstance(value, str):
        raw_values = value.split(",")
    else:
        raw_values = value
    splits: list[CaseSplit] = []
    seen: set[CaseSplit] = set()
    for raw_value in raw_values:
        if isinstance(raw_value, CaseSplit):
            split = raw_value
        else:
            normalized_value = str(raw_value).strip()
            if not normalized_value:
                continue
            split = CaseSplit(normalized_value)
        if split not in seen:
            splits.append(split)
            seen.add(split)
    return tuple(splits)


__all__ = [
    "DEFAULT_REWARD_THRESHOLD",
    "SFT_EXPORT_VERSION",
    "SftExportConfig",
    "SftExportManifest",
    "SftExportResult",
    "SftPlannerSample",
    "export_sft_samples",
    "export_sft_samples_from_files",
    "load_baseline_eval_output",
    "parse_allowed_splits",
    "write_sft_manifest",
    "write_sft_samples",
]
