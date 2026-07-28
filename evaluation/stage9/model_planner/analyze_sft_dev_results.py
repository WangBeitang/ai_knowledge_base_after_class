"""任务 9.3.11：分析 SFT v1 的 dev（开发集）结果并形成可审计失败归因。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_VERSION = "stage9-sft-v1-dev-analysis-v1"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "evaluation/stage9/artifacts/sft/sft_v1_dev_case_analysis.json"
DEFAULT_OUTPUT_REPORT = PROJECT_ROOT / "evaluation/stage9/artifacts/reports/阶段9-SFT-v1-dev分析报告.md"
_CASE_DURATION_PATTERN = re.compile(
    r"case_id=(?P<case_id>\S+) status=completed "
    r"duration_ms=(?P<duration_ms>\d+) action_path=(?P<action_path>.+)"
)


class AnalysisModel(BaseModel):
    """9.3.11 分析产物的公共 schema（数据结构）；拒绝未声明字段，避免报告静默漂移。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class FailureAttribution(str, Enum):
    """失败归因枚举；只描述可由当前产物支持的责任边界。"""

    MODEL_ERROR = "model_error"  # 模型错误：Planner 在合法上下文中选择了错误 Action 或终态。
    EVALUATOR_LIMITATION = "evaluator_limitation"  # 评测器限制：评分对象不是 Planner 真正负责的能力。
    PROVIDER_LIMITATION = "provider_limitation"  # 执行器限制：离线 Provider 不能证明真实 Milvus/Web 质量。
    LABEL_ISSUE = "label_issue"  # 标签问题：期望行为自相矛盾或经人工复核确认不合理。
    INSUFFICIENT_COVERAGE = "insufficient_coverage"  # 覆盖不足：样本量或 Action 路线不足以支持泛化结论。


class CaseAnalysisStatus(str, Enum):
    """逐 case（样本）分析状态；它不是训练通过/失败状态。"""

    PASSED = "passed"  # 期望路径、终态和关键行为均匹配，且没有额外观察。
    PASSED_WITH_COST_OBSERVATION = "passed_with_cost_observation"  # 路线合法，但比最短路径多走了 Action。
    EVALUATOR_LIMITED = "evaluator_limited"  # Planner 路由正确，低分主要来自评测器/Provider 能力边界。
    MODEL_ROUTE_FAILURE = "model_route_failure"  # Planner 路径、终态或 Web 行为与标签不一致。


class SourceFileRecord(AnalysisModel):
    """输入文件身份；logical_path（逻辑路径）用于归档复现，SHA256 用于防止内容漂移。"""

    logical_path: str = Field(description="文件在冻结归档或项目中的逻辑路径，不写本机绝对路径。")
    sha256: str = Field(min_length=64, max_length=64, description="输入文件内容的 SHA256。")


class DevAnalysisIdentity(AnalysisModel):
    """本次分析绑定的 dev run、checkpoint、snapshot、Reward 和输入文件身份。"""

    dev_run_id: str = Field(description="被分析的 dev eval 运行身份。")
    checkpoint_run_id: str = Field(description="被评测 checkpoint 的目录身份。")
    checkpoint: str = Field(description="dev eval 原始结果中记录的 checkpoint 逻辑路径。")
    snapshot_id: str = Field(description="离线环境快照身份；不能等同于真实线上语料状态。")
    reward_version: str = Field(description="生成原始分数的 Reward（奖励函数）版本。")
    planner_mode: str = Field(description="被分析的 Planner 模式，本任务固定为 sft。")
    action_provider: str = Field(description="原始评测使用的 Action Provider（动作执行器）。")
    source_eval: SourceFileRecord
    source_cases: SourceFileRecord
    source_dev_log: SourceFileRecord | None = None
    source_archive_name: str = Field(default="", description="冻结归档文件名；为空表示调用方未提供。")
    source_archive_sha256: str = Field(default="", description="冻结归档整体 SHA256；为空表示调用方未提供。")


class DevCaseAnalysis(AnalysisModel):
    """单条 dev case 的期望、实际、得分、证据边界和失败归因。"""

    case_id: str
    case_group: str
    query: str
    label_source: str
    human_review_status: str
    label_review_required: bool = Field(description="标签是否仍为 pending，提示结论需要人工标签复核。")
    expected_action_paths: list[list[str]]
    expected_terminal_action: str
    should_call_web: bool
    actual_action_path: list[str]
    actual_terminal_action: str
    terminal_reason_code: str
    path_match: bool
    terminal_match: bool
    web_behavior_match: bool
    total_reward: float
    component_scores: dict[str, float]
    retrieved_chunk_ids: list[int]
    citation_chunk_ids: list[int]
    trace_duration_ms: int = Field(description="离线 Environment step 计时；不包含完整模型推理墙钟时间。")
    wall_duration_ms: int | None = Field(
        default=None,
        description="从 dev_eval.log 提取的单 case 墙钟耗时，包含模型生成等待。",
    )
    analysis_status: CaseAnalysisStatus
    primary_attribution: FailureAttribution | None = Field(
        default=None,
        description="主要归因；正常样本为空，不能为了填字段制造失败。",
    )
    attributions: list[FailureAttribution] = Field(
        default_factory=list,
        description="当前证据支持的全部归因；允许评测器与 Provider 限制同时存在。",
    )
    model_route_failure: bool
    answer_score_interpretable_as_planner_quality: bool = Field(
        description="当前 answer 分是否能解释 Planner 质量；离线占位回答时为 false。",
    )
    real_retrieval_quality_verified: bool = Field(
        description="是否使用真实 Milvus/Web 验证检索质量；snapshot_expected_chunks 下固定为 false。",
    )
    findings: list[str]


class SftDevAnalysis(AnalysisModel):
    """9.3.11 完整机器可读产物；summary 保存聚合结论，cases 保存逐条证据。"""

    analysis_version: str = ANALYSIS_VERSION
    created_at: str
    identity: DevAnalysisIdentity
    case_count: int
    summary: dict[str, Any]
    terminal_confusion_matrix: dict[str, dict[str, int]]
    case_attribution_counts: dict[str, int]
    dataset_attributions: list[FailureAttribution]
    proved: list[str]
    not_proved: list[str]
    forbidden_inferences: list[str]
    recommended_next_step: str
    cases: list[DevCaseAnalysis]


def analyze_sft_dev_results(
        *,
        eval_path: Path,
        cases_path: Path,
        dev_log_path: Path | None = None,
        eval_logical_path: str | None = None,
        cases_logical_path: str | None = None,
        dev_log_logical_path: str | None = None,
        source_archive_name: str = "",
        source_archive_sha256: str = "",
) -> SftDevAnalysis:
    """读取已冻结的原始产物，校验身份后生成逐 case 归因；不运行模型、不修改 checkpoint。"""

    eval_payload = _read_json_object(eval_path)
    case_payloads = _read_jsonl_objects(cases_path)
    _validate_eval_header(eval_payload)
    eval_results = eval_payload["results"]
    dev_cases = {case["case_id"]: case for case in case_payloads if case.get("split") == "dev"}
    result_case_ids = [str(result["case_id"]) for result in eval_results]
    if len(result_case_ids) != len(set(result_case_ids)):
        raise ValueError("dev eval 结果包含重复 case_id")
    missing_cases = sorted(set(result_case_ids) - set(dev_cases))
    if missing_cases:
        raise ValueError(f"dev eval case 在 cases 文件中不存在：{missing_cases}")
    if int(eval_payload["case_count"]) != len(eval_results):
        raise ValueError("case_count 与 results 数量不一致")

    wall_durations = _parse_dev_log(dev_log_path) if dev_log_path else {}
    if dev_log_path:
        missing_log_cases = sorted(set(result_case_ids) - set(wall_durations))
        if missing_log_cases:
            raise ValueError(f"dev log 缺少 completed 记录：{missing_log_cases}")

    provider = str(eval_payload["planner_summaries"][0]["config"]["action_provider"])
    case_analyses = [
        _analyze_case(
            case=dev_cases[result["case_id"]],
            result=result,
            provider=provider,
            wall_duration_ms=wall_durations.get(result["case_id"]),
            dev_run_id=str(eval_payload["run_id"]),
        )
        for result in eval_results
    ]
    identity = _identity(
        eval_payload=eval_payload,
        eval_path=eval_path,
        cases_path=cases_path,
        dev_log_path=dev_log_path,
        eval_logical_path=eval_logical_path,
        cases_logical_path=cases_logical_path,
        dev_log_logical_path=dev_log_logical_path,
        source_archive_name=source_archive_name,
        source_archive_sha256=source_archive_sha256,
    )
    return _build_analysis(identity=identity, cases=case_analyses, eval_payload=eval_payload)


def _analyze_case(
        *,
        case: dict[str, Any],
        result: dict[str, Any],
        provider: str,
        wall_duration_ms: int | None,
        dev_run_id: str,
) -> DevCaseAnalysis:
    if str(result.get("run_id")) != dev_run_id:
        raise ValueError(f"case={case['case_id']} 的 run_id 与顶层不一致")
    expected_paths = [[str(action) for action in path] for path in case["acceptable_action_paths"]]
    actual_path = [str(action) for action in result["action_path"]]
    expected_terminal = _expected_terminal(case["expected_behavior"])
    actual_terminal = str(result.get("terminal_action") or "")
    path_match = actual_path in expected_paths
    metric_path_match = bool(result["metrics"]["path_match"])
    if path_match != metric_path_match:
        raise ValueError(f"case={case['case_id']} 的 path_match 与原始 metrics 不一致")
    terminal_match = actual_terminal == expected_terminal
    used_web = "web_search" in actual_path
    should_call_web = bool(case["expected_behavior"]["should_call_web"])
    web_behavior_match = used_web == should_call_web
    behavior_details = result["reward"]["components"]["behavior"]["details"]
    if str(behavior_details["expected_terminal"]) != expected_terminal:
        raise ValueError(f"case={case['case_id']} 的 expected_terminal 与 Reward 不一致")

    component_scores = {
        str(name): float(component["score"])
        for name, component in result["reward"]["components"].items()
    }
    trace_duration = int(result["usage"].get("duration_ms") or 0)
    findings: list[str] = []
    attributions: list[FailureAttribution] = []
    primary: FailureAttribution | None = None
    model_route_failure = not (path_match and terminal_match and web_behavior_match)
    answer_interpretable = not (
        bool(case["expected_behavior"]["should_answer"])
        and provider == "snapshot_expected_chunks"
    )
    real_retrieval_verified = provider not in {"snapshot_expected_chunks", "SnapshotExpectedChunkActionProvider"}

    if model_route_failure:
        status = CaseAnalysisStatus.MODEL_ROUTE_FAILURE
        primary = FailureAttribution.MODEL_ERROR
        attributions.append(primary)
        findings.append(
            f"期望路径={_path_set_text(expected_paths)}，实际路径={_path_text(actual_path)}。"
        )
        if not terminal_match:
            findings.append(f"期望终态={expected_terminal}，实际终态={actual_terminal}。")
        if not web_behavior_match:
            findings.append(
                f"should_call_web={str(should_call_web).lower()}，"
                f"实际 used_web={str(used_web).lower()}。"
            )
        if str(case.get("human_review_status")) != "reviewed":
            findings.append("该标签仍为 pending；模型错误候选成立，但严重程度需在 9.3.12 人工复核标签后冻结。")
    elif not answer_interpretable and component_scores.get("answer", 1.0) < 1.0:
        status = CaseAnalysisStatus.EVALUATOR_LIMITED
        primary = FailureAttribution.EVALUATOR_LIMITATION
        attributions.extend([
            FailureAttribution.EVALUATOR_LIMITATION,
            FailureAttribution.PROVIDER_LIMITATION,
        ])
        findings.extend([
            "Planner 路径与终态匹配，不能把 answer 低分归因为路由错误。",
            "OfflineRagEnvironment 只生成固定占位回答，没有调用正式答案生成模型。",
            "snapshot_expected_chunks 按 expected_chunks 构造候选，retrieval/citation 满分不代表真实召回质量。",
        ])
    else:
        cost_details = result["reward"]["components"]["cost"]["details"]
        extra_steps = int(cost_details.get("extra_steps") or 0)
        if extra_steps:
            status = CaseAnalysisStatus.PASSED_WITH_COST_OBSERVATION
            findings.append(
                f"实际路径属于 acceptable_action_paths，但比最短可接受路径多 {extra_steps} 个 Action。"
            )
        else:
            status = CaseAnalysisStatus.PASSED
            findings.append("实际路径、终态和 Web 行为均符合当前标签。")

    if not real_retrieval_verified and not bool(case["expected_behavior"]["should_answer"]):
        findings.append("本样本未形成真实 Milvus/Web 执行质量证据。")
    if wall_duration_ms is not None and wall_duration_ms > max(100, trace_duration * 10):
        findings.append(
            f"墙钟耗时={wall_duration_ms}ms，而结果内 trace_duration={trace_duration}ms；"
            "后者没有覆盖完整模型推理等待，不能用于真实延迟结论。"
        )

    return DevCaseAnalysis(
        case_id=str(case["case_id"]),
        case_group=str(case["case_group"]),
        query=str(case["query"]),
        label_source=str(case["label_source"]),
        human_review_status=str(case["human_review_status"]),
        label_review_required=str(case["human_review_status"]) != "reviewed",
        expected_action_paths=expected_paths,
        expected_terminal_action=expected_terminal,
        should_call_web=should_call_web,
        actual_action_path=actual_path,
        actual_terminal_action=actual_terminal,
        terminal_reason_code=str(result.get("terminal_reason_code") or ""),
        path_match=path_match,
        terminal_match=terminal_match,
        web_behavior_match=web_behavior_match,
        total_reward=float(result["metrics"]["total_reward"]),
        component_scores=component_scores,
        retrieved_chunk_ids=[int(value) for value in result.get("retrieved_chunk_ids", [])],
        citation_chunk_ids=[int(value) for value in result.get("citation_chunk_ids", [])],
        trace_duration_ms=trace_duration,
        wall_duration_ms=wall_duration_ms,
        analysis_status=status,
        primary_attribution=primary,
        attributions=attributions,
        model_route_failure=model_route_failure,
        answer_score_interpretable_as_planner_quality=answer_interpretable,
        real_retrieval_quality_verified=real_retrieval_verified,
        findings=findings,
    )


def _build_analysis(
        *,
        identity: DevAnalysisIdentity,
        cases: list[DevCaseAnalysis],
        eval_payload: dict[str, Any],
) -> SftDevAnalysis:
    case_count = len(cases)
    path_match_count = sum(case.path_match for case in cases)
    terminal_match_count = sum(case.terminal_match for case in cases)
    web_match_count = sum(case.web_behavior_match for case in cases)
    model_failure_count = sum(case.model_route_failure for case in cases)
    reviewed_count = sum(case.human_review_status == "reviewed" for case in cases)
    attribution_counts = Counter(
        attribution.value
        for case in cases
        for attribution in case.attributions
    )
    status_counts = Counter(case.analysis_status.value for case in cases)
    actual_path_counts = Counter(_path_text(case.actual_action_path) for case in cases)
    expected_path_family_counts = Counter(_expected_route_family(case) for case in cases)
    component_averages = eval_payload["planner_summaries"][0]["reward"]["component_average_scores"]

    summary = {
        "completed_case_count": int(eval_payload["planner_summaries"][0]["completed_case_count"]),
        "failed_execution_case_count": int(eval_payload["planner_summaries"][0]["failed_case_count"]),
        "path_match_count": path_match_count,
        "path_match_rate": _rate(path_match_count, case_count),
        "terminal_match_count": terminal_match_count,
        "terminal_match_rate": _rate(terminal_match_count, case_count),
        "web_behavior_match_count": web_match_count,
        "web_behavior_match_rate": _rate(web_match_count, case_count),
        "model_route_failure_case_count": model_failure_count,
        "reviewed_case_count": reviewed_count,
        "pending_review_case_count": case_count - reviewed_count,
        "average_total_reward": float(
            eval_payload["planner_summaries"][0]["reward"]["average_total_reward"]
        ),
        "component_average_scores": {
            str(name): float(value)
            for name, value in component_averages.items()
        },
        "analysis_status_counts": dict(sorted(status_counts.items())),
        "actual_path_counts": dict(sorted(actual_path_counts.items())),
        "expected_route_family_counts": dict(sorted(expected_path_family_counts.items())),
        "hyde_actual_case_count": sum("hyde_search" in case.actual_action_path for case in cases),
        "web_expected_case_count": sum(case.should_call_web for case in cases),
        "web_actual_case_count": sum("web_search" in case.actual_action_path for case in cases),
        "real_retrieval_quality_verified_case_count": sum(
            case.real_retrieval_quality_verified for case in cases
        ),
    }
    return SftDevAnalysis(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        identity=identity,
        case_count=case_count,
        summary=summary,
        terminal_confusion_matrix=_terminal_confusion_matrix(cases),
        case_attribution_counts=dict(sorted(attribution_counts.items())),
        dataset_attributions=[FailureAttribution.INSUFFICIENT_COVERAGE],
        proved=[
            f"{case_count} 条 dev case 均完成执行，没有 execution failure（执行失败）。",
            f"Action path 命中 {path_match_count}/{case_count}，终态命中 {terminal_match_count}/{case_count}。",
            "结构化 Planner 输出均通过原评测 format（格式）校验。",
        ],
        not_proved=[
            "snapshot_expected_chunks 不证明真实 Milvus/Web 召回、排序或引用质量。",
            "离线占位 answer builder 不证明 SFT 模型的最终答案生成质量。",
            "当前 dev 没有实际命中的 HyDE 路线，Web 期望样本也只有 1 条且标签 pending。",
            f"只有 {reviewed_count}/{case_count} 条 dev 标签为 reviewed，不能据此证明独立泛化。",
            "结果内 step duration 不覆盖完整模型推理墙钟时间，不能作为正式延迟指标。",
        ],
        forbidden_inferences=[
            f"不能把 average_total_reward={summary['average_total_reward']:.4f} 表述为模型质量已通过。",
            "不能把 answer="
            f"{summary['component_average_scores']['answer']:.4f} 表述为 Qwen3.5-4B 回答能力差。",
            "不能把 retrieval/citation="
            f"{summary['component_average_scores']['retrieval']:.4f}/"
            f"{summary['component_average_scores']['citation']:.4f} "
            "表述为真实检索与引用质量满分。",
            "不能仅凭 1 条实时路线错误决定立即重训或修改 checkpoint。",
        ],
        recommended_next_step=(
            "进入 9.3.12：审计 train/dev/test 来源、审核状态、泄漏组与 Action 路线分布，"
            "先冻结 balanced dev/test（路线均衡开发集/测试集）补数矩阵，不修改 SFT v1 checkpoint。"
        ),
        cases=cases,
    )


def render_markdown_report(analysis: SftDevAnalysis) -> str:
    """从结构化分析生成中文报告；报告只复述 JSON 中已固化的事实和边界。"""

    summary = analysis.summary
    identity = analysis.identity
    evaluator_limited_count = int(
        summary["analysis_status_counts"].get(CaseAnalysisStatus.EVALUATOR_LIMITED.value, 0)
    )
    lines = [
        "# 阶段 9 SFT v1 开发集结果分析与失败归因报告",
        "",
        "## 结论",
        "",
        f"- {analysis.case_count} 条 dev（开发集）均完成执行，但这只证明评测链路可运行，不等于模型质量通过。",
        f"- Action path（动作路径）命中 `{summary['path_match_count']}/{analysis.case_count}`"
        f"（`{summary['path_match_rate']:.4f}`）；终态命中 "
        f"`{summary['terminal_match_count']}/{analysis.case_count}`"
        f"（`{summary['terminal_match_rate']:.4f}`）。",
        f"- 发现 `{summary['model_route_failure_case_count']}` 条模型路线错误候选：实时召回公告问题"
        "没有进入 Web，而是走到 `local_search -> ask_clarification`。",
        f"- {evaluator_limited_count} 条回答型样本的 Planner 路由均正确；"
        "`answer=0`来自离线占位回答，不能归因为 Planner 路由失败。",
        "- 原评测使用 `snapshot_expected_chunks`，因此 `retrieval/citation=1.0`不代表真实 Milvus/Web 质量。",
        "- 当前证据不足以决定重训；下一步应先完成 9.3.12 的评测数据与路线覆盖审计。",
        "",
        "## 分析身份",
        "",
        "| 字段 | 值 |",
        "|---|---|",
        f"| `analysis_version（分析格式版本）` | `{analysis.analysis_version}` |",
        f"| `dev_run_id（开发集运行身份）` | `{identity.dev_run_id}` |",
        f"| `checkpoint_run_id（检查点身份）` | `{identity.checkpoint_run_id}` |",
        f"| `snapshot_id（环境快照身份）` | `{identity.snapshot_id}` |",
        f"| `reward_version（奖励版本）` | `{identity.reward_version}` |",
        f"| `action_provider（动作执行器）` | `{identity.action_provider}` |",
        f"| `source_eval_sha256（原始结果哈希）` | `{identity.source_eval.sha256}` |",
        f"| `source_cases_sha256（样本文件哈希）` | `{identity.source_cases.sha256}` |",
    ]
    if identity.source_dev_log:
        lines.append(f"| `source_dev_log_sha256（开发集日志哈希）` | `{identity.source_dev_log.sha256}` |")
    if identity.source_archive_name:
        lines.append(f"| `source_archive（来源归档）` | `{identity.source_archive_name}` |")
    if identity.source_archive_sha256:
        lines.append(f"| `source_archive_sha256（归档哈希）` | `{identity.source_archive_sha256}` |")

    lines.extend([
        "",
        "## 聚合结果",
        "",
        "| 指标 | 结果 | 解释边界 |",
        "|---|---:|---|",
        f"| `average_total_reward` | `{summary['average_total_reward']:.4f}` | 多个不同能力分项的加权平均，不能单独决定通过 |",
        f"| `path_match` | `{summary['path_match_count']}/{analysis.case_count}` | 只比较当前标签允许的 Action 路径 |",
        f"| `terminal_match` | `{summary['terminal_match_count']}/{analysis.case_count}` | 只比较 answer/clarify/refuse 终态 |",
        f"| `model_route_failure` | `{summary['model_route_failure_case_count']}` | 仍需考虑 pending 标签状态 |",
        f"| `reviewed labels` | `{summary['reviewed_case_count']}/{analysis.case_count}` | 其余标签尚未完成正式人工复核 |",
        f"| `HyDE actual` | `{summary['hyde_actual_case_count']}` | 没有 HyDE 实际路线证据 |",
        f"| `Web expected / actual` | `{summary['web_expected_case_count']} / {summary['web_actual_case_count']}` | 唯一 Web 期望样本没有调用 Web |",
        f"| `real retrieval verified` | `{summary['real_retrieval_quality_verified_case_count']}` | 当前 Provider 不能证明真实召回质量 |",
        "",
        "### Reward 分项",
        "",
        "| 分项 | 平均分 | 本次可以怎样解释 |",
        "|---|---:|---|",
    ])
    score_boundaries = {
        "format": "可证明结构化输出格式有效。",
        "retrieval": "由 expected_chunks 构造候选，只证明离线链路，不证明真实召回。",
        "citation": "引用来自离线期望候选，只证明引用链路，不证明线上引用质量。",
        "answer": "回答型样本使用固定占位回答，不能解释最终回答能力。",
        "behavior": "可用于当前标签下的路线/终态诊断，但 3 条标签仍为 pending。",
        "cost": "可比较路径步数；结果内 duration 不含完整模型推理时间。",
    }
    for name, value in summary["component_average_scores"].items():
        lines.append(f"| `{name}` | `{value:.4f}` | {score_boundaries[name]} |")

    lines.extend([
        "",
        "## Terminal action（终态动作）混淆矩阵",
        "",
        "| 期望终态 \\\\ 实际终态 | `answer` | `ask_clarification` | `refuse` |",
        "|---|---:|---:|---:|",
    ])
    for expected in ("answer", "ask_clarification", "refuse"):
        row = analysis.terminal_confusion_matrix.get(expected, {})
        lines.append(
            f"| `{expected}` | {row.get('answer', 0)} | "
            f"{row.get('ask_clarification', 0)} | {row.get('refuse', 0)} |"
        )

    lines.extend([
        "",
        "## 逐 case 分析",
        "",
        "| case_id | 标签状态 | 期望路径 | 实际路径 | Reward | 分析状态 | 主要归因 |",
        "|---|---|---|---|---:|---|---|",
    ])
    for case in analysis.cases:
        primary = case.primary_attribution.value if case.primary_attribution else "-"
        lines.append(
            f"| `{case.case_id}` | `{case.human_review_status}` | "
            f"{_markdown_path_set(case.expected_action_paths)} | `{_path_text(case.actual_action_path)}` | "
            f"`{case.total_reward:.3f}` | `{case.analysis_status.value}` | `{primary}` |"
        )

    model_failures = [case for case in analysis.cases if case.model_route_failure]
    evaluator_limited = [
        case for case in analysis.cases
        if case.analysis_status == CaseAnalysisStatus.EVALUATOR_LIMITED
    ]
    cost_observations = [
        case for case in analysis.cases
        if case.analysis_status == CaseAnalysisStatus.PASSED_WITH_COST_OBSERVATION
    ]
    lines.extend([
        "",
        "## 失败归因",
        "",
        "### 1. Model error（模型路线错误候选）",
        "",
    ])
    for case in model_failures:
        lines.append(f"- `{case.case_id}`：")
        lines.extend(f"  - {finding}" for finding in case.findings)
    lines.extend([
        "",
        "### 2. Evaluator limitation（评测器限制）",
        "",
        f"- 共 `{len(evaluator_limited)}` 条回答型样本属于这一类。",
        "- 它们的 `local_search -> answer`、终态、expected chunk 和 citation 均匹配。",
        "- `OfflineRagEnvironment._build_offline_answer()`只生成固定占位文本，没有运行正式答案生成。",
        f"- 因此这 {len(evaluator_limited)} 条的 `answer=0`应从 Planner 路由判断中剥离，"
        "不能解释为模型不会回答。",
        "",
        "### 3. Provider limitation（动作执行器限制）",
        "",
        "- `snapshot_expected_chunks`直接按 case 的 expected_chunks 构造本地候选。",
        "- 所以 `retrieval=1.0`与`citation=1.0`只证明离线状态机、Reward 和引用落盘链路可运行。",
        "- 本次没有真实 Milvus 召回，也没有真实 Web 结果，不能形成线上检索质量结论。",
        "",
        "### 4. Cost observation（路径成本观察）",
        "",
    ])
    for case in cost_observations:
        lines.append(f"- `{case.case_id}`：{case.findings[0]}")
    lines.extend([
        "",
        "## 已证明",
        "",
    ])
    lines.extend(f"- {item}" for item in analysis.proved)
    lines.extend([
        "",
        "## 未证明",
        "",
    ])
    lines.extend(f"- {item}" for item in analysis.not_proved)
    lines.extend([
        "",
        "## 禁止推导",
        "",
    ])
    lines.extend(f"- {item}" for item in analysis.forbidden_inferences)
    lines.extend([
        "",
        "## 下一步",
        "",
        analysis.recommended_next_step,
        "",
    ])
    return "\n".join(lines)


def write_analysis_outputs(
        *,
        analysis: SftDevAnalysis,
        output_json: Path,
        output_report: Path,
        overwrite: bool,
) -> None:
    """同时写机器可读 JSON 和人类可读报告；默认拒绝覆盖，保留首次正式归因。"""

    for path in (output_json, output_report):
        if path.exists() and not overwrite:
            raise FileExistsError(f"分析产物已存在；如需明确覆盖请传 --overwrite：{path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(analysis.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_report.write_text(render_markdown_report(analysis), encoding="utf-8")


def _identity(
        *,
        eval_payload: dict[str, Any],
        eval_path: Path,
        cases_path: Path,
        dev_log_path: Path | None,
        eval_logical_path: str | None,
        cases_logical_path: str | None,
        dev_log_logical_path: str | None,
        source_archive_name: str,
        source_archive_sha256: str,
) -> DevAnalysisIdentity:
    summary = eval_payload["planner_summaries"][0]
    checkpoint = str(summary["config"]["checkpoint"])
    return DevAnalysisIdentity(
        dev_run_id=str(eval_payload["run_id"]),
        checkpoint_run_id=Path(checkpoint).name,
        checkpoint=checkpoint,
        snapshot_id=str(eval_payload["snapshot_id"]),
        reward_version=str(eval_payload["reward_version"]),
        planner_mode=str(summary["planner_mode"]),
        action_provider=str(summary["config"]["action_provider"]),
        source_eval=SourceFileRecord(
            logical_path=eval_logical_path or eval_path.name,
            sha256=_sha256(eval_path),
        ),
        source_cases=SourceFileRecord(
            logical_path=cases_logical_path or cases_path.name,
            sha256=_sha256(cases_path),
        ),
        source_dev_log=SourceFileRecord(
            logical_path=dev_log_logical_path or dev_log_path.name,
            sha256=_sha256(dev_log_path),
        ) if dev_log_path else None,
        source_archive_name=source_archive_name,
        source_archive_sha256=source_archive_sha256,
    )


def _validate_eval_header(payload: dict[str, Any]) -> None:
    required = {"run_id", "split", "snapshot_id", "reward_version", "case_count", "planner_summaries", "results"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"dev eval 缺少字段：{missing}")
    if payload["split"] != "dev":
        raise ValueError("9.3.11 只分析 split=dev 的结果")
    if len(payload["planner_summaries"]) != 1:
        raise ValueError("9.3.11 要求恰好一个 SFT planner summary")
    summary = payload["planner_summaries"][0]
    if summary.get("planner_mode") != "sft":
        raise ValueError("9.3.11 只分析 planner_mode=sft")
    if summary.get("status") != "completed":
        raise ValueError("dev eval summary 尚未 completed")


def _expected_terminal(expected_behavior: dict[str, Any]) -> str:
    candidates = [
        action
        for flag, action in (
            ("should_answer", "answer"),
            ("should_ask_clarification", "ask_clarification"),
            ("should_refuse", "refuse"),
        )
        if bool(expected_behavior.get(flag))
    ]
    if len(candidates) != 1:
        raise ValueError(f"expected_behavior 必须且只能声明一个终态：{expected_behavior}")
    return candidates[0]


def _terminal_confusion_matrix(cases: list[DevCaseAnalysis]) -> dict[str, dict[str, int]]:
    actions = ("answer", "ask_clarification", "refuse")
    matrix = {expected: {actual: 0 for actual in actions} for expected in actions}
    for case in cases:
        if case.actual_terminal_action not in actions:
            raise ValueError(f"未知 terminal_action：{case.actual_terminal_action}")
        matrix[case.expected_terminal_action][case.actual_terminal_action] += 1
    return matrix


def _expected_route_family(case: DevCaseAnalysis) -> str:
    if case.should_call_web:
        return "web_required"
    if case.expected_terminal_action == "ask_clarification":
        return "ask_clarification"
    if case.expected_terminal_action == "refuse":
        return "refuse"
    if any("hyde_search" in path for path in case.expected_action_paths):
        return "local_answer_or_hyde"
    return "local_answer"


def _parse_dev_log(path: Path) -> dict[str, int]:
    durations: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _CASE_DURATION_PATTERN.search(line)
        if not match:
            continue
        case_id = match.group("case_id")
        if case_id in durations:
            raise ValueError(f"dev log 包含重复 completed case：{case_id}")
        durations[case_id] = int(match.group("duration_ms"))
    return durations


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是 object：{path}")
    return payload


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL 第 {line_number} 行不是 object：{path}")
        records.append(payload)
    return records


def _path_text(path: list[str]) -> str:
    return " -> ".join(path)


def _path_set_text(paths: list[list[str]]) -> str:
    return " | ".join(_path_text(path) for path in paths)


def _markdown_path_set(paths: list[list[str]]) -> str:
    """在 Markdown table（表格）中用换行分隔多条允许路径，避免竖线破坏列结构。"""

    return "<br>".join(f"`{_path_text(path)}`" for path in paths)


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="分析 SFT v1 的 dev 结果并生成 9.3.11 失败归因。")
    parser.add_argument("--eval", type=Path, required=True, help="冻结归档中的 sft_eval_dev.json。")
    parser.add_argument("--cases", type=Path, required=True, help="与 dev eval 配套的 planner_cases.jsonl。")
    parser.add_argument("--dev-log", type=Path, help="可选 dev_eval.log，用于恢复单 case 墙钟耗时。")
    parser.add_argument("--eval-logical-path")
    parser.add_argument("--cases-logical-path")
    parser.add_argument("--dev-log-logical-path")
    parser.add_argument("--source-archive-name", default="")
    parser.add_argument("--source-archive-sha256", default="")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_OUTPUT_REPORT)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    analysis = analyze_sft_dev_results(
        eval_path=args.eval,
        cases_path=args.cases,
        dev_log_path=args.dev_log,
        eval_logical_path=args.eval_logical_path,
        cases_logical_path=args.cases_logical_path,
        dev_log_logical_path=args.dev_log_logical_path,
        source_archive_name=args.source_archive_name,
        source_archive_sha256=args.source_archive_sha256,
    )
    write_analysis_outputs(
        analysis=analysis,
        output_json=args.output_json,
        output_report=args.output_report,
        overwrite=args.overwrite,
    )
    print(json.dumps({
        "ok": True,
        "analysis_version": analysis.analysis_version,
        "case_count": analysis.case_count,
        "path_match_count": analysis.summary["path_match_count"],
        "model_route_failure_case_count": analysis.summary["model_route_failure_case_count"],
        "output_json": str(args.output_json),
        "output_report": str(args.output_report),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
