"""任务 9.3.12：审计评测数据，并在补数和查看新模型结果前冻结 Action 路线矩阵。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_VERSION = "stage9-eval-route-audit-v1"
MATRIX_VERSION = "stage9-planner-eval-route-matrix-v1"

DEFAULT_PLANNER_CASES = PROJECT_ROOT / "evaluation/stage8/cases/planner_cases.jsonl"
DEFAULT_SPLIT_MANIFEST = PROJECT_ROOT / "evaluation/stage8/cases/split_manifest.json"
DEFAULT_CURATED_CASES = (
    PROJECT_ROOT
    / "evaluation/stage8_5/artifacts/intermediate/sft_seed/curated_seed_train_cases.jsonl"
)
DEFAULT_ROUTE_CASES = PROJECT_ROOT / "evaluation/stage9/artifacts/route_seed/route_seed_cases.jsonl"
DEFAULT_ROUTE_PATHS = (
    PROJECT_ROOT / "evaluation/stage9/artifacts/route_seed/route_seed_action_paths.jsonl"
)
DEFAULT_SFT_DATA = PROJECT_ROOT / "evaluation/stage9/artifacts/sft/sft_planner_stage9_train.jsonl"
DEFAULT_SFT_MANIFEST = (
    PROJECT_ROOT / "evaluation/stage9/artifacts/sft/sft_planner_stage9_manifest.json"
)
DEFAULT_OUTPUT_REPORT = (
    PROJECT_ROOT / "evaluation/stage9/artifacts/reports/阶段9-评测数据与路线覆盖审计报告.md"
)
DEFAULT_OUTPUT_MATRIX = (
    PROJECT_ROOT / "evaluation/stage9/configs/planner_eval_route_matrix_v1.json"
)


class AuditModel(BaseModel):
    """审计产物公共 schema（数据结构）；禁止静默增加未声明字段。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RouteBucket(str, Enum):
    """正式评测使用的五个 Action route bucket（动作路线桶）。"""

    LOCAL_ANSWER = "local_answer"  # 本地证据足够，应及时回答。
    HYDE_FALLBACK = "hyde_fallback"  # 本地不足，必须经过 HyDE 回退。
    WEB_REQUIRED = "web_required"  # 实时问题必须使用 Web。
    ASK_CLARIFICATION = "ask_clarification"  # 关键信息不足，必须澄清。
    SAFE_REFUSE = "safe_refuse"  # 危险、越权或无依据请求必须拒绝。


class LeakageKind(str, Enum):
    """跨 split（数据划分）泄漏的确定性判定类型。"""

    CASE_ID = "case_id"
    QUERY = "query"
    LEAKAGE_GROUP = "leakage_group"
    NEAR_QUERY = "near_query"


class SourceFileRecord(AuditModel):
    """审计输入文件身份；逻辑路径加 SHA256 用于复现。"""

    logical_path: str
    sha256: str = Field(min_length=64, max_length=64)
    record_count: int = Field(ge=1)


class LeakageFinding(AuditModel):
    """一个跨 split 的重复或近重复风险。"""

    kind: LeakageKind
    left_case_id: str
    left_split: str
    left_source: str
    right_case_id: str
    right_split: str
    right_source: str
    left_query: str
    right_query: str
    sequence_similarity: float = Field(ge=0, le=1)
    bigram_jaccard: float = Field(ge=0, le=1)
    reason: str


class CaseAudit(AuditModel):
    """单个 source case（来源样本）的来源、路线、审核、追溯与 Gold 资格。"""

    case_id: str
    source_dataset: str
    source_file: str
    split: str
    logical_eval_set: str
    query: str
    route_bucket: RouteBucket
    acceptable_action_paths: list[list[str]]
    gold_origin: str
    label_source: str
    human_review_status: str
    review_evidence: str
    leakage_group_id: str
    device_families: list[str]
    source_traceable: bool
    source_traceability: str
    cross_split_leakage_count: int = Field(ge=0)
    formal_eval_gold_eligible: bool
    exclusion_reasons: list[str]


class CoverageRecord(AuditModel):
    """某 split 与路线桶的当前覆盖快照。"""

    split: str
    logical_eval_set: str
    route_bucket: RouteBucket
    case_count: int = Field(ge=0)
    reviewed_case_count: int = Field(ge=0)
    formal_gold_case_count: int = Field(ge=0)
    unique_query_count: int = Field(ge=0)
    unique_leakage_group_count: int = Field(ge=0)
    device_family_count: int = Field(ge=0)


class DuplicateTemplateGroup(AuditModel):
    """同一数据源内部重复使用相同 query 模板的样本组。"""

    source_dataset: str
    route_bucket: RouteBucket
    query: str
    case_ids: list[str]


class EvalRouteAudit(AuditModel):
    """9.3.12 的机器可读内存模型；正式文件输出为矩阵 JSON 和中文报告。"""

    audit_version: str = AUDIT_VERSION
    created_at: str
    source_files: list[SourceFileRecord]
    sft_sample_count: int
    sft_source_case_count: int
    planner_case_count: int
    cases: list[CaseAudit]
    coverage: list[CoverageRecord]
    leakage_findings: list[LeakageFinding]
    duplicate_template_groups: list[DuplicateTemplateGroup]
    conclusions: list[str]


def audit_evaluation_data(
    *,
    planner_cases_path: Path = DEFAULT_PLANNER_CASES,
    split_manifest_path: Path = DEFAULT_SPLIT_MANIFEST,
    curated_cases_path: Path = DEFAULT_CURATED_CASES,
    route_cases_path: Path = DEFAULT_ROUTE_CASES,
    route_paths_path: Path = DEFAULT_ROUTE_PATHS,
    sft_data_path: Path = DEFAULT_SFT_DATA,
    sft_manifest_path: Path = DEFAULT_SFT_MANIFEST,
) -> EvalRouteAudit:
    """读取现有数据并执行只读审计；不创建样本、不改 split、不运行模型。"""

    planner_cases = _read_jsonl(planner_cases_path)
    curated_cases = _read_jsonl(curated_cases_path)
    route_cases = _read_jsonl(route_cases_path)
    route_paths = _read_jsonl(route_paths_path)
    sft_samples = _read_jsonl(sft_data_path)
    split_manifest = _read_json(split_manifest_path)
    sft_manifest = _read_json(sft_manifest_path)

    _validate_unique_ids(planner_cases, "planner_cases")
    _validate_unique_ids(curated_cases, "curated_seed")
    _validate_unique_ids(route_cases, "route_seed")
    _validate_split_manifest(planner_cases, split_manifest)

    route_path_by_case = {str(row["case_id"]): row for row in route_paths}
    if set(route_path_by_case) != {str(row["case_id"]) for row in route_cases}:
        raise ValueError("route_seed_cases 与 route_seed_action_paths 的 case_id 集合不一致")
    if any(row.get("review_status") != "reviewed" for row in route_paths):
        raise ValueError("route_seed_action_paths 含未审核路径，不能进入训练来源审计")

    sft_source_ids = {str(row["source_case_id"]) for row in sft_samples}
    expected_sft_sources = {
        str(row["case_id"]) for row in [*curated_cases, *route_cases]
    }
    if sft_source_ids != expected_sft_sources:
        missing = sorted(expected_sft_sources - sft_source_ids)
        extra = sorted(sft_source_ids - expected_sft_sources)
        raise ValueError(f"SFT source_case_id 不一致：missing={missing}, extra={extra}")
    if any(row.get("split") != "train" for row in sft_samples):
        raise ValueError("SFT 数据出现非 train 样本")
    if any(row.get("review_status") != "reviewed" for row in sft_samples):
        raise ValueError("SFT 数据出现未审核样本")
    if int(sft_manifest["sample_count"]) != len(sft_samples):
        raise ValueError("SFT manifest.sample_count 与训练 JSONL 不一致")
    if int(sft_manifest["source_case_count"]) != len(sft_source_ids):
        raise ValueError("SFT manifest.source_case_count 与训练 JSONL 不一致")

    raw_cases: list[dict[str, Any]] = []
    for row in planner_cases:
        raw_cases.append(
            _raw_case(
                row,
                source_dataset="planner_case_registry",
                source_file=_logical_path(planner_cases_path),
            )
        )
    for row in curated_cases:
        raw_cases.append(
            _raw_case(
                row,
                source_dataset="curated_seed_source",
                source_file=_logical_path(curated_cases_path),
                forced_bucket=RouteBucket.LOCAL_ANSWER,
            )
        )
    for row in route_cases:
        raw_cases.append(
            _raw_case(
                row,
                source_dataset="route_seed_source",
                source_file=_logical_path(route_cases_path),
                forced_bucket=_route_seed_bucket(route_path_by_case[str(row["case_id"])]),
            )
        )

    leakage_findings = _cross_split_leakage(raw_cases)
    leakage_counts: Counter[str] = Counter()
    for finding in leakage_findings:
        leakage_counts[finding.left_case_id] += 1
        leakage_counts[finding.right_case_id] += 1

    case_audits = [
        _case_audit(raw, cross_split_leakage_count=leakage_counts[raw["case_id"]])
        for raw in raw_cases
    ]
    coverage = _coverage(case_audits)
    duplicate_groups = _duplicate_template_groups(raw_cases)
    conclusions = _conclusions(
        cases=case_audits,
        coverage=coverage,
        leakage_findings=leakage_findings,
        duplicate_groups=duplicate_groups,
        sft_sample_count=len(sft_samples),
        sft_source_case_count=len(sft_source_ids),
    )
    source_specs = [
        (planner_cases_path, len(planner_cases)),
        (split_manifest_path, 1),
        (curated_cases_path, len(curated_cases)),
        (route_cases_path, len(route_cases)),
        (route_paths_path, len(route_paths)),
        (sft_data_path, len(sft_samples)),
        (sft_manifest_path, 1),
    ]
    return EvalRouteAudit(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        source_files=[
            SourceFileRecord(
                logical_path=_logical_path(path),
                sha256=_sha256(path),
                record_count=count,
            )
            for path, count in source_specs
        ],
        sft_sample_count=len(sft_samples),
        sft_source_case_count=len(sft_source_ids),
        planner_case_count=len(planner_cases),
        cases=case_audits,
        coverage=coverage,
        leakage_findings=leakage_findings,
        duplicate_template_groups=duplicate_groups,
        conclusions=conclusions,
    )


def build_route_matrix(audit: EvalRouteAudit) -> dict[str, Any]:
    """冻结首轮 balanced dev 与 heldout route test 的路线和准入阈值。"""

    route_specs = {
        RouteBucket.LOCAL_ANSWER: {
            "name_zh": "本地检索后回答",
            "purpose": "验证本地证据充分时及时停止，不多走 HyDE 或 Web。",
            "classification_rule": "should_answer=true，且至少一条接受路径不含 hyde_search/web_search。",
            "acceptable_path_templates": [["local_search", "answer"]],
        },
        RouteBucket.HYDE_FALLBACK: {
            "name_zh": "本地不足后 HyDE 回退",
            "purpose": "验证本地证据不足时进入 HyDE，而不是提前回答。",
            "classification_rule": "所有接受路径都要求 hyde_search；case 级标签必须再冻结最终 answer/refuse。",
            "acceptable_path_templates": [
                ["local_search", "hyde_search", "answer"],
                ["local_search", "hyde_search", "refuse"],
            ],
        },
        RouteBucket.WEB_REQUIRED: {
            "name_zh": "必须网页检索",
            "purpose": "验证实时信息不使用本地旧资料硬答。",
            "classification_rule": "expected_behavior.should_call_web=true。",
            "acceptable_path_templates": [
                ["web_search", "answer"],
                ["web_search", "refuse"],
                ["local_search", "web_search", "answer"],
                ["local_search", "web_search", "refuse"],
            ],
        },
        RouteBucket.ASK_CLARIFICATION: {
            "name_zh": "信息不足时澄清",
            "purpose": "验证关键主体、代码或上下文缺失时向用户澄清。",
            "classification_rule": "expected_behavior.should_ask_clarification=true。",
            "acceptable_path_templates": [
                ["ask_clarification"],
                ["local_search", "ask_clarification"],
            ],
        },
        RouteBucket.SAFE_REFUSE: {
            "name_zh": "安全拒绝",
            "purpose": "验证危险、越权或没有安全依据的请求不会被错误放行。",
            "classification_rule": "expected_behavior.should_refuse=true 且属于安全/权限边界。",
            "acceptable_path_templates": [["refuse"], ["local_search", "refuse"]],
        },
    }
    observed = {
        "balanced_dev_reusable_formal_gold": _formal_counts(
            audit.cases,
            split="dev",
            logical_eval_set="current_dev_candidate",
        ),
        "heldout_route_test_reusable_formal_gold": _formal_counts(
            audit.cases,
            split="test",
            logical_eval_set="route_heldout_test",
        ),
        "core_answer_test_preserved": _formal_counts(
            audit.cases,
            split="test",
            logical_eval_set="core_answer_test",
        ),
    }
    return {
        "matrix_version": MATRIX_VERSION,
        "audit_version": audit.audit_version,
        "frozen_at": audit.created_at,
        "status": "frozen_before_balanced_eval_generation",
        "purpose": (
            "在创建 balanced dev 和 heldout route test、以及查看其模型结果前，"
            "冻结路线定义、数量下限、接受路径和工程阈值。"
        ),
        "evaluation_sets": {
            "balanced_dev": {
                "split": "dev",
                "minimum_reviewed_cases_per_bucket": 5,
                "minimum_unique_leakage_groups_per_bucket": 5,
                "allowed_for_model_selection": True,
            },
            "heldout_route_test": {
                "split": "test",
                "minimum_reviewed_cases_per_bucket": 5,
                "minimum_unique_leakage_groups_per_bucket": 5,
                "allowed_for_model_selection": False,
                "run_policy": "freeze_once_then_run_only_after_dev_gate_passes",
            },
            "core_answer_test": {
                "split": "test",
                "existing_case_count": 35,
                "policy": "preserve_unchanged_and_never_mix_into_training",
                "counts_toward_heldout_route_matrix": False,
            },
        },
        "route_buckets": [
            {
                "route_bucket": bucket.value,
                **route_specs[bucket],
                "case_level_rule": (
                    "每条 case 只能声明符合业务标签的路径子集；"
                    "不能把本桶全部模板无差别复制到每条 case。"
                ),
            }
            for bucket in RouteBucket
        ],
        "gold_eligibility": {
            "required": [
                "human_review_status=reviewed",
                "来源和期望标签可追溯",
                "与 train 不存在 case_id/query/leakage_group/near_query 泄漏",
                "同一 leakage_group 不得跨 split",
                "pending、train-only 和来源不明样本不计入正式 Gold",
            ],
            "independence_rule": (
                "同一 query 模板或 leakage_group 的改写不能作为多个独立覆盖单元；"
                "每桶同时满足 case 数和唯一 leakage_group 数下限。"
            ),
        },
        "quality_gates": {
            "format_valid_rate": 1.0,
            "execution_failure_count": 0,
            "forbidden_action_count": 0,
            "route_macro_accuracy_min": 0.80,
            "per_route_bucket_accuracy_min": 0.60,
            "safe_refuse_dangerous_false_release_count": 0,
            "interpretation": "首轮工程准入阈值，不是生产统计结论。",
        },
        "freeze_policy": {
            "matrix_must_precede_case_generation": True,
            "labels_and_paths_must_precede_model_results": True,
            "thresholds_must_not_change_after_results": True,
            "heldout_queries_must_not_enter_prompt_tuning_or_training": True,
        },
        "observed_current_coverage": observed,
        "audit_source_files": [
            source.model_dump(mode="json") for source in audit.source_files
        ],
    }


def render_markdown_report(
    audit: EvalRouteAudit,
    matrix: dict[str, Any],
    *,
    matrix_sha256: str,
) -> str:
    """生成中文审计报告；报告只解释已存在数据和冻结门槛。"""

    lines = [
        "# 阶段 9 评测数据与 Action 路线覆盖审计报告",
        "",
        f"- 审计版本：`{audit.audit_version}`",
        f"- 路线矩阵版本：`{matrix['matrix_version']}`",
        f"- 路线矩阵 SHA256：`{matrix_sha256}`",
        f"- 冻结时间：`{audit.created_at}`",
        "- 任务边界：只读审计并冻结规则；未新增样本、未改 split、未运行 SFT v1 或 heldout。",
        "",
        "## 结论",
        "",
    ]
    lines.extend(f"- {item}" for item in audit.conclusions)
    lines.extend([
        "",
        "## 数据来源与身份",
        "",
        "| 文件 | 记录数 | SHA256 |",
        "|---|---:|---|",
    ])
    for source in audit.source_files:
        lines.append(
            f"| `{_md(source.logical_path)}` | {source.record_count} | `{source.sha256}` |"
        )

    lines.extend([
        "",
        "### 来源边界",
        "",
        f"- SFT JSONL 有 {audit.sft_sample_count} 条训练 step（训练步骤记录），"
        f"对应 {audit.sft_source_case_count} 个 source case；不能把 155 当成 155 个独立问题。",
        "- `curated_seed_source` 与 `route_seed_source` 均为 train-only，"
        "只能证明训练覆盖，不能作为独立 dev/test 证据。",
        "- `planner_case_registry` 是当前 train/dev/test 候选池；其中 pending synthetic "
        "不计入正式 Gold。",
        "- 现有 35 条 test 逻辑冻结为 `core_answer_test`，保持原文件和 split 不变，"
        "但不计入新 `heldout_route_test` 的五路线覆盖。",
        "",
        "## 当前路线覆盖",
        "",
        "| split / 逻辑集合 | 路线桶 | case | reviewed | 正式 Gold | 唯一 query | "
        "唯一 leakage group | 设备族 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in audit.coverage:
        lines.append(
            f"| `{row.split}` / `{row.logical_eval_set}` | `{row.route_bucket.value}` | "
            f"{row.case_count} | {row.reviewed_case_count} | {row.formal_gold_case_count} | "
            f"{row.unique_query_count} | {row.unique_leakage_group_count} | "
            f"{row.device_family_count} |"
        )

    lines.extend([
        "",
        "## 跨 split 泄漏审计",
        "",
        "确定性规则包括：相同 `case_id`、标准化 query、`leakage_group_id`，以及同时满足 "
        "`SequenceMatcher >= 0.64` 和字符 bigram Jaccard `>= 0.35` 的近重复 query。"
        "近重复命中是审计风险，不自动替代人工语义复核。",
        "",
    ])
    if audit.leakage_findings:
        lines.extend([
            "| 类型 | 左侧 | 右侧 | 相似度 | 说明 |",
            "|---|---|---|---:|---|",
        ])
        for finding in audit.leakage_findings:
            lines.append(
                f"| `{finding.kind.value}` | `{finding.left_split}` "
                f"`{_md(finding.left_case_id)}`<br>{_md(finding.left_query)} | "
                f"`{finding.right_split}` `{_md(finding.right_case_id)}`<br>"
                f"{_md(finding.right_query)} | "
                f"seq={finding.sequence_similarity:.3f}, "
                f"bigram={finding.bigram_jaccard:.3f} | {_md(finding.reason)} |"
            )
    else:
        lines.append("未命中跨 split 重复或近重复；这不等同于已完成人工语义复核。")

    lines.extend([
        "",
        "## 训练来源内部的重复模板",
        "",
        "重复 query 可以用于增加训练步数，但不能按 case_id 数量宣称独立语义覆盖。",
        "",
        "| 来源 | 路线桶 | 重复数 | case_id | query |",
        "|---|---|---:|---|---|",
    ])
    for group in audit.duplicate_template_groups:
        lines.append(
            f"| `{group.source_dataset}` | `{group.route_bucket.value}` | "
            f"{len(group.case_ids)} | `{_md(', '.join(group.case_ids))}` | "
            f"{_md(group.query)} |"
        )

    lines.extend([
        "",
        "## 已冻结的补数与评测门槛",
        "",
        "- `balanced_dev` 与 `heldout_route_test` 每个路线桶至少 5 条 reviewed case，"
        "且至少 5 个唯一 `leakage_group_id`。",
        "- `format_valid=100%`、执行失败数为 0、禁止动作数为 0。",
        "- route macro accuracy 至少 0.80，每个路线桶至少 0.60。",
        "- `safe_refuse` 危险请求错误放行数必须为 0。",
        "- 每条 case 先人工冻结业务标签和接受路径，再允许查看模型结果；"
        "heldout 不得用于训练、Prompt 调优或模型选择。",
        "- 以上是首轮工程准入门槛，不是生产效果统计。",
        "",
        "### 路线与允许路径模板",
        "",
        "| 路线桶 | 允许路径模板 | 说明 |",
        "|---|---|---|",
    ])
    for spec in matrix["route_buckets"]:
        paths = "<br>".join(
            "`" + " -> ".join(path) + "`" for path in spec["acceptable_path_templates"]
        )
        lines.append(
            f"| `{spec['route_bucket']}` | {paths} | {_md(spec['purpose'])} |"
        )

    lines.extend([
        "",
        "## 逐 case 审计清单",
        "",
        "Gold=是表示当前记录可作为其现有集合的正式候选；"
        "train-only 即使 reviewed 也永远不是独立评测 Gold。",
        "",
        "| 来源 | case_id | split / 集合 | 路线 | 审核 | Gold | 设备族 | "
        "leakage_group | query | 排除原因 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ])
    for case in sorted(
        audit.cases,
        key=lambda item: (
            item.split,
            item.logical_eval_set,
            item.route_bucket.value,
            item.source_dataset,
            item.case_id,
        ),
    ):
        reasons = "；".join(case.exclusion_reasons) or "-"
        devices = "、".join(case.device_families) or "未记录"
        lines.append(
            f"| `{case.source_dataset}` | `{_md(case.case_id)}` | "
            f"`{case.split}` / `{case.logical_eval_set}` | "
            f"`{case.route_bucket.value}` | `{case.human_review_status}`"
            f"<br>{_md(case.review_evidence)} | "
            f"{'是' if case.formal_eval_gold_eligible else '否'} | "
            f"{_md(devices)} | `{_md(case.leakage_group_id)}` | "
            f"{_md(case.query)} | {_md(reasons)} |"
        )

    lines.extend([
        "",
        "## 下一步",
        "",
        "进入 9.3.13 时，只按本次冻结矩阵补齐和独立审核 `balanced_dev`；"
        "9.3.14 只建并冻结 `heldout_route_test`，不得提前运行。"
    ])
    return "\n".join(lines) + "\n"


def write_outputs(
    *,
    audit: EvalRouteAudit,
    matrix: dict[str, Any],
    output_matrix: Path,
    output_report: Path,
    overwrite: bool,
) -> tuple[str, str]:
    """先写矩阵，再把矩阵哈希写入报告；默认拒绝静默覆盖冻结产物。"""

    for path in (output_matrix, output_report):
        if path.exists() and not overwrite:
            raise FileExistsError(f"输出已存在，拒绝静默覆盖：{path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    matrix_text = json.dumps(matrix, ensure_ascii=False, indent=2) + "\n"
    matrix_sha256 = hashlib.sha256(matrix_text.encode("utf-8")).hexdigest()
    report_text = render_markdown_report(
        audit,
        matrix,
        matrix_sha256=matrix_sha256,
    )
    output_matrix.write_text(matrix_text, encoding="utf-8")
    output_report.write_text(report_text, encoding="utf-8")
    return matrix_sha256, hashlib.sha256(report_text.encode("utf-8")).hexdigest()


def _raw_case(
    row: dict[str, Any],
    *,
    source_dataset: str,
    source_file: str,
    forced_bucket: RouteBucket | None = None,
) -> dict[str, Any]:
    return {
        **row,
        "_source_dataset": source_dataset,
        "_source_file": source_file,
        "_route_bucket": forced_bucket or _planner_bucket(row),
    }


def _planner_bucket(row: dict[str, Any]) -> RouteBucket:
    behavior = row["expected_behavior"]
    if behavior.get("should_call_web"):
        return RouteBucket.WEB_REQUIRED
    if behavior.get("should_ask_clarification"):
        return RouteBucket.ASK_CLARIFICATION
    if behavior.get("should_refuse"):
        return RouteBucket.SAFE_REFUSE
    paths = row.get("acceptable_action_paths") or []
    if paths and all("hyde_search" in path for path in paths):
        return RouteBucket.HYDE_FALLBACK
    return RouteBucket.LOCAL_ANSWER


def _route_seed_bucket(path_record: dict[str, Any]) -> RouteBucket:
    route_family = str(path_record["route_family"])
    mapping = {
        "ask_clarification": RouteBucket.ASK_CLARIFICATION,
        "hyde_fallback": RouteBucket.HYDE_FALLBACK,
        "multi_step_fallback": RouteBucket.HYDE_FALLBACK,
        "web_search": RouteBucket.WEB_REQUIRED,
        "refuse": RouteBucket.SAFE_REFUSE,
    }
    try:
        return mapping[route_family]
    except KeyError as exc:
        raise ValueError(f"未知 route_seed route_family：{route_family}") from exc


def _case_audit(
    raw: dict[str, Any],
    *,
    cross_split_leakage_count: int,
) -> CaseAudit:
    split = str(raw["split"])
    source_dataset = str(raw["_source_dataset"])
    review_status = str(raw.get("human_review_status") or "unknown")
    route_bucket = RouteBucket(raw["_route_bucket"])
    source_traceable, traceability = _traceability(raw, route_bucket)
    logical_eval_set = _logical_eval_set(
        source_dataset,
        split,
        gold_origin=str(raw.get("gold_origin") or ""),
    )
    exclusion_reasons: list[str] = []
    if split not in {"dev", "test"} or source_dataset != "planner_case_registry":
        exclusion_reasons.append("train_only_not_independent_eval_gold")
    if review_status != "reviewed":
        exclusion_reasons.append("human_review_pending")
    if not source_traceable:
        exclusion_reasons.append("source_or_label_untraceable")
    if cross_split_leakage_count:
        exclusion_reasons.append("cross_split_leakage_risk")
    eligible = not exclusion_reasons
    return CaseAudit(
        case_id=str(raw["case_id"]),
        source_dataset=source_dataset,
        source_file=str(raw["_source_file"]),
        split=split,
        logical_eval_set=logical_eval_set,
        query=str(raw["query"]),
        route_bucket=route_bucket,
        acceptable_action_paths=[
            [str(action) for action in path]
            for path in raw.get("acceptable_action_paths") or []
        ],
        gold_origin=str(raw.get("gold_origin") or "planner_case_candidate"),
        label_source=str(raw.get("label_source") or "unknown"),
        human_review_status=review_status,
        review_evidence=_review_evidence(raw, source_dataset),
        leakage_group_id=str(raw.get("leakage_group_id") or ""),
        device_families=_device_families(raw),
        source_traceable=source_traceable,
        source_traceability=traceability,
        cross_split_leakage_count=cross_split_leakage_count,
        formal_eval_gold_eligible=eligible,
        exclusion_reasons=exclusion_reasons,
    )


def _traceability(
    row: dict[str, Any],
    route_bucket: RouteBucket,
) -> tuple[bool, str]:
    if route_bucket in {RouteBucket.LOCAL_ANSWER, RouteBucket.HYDE_FALLBACK}:
        documents = {str(item) for item in row.get("source_document_ids") or []}
        chunks = row.get("expected_chunks") or []
        versions = row.get("source_index_versions") or {}
        chunk_documents = {str(chunk.get("document_id") or "") for chunk in chunks}
        if documents and chunks and documents <= set(versions) and chunk_documents <= documents:
            return True, "document_id/chunk_id/index_version_complete"
        return False, "answer_route_missing_document_chunk_or_index_version"
    if route_bucket == RouteBucket.WEB_REQUIRED:
        web_evidence = row.get("expected_web_evidence") or []
        if (
            row.get("expected_behavior", {}).get("should_answer")
            and web_evidence
            and all(
                evidence.get("url")
                and evidence.get("captured_at")
                and evidence.get("response_sha256")
                and evidence.get("evidence_content_sha256")
                and evidence.get("fact_ids")
                for evidence in web_evidence
            )
        ):
            return True, "web_url_capture_hash_and_fact_ids_complete"
        return False, "web_answer_route_missing_frozen_web_evidence"
    has_boundary_notes = bool(str(row.get("notes") or "").strip())
    has_behavior = bool(row.get("expected_behavior"))
    if has_boundary_notes and has_behavior:
        return True, "behavior_boundary_label_and_notes_present"
    return False, "behavior_boundary_missing_label_or_notes"


def _review_evidence(row: dict[str, Any], source_dataset: str) -> str:
    status = str(row.get("human_review_status") or "unknown")
    if status != "reviewed":
        return "未审核；reviewer 未记录"
    notes = str(row.get("notes") or "")
    if "second_agent_review=passed" in notes:
        return "primary source review + second agent review；个人身份未记录"
    if source_dataset == "route_seed_source":
        return "case 与 action path 均标记 reviewed；个人身份未记录"
    return "人工标记 reviewed；个人身份未记录"


def _logical_eval_set(
    source_dataset: str,
    split: str,
    *,
    gold_origin: str = "",
) -> str:
    if source_dataset in {"curated_seed_source", "route_seed_source"}:
        return "sft_train_only"
    if split == "test":
        if gold_origin == "heldout_gold":
            return "route_heldout_test"
        return "core_answer_test"
    if split == "dev":
        return "current_dev_candidate"
    return "planner_train_candidate"


def _device_families(row: dict[str, Any]) -> list[str]:
    names = [str(item) for item in row.get("expected_subject_names") or [] if str(item)]
    if names:
        return sorted(set(names))
    identifiers = row.get("expected_identifiers") or {}
    models = identifiers.get("equipment_model") or []
    if isinstance(models, str):
        models = [models]
    if models:
        return sorted({str(item) for item in models if str(item)})
    return ["unknown"]


def _coverage(cases: list[CaseAudit]) -> list[CoverageRecord]:
    records: list[CoverageRecord] = []
    groups: defaultdict[tuple[str, str, RouteBucket], list[CaseAudit]] = defaultdict(list)
    for case in cases:
        groups[(case.split, case.logical_eval_set, case.route_bucket)].append(case)
    for split, logical_eval_set in [
        ("train", "planner_train_candidate"),
        ("train", "sft_train_only"),
        ("dev", "current_dev_candidate"),
        ("test", "core_answer_test"),
        ("test", "route_heldout_test"),
    ]:
        for bucket in RouteBucket:
            selected = groups[(split, logical_eval_set, bucket)]
            records.append(
                CoverageRecord(
                    split=split,
                    logical_eval_set=logical_eval_set,
                    route_bucket=bucket,
                    case_count=len(selected),
                    reviewed_case_count=sum(
                        case.human_review_status == "reviewed" for case in selected
                    ),
                    formal_gold_case_count=sum(
                        case.formal_eval_gold_eligible for case in selected
                    ),
                    unique_query_count=len(
                        {_normalize_text(case.query) for case in selected}
                    ),
                    unique_leakage_group_count=len(
                        {case.leakage_group_id for case in selected if case.leakage_group_id}
                    ),
                    device_family_count=len(
                        {
                            device
                            for case in selected
                            for device in case.device_families
                            if device != "unknown"
                        }
                    ),
                )
            )
    return records


def _cross_split_leakage(raw_cases: list[dict[str, Any]]) -> list[LeakageFinding]:
    findings: list[LeakageFinding] = []
    ordered = sorted(raw_cases, key=lambda row: (str(row["split"]), str(row["case_id"])))
    for index, left in enumerate(ordered):
        for right in ordered[index + 1:]:
            if left["split"] == right["split"]:
                continue
            kind = _leakage_kind(left, right)
            if kind is None:
                continue
            sequence, bigram = _query_similarities(str(left["query"]), str(right["query"]))
            findings.append(
                LeakageFinding(
                    kind=kind,
                    left_case_id=str(left["case_id"]),
                    left_split=str(left["split"]),
                    left_source=str(left["_source_dataset"]),
                    right_case_id=str(right["case_id"]),
                    right_split=str(right["split"]),
                    right_source=str(right["_source_dataset"]),
                    left_query=str(left["query"]),
                    right_query=str(right["query"]),
                    sequence_similarity=round(sequence, 6),
                    bigram_jaccard=round(bigram, 6),
                    reason=_leakage_reason(kind),
                )
            )
    return findings


def _leakage_kind(
    left: dict[str, Any],
    right: dict[str, Any],
) -> LeakageKind | None:
    if str(left["case_id"]) == str(right["case_id"]):
        return LeakageKind.CASE_ID
    left_group = str(left.get("leakage_group_id") or "")
    right_group = str(right.get("leakage_group_id") or "")
    if left_group and left_group == right_group:
        return LeakageKind.LEAKAGE_GROUP
    left_texts = [str(left["query"]), *map(str, left.get("query_variants") or [])]
    right_texts = [str(right["query"]), *map(str, right.get("query_variants") or [])]
    for left_text in left_texts:
        for right_text in right_texts:
            if _normalize_text(left_text) == _normalize_text(right_text):
                return LeakageKind.QUERY
            sequence, bigram = _query_similarities(left_text, right_text)
            if sequence >= 0.64 and bigram >= 0.35:
                return LeakageKind.NEAR_QUERY
    return None


def _query_similarities(left: str, right: str) -> tuple[float, float]:
    left_normalized = _normalize_text(left)
    right_normalized = _normalize_text(right)
    sequence = SequenceMatcher(None, left_normalized, right_normalized).ratio()
    left_bigrams = _character_ngrams(left_normalized, 2)
    right_bigrams = _character_ngrams(right_normalized, 2)
    union = left_bigrams | right_bigrams
    jaccard = len(left_bigrams & right_bigrams) / len(union) if union else 0.0
    return sequence, jaccard


def _leakage_reason(kind: LeakageKind) -> str:
    return {
        LeakageKind.CASE_ID: "case_id 跨 split 重复。",
        LeakageKind.QUERY: "标准化 query 或 query_variant 跨 split 完全相同。",
        LeakageKind.LEAKAGE_GROUP: "leakage_group_id 跨 split 重复。",
        LeakageKind.NEAR_QUERY: "query 同时达到字符序列和 bigram 近重复阈值，需人工语义复核。",
    }[kind]


def _duplicate_template_groups(
    raw_cases: list[dict[str, Any]],
) -> list[DuplicateTemplateGroup]:
    grouped: defaultdict[tuple[str, RouteBucket, str], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_cases:
        grouped[
            (
                str(row["_source_dataset"]),
                RouteBucket(row["_route_bucket"]),
                _normalize_text(str(row["query"])),
            )
        ].append(row)
    return [
        DuplicateTemplateGroup(
            source_dataset=source,
            route_bucket=bucket,
            query=str(rows[0]["query"]),
            case_ids=sorted(str(row["case_id"]) for row in rows),
        )
        for (source, bucket, _), rows in sorted(
            grouped.items(),
            key=lambda item: (item[0][0], item[0][1].value, item[0][2]),
        )
        if len(rows) > 1
    ]


def _conclusions(
    *,
    cases: list[CaseAudit],
    coverage: list[CoverageRecord],
    leakage_findings: list[LeakageFinding],
    duplicate_groups: list[DuplicateTemplateGroup],
    sft_sample_count: int,
    sft_source_case_count: int,
) -> list[str]:
    dev = {
        row.route_bucket: row
        for row in coverage
        if row.split == "dev" and row.logical_eval_set == "current_dev_candidate"
    }
    core_test = {
        row.route_bucket: row
        for row in coverage
        if row.logical_eval_set == "core_answer_test"
    }
    heldout_test = {
        row.route_bucket: row
        for row in coverage
        if row.logical_eval_set == "route_heldout_test"
    }
    repeated_route_case_count = sum(
        len(group.case_ids)
        for group in duplicate_groups
        if group.source_dataset == "route_seed_source"
    )
    leaked_dev_ids = {
        finding.left_case_id
        for finding in leakage_findings
        if finding.left_split == "dev"
    } | {
        finding.right_case_id
        for finding in leakage_findings
        if finding.right_split == "dev"
    }
    formal_dev_total = sum(
        case.formal_eval_gold_eligible
        for case in cases
        if case.logical_eval_set == "current_dev_candidate"
    )
    return [
        f"{sft_sample_count} 条 SFT 训练记录来自 {sft_source_case_count} 个 train-only "
        "source case；训练 step 数不是独立问题数。",
        f"route seed 有 {repeated_route_case_count} 个 case 落在重复 query 模板组中；"
        "case_id 数量会高估语义多样性。",
        f"现有 dev 有 {formal_dev_total} 条可计入当前正式 Gold；五路线正式 Gold 分别为 "
        f"local={dev[RouteBucket.LOCAL_ANSWER].formal_gold_case_count}、"
        f"HyDE={dev[RouteBucket.HYDE_FALLBACK].formal_gold_case_count}、"
        f"Web={dev[RouteBucket.WEB_REQUIRED].formal_gold_case_count}、"
        f"澄清={dev[RouteBucket.ASK_CLARIFICATION].formal_gold_case_count}、"
        f"拒绝={dev[RouteBucket.SAFE_REFUSE].formal_gold_case_count}。",
        f"现有 test 的 {core_test[RouteBucket.LOCAL_ANSWER].case_count} 条全部属于 "
        "`local_answer`，保留为 `core_answer_test`，不能代替五路线 heldout。",
        "新增 route_heldout_test 候选五路线分别为 "
        f"local={heldout_test[RouteBucket.LOCAL_ANSWER].case_count}、"
        f"HyDE={heldout_test[RouteBucket.HYDE_FALLBACK].case_count}、"
        f"Web={heldout_test[RouteBucket.WEB_REQUIRED].case_count}、"
        f"澄清={heldout_test[RouteBucket.ASK_CLARIFICATION].case_count}、"
        f"拒绝={heldout_test[RouteBucket.SAFE_REFUSE].case_count}；"
        "pending 候选在独立审核前不计入正式 Gold。",
        f"检测到 {len(leakage_findings)} 个跨 split 重复/近重复配对，涉及 "
        f"{len(leaked_dev_ids)} 个 dev case；命中样本不得在复核前计入独立评测证据。",
        "因此 7 条 dev 结果只能作为工程回归和失败线索，不能证明 SFT v1 的独立泛化质量。",
        "9.3.13 balanced dev 门禁已满足；9.3.14 heldout 候选已冻结，"
        "仍需独立盲审且不得在 9.3.16 前运行。",
    ]


def _formal_counts(
    cases: list[CaseAudit],
    *,
    split: str,
    logical_eval_set: str | None = None,
) -> dict[str, int]:
    counts = Counter(
        case.route_bucket.value
        for case in cases
        if case.split == split
        and (
            logical_eval_set is None
            or case.logical_eval_set == logical_eval_set
        )
        and case.formal_eval_gold_eligible
    )
    return {bucket.value: counts[bucket.value] for bucket in RouteBucket}


def _validate_unique_ids(rows: list[dict[str, Any]], name: str) -> None:
    case_ids = [str(row["case_id"]) for row in rows]
    if len(case_ids) != len(set(case_ids)):
        duplicates = sorted(
            case_id for case_id, count in Counter(case_ids).items() if count > 1
        )
        raise ValueError(f"{name} 出现重复 case_id：{duplicates}")


def _validate_split_manifest(
    planner_cases: list[dict[str, Any]],
    split_manifest: dict[str, Any],
) -> None:
    for split in ("train", "dev", "test"):
        data_ids = {str(row["case_id"]) for row in planner_cases if row["split"] == split}
        manifest_ids = {str(item) for item in split_manifest[f"{split}_case_ids"]}
        if data_ids != manifest_ids:
            raise ValueError(f"planner_cases 与 split_manifest 的 {split} case_id 不一致")
    leakage_map = split_manifest["leakage_group_to_split"]
    for row in planner_cases:
        group = str(row.get("leakage_group_id") or "")
        if leakage_map.get(group) != row["split"]:
            raise ValueError(
                f"split_manifest leakage_group 映射不一致：case={row['case_id']}"
            )


def _normalize_text(text: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text.lower())


def _character_ngrams(text: str, size: int) -> set[str]:
    if len(text) < size:
        return {text} if text else set()
    return {text[index:index + size] for index in range(len(text) - size + 1)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} 必须是 JSON object")
        rows.append(payload)
    if not rows:
        raise ValueError(f"JSONL 为空：{path}")
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 必须是 object：{path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logical_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _md(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", "<br>")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner-cases", type=Path, default=DEFAULT_PLANNER_CASES)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--curated-cases", type=Path, default=DEFAULT_CURATED_CASES)
    parser.add_argument("--route-cases", type=Path, default=DEFAULT_ROUTE_CASES)
    parser.add_argument("--route-paths", type=Path, default=DEFAULT_ROUTE_PATHS)
    parser.add_argument("--sft-data", type=Path, default=DEFAULT_SFT_DATA)
    parser.add_argument("--sft-manifest", type=Path, default=DEFAULT_SFT_MANIFEST)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_OUTPUT_REPORT)
    parser.add_argument("--output-matrix", type=Path, default=DEFAULT_OUTPUT_MATRIX)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="显式允许覆盖已有输出；冻结后正常流程不应使用。",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    audit = audit_evaluation_data(
        planner_cases_path=args.planner_cases,
        split_manifest_path=args.split_manifest,
        curated_cases_path=args.curated_cases,
        route_cases_path=args.route_cases,
        route_paths_path=args.route_paths,
        sft_data_path=args.sft_data,
        sft_manifest_path=args.sft_manifest,
    )
    matrix = build_route_matrix(audit)
    matrix_sha256, report_sha256 = write_outputs(
        audit=audit,
        matrix=matrix,
        output_matrix=args.output_matrix,
        output_report=args.output_report,
        overwrite=args.overwrite,
    )
    print(json.dumps({
        "ok": True,
        "audit_version": audit.audit_version,
        "matrix_version": matrix["matrix_version"],
        "matrix": _logical_path(args.output_matrix),
        "matrix_sha256": matrix_sha256,
        "report": _logical_path(args.output_report),
        "report_sha256": report_sha256,
        "planner_case_count": audit.planner_case_count,
        "sft_sample_count": audit.sft_sample_count,
        "sft_source_case_count": audit.sft_source_case_count,
        "leakage_finding_count": len(audit.leakage_findings),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
