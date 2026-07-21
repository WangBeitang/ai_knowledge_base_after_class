"""把已入库、已二审的 20 条 Gold 转成阶段 9 可用的 Planner SFT 训练 case。

本脚本不生成新问题，也不改写答案事实。它只修正训练运行所需的数据契约：

1. 给两个公开来源分配稳定 ``subject_id`` 和唯一标准主题名，使 Rule Planner 能在
   “主体已确认”的安全前提下执行 local_search；
2. 移除 ``dataset_name/failure_mode/profile_column`` 这类普通语义标签。它们不是设备
   型号、报警码或部件编号，不能放进 query_identifiers 触发同码安全校验；
3. 写入机器可读 ``gold_origin=curated_seed_gold``；
4. 只接受 train + reviewed + 独立二审 passed 的 case，并生成独立 split manifest。

输入 ``gold_cases_indexed.jsonl`` 已经绑定真实 Milvus 整数 chunk_id；输出不会修改这些
document/chunk/index_version 身份。原始 indexed Gold 继续保留为入库审计产物，训练就绪
case 单独落盘，避免为了 SFT 状态字段改写已经冻结的 v1 快照输入。
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.evaluation.case_schema import (  # noqa: E402
    CaseSplit,
    GoldOrigin,
    HumanReviewStatus,
    PlannerEvalCase,
    SplitManifest,
    load_planner_cases,
)
from evaluation.stage8_5.pipelines.common.paths import (  # noqa: E402
    DEFAULT_STAGE85_ROOT,
    stage85_layout,
)
from evaluation.stage8_5.pipelines.common.stage85_schema import (  # noqa: E402
    read_jsonl,
    write_json,
    write_jsonl,
)
from evaluation.stage8_5.pipelines.curated_gold.build_source_grounded_gold import (  # noqa: E402
    GoldCaseAudit,
)


CURATED_SEED_VERSION = "stage85-curated-seed-v1"
DEFAULT_SNAPSHOT_ID = "stage85-env-20260721-v2"

# subject_id 是 Planner 安全路由使用的稳定主体身份；subject_name 只负责展示和 Prompt。
# 两个值按来源文档固定，不能按 TWF/HDF 等问题主题临时变化，否则同一文档会被伪装成
# 多个设备主体，也无法稳定复现 local_search 的权限/主题范围。
SUBJECT_BY_DOCUMENT = {
    "doc_stage85_uci_ai4i_official_description_v1": (
        "subject_uci_ai4i_2020",
        "AI4I 2020 Predictive Maintenance Dataset",
    ),
    "doc_stage85_uci_hydraulic_official_description_v1": (
        "subject_uci_hydraulic_condition",
        "Condition Monitoring of Hydraulic Systems",
    ),
}


def prepare_curated_seed_cases(
        *,
        indexed_cases: list[PlannerEvalCase],
        audits: list[GoldCaseAudit],
        snapshot_id: str,
        created_at: str | None = None,
) -> tuple[list[PlannerEvalCase], SplitManifest]:
    """校验审核门禁并返回训练就绪 case 和 train-only split manifest。"""

    audit_by_case = {audit.case_id: audit for audit in audits}
    case_ids = {case.case_id for case in indexed_cases}
    if set(audit_by_case) != case_ids:
        raise ValueError("gold_case_audit 与 indexed Gold 的 case_id 集合不一致")

    prepared_cases: list[PlannerEvalCase] = []
    for case in indexed_cases:
        if case.split != CaseSplit.TRAIN:
            raise ValueError(f"case_id={case.case_id} 不是 train，不能进入 curated SFT seed")
        if case.human_review_status != HumanReviewStatus.REVIEWED:
            raise ValueError(f"case_id={case.case_id} 尚未 reviewed，不能进入 curated SFT seed")

        audit = audit_by_case[case.case_id]
        if audit.second_review_status != "passed":
            raise ValueError(f"case_id={case.case_id} 独立二审未通过")
        if len(case.source_document_ids) != 1:
            raise ValueError(f"case_id={case.case_id} 必须且只能绑定一个 curated source document")

        document_id = case.source_document_ids[0]
        subject_identity = SUBJECT_BY_DOCUMENT.get(document_id)
        if subject_identity is None:
            raise ValueError(f"case_id={case.case_id} 的 document_id={document_id} 没有主题身份配置")
        subject_id, subject_name = subject_identity

        payload = case.model_dump(mode="json")
        payload["expected_subject_ids"] = [subject_id]
        payload["expected_subject_names"] = [subject_name]
        # 当前 Gold 的 dataset/failure/profile 标签属于语义检索词，不是需要“同码确认”的
        # 结构化设备标识。清空后仍保留原始 query 和证据，不会降低事实可追溯性。
        payload["expected_identifiers"] = {}
        payload["gold_origin"] = GoldOrigin.CURATED_SEED_GOLD.value
        old_notes = str(payload.get("notes") or "").replace(
            "second_agent_review=pending",
            "second_agent_review=passed",
        )
        payload["notes"] = (
            f"{old_notes}; curated_seed_version={CURATED_SEED_VERSION}; "
            f"training_status=approved; snapshot_id={snapshot_id}"
        ).strip("; ")
        prepared_cases.append(PlannerEvalCase.model_validate(payload))

    manifest = SplitManifest(
        manifest_id=f"{CURATED_SEED_VERSION}-split-manifest",
        created_at=created_at or datetime.now(UTC).isoformat(timespec="seconds"),
        snapshot_id=snapshot_id,
        train_case_ids=[case.case_id for case in prepared_cases],
        dev_case_ids=[],
        test_case_ids=[],
        demo_regression_case_ids=[],
        leakage_group_to_split={case.leakage_group_id: case.split for case in prepared_cases},
        notes=(
            "20 条 curated_seed_gold 全部为 approved train seed；"
            "dev/test 必须来自独立 production chunk。"
        ),
    )
    return prepared_cases, manifest


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    layout = stage85_layout(args.base_dir)
    cases_path = layout.curated_intermediate / "gold_cases_indexed.jsonl"
    audit_path = layout.curated_review / "gold_case_audit.jsonl"
    output_path = args.output or layout.sft_intermediate / "curated_seed_train_cases.jsonl"
    manifest_path = args.manifest or layout.sft_intermediate / "curated_seed_split_manifest.json"

    existing_created_at: str | None = None
    if manifest_path.exists():
        existing = SplitManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if existing.snapshot_id != args.snapshot_id:
            raise ValueError(
                f"已有 curated split manifest 绑定 {existing.snapshot_id}，"
                f"不能静默改成 {args.snapshot_id}"
            )
        existing_created_at = existing.created_at

    cases, manifest = prepare_curated_seed_cases(
        indexed_cases=load_planner_cases(cases_path),
        audits=read_jsonl(audit_path, GoldCaseAudit),
        snapshot_id=args.snapshot_id,
        created_at=existing_created_at,
    )
    write_jsonl(output_path, cases)
    write_json(manifest_path, manifest)
    print(f"snapshot_id={manifest.snapshot_id}")
    print(f"approved_case_count={len(cases)}")
    print(f"output={output_path}")
    print(f"manifest={manifest_path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成阶段 8.5 curated Gold 的正式 SFT 训练 case。")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_STAGE85_ROOT)
    parser.add_argument("--snapshot-id", default=DEFAULT_SNAPSHOT_ID)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
