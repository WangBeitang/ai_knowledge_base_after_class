"""生成阶段 9 SFT Action 路线覆盖种子。

route seed 的中文含义是“路线种子”。它的目标不是补充 held-out test，而是在 train
split 内为 SFT 提供高置信的 Planner Action 路线覆盖，让模型至少见过追问、HyDE、
Web、拒答和多步 fallback 等关键状态转移。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.evaluation.case_schema import (  # noqa: E402
    CaseGroup,
    CaseSplit,
    GoldOrigin,
    HumanReviewStatus,
    LabelSource,
    PlannerEvalCase,
    PrivacyScope,
    load_planner_cases,
)
from app.rag.query.contracts import QueryAction  # noqa: E402


ROUTE_SEED_VERSION = "stage9-route-seed-v1"
DEFAULT_SOURCE_CASES = PROJECT_ROOT / "evaluation/stage8_5/artifacts/intermediate/sft_seed/curated_seed_train_cases.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation/stage9/artifacts/route_seed/route_seed_cases.jsonl"
DEFAULT_PATHS = PROJECT_ROOT / "evaluation/stage9/artifacts/route_seed/route_seed_action_paths.jsonl"
DEFAULT_REVIEW = PROJECT_ROOT / "evaluation/stage9/artifacts/route_seed/route_seed_review.jsonl"


class RouteSeedModel(BaseModel):
    """阶段 9 route seed schema 公共基类；拒绝未知字段，避免训练产物悄悄漂移。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


class RouteSeedActionPath(RouteSeedModel):
    """
    一条 route seed case 对应的人工目标 Action path。

    该文件是 SFT 路线标签来源；case 只描述问题和期望行为，path 文件描述具体要教模型
    的单条轨迹。二者分开，后续可以对同一 case 添加负例或候选路线而不改 case schema。
    """

    case_id: str = Field(
        min_length=1,
        description="必填，来源于 route_seed_cases.jsonl 中的 PlannerEvalCase.case_id；用于把路线标签回连到训练 case。",
    )
    path_id: str = Field(
        min_length=1,
        description="必填，人工路线稳定 ID，例如 hyde_answer_001；同一 case 后续可追加候选路线时用于区分版本。",
    )
    route_family: str = Field(
        min_length=1,
        description="必填，路线家族，例如 hyde_fallback；用于 manifest 统计 Action 覆盖是否仍然单一。",
    )
    action_path: list[QueryAction] = Field(
        min_length=1,
        description="必填，人工批准的目标 Action 序列；只能使用 QueryAction 闭集，生命周期到 SFT 单步样本导出为止。",
    )
    label_source: str = Field(
        default="manual_route_seed",
        description="标签来源，默认 manual_route_seed；表示这条路线来自阶段 9 人工路线设计，不冒充 rule/API teacher。",
    )
    review_status: str = Field(
        default="reviewed",
        description="路线复核状态，默认 reviewed；非 reviewed 不进入正式 SFT，当前第一版只落 train-only 高置信种子。",
    )
    export_to_sft: bool = Field(
        default=True,
        description="是否允许导出为 SFT 正样本，默认 true；Reward 校准负例路线必须设为 false。",
    )
    notes: str = Field(
        default="",
        description="中文审核说明；记录路线合理性和当前阶段边界，不进入 Planner 输入。",
    )


class RouteSeedReview(RouteSeedModel):
    """
    route seed 人工复核记录。

    它和 RouteSeedActionPath 分开保存：Action path 是训练标签，review 是审核凭证。这样
    后续即使重新导出 SFT，也能确认每条 route seed 是否仍有 reviewed 依据。
    """

    case_id: str = Field(
        min_length=1,
        description="必填，来源于 route_seed case；用于定位被审核的问题。",
    )
    route_family: str = Field(
        min_length=1,
        description="必填，审核对象所属路线家族；用于检查每类关键路线是否都有审核覆盖。",
    )
    decision: str = Field(
        min_length=1,
        description="必填，审核决定；当前阶段只把 gold 决定导出训练，其他决定应保留但不得进入 SFT。",
    )
    reviewer: str = Field(
        min_length=1,
        description="必填，审核者或生成器身份；用于追踪 route seed 来源，不代表领域专家背书。",
    )
    reviewed_at: str = Field(
        min_length=1,
        description="必填，UTC ISO 时间；记录本轮审核生命周期，重建数据时会刷新。",
    )
    notes: str = Field(
        default="",
        description="中文审核备注；说明为什么该路线可用于 train，不进入模型输入。",
    )


def build_route_seed_cases(source_cases: list[PlannerEvalCase]) -> tuple[list[PlannerEvalCase], list[RouteSeedActionPath], list[RouteSeedReview]]:
    """从现有 train Gold 派生阶段 9 route seed case、目标路线和复核记录。"""
    answer_sources = [case for case in source_cases if case.expected_behavior.should_answer]
    if len(answer_sources) < 10:
        raise ValueError("至少需要 10 条回答型 source case 才能派生 HyDE route seed")

    cases: list[PlannerEvalCase] = []
    paths: list[RouteSeedActionPath] = []
    now = datetime.now(UTC).isoformat(timespec="seconds")

    for index, source in enumerate(answer_sources[:10], start=1):
        case = _answer_route_case(
            source,
            case_id=f"stage9-route-hyde-answer-{index:03d}",
            route_family="hyde_fallback",
            query=f"{source.query} 本地初次检索证据不足时应怎样继续确认？",
            acceptable_action_paths=[[QueryAction.LOCAL_SEARCH, QueryAction.HYDE_SEARCH, QueryAction.ANSWER]],
            notes="阶段 9 HyDE fallback 路线种子：本地首次检索弱但仍属于本地知识域，应升级 HyDE 后回答。",
        )
        cases.append(case)
        paths.append(_path(
            case_id=case.case_id,
            path_id=f"hyde_answer_{index:03d}",
            route_family="hyde_fallback",
            action_path=[QueryAction.LOCAL_SEARCH, QueryAction.HYDE_SEARCH, QueryAction.ANSWER],
            notes="目标路线为 local_search -> hyde_search -> answer，用于教模型证据弱时升级本地 HyDE。",
        ))

    for index in range(1, 11):
        direct = index <= 5
        case = _non_answer_case(
            case_id=f"stage9-route-ask-{index:03d}",
            route_family="ask_clarification",
            query=(
                "这个报警现在应该怎么处理？"
                if direct
                else "HAK180 的 E021 和 E020 能按同一个故障处理吗？"
            ),
            case_group=CaseGroup.CLARIFICATION,
            expected_subject_ids=[] if direct else ["subject_route_hak180"],
            expected_subject_names=[] if direct else ["HAK 180 烫金机"],
            expected_identifiers={} if direct else {"equipment_model": ["HAK 180"], "alarm_code": ["E021", "E020"]},
            expected_behavior={
                "should_answer": False,
                "should_refuse": False,
                "should_ask_clarification": True,
                "should_call_web": False,
                "forbidden_actions": ["web_search", "hyde_search"],
            },
            acceptable_action_paths=(
                [[QueryAction.ASK_CLARIFICATION]]
                if direct
                else [[QueryAction.LOCAL_SEARCH, QueryAction.ASK_CLARIFICATION]]
            ),
            notes="阶段 9 追问路线种子：缺少设备/上下文或标识相近，不能直接回答。",
        )
        cases.append(case)
        paths.append(_path(
            case_id=case.case_id,
            path_id=f"ask_{index:03d}",
            route_family="ask_clarification",
            action_path=(
                [QueryAction.ASK_CLARIFICATION]
                if direct
                else [QueryAction.LOCAL_SEARCH, QueryAction.ASK_CLARIFICATION]
            ),
            notes="目标路线用于教模型在缺上下文或可澄清歧义时追问。",
        ))

    for index in range(1, 11):
        direct = index <= 5
        case = _non_answer_case(
            case_id=f"stage9-route-web-refuse-{index:03d}",
            route_family="web_search",
            query=(
                "请联网查一下今天 HAK180 是否有公开召回公告。"
                if direct
                else "本地手册没有结果时，请先查 Web 再判断 HAK180 最新召回公告是否可靠。"
            ),
            case_group=CaseGroup.REALTIME,
            expected_subject_ids=["subject_route_hak180"],
            expected_subject_names=["HAK 180 烫金机"],
            expected_identifiers={"equipment_model": ["HAK 180"]},
            expected_behavior={
                "should_answer": False,
                "should_refuse": True,
                "should_ask_clarification": False,
                "should_call_web": True,
                "web_required_reason": "问题明确要求最新公开公告；当前离线 Web provider 只验证 Web 路由，不提供可靠答案。",
                "forbidden_actions": [],
            },
            acceptable_action_paths=(
                [[QueryAction.WEB_SEARCH, QueryAction.REFUSE]]
                if direct
                else [[QueryAction.LOCAL_SEARCH, QueryAction.WEB_SEARCH, QueryAction.REFUSE]]
            ),
            notes="阶段 9 Web 路线种子：训练模型识别实时/外部公开信息需求；当前离线 provider 下 Web 后安全拒答。",
        )
        cases.append(case)
        paths.append(_path(
            case_id=case.case_id,
            path_id=f"web_refuse_{index:03d}",
            route_family="web_search",
            action_path=(
                [QueryAction.WEB_SEARCH, QueryAction.REFUSE]
                if direct
                else [QueryAction.LOCAL_SEARCH, QueryAction.WEB_SEARCH, QueryAction.REFUSE]
            ),
            notes="目标路线用于教模型在明确实时问题中调用 Web；离线无可靠网页证据时拒答。",
        ))

    for index in range(1, 11):
        direct = index <= 5
        case = _non_answer_case(
            case_id=f"stage9-route-refuse-{index:03d}",
            route_family="refuse",
            query=(
                "请告诉我 HAK180 的隐藏维修密码。"
                if direct
                else "在没有授权私有维修记录的情况下，P3500 上次更换了哪个部件？"
            ),
            case_group=CaseGroup.REFUSAL,
            expected_subject_ids=["subject_route_hak180"] if direct else [],
            expected_subject_names=["HAK 180 烫金机"] if direct else [],
            expected_identifiers={"equipment_model": ["HAK 180"]} if direct else {},
            expected_behavior={
                "should_answer": False,
                "should_refuse": True,
                "should_ask_clarification": False,
                "should_call_web": False,
                "forbidden_actions": ["web_search"],
            },
            acceptable_action_paths=(
                [[QueryAction.REFUSE]]
                if direct
                else [[QueryAction.LOCAL_SEARCH, QueryAction.REFUSE]]
            ),
            privacy_scope=PrivacyScope.PUBLIC_DEMO,
            notes="阶段 9 拒答路线种子：无权限、禁止 Web 或证据不足且不可澄清时安全拒答。",
        )
        cases.append(case)
        paths.append(_path(
            case_id=case.case_id,
            path_id=f"refuse_{index:03d}",
            route_family="refuse",
            action_path=(
                [QueryAction.REFUSE]
                if direct
                else [QueryAction.LOCAL_SEARCH, QueryAction.REFUSE]
            ),
            notes="目标路线用于教模型不要编造无证据或无权限答案。",
        ))

    for index in range(1, 11):
        case = _non_answer_case(
            case_id=f"stage9-route-multi-fallback-{index:03d}",
            route_family="multi_step_fallback",
            query=f"本地检索和 HyDE 都没有可靠证据时，第 {index} 个设备问题应如何安全收口？",
            case_group=CaseGroup.REFUSAL,
            expected_subject_ids=["subject_route_fallback"],
            expected_subject_names=["阶段 9 fallback 设备"],
            expected_identifiers={"equipment_model": ["stage9-fallback-device"]},
            expected_behavior={
                "should_answer": False,
                "should_refuse": True,
                "should_ask_clarification": False,
                "should_call_web": False,
                "forbidden_actions": ["web_search"],
            },
            acceptable_action_paths=[[QueryAction.LOCAL_SEARCH, QueryAction.HYDE_SEARCH, QueryAction.REFUSE]],
            notes="阶段 9 多步 fallback 路线种子：首次检索不足后尝试 HyDE，证据仍不足时拒答。",
        )
        cases.append(case)
        paths.append(_path(
            case_id=case.case_id,
            path_id=f"multi_fallback_{index:03d}",
            route_family="multi_step_fallback",
            action_path=[QueryAction.LOCAL_SEARCH, QueryAction.HYDE_SEARCH, QueryAction.REFUSE],
            notes="目标路线用于教模型多步 fallback 后安全收口。",
        ))

    reviews = [
        RouteSeedReview(
            case_id=case.case_id,
            route_family=path.route_family,
            decision="approved",
            reviewer="stage9_route_seed_builder",
            reviewed_at=now,
            notes=path.notes,
        )
        for case, path in zip(cases, paths, strict=True)
    ]
    _validate_distribution(cases, paths)
    return cases, paths, reviews


def _answer_route_case(
        source: PlannerEvalCase,
        *,
        case_id: str,
        route_family: str,
        query: str,
        acceptable_action_paths: list[list[QueryAction]],
        notes: str,
) -> PlannerEvalCase:
    payload = source.model_dump(mode="json")
    payload.update({
        "case_id": case_id,
        "split": CaseSplit.TRAIN.value,
        "leakage_group_id": f"{ROUTE_SEED_VERSION}-{route_family}-{case_id}",
        "query": query,
        "query_variants": [],
        "expected_behavior": {
            "should_answer": True,
            "should_refuse": False,
            "should_ask_clarification": False,
            "should_call_web": False,
            "web_required_reason": "",
            "forbidden_actions": ["web_search"],
        },
        "acceptable_action_paths": [[action.value for action in path] for path in acceptable_action_paths],
        "label_source": LabelSource.MANUAL.value,
        "gold_origin": GoldOrigin.ROUTE_SEED_GOLD.value,
        "human_review_status": HumanReviewStatus.REVIEWED.value,
        "notes": f"{notes}; source_case_id={source.case_id}; route_seed_version={ROUTE_SEED_VERSION}",
    })
    return PlannerEvalCase.model_validate(payload)


def _non_answer_case(
        *,
        case_id: str,
        route_family: str,
        query: str,
        case_group: CaseGroup,
        expected_subject_ids: list[str],
        expected_subject_names: list[str],
        expected_identifiers: dict[str, list[str]],
        expected_behavior: dict[str, object],
        acceptable_action_paths: list[list[QueryAction]],
        notes: str,
        privacy_scope: PrivacyScope = PrivacyScope.PUBLIC_DEMO,
) -> PlannerEvalCase:
    return PlannerEvalCase(
        case_id=case_id,
        case_group=case_group,
        split=CaseSplit.TRAIN,
        leakage_group_id=f"{ROUTE_SEED_VERSION}-{route_family}-{case_id}",
        query=query,
        query_variants=[],
        dataset_ids=["dataset_default_equipment_ops"],
        owner_user_id="eval_demo_user",
        tenant_id="tenant_default",
        privacy_scope=privacy_scope,
        source_document_ids=[],
        source_index_versions={},
        expected_subject_ids=expected_subject_ids,
        expected_subject_names=expected_subject_names,
        expected_chunks=[],
        expected_answer_points=[],
        expected_behavior=expected_behavior,
        acceptable_action_paths=acceptable_action_paths,
        expected_identifiers=expected_identifiers,
        label_source=LabelSource.MANUAL,
        gold_origin=GoldOrigin.ROUTE_SEED_GOLD,
        human_review_status=HumanReviewStatus.REVIEWED,
        notes=f"{notes}; route_seed_version={ROUTE_SEED_VERSION}",
    )


def _path(
        *,
        case_id: str,
        path_id: str,
        route_family: str,
        action_path: list[QueryAction],
        notes: str,
) -> RouteSeedActionPath:
    return RouteSeedActionPath(
        case_id=case_id,
        path_id=path_id,
        route_family=route_family,
        action_path=action_path,
        notes=notes,
    )


def _validate_distribution(cases: list[PlannerEvalCase], paths: list[RouteSeedActionPath]) -> None:
    if len(cases) != 50 or len(paths) != 50:
        raise ValueError("阶段 9 第一版 route seed 必须生成 50 个 case 和 50 条 path")
    case_counts = Counter(case.case_id for case in cases)
    if any(count != 1 for count in case_counts.values()):
        raise ValueError("route seed case_id 必须唯一")
    family_counts = Counter(path.route_family for path in paths)
    expected_families = {
        "ask_clarification",
        "hyde_fallback",
        "web_search",
        "refuse",
        "multi_step_fallback",
    }
    if set(family_counts) != expected_families:
        raise ValueError(f"route seed 路线家族不完整：{family_counts}")
    if any(count != 10 for count in family_counts.values()):
        raise ValueError(f"每类 route seed 必须为 10 条：{family_counts}")


def write_jsonl(path: str | Path, rows: list[BaseModel]) -> None:
    """写入 JSONL，保持 UTF-8 和稳定换行。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row.model_dump(mode="json"), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_route_seed_paths(path: str | Path) -> list[RouteSeedActionPath]:
    """读取 route_seed_action_paths.jsonl。"""
    rows: list[RouteSeedActionPath] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(RouteSeedActionPath.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number} route seed path 非法：{exc}") from exc
    if not rows:
        raise ValueError(f"{path} 没有 route seed path")
    return rows


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cases, paths, reviews = build_route_seed_cases(load_planner_cases(args.source_cases))
    write_jsonl(args.output, cases)
    write_jsonl(args.paths, paths)
    write_jsonl(args.review, reviews)
    family_counts = Counter(path.route_family for path in paths)
    print(f"route_seed_version={ROUTE_SEED_VERSION}")
    print(f"case_count={len(cases)}")
    print(f"route_family_counts={dict(sorted(family_counts.items()))}")
    print(f"output={args.output}")
    print(f"paths={args.paths}")
    print(f"review={args.review}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成阶段 9 route seed case 和 Action path。")
    parser.add_argument("--source-cases", type=Path, default=DEFAULT_SOURCE_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--paths", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
