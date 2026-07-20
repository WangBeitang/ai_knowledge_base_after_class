"""阶段 8.5 第一批公开数据候选池物化脚本。

这个脚本只生成可审计的“小规模种子池”，不是公开数据大规模采集器。它把已经人工核验
过许可证边界的公开来源，整理成故障场景卡片，再复用阶段 8 的 `PlannerEvalCase`
schema 生成候选样本。

关键边界：
- 来源必须先写入 source/license manifest，并通过 approved 门禁。
- 表格和时序数据不能直接当 RAG 文档；这里先转成可解释的 FaultScenarioCard。
- 自动生成的 case 默认不能进训练；本脚本只把少量人工抽查过的核心样本标为 reviewed。
- approved/review/rejected 三个池分开写，pending/rejected 不会出现在 approved_cases.jsonl。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.evaluation.case_schema import CaseSplit, HumanReviewStatus  # noqa: E402
from app.shared.config.knowledge_base_config import (  # noqa: E402
    DEFAULT_DATASET_ID,
    DEFAULT_TENANT_ID,
)
from evaluation.stage8_5.generate_stage85_report import build_markdown_report  # noqa: E402
from evaluation.stage8_5.stage85_schema import (  # noqa: E402
    FaultScenarioCard,
    LicenseRecord,
    SourceRecord,
    Stage85Issue,
    Stage85QualityReport,
    build_case_payloads_from_cards,
    build_split_manifest,
    filter_fault_cards_by_sources,
    validate_candidate_payloads,
    validate_source_records,
    write_json,
    write_jsonl,
)


DEFAULT_BASE_DIR = PROJECT_ROOT / "evaluation/stage8_5"
COLLECTED_AT = "2026-07-20T00:00:00+00:00"
SEED_VERSION = "stage85-public-seed-v1"
SNAPSHOT_ID = "stage85-public-seed-snapshot-v1"


@dataclass(frozen=True, slots=True)
class CardSeed:
    """一张种子卡片及其处理元数据。

    `split` 是候选 case 进入 train/dev/test 的预分配边界。`reviewed` 表示这张卡片的
    两条候选问题已经经过第一轮人工抽查，可以进入 approved 池；False 时只进入 review
    queue，后续人工确认前不能导出训练。
    """

    payload: dict[str, Any]
    split: CaseSplit
    reviewed: bool


def main(argv: list[str] | None = None) -> int:
    """生成第一批公开数据候选池和质量报告。"""

    args = _build_parser().parse_args(argv)
    base_dir = args.base_dir
    paths = _stage85_paths(base_dir)
    _ensure_directories(paths)

    licenses = _license_records()
    sources = _source_records()
    source_report = validate_source_records(sources, licenses)
    approved_source_ids = source_report.approved_source_ids

    seed_cards = _card_seeds()
    raw_cards = [FaultScenarioCard.model_validate(seed.payload) for seed in seed_cards]
    accepted_cards, card_issues = filter_fault_cards_by_sources(raw_cards, approved_source_ids)
    accepted_by_id = {card.card_id: card for card in accepted_cards}

    case_payloads: list[dict[str, Any]] = []
    rejected_card_issues: list[Stage85Issue] = []
    for seed in seed_cards:
        card = accepted_by_id.get(str(seed.payload["card_id"]))
        if card is None:
            continue
        generated_payloads, rejected_cards = build_case_payloads_from_cards(
            [card],
            dataset_id=args.dataset_id,
            owner_user_id=args.owner_user_id,
            tenant_id=args.tenant_id,
            split=seed.split,
        )
        rejected_card_issues.extend(issue for record in rejected_cards for issue in record.issues)
        for payload in generated_payloads:
            payload["human_review_status"] = (
                HumanReviewStatus.REVIEWED.value if seed.reviewed else HumanReviewStatus.PENDING.value
            )
            payload["notes"] = (
                f"{payload.get('notes', '')}; seed_version={SEED_VERSION}; "
                f"review_status={'reviewed_seed' if seed.reviewed else 'pending_manual_review'}"
            )
            case_payloads.append(payload)

    approved_cases, review_cases, rejected_cases, case_report = validate_candidate_payloads(case_payloads)
    split_manifest = build_split_manifest(
        approved_cases,
        manifest_id=f"{SEED_VERSION}-split-manifest",
        snapshot_id=SNAPSHOT_ID,
        notes=(
            "阶段 8.5 第一批公开数据 approved case split 清单；"
            "review/rejected 样本不进入本 manifest。"
        ),
    )
    quality_report = _build_quality_report(
        paths=paths,
        sources=sources,
        cards=accepted_cards,
        approved_cases=approved_cases,
        review_cases=review_cases,
        rejected_cases=rejected_cases,
        source_report=source_report,
        card_issues=card_issues,
        rejected_card_issues=rejected_card_issues,
        candidate_issues=case_report.issues,
    )

    write_jsonl(paths["licenses"], licenses)
    write_jsonl(paths["sources"], sources)
    write_jsonl(paths["raw_cards"], raw_cards)
    write_jsonl(paths["cards"], accepted_cards)
    write_jsonl(paths["documents"], _public_document_records(sources, accepted_cards))
    write_jsonl(paths["chunk_map"], _chunk_source_records(accepted_cards))
    write_jsonl(paths["candidates"], case_payloads)
    write_jsonl(paths["approved"], approved_cases)
    write_jsonl(paths["review"], review_cases)
    write_jsonl(paths["rejected"], rejected_cases)
    write_json(paths["split_manifest"], split_manifest)
    write_json(paths["quality_report"], quality_report)
    paths["markdown_report"].write_text(
        build_markdown_report(quality_report.model_dump(mode="json")),
        encoding="utf-8",
    )

    error_count = sum(1 for issue in quality_report.issues if issue.severity.value == "error")
    print(
        "stage85 public seed pool: "
        f"sources={len(sources)}, cards={len(accepted_cards)}, "
        f"candidates={len(case_payloads)}, approved={len(approved_cases)}, "
        f"review={len(review_cases)}, rejected={len(rejected_cases)}, errors={error_count}"
    )
    return 1 if error_count else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成阶段 8.5 第一批公开数据候选池。")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR, help="stage8_5 根目录。")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID, help="候选 case 默认 dataset_id。")
    parser.add_argument("--owner-user-id", default="eval_demo_user", help="候选 case 固定测试用户。")
    parser.add_argument("--tenant-id", default=DEFAULT_TENANT_ID, help="候选 case 固定租户。")
    return parser


def _stage85_paths(base_dir: Path) -> dict[str, Path]:
    """集中定义输出路径，避免脚本里散落字符串。"""

    return {
        "licenses": base_dir / "sources/license_manifest.jsonl",
        "sources": base_dir / "sources/source_manifest.jsonl",
        "raw_cards": base_dir / "processed/fault_scenario_cards.raw.jsonl",
        "cards": base_dir / "processed/fault_scenario_cards.jsonl",
        "documents": base_dir / "processed/public_documents_manifest.jsonl",
        "chunk_map": base_dir / "processed/chunk_source_map.jsonl",
        "candidates": base_dir / "candidates/planner_case_candidates.jsonl",
        "approved": base_dir / "candidates/approved_cases.jsonl",
        "review": base_dir / "candidates/review_queue.jsonl",
        "rejected": base_dir / "candidates/rejected_cases.jsonl",
        "split_manifest": base_dir / "candidates/split_manifest.json",
        "quality_report": base_dir / "results/data_quality_report.json",
        "markdown_report": base_dir / "reports/阶段8.5数据处理报告.md",
    }


def _ensure_directories(paths: dict[str, Path]) -> None:
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)


def _license_records() -> list[LicenseRecord]:
    """第一批来源都使用 UCI 页面标注的 CC BY 4.0。"""

    return [
        LicenseRecord(
            license_id="cc-by-4.0",
            license_name="CC BY 4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            redistribution_allowed=True,
            training_allowed=True,
            commercial_use_allowed=True,
            notes="需要署名；本阶段只生成可审计候选，不再分发原始数据文件。",
        )
    ]


def _source_records() -> list[SourceRecord]:
    """第一批只选择许可证明确、字段解释公开、适合故障场景卡片化的来源。"""

    return [
        SourceRecord(
            source_id="uci-metropt3",
            source_type="timeseries",
            title="MetroPT-3 Dataset",
            publisher="UCI Machine Learning Repository",
            url_or_path="https://archive.ics.uci.edu/dataset/791/metropt%2B3%2Bdataset",
            collected_at=COLLECTED_AT,
            source_hash="",
            license_name="CC BY 4.0",
            license_url="",
            redistribution_allowed=None,
            training_allowed=None,
            commercial_use_allowed=None,
            approval_status="approved",
            reject_reason="",
            notes="时序传感器数据；本阶段只抽取字段语义和故障现象生成卡片，不复制原始 CSV。",
        ),
        SourceRecord(
            source_id="uci-ai4i-2020",
            source_type="table",
            title="AI4I 2020 Predictive Maintenance Dataset",
            publisher="UCI Machine Learning Repository",
            url_or_path="https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset",
            collected_at=COLLECTED_AT,
            source_hash="",
            license_name="CC BY 4.0",
            license_url="",
            redistribution_allowed=None,
            training_allowed=None,
            commercial_use_allowed=None,
            approval_status="approved",
            reject_reason="",
            notes="表格型预测维护数据；只使用公开字段和失效类型生成候选场景。",
        ),
        SourceRecord(
            source_id="uci-hydraulic-condition",
            source_type="timeseries",
            title="Condition Monitoring of Hydraulic Systems",
            publisher="UCI Machine Learning Repository",
            url_or_path="https://archive.ics.uci.edu/dataset/447/condition%2Bmonitoring%2Bof%2Bhydraulic%2Bsystems",
            collected_at=COLLECTED_AT,
            source_hash="",
            license_name="CC BY 4.0",
            license_url="",
            redistribution_allowed=None,
            training_allowed=None,
            commercial_use_allowed=None,
            approval_status="approved",
            reject_reason="",
            notes="液压系统状态监测数据；先转为阀、泵、蓄能器和冷却回路故障卡片。",
        ),
    ]


def _card_seeds() -> list[CardSeed]:
    """返回 26 张卡片：每张 2 个候选问题，共 52 条候选 case。"""

    seeds: list[CardSeed] = []
    seeds.extend(_metropt_cards())
    seeds.extend(_ai4i_cards())
    seeds.extend(_hydraulic_cards())
    return seeds


def _metropt_cards() -> list[CardSeed]:
    source = "uci-metropt3"
    document_id = "doc_stage85_uci_metropt3_seed_v1"
    equipment = "MetroPT-3 APU"
    specs = [
        ("air-leak-pressure-recovery", "air_leak_pressure_recovery", "air compressor",
         "储气罐压力恢复慢，压缩机运行周期变长。", ["气路泄漏", "排放阀或管路密封异常"],
         ["查看 Reservoirs、COMP 和 DV_pressure 趋势", "确认压力下降是否伴随压缩机频繁启动"],
         ["停机泄压后检查接头和阀组密封", "修复泄漏后复测压力恢复时间"]),
        ("valve-lag-response", "valve_response_lag", "pneumatic valve",
         "阀动作后压力变化滞后，执行机构响应慢。", ["阀芯卡滞", "控制阀响应延迟"],
         ["对齐阀命令时间和 DV_pressure 变化", "检查动作前后压力阶跃是否变钝"],
         ["清洁或更换阀组件", "复核控制气路是否堵塞"]),
        ("compressor-over-cycling", "compressor_over_cycling", "air compressor",
         "压缩机频繁启停，压力波动幅度增大。", ["储气罐容量不足", "压力开关或泄漏导致频繁补气"],
         ["统计 COMP 启停频率", "比较 Reservoirs 压力上下限"],
         ["检查压力开关阈值", "排查储气罐和气路泄漏"]),
        ("oil-temperature-rise", "oil_temperature_rise", "oil circuit",
         "运行期间油温持续升高，恢复速度变慢。", ["润滑不足", "散热条件下降"],
         ["查看 Oil_temperature 与 COMP 负载趋势", "确认环境温度是否同步升高"],
         ["检查润滑油状态", "清理散热通道并复测温升"]),
        ("abnormal-current-draw", "abnormal_current_draw", "compressor motor",
         "压缩机电流高于同工况基线。", ["电机负载异常", "机械阻力增大"],
         ["比较 Motor_current 与压力恢复速度", "确认异常是否只在压缩机启动时出现"],
         ["检查电机轴承和联轴器", "确认供电端子无松动"]),
        ("pressure-drop-after-stop", "pressure_drop_after_stop", "reservoir",
         "停机后储气罐压力快速下降。", ["单向阀密封不良", "储气罐或管路微漏"],
         ["记录停机后 Reservoirs 压力衰减曲线", "隔离阀组确认泄漏段"],
         ["检查单向阀和储气罐接头", "完成泄漏修复后做保压测试"]),
        ("cooling-fan-ineffective", "cooling_fan_ineffective", "cooling path",
         "负载不高但温度回落慢。", ["冷却风道堵塞", "风扇效率下降"],
         ["比较油温回落斜率和压缩机停机时间", "检查冷却风道是否积尘"],
         ["清洁风道", "检查风扇供电和转速"]),
        ("sensor-flatline", "sensor_flatline", "pressure sensor",
         "压力读数长时间不变但压缩机状态在变化。", ["压力传感器失效", "采集通道冻结"],
         ["比对 COMP 状态和压力信号", "检查同一时段其他压力测点是否变化"],
         ["检查传感器接线", "重启采集通道并复测"]),
        ("maintenance-after-leak", "maintenance_after_leak", "air circuit",
         "空气泄漏处理后仍偶发压力恢复慢。", ["泄漏点未完全排除", "阀门复位不稳定"],
         ["复查维修前后压力曲线", "检查阀门复位后的压力保持时间"],
         ["二次保压测试", "记录维修后基线用于后续对比"]),
    ]
    return _card_seed_group(source, document_id, equipment, specs)


def _ai4i_cards() -> list[CardSeed]:
    source = "uci-ai4i-2020"
    document_id = "doc_stage85_uci_ai4i_seed_v1"
    equipment = "AI4I milling machine"
    specs = [
        ("tool-wear-failure", "tool_wear_failure", "cutting tool",
         "刀具磨损时间接近失效区间，表面质量波动。", ["刀具寿命接近上限", "进给或材料负载偏高"],
         ["检查 Tool wear min 与加工质量趋势", "确认同批次材料负载是否升高"],
         ["安排换刀", "降低负载后复测加工质量"]),
        ("heat-dissipation-failure", "heat_dissipation_failure", "spindle thermal path",
         "工艺温度与空气温度差异常，散热能力下降。", ["散热通道堵塞", "冷却条件不足"],
         ["比较 Process temperature 和 Air temperature 差值", "检查散热风道"],
         ["清理散热通道", "确认冷却风量和环境温度"]),
        ("power-failure-high-load", "power_failure_high_load", "drive system",
         "转矩和转速组合导致功率负载异常。", ["负载过高", "传动阻力增大"],
         ["计算转矩与转速对应功率", "比对同产品类型历史负载"],
         ["降低进给负载", "检查传动链和轴承状态"]),
        ("overstrain-failure", "overstrain_failure", "mechanical structure",
         "高转矩叠加刀具磨损，设备过载风险上升。", ["刀具磨损导致切削力升高", "工件夹持或材料异常"],
         ["同时查看 Torque 和 Tool wear", "检查加工批次是否更换材料"],
         ["换刀并复核夹具", "降低单次切削深度"]),
        ("random-failure-review", "random_failure_review", "machine health",
         "存在随机失效标签但缺少单一可解释传感器原因。", ["未知偶发故障", "未采集到关键变量"],
         ["检查是否存在 TWF/HDF/PWF/OSF 以外异常", "复核维护记录"],
         ["进入人工复核队列", "补充故障前后上下文"]),
        ("low-speed-high-torque", "low_speed_high_torque", "drive shaft",
         "低转速高转矩时振动和加工负载风险升高。", ["切削阻力增加", "传动轴润滑不足"],
         ["按转速分桶比较 Torque", "检查是否伴随温升"],
         ["检查润滑状态", "降低切削负载后观察"]),
        ("temperature-drift", "temperature_drift", "thermal sensor",
         "工艺温度缓慢漂移，未必马上触发 HDF。", ["温度传感器漂移", "散热效率下降早期"],
         ["比较空气温度和工艺温度同步性", "复测温度传感器"],
         ["校准温度传感器", "清理散热区域"]),
        ("product-type-risk", "product_type_risk", "process configuration",
         "不同产品类型下同一负载阈值风险不同。", ["产品类型影响工艺窗口", "统一阈值不适用"],
         ["按 Product ID/Type 分组查看失效率", "复核同类型历史负载"],
         ["按产品类型维护阈值", "人工确认工艺窗口"]),
        ("post-maintenance-baseline", "post_maintenance_baseline", "maintenance record",
         "维修后负载恢复正常但刀具磨损仍需跟踪。", ["维修后基线未重新冻结", "刀具剩余寿命不足"],
         ["比较维修前后 Torque 和 Temperature", "记录 Tool wear 新基线"],
         ["冻结维修后基线", "安排后续刀具检查"]),
    ]
    return _card_seed_group(source, document_id, equipment, specs)


def _hydraulic_cards() -> list[CardSeed]:
    source = "uci-hydraulic-condition"
    document_id = "doc_stage85_uci_hydraulic_seed_v1"
    equipment = "Hydraulic test rig"
    specs = [
        ("cooler-efficiency-low", "cooler_efficiency_low", "cooler",
         "冷却效率下降，温度传感器显示回落慢。", ["冷却器堵塞", "冷却回路流量不足"],
         ["比较 TS 温度序列和冷却器状态标签", "检查 FS 流量是否下降"],
         ["清洁冷却器", "检查冷却泵和管路"]),
        ("valve-switching-delay", "valve_switching_delay", "directional valve",
         "阀切换后压力响应延迟。", ["阀芯磨损", "控制信号或液压油污染"],
         ["对齐阀命令与 PS 压力变化", "检查 VS 振动是否升高"],
         ["检查阀芯和油液污染", "更换异常阀件"]),
        ("internal-pump-leakage", "internal_pump_leakage", "hydraulic pump",
         "泵效率下降，同负载下压力建立慢。", ["泵内泄漏", "密封磨损"],
         ["比较 PS 压力建立速度和 CP 功率", "检查流量与压力是否同时下降"],
         ["检查泵密封件", "做泵效率测试"]),
        ("accumulator-pressure-low", "accumulator_pressure_low", "accumulator",
         "蓄能器压力偏低，系统压力波动变大。", ["蓄能器预充压力不足", "隔膜老化"],
         ["检查 EPS/PS 压力波动", "比对蓄能器状态标签"],
         ["检查预充压力", "必要时更换蓄能器"]),
        ("stable-flag-mismatch", "stable_flag_mismatch", "system state",
         "系统处于非稳定状态时仍被用于评估故障。", ["工况未稳定", "采样窗口选择错误"],
         ["检查 stable flag", "过滤启动和切换阶段数据"],
         ["只在稳定窗口评估状态", "重新抽取稳定样本"]),
        ("flow-sensor-drop", "flow_sensor_drop", "flow sensor",
         "流量传感器读数下降但压力未同步下降。", ["流量传感器漂移", "局部堵塞早期"],
         ["比较 FS 与 PS 信号一致性", "检查是否只单一传感器异常"],
         ["校验流量传感器", "检查过滤器和局部管路"]),
        ("vibration-rise", "vibration_rise", "pump bearing",
         "振动信号升高并伴随功率波动。", ["泵轴承磨损", "联轴器不对中"],
         ["查看 VS 与 CP/SE 功率信号", "确认是否与压力脉动同步"],
         ["检查轴承和联轴器", "复测振动基线"]),
        ("oil-degradation-suspected", "oil_degradation_suspected", "hydraulic oil",
         "多传感器出现缓慢漂移，怀疑油液劣化。", ["油液污染", "过滤器失效"],
         ["比较温度、压力、流量长期趋势", "检查维护记录中的换油周期"],
         ["抽样检测油液", "更换过滤器并更新维护记录"]),
    ]
    return _card_seed_group(source, document_id, equipment, specs)


def _card_seed_group(
        source_id: str,
        document_id: str,
        equipment_model: str,
        specs: list[tuple[str, str, str, str, list[str], list[str], list[str]]],
) -> list[CardSeed]:
    """把紧凑规格转成完整 FaultScenarioCard seed。"""

    seeds: list[CardSeed] = []
    for index, (slug, section, component, symptom, causes, diagnostics, actions) in enumerate(specs, start=1):
        card_id = f"{source_id.replace('uci-', '')}-{slug}"
        reviewed = index <= 4
        split = _seed_split(index, reviewed=reviewed)
        seeds.append(CardSeed(
            split=split,
            reviewed=reviewed,
            payload=_card_payload(
                card_id=card_id,
                source_id=source_id,
                document_id=document_id,
                source_section=section,
                equipment_model=equipment_model,
                component_name=component,
                symptom=symptom,
                possible_causes=causes,
                diagnostic_steps=diagnostics,
                maintenance_actions=actions,
            ),
        ))
    return seeds


def _seed_split(index: int, *, reviewed: bool) -> CaseSplit:
    """同一张卡片的两个问题使用同一 split，避免 leakage_group 跨 split。"""

    if reviewed:
        if index <= 2:
            return CaseSplit.TRAIN
        if index == 3:
            return CaseSplit.DEV
        return CaseSplit.TEST
    return CaseSplit.TRAIN if index % 3 else CaseSplit.DEV


def _card_payload(
        *,
        card_id: str,
        source_id: str,
        document_id: str,
        source_section: str,
        equipment_model: str,
        component_name: str,
        symptom: str,
        possible_causes: list[str],
        diagnostic_steps: list[str],
        maintenance_actions: list[str],
) -> dict[str, Any]:
    answer_point_ids = [f"{card_id}-symptom", f"{card_id}-diagnosis", f"{card_id}-action"]
    return {
        "card_id": card_id,
        "source_id": source_id,
        "source_document_id": document_id,
        "source_section": source_section,
        "equipment_model": equipment_model,
        "component_name": component_name,
        "alarm_code": "",
        "symptom": symptom,
        "possible_causes": possible_causes,
        "diagnostic_steps": diagnostic_steps,
        "maintenance_actions": maintenance_actions,
        "safety_notes": ["执行维护前先停机、隔离能量并确认压力或温度处于安全范围。"],
        "evidence_text": (
            "公开数据字段和状态标签经人工整理为故障场景卡片；"
            f"source_id={source_id}, section={source_section}。"
        ),
        "evidence_chunk_ids": [
            {
                "document_id": document_id,
                "chunk_id": f"chunk_{card_id}",
                "index_version": 1,
                "relevance": "required",
                "answer_point_ids": answer_point_ids,
            }
        ],
        "quality_flags": ["seed_interpretation_requires_domain_review"],
        "candidate_queries": [
            f"{equipment_model} 的 {component_name} 出现{symptom}应该先排查什么？",
            f"{equipment_model} 中 {component_name} 相关的现象“{symptom}”可能是什么原因？",
        ],
        "expected_answer_points": [
            symptom,
            *possible_causes[:2],
            *diagnostic_steps[:2],
            *maintenance_actions[:2],
            "维护前需要先停机并确认安全边界。",
        ],
    }


def _public_document_records(
        sources: list[SourceRecord],
        cards: list[FaultScenarioCard],
) -> list[dict[str, Any]]:
    """生成公开文档处理清单；这里记录的是卡片化后的 RAG 候选文档版本。"""

    cards_by_source = Counter(card.source_id for card in cards)
    document_by_source = {
        "uci-metropt3": "doc_stage85_uci_metropt3_seed_v1",
        "uci-ai4i-2020": "doc_stage85_uci_ai4i_seed_v1",
        "uci-hydraulic-condition": "doc_stage85_uci_hydraulic_seed_v1",
    }
    return [
        {
            "source_id": source.source_id,
            "document_id": document_by_source[source.source_id],
            "title": f"{source.title} - stage 8.5 fault card digest",
            "url_or_path": source.url_or_path,
            "license_name": source.license_name,
            "index_version": 1,
            "chunk_count": cards_by_source[source.source_id],
            "processing_status": "fault_card_digest_seeded",
            "generation_method": SEED_VERSION,
            "notes": "卡片化摘要用于候选生成；不保存或再分发原始公开数据文件。",
        }
        for source in sources
    ]


def _chunk_source_records(cards: list[FaultScenarioCard]) -> list[dict[str, Any]]:
    """生成 chunk 到来源卡片的映射，满足 approved case 可追溯要求。"""

    records: list[dict[str, Any]] = []
    for card in cards:
        for chunk in card.evidence_chunk_ids:
            records.append({
                "source_id": card.source_id,
                "card_id": card.card_id,
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "index_version": chunk.index_version,
                "source_section": card.source_section,
                "relevance": chunk.relevance,
                "answer_point_ids": chunk.answer_point_ids,
                "generation_method": SEED_VERSION,
                "evidence_summary": card.evidence_text,
            })
    return records


def _build_quality_report(
        *,
        paths: dict[str, Path],
        sources: list[SourceRecord],
        cards: list[FaultScenarioCard],
        approved_cases: list[Any],
        review_cases: list[Any],
        rejected_cases: list[Any],
        source_report: Stage85QualityReport,
        card_issues: list[Stage85Issue],
        rejected_card_issues: list[Stage85Issue],
        candidate_issues: list[Stage85Issue],
) -> Stage85QualityReport:
    """汇总机器可读质量报告，供 Markdown 报告和人工抽查使用。"""

    approved_split_counts = Counter(case.split.value for case in approved_cases)
    return Stage85QualityReport(
        report_version=SEED_VERSION,
        files={key: str(path) for key, path in sorted(paths.items())},
        source_counts={
            "total": len(sources),
            "approved": len(source_report.approved_source_ids),
            "pending": 0,
            "rejected": 0,
        },
        card_counts={
            "raw_total": len(cards),
            "accepted": len(cards),
            "rejected": 0,
        },
        case_counts={
            "total": len(approved_cases) + len(review_cases) + len(rejected_cases),
            "approved": len(approved_cases),
            "review": len(review_cases),
            "rejected": len(rejected_cases),
        },
        split_counts=dict(sorted(approved_split_counts.items())),
        approved_source_ids=source_report.approved_source_ids,
        issues=[
            *source_report.issues,
            *card_issues,
            *rejected_card_issues,
            *candidate_issues,
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
