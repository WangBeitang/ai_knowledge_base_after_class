"""把第一批公开数据候选重写为可逐答案点追溯的 source-grounded gold。

本脚本只处理已经从 UCI 官方数据集说明中核实的两类事实：

- AI4I 的 TWF/HDF/PWF/OSF/RNF 五类失效规则。
- Hydraulic 数据集 ``profile.txt`` 的五列状态标签含义。

这里的 ``source-grounded`` 表示每个标准答案要点都直接来自已记录的来源事实，不包含
维修动作、未公开根因或领域经验推断。脚本不会修改原 52 条候选，而是生成独立 gold 文件，
因此旧候选、第一轮审核结论和重写结果可以并存审计。

几个关键边界：

- ``human_review_status=reviewed`` 表示这 20 条已经通过当前阶段的来源事实复核门禁；
  ``label_source=api_assisted`` 仍如实说明重写由 agent 辅助完成，不冒充领域专家人工标注。
- ``second_review_status=pending`` 保存在审计文件中，供另一个 agent 独立复核。
- 证据文档当前是待导入的离线证据摘要；在阶段 8.5.4 真正跑检索评测前，必须先导入知识库
  并生成包含这些 document/chunk 身份的新 Environment Snapshot（环境快照）。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.evaluation.case_schema import PlannerEvalCase  # noqa: E402
from app.shared.config.knowledge_base_config import (  # noqa: E402
    DEFAULT_DATASET_ID,
    DEFAULT_TENANT_ID,
)
from evaluation.stage8_5.pipelines.common.paths import (  # noqa: E402
    DEFAULT_STAGE85_ROOT,
    stage85_layout,
)
from evaluation.stage8_5.pipelines.common.stage85_schema import write_jsonl  # noqa: E402


GOLD_VERSION = "stage85-source-grounded-gold-v1"
SOURCE_CHECKED_AT = "2026-07-20"


class GoldModel(BaseModel):
    """Gold 审计文件公共基类；拒绝未知字段，防止证据字段拼错后被静默丢弃。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class EvidenceFact(GoldModel):
    """一个最小可验证来源事实。

    ``fact_id`` 是答案要点与证据之间的稳定关联键；``statement_zh`` 是中文事实摘要，
    它既写入证据 chunk，也直接生成 case 的 ``expected_answer_points``。这种单一来源设计
    可以避免证据改了而标准答案仍保留旧说法。
    """

    fact_id: str = Field(min_length=1, description="来源事实稳定 ID，在同一证据 chunk 内唯一。")
    statement_zh: str = Field(min_length=1, description="仅包含官方来源直接支持内容的中文事实摘要。")


class GoldEvidenceChunk(GoldModel):
    """可导入 RAG 的 gold 证据片段及其来源定位信息。"""

    source_id: str = Field(min_length=1, description="关联 source_manifest.jsonl 的公开来源 ID。")
    source_title: str = Field(min_length=1, description="UCI 官方数据集标题，供报告和人工复核展示。")
    source_url: str = Field(min_length=1, description="UCI 官方数据集页面 URL。")
    source_locator: str = Field(min_length=1, description="页面内可定位章节或文件字段，例如 Additional Variable Information。")
    source_checked_at: str = Field(min_length=1, description="本轮核对官方来源的日期，格式 YYYY-MM-DD。")
    document_id: str = Field(min_length=1, description="待导入 gold 证据文档 ID；导入后必须保持一致。")
    chunk_id: str = Field(min_length=1, description="证据 chunk 稳定 ID，PlannerEvalCase.expected_chunks 会引用它。")
    index_version: int = Field(ge=1, description="计划导入版本；重建文档后必须同步更新 gold case。")
    topic: str = Field(min_length=1, description="本 chunk 覆盖的规则或 profile 字段主题。")
    evidence_text_zh: str = Field(min_length=1, description="用于本地检索与人工阅读的中文证据摘要。")
    facts: list[EvidenceFact] = Field(min_length=1, description="该 chunk 直接支撑的原子事实列表。")
    license_name: str = Field(default="CC BY 4.0", description="来源许可证；使用时仍需保留 UCI 署名。")
    generation_method: str = Field(default=GOLD_VERSION, description="证据摘要生成版本，用于审计和重建。")

    @model_validator(mode="after")
    def validate_unique_fact_ids(self) -> "GoldEvidenceChunk":
        fact_ids = [fact.fact_id for fact in self.facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("同一 gold evidence chunk 的 fact_id 不能重复")
        return self


class GoldAnswerEvidence(GoldModel):
    """一条标准答案要点与来源事实的逐点映射。"""

    answer_point_id: str = Field(min_length=1, description="当前 case 内稳定的答案要点 ID。")
    answer_point: str = Field(min_length=1, description="写入 PlannerEvalCase.expected_answer_points 的标准答案。")
    evidence_fact_ids: list[str] = Field(min_length=1, description="直接支撑该答案点的 EvidenceFact ID。")


class GoldCaseAudit(GoldModel):
    """重写后 gold case 的审计记录，不参与 Planner 运行时评分。"""

    case_id: str = Field(min_length=1, description="关联 gold_cases_authoring.jsonl 的新 case_id。")
    rewritten_from_case_id: str = Field(min_length=1, description="本条由原 52 条候选中的哪条 case 重写。")
    source_id: str = Field(min_length=1, description="公开来源 ID。")
    document_id: str = Field(min_length=1, description="gold 证据文档 ID。")
    chunk_id: str = Field(min_length=1, description="gold 证据 chunk ID。")
    gold_status: str = Field(default="source_verified", description="source_verified 表示所有答案点已完成来源核对。")
    reviewer_type: str = Field(default="primary_agent", description="当前复核者类型；不伪装成人类领域专家。")
    second_review_status: str = Field(default="pending", description="独立第二轮 agent 复审状态，当前默认 pending。")
    second_reviewer_type: str = Field(default="", description="二审执行者类型；通过前为空，通过后记录 independent_agent。")
    second_review_artifact: str = Field(default="", description="二审结果文件路径；用于证明 passed 状态来自哪份独立审核。")
    answer_evidence: list[GoldAnswerEvidence] = Field(min_length=1, description="每个答案要点到来源事实的映射。")
    excluded_content: list[str] = Field(default_factory=list, description="从原候选删除的无来源维修动作、根因或推断。")
    review_note: str = Field(min_length=1, description="说明为何本条可以达到 source-grounded gold。")


@dataclass(frozen=True)
class RewriteCase:
    """一条旧 case 到新 gold case 的人工策划规格。"""

    old_case_id: str
    new_case_id: str
    query: str
    fact_ids: tuple[str, ...]
    excluded_content: tuple[str, ...]


@dataclass(frozen=True)
class TopicSpec:
    """同一来源规则或 profile 字段的证据与两条问题规格。"""

    source_id: str
    source_title: str
    source_url: str
    source_locator: str
    document_id: str
    chunk_id: str
    topic: str
    subject_names: tuple[str, ...]
    identifier_key: str
    identifier_values: tuple[str, ...]
    facts: tuple[tuple[str, str], ...]
    rewrites: tuple[RewriteCase, RewriteCase]


def _topic_specs() -> tuple[TopicSpec, ...]:
    """返回 10 个已核实主题，每个主题重写 2 条旧候选，共 20 条 gold。"""

    ai4i_url = "https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset"
    ai4i_doc = "doc_stage85_uci_ai4i_official_description_v1"
    hydraulic_url = "https://archive.ics.uci.edu/dataset/447/condition%2Bmonitoring%2Bof%2Bhydraulic%2Bsystems"
    hydraulic_doc = "doc_stage85_uci_hydraulic_official_description_v1"

    return (
        TopicSpec(
            source_id="uci-ai4i-2020",
            source_title="AI4I 2020 Predictive Maintenance Dataset",
            source_url=ai4i_url,
            source_locator="Additional Variable Information > tool wear failure (TWF)",
            document_id=ai4i_doc,
            chunk_id="chunk_ai4i_twf_rule",
            topic="AI4I TWF rule",
            subject_names=("AI4I 2020", "tool wear failure (TWF)"),
            identifier_key="failure_mode",
            identifier_values=("TWF",),
            facts=(
                ("twf-1", "TWF 的触发时点是在 200 至 240 分钟之间随机选定的刀具磨损时间。"),
                ("twf-2", "到达该时点时，数据生成过程会让刀具被更换或发生失效。"),
                ("twf-3", "数据集中共有 120 个这样的时点，其中 69 次更换刀具、51 次标为失效。"),
                ("twf-4", "更换还是失效是随机分配的。"),
            ),
            rewrites=(
                RewriteCase(
                    old_case_id="stage85-ai4i-2020-tool-wear-failure-001",
                    new_case_id="stage85-gold-ai4i-twf-rule-001",
                    query="AI4I 2020 数据集中的 TWF 在什么条件下产生？",
                    fact_ids=("twf-1", "twf-2"),
                    excluded_content=("表面质量波动", "检查材料负载", "安排换刀"),
                ),
                RewriteCase(
                    old_case_id="stage85-ai4i-2020-tool-wear-failure-002",
                    new_case_id="stage85-gold-ai4i-twf-rule-002",
                    query="AI4I 2020 的 120 个刀具磨损时点中，更换和失效各有多少次？",
                    fact_ids=("twf-3", "twf-4"),
                    excluded_content=("刀具寿命接近上限", "降低负载后复测加工质量"),
                ),
            ),
        ),
        TopicSpec(
            source_id="uci-ai4i-2020",
            source_title="AI4I 2020 Predictive Maintenance Dataset",
            source_url=ai4i_url,
            source_locator="Additional Variable Information > heat dissipation failure (HDF)",
            document_id=ai4i_doc,
            chunk_id="chunk_ai4i_hdf_rule",
            topic="AI4I HDF rule",
            subject_names=("AI4I 2020", "heat dissipation failure (HDF)"),
            identifier_key="failure_mode",
            identifier_values=("HDF",),
            facts=(
                ("hdf-1", "HDF 要求工艺温度与空气温度的差值低于 8.6 K。"),
                ("hdf-2", "HDF 同时要求转速低于 1380 rpm。"),
                ("hdf-3", "两个条件必须同时满足才按该规则发生散热失效。"),
                ("hdf-4", "AI4I 2020 数据集中有 115 个数据点满足 HDF 规则。"),
            ),
            rewrites=(
                RewriteCase(
                    old_case_id="stage85-ai4i-2020-heat-dissipation-failure-001",
                    new_case_id="stage85-gold-ai4i-hdf-rule-001",
                    query="AI4I 2020 数据集如何判定 HDF？",
                    fact_ids=("hdf-1", "hdf-2", "hdf-3"),
                    excluded_content=("散热通道堵塞", "清理散热通道", "确认冷却风量"),
                ),
                RewriteCase(
                    old_case_id="stage85-ai4i-2020-heat-dissipation-failure-002",
                    new_case_id="stage85-gold-ai4i-hdf-rule-002",
                    query="AI4I 2020 中有多少个数据点满足 HDF 规则，这个规则是单条件还是组合条件？",
                    fact_ids=("hdf-4", "hdf-3"),
                    excluded_content=("spindle thermal path", "冷却条件不足"),
                ),
            ),
        ),
        TopicSpec(
            source_id="uci-ai4i-2020",
            source_title="AI4I 2020 Predictive Maintenance Dataset",
            source_url=ai4i_url,
            source_locator="Additional Variable Information > power failure (PWF)",
            document_id=ai4i_doc,
            chunk_id="chunk_ai4i_pwf_rule",
            topic="AI4I PWF rule",
            subject_names=("AI4I 2020", "power failure (PWF)"),
            identifier_key="failure_mode",
            identifier_values=("PWF",),
            facts=(
                ("pwf-1", "PWF 使用转矩与以 rad/s 表示的转速相乘，得到过程所需功率。"),
                ("pwf-2", "计算功率低于 3500 W 时，过程按 PWF 规则失效。"),
                ("pwf-3", "计算功率高于 9000 W 时，过程按 PWF 规则失效。"),
                ("pwf-4", "AI4I 2020 数据集中有 95 个数据点满足 PWF 规则。"),
            ),
            rewrites=(
                RewriteCase(
                    old_case_id="stage85-ai4i-2020-power-failure-high-load-001",
                    new_case_id="stage85-gold-ai4i-pwf-rule-001",
                    query="AI4I 2020 的 PWF 如何根据转矩和转速判定？",
                    fact_ids=("pwf-1", "pwf-2", "pwf-3"),
                    excluded_content=("传动阻力增大", "降低进给负载", "检查轴承"),
                ),
                RewriteCase(
                    old_case_id="stage85-ai4i-2020-power-failure-high-load-002",
                    new_case_id="stage85-gold-ai4i-pwf-rule-002",
                    query="AI4I 2020 的 PWF 功率上下限分别是多少，数据集中有多少条 PWF？",
                    fact_ids=("pwf-2", "pwf-3", "pwf-4"),
                    excluded_content=("drive system 根因", "检查传动链状态"),
                ),
            ),
        ),
        TopicSpec(
            source_id="uci-ai4i-2020",
            source_title="AI4I 2020 Predictive Maintenance Dataset",
            source_url=ai4i_url,
            source_locator="Additional Variable Information > overstrain failure (OSF)",
            document_id=ai4i_doc,
            chunk_id="chunk_ai4i_osf_rule",
            topic="AI4I OSF rule",
            subject_names=("AI4I 2020", "overstrain failure (OSF)"),
            identifier_key="failure_mode",
            identifier_values=("OSF",),
            facts=(
                ("osf-1", "OSF 比较刀具磨损时间与转矩的乘积。"),
                ("osf-2", "L 产品类型的 OSF 阈值为 11000 minNm。"),
                ("osf-3", "M 产品类型的 OSF 阈值为 12000 minNm。"),
                ("osf-4", "H 产品类型的 OSF 阈值为 13000 minNm。"),
                ("osf-5", "AI4I 2020 数据集中有 98 个数据点满足 OSF 规则。"),
            ),
            rewrites=(
                RewriteCase(
                    old_case_id="stage85-ai4i-2020-overstrain-failure-001",
                    new_case_id="stage85-gold-ai4i-osf-rule-001",
                    query="AI4I 2020 的 OSF 使用哪些变量，L/M/H 三类产品的阈值分别是多少？",
                    fact_ids=("osf-1", "osf-2", "osf-3", "osf-4"),
                    excluded_content=("工件夹持异常", "换刀并复核夹具", "降低切削深度"),
                ),
                RewriteCase(
                    old_case_id="stage85-ai4i-2020-overstrain-failure-002",
                    new_case_id="stage85-gold-ai4i-osf-rule-002",
                    query="AI4I 2020 中 H 类产品的刀具磨损与转矩乘积超过多少会触发 OSF，共有多少条 OSF？",
                    fact_ids=("osf-4", "osf-5"),
                    excluded_content=("设备过载维修建议", "检查材料批次"),
                ),
            ),
        ),
        TopicSpec(
            source_id="uci-ai4i-2020",
            source_title="AI4I 2020 Predictive Maintenance Dataset",
            source_url=ai4i_url,
            source_locator="Additional Variable Information > random failures (RNF)",
            document_id=ai4i_doc,
            chunk_id="chunk_ai4i_rnf_rule",
            topic="AI4I RNF rule",
            subject_names=("AI4I 2020", "random failure (RNF)"),
            identifier_key="failure_mode",
            identifier_values=("RNF",),
            facts=(
                ("rnf-1", "AI4I 2020 中每个过程有 0.1% 的概率发生 RNF。"),
                ("rnf-2", "RNF 与该过程的工艺参数无关。"),
                ("rnf-3", "数据集中实际有 5 个 RNF 数据点。"),
                ("rnf-4", "五种失效模式中任意一种为真时，Machine failure 标签都会设为 1。"),
            ),
            rewrites=(
                RewriteCase(
                    old_case_id="stage85-ai4i-2020-random-failure-review-001",
                    new_case_id="stage85-gold-ai4i-rnf-rule-001",
                    query="AI4I 2020 的 RNF 与工艺参数是什么关系，设定概率是多少？",
                    fact_ids=("rnf-1", "rnf-2"),
                    excluded_content=("未知偶发故障", "补充故障上下文", "人工维修复核"),
                ),
                RewriteCase(
                    old_case_id="stage85-ai4i-2020-random-failure-review-002",
                    new_case_id="stage85-gold-ai4i-rnf-rule-002",
                    query="AI4I 2020 实际有多少个 RNF 数据点，RNF 发生时 Machine failure 如何标记？",
                    fact_ids=("rnf-3", "rnf-4"),
                    excluded_content=("复核维护记录", "未采集关键变量"),
                ),
            ),
        ),
        TopicSpec(
            source_id="uci-hydraulic-condition",
            source_title="Condition Monitoring of Hydraulic Systems",
            source_url=hydraulic_url,
            source_locator="Attribute Information > profile.txt column 1: Cooler condition / %",
            document_id=hydraulic_doc,
            chunk_id="chunk_hydraulic_profile_cooler",
            topic="Hydraulic profile cooler labels",
            subject_names=("Hydraulic test rig", "profile.txt cooler condition"),
            identifier_key="profile_column",
            identifier_values=("1", "cooler condition"),
            facts=(
                ("cooler-1", "profile.txt 第 1 列表示冷却器状态，数值单位为百分比。"),
                ("cooler-2", "冷却器标签 3 表示接近完全失效。"),
                ("cooler-3", "冷却器标签 20 表示效率降低。"),
                ("cooler-4", "冷却器标签 100 表示完全有效。"),
            ),
            rewrites=(
                RewriteCase(
                    old_case_id="stage85-hydraulic-condition-cooler-efficiency-low-001",
                    new_case_id="stage85-gold-hydraulic-cooler-profile-001",
                    query="Hydraulic 数据集的 profile.txt 第 1 列表示什么，3/20/100 分别代表什么？",
                    fact_ids=("cooler-1", "cooler-2", "cooler-3", "cooler-4"),
                    excluded_content=("冷却器堵塞", "检查冷却泵和管路", "清洁冷却器"),
                ),
                RewriteCase(
                    old_case_id="stage85-hydraulic-condition-cooler-efficiency-low-002",
                    new_case_id="stage85-gold-hydraulic-cooler-profile-002",
                    query="Hydraulic 的 profile.txt 中 cooler condition=20 应解释为什么状态？",
                    fact_ids=("cooler-1", "cooler-3"),
                    excluded_content=("温度回落慢的根因", "冷却回路流量不足"),
                ),
            ),
        ),
        TopicSpec(
            source_id="uci-hydraulic-condition",
            source_title="Condition Monitoring of Hydraulic Systems",
            source_url=hydraulic_url,
            source_locator="Attribute Information > profile.txt column 2: Valve condition / %",
            document_id=hydraulic_doc,
            chunk_id="chunk_hydraulic_profile_valve",
            topic="Hydraulic profile valve labels",
            subject_names=("Hydraulic test rig", "profile.txt valve condition"),
            identifier_key="profile_column",
            identifier_values=("2", "valve condition"),
            facts=(
                ("valve-1", "profile.txt 第 2 列表示阀门状态，数值单位为百分比。"),
                ("valve-2", "阀门标签 100 表示最佳切换行为。"),
                ("valve-3", "阀门标签 90 表示轻微延迟。"),
                ("valve-4", "阀门标签 80 表示严重延迟。"),
                ("valve-5", "阀门标签 73 表示接近完全失效。"),
            ),
            rewrites=(
                RewriteCase(
                    old_case_id="stage85-hydraulic-condition-valve-switching-delay-001",
                    new_case_id="stage85-gold-hydraulic-valve-profile-001",
                    query="Hydraulic 数据集 profile.txt 的 valve condition 有哪些标签及含义？",
                    fact_ids=("valve-1", "valve-2", "valve-3", "valve-4", "valve-5"),
                    excluded_content=("阀芯磨损", "油液污染", "更换异常阀件"),
                ),
                RewriteCase(
                    old_case_id="stage85-hydraulic-condition-valve-switching-delay-002",
                    new_case_id="stage85-gold-hydraulic-valve-profile-002",
                    query="Hydraulic 的 profile.txt 中 valve condition=80 代表什么？",
                    fact_ids=("valve-1", "valve-4"),
                    excluded_content=("控制信号异常", "检查阀芯"),
                ),
            ),
        ),
        TopicSpec(
            source_id="uci-hydraulic-condition",
            source_title="Condition Monitoring of Hydraulic Systems",
            source_url=hydraulic_url,
            source_locator="Attribute Information > profile.txt column 3: Internal pump leakage",
            document_id=hydraulic_doc,
            chunk_id="chunk_hydraulic_profile_pump_leakage",
            topic="Hydraulic profile internal pump leakage labels",
            subject_names=("Hydraulic test rig", "profile.txt internal pump leakage"),
            identifier_key="profile_column",
            identifier_values=("3", "internal pump leakage"),
            facts=(
                ("pump-1", "profile.txt 第 3 列表示泵内部泄漏状态。"),
                ("pump-2", "泵内部泄漏标签 0 表示无泄漏。"),
                ("pump-3", "泵内部泄漏标签 1 表示轻微泄漏。"),
                ("pump-4", "泵内部泄漏标签 2 表示严重泄漏。"),
            ),
            rewrites=(
                RewriteCase(
                    old_case_id="stage85-hydraulic-condition-internal-pump-leakage-001",
                    new_case_id="stage85-gold-hydraulic-pump-profile-001",
                    query="Hydraulic 数据集 profile.txt 的 internal pump leakage 标签 0/1/2 各代表什么？",
                    fact_ids=("pump-1", "pump-2", "pump-3", "pump-4"),
                    excluded_content=("泵密封磨损", "做泵效率测试", "检查泵密封件"),
                ),
                RewriteCase(
                    old_case_id="stage85-hydraulic-condition-internal-pump-leakage-002",
                    new_case_id="stage85-gold-hydraulic-pump-profile-002",
                    query="Hydraulic 的 profile.txt 中 internal pump leakage=1 应如何解释？",
                    fact_ids=("pump-1", "pump-3"),
                    excluded_content=("压力建立慢", "流量与压力诊断"),
                ),
            ),
        ),
        TopicSpec(
            source_id="uci-hydraulic-condition",
            source_title="Condition Monitoring of Hydraulic Systems",
            source_url=hydraulic_url,
            source_locator="Attribute Information > profile.txt column 4: Hydraulic accumulator / bar",
            document_id=hydraulic_doc,
            chunk_id="chunk_hydraulic_profile_accumulator",
            topic="Hydraulic profile accumulator labels",
            subject_names=("Hydraulic test rig", "profile.txt hydraulic accumulator"),
            identifier_key="profile_column",
            identifier_values=("4", "hydraulic accumulator"),
            facts=(
                ("acc-1", "profile.txt 第 4 列表示液压蓄能器压力，单位为 bar。"),
                ("acc-2", "蓄能器标签 130 表示最佳压力。"),
                ("acc-3", "蓄能器标签 115 表示压力略有降低。"),
                ("acc-4", "蓄能器标签 100 表示压力严重降低。"),
                ("acc-5", "蓄能器标签 90 表示接近完全失效。"),
            ),
            rewrites=(
                RewriteCase(
                    old_case_id="stage85-hydraulic-condition-accumulator-pressure-low-001",
                    new_case_id="stage85-gold-hydraulic-accumulator-profile-001",
                    query="Hydraulic 数据集 profile.txt 的 accumulator 标签 130/115/100/90 分别代表什么？",
                    fact_ids=("acc-1", "acc-2", "acc-3", "acc-4", "acc-5"),
                    excluded_content=("隔膜老化", "检查预充压力", "更换蓄能器"),
                ),
                RewriteCase(
                    old_case_id="stage85-hydraulic-condition-accumulator-pressure-low-002",
                    new_case_id="stage85-gold-hydraulic-accumulator-profile-002",
                    query="Hydraulic 的 profile.txt 中 accumulator=100 bar 表示什么状态？",
                    fact_ids=("acc-1", "acc-4"),
                    excluded_content=("系统压力波动根因", "EPS/PS 诊断"),
                ),
            ),
        ),
        TopicSpec(
            source_id="uci-hydraulic-condition",
            source_title="Condition Monitoring of Hydraulic Systems",
            source_url=hydraulic_url,
            source_locator="Attribute Information > profile.txt column 5: stable flag",
            document_id=hydraulic_doc,
            chunk_id="chunk_hydraulic_profile_stable_flag",
            topic="Hydraulic profile stable flag labels",
            subject_names=("Hydraulic test rig", "profile.txt stable flag"),
            identifier_key="profile_column",
            identifier_values=("5", "stable flag"),
            facts=(
                ("stable-1", "profile.txt 第 5 列是 stable flag，用于描述该周期是否达到稳定状态。"),
                ("stable-2", "stable flag=0 表示条件稳定。"),
                ("stable-3", "stable flag=1 表示可能尚未达到静态条件。"),
                ("stable-4", "stable flag 与前四列组件状态标签分列记录。"),
            ),
            rewrites=(
                RewriteCase(
                    old_case_id="stage85-hydraulic-condition-stable-flag-mismatch-001",
                    new_case_id="stage85-gold-hydraulic-stable-profile-001",
                    query="Hydraulic 数据集 profile.txt 的 stable flag=0 和 1 分别表示什么？",
                    fact_ids=("stable-1", "stable-2", "stable-3"),
                    excluded_content=("过滤启动阶段", "重新抽取稳定样本"),
                ),
                RewriteCase(
                    old_case_id="stage85-hydraulic-condition-stable-flag-mismatch-002",
                    new_case_id="stage85-gold-hydraulic-stable-profile-002",
                    query="Hydraulic 的 stable flag=1 表示什么，它与前四列组件状态标签是什么关系？",
                    fact_ids=("stable-3", "stable-4"),
                    excluded_content=("采样窗口选择错误", "故障评估处置建议"),
                ),
            ),
        ),
    )


def _build_outputs() -> tuple[list[GoldEvidenceChunk], list[PlannerEvalCase], list[GoldCaseAudit]]:
    """从同一组来源事实同时构建证据、Planner case 和审计映射。"""

    evidence_chunks: list[GoldEvidenceChunk] = []
    cases: list[PlannerEvalCase] = []
    audits: list[GoldCaseAudit] = []

    for spec in _topic_specs():
        facts = [EvidenceFact(fact_id=fact_id, statement_zh=statement) for fact_id, statement in spec.facts]
        fact_by_id = {fact.fact_id: fact.statement_zh for fact in facts}
        evidence_chunks.append(GoldEvidenceChunk(
            source_id=spec.source_id,
            source_title=spec.source_title,
            source_url=spec.source_url,
            source_locator=spec.source_locator,
            source_checked_at=SOURCE_CHECKED_AT,
            document_id=spec.document_id,
            chunk_id=spec.chunk_id,
            index_version=1,
            topic=spec.topic,
            evidence_text_zh=" ".join(fact.statement_zh for fact in facts),
            facts=facts,
        ))

        for rewrite in spec.rewrites:
            unknown_fact_ids = set(rewrite.fact_ids) - set(fact_by_id)
            if unknown_fact_ids:
                raise ValueError(f"{rewrite.new_case_id} 引用了不存在的 fact_id: {sorted(unknown_fact_ids)}")

            answer_evidence = [
                GoldAnswerEvidence(
                    answer_point_id=f"{rewrite.new_case_id}-ap{index}",
                    answer_point=fact_by_id[fact_id],
                    evidence_fact_ids=[fact_id],
                )
                for index, fact_id in enumerate(rewrite.fact_ids, start=1)
            ]
            answer_point_ids = [item.answer_point_id for item in answer_evidence]
            cases.append(PlannerEvalCase.model_validate({
                "case_id": rewrite.new_case_id,
                "case_group": "core",
                # 这 20 条先作为训练 gold；不把同一官方说明拆到 dev/test，避免伪造 held-out 边界。
                "split": "train",
                "leakage_group_id": f"{spec.source_id}-{spec.chunk_id}",
                "query": rewrite.query,
                "query_variants": [],
                "dataset_ids": [DEFAULT_DATASET_ID],
                "owner_user_id": "eval_demo_user",
                "tenant_id": DEFAULT_TENANT_ID,
                "privacy_scope": "public_demo",
                "source_document_ids": [spec.document_id],
                "source_index_versions": {spec.document_id: 1},
                "expected_subject_ids": [],
                "expected_subject_names": list(spec.subject_names),
                "expected_chunks": [{
                    "document_id": spec.document_id,
                    "chunk_id": spec.chunk_id,
                    "index_version": 1,
                    "relevance": "required",
                    "answer_point_ids": answer_point_ids,
                }],
                "expected_answer_points": [item.answer_point for item in answer_evidence],
                "expected_behavior": {
                    "should_answer": True,
                    "should_refuse": False,
                    "should_ask_clarification": False,
                    "should_call_web": False,
                    "web_required_reason": "",
                    # 这些问题含有官方固定术语或数值，本地精确检索足够；HyDE/Web 会增加成本且无证据收益。
                    "forbidden_actions": ["hyde_search", "web_search"],
                },
                "acceptable_action_paths": [["local_search", "answer"]],
                "expected_identifiers": {
                    "dataset_name": [spec.source_title],
                    spec.identifier_key: list(spec.identifier_values),
                },
                "label_source": "api_assisted",
                "human_review_status": "reviewed",
                "notes": (
                    f"source-grounded gold; gold_version={GOLD_VERSION}; "
                    f"source_id={spec.source_id}; rewritten_from={rewrite.old_case_id}; "
                    "primary_agent_source_reviewed; second_agent_review=pending"
                ),
            }))
            audits.append(GoldCaseAudit(
                case_id=rewrite.new_case_id,
                rewritten_from_case_id=rewrite.old_case_id,
                source_id=spec.source_id,
                document_id=spec.document_id,
                chunk_id=spec.chunk_id,
                answer_evidence=answer_evidence,
                excluded_content=list(rewrite.excluded_content),
                review_note="query 与全部 expected_answer_points 均由同一 UCI 官方说明 chunk 的原子事实生成。",
            ))

    return evidence_chunks, cases, audits


def _document_manifest(evidence_chunks: list[GoldEvidenceChunk]) -> list[dict[str, object]]:
    """生成待导入证据文档清单；这里不宣称 Milvus 中已经存在这些 chunk。"""

    documents: dict[str, dict[str, object]] = {}
    for chunk in evidence_chunks:
        record = documents.setdefault(chunk.document_id, {
            "document_id": chunk.document_id,
            "source_id": chunk.source_id,
            "source_title": chunk.source_title,
            "source_url": chunk.source_url,
            "license_name": chunk.license_name,
            "index_version": chunk.index_version,
            "chunk_count": 0,
            "processing_status": "gold_evidence_ready_for_import",
            "generation_method": GOLD_VERSION,
            "notes": "UCI 官方说明的中文事实摘要；导入并生成新环境快照后才可执行检索评测。",
        })
        record["chunk_count"] = int(record["chunk_count"]) + 1
    return list(documents.values())


def _write_report(path: Path, *, cases: list[PlannerEvalCase], audits: list[GoldCaseAudit]) -> None:
    """写入面向人工阅读的重写报告。"""

    ai4i_count = sum("ai4i" in case.case_id for case in cases)
    hydraulic_count = len(cases) - ai4i_count
    lines = [
        "# 阶段 8.5 source-grounded gold 重写报告",
        "",
        "## 结果",
        "",
        f"- 共生成 {len(cases)} 条 source-grounded gold：AI4I {ai4i_count} 条，Hydraulic {hydraulic_count} 条。",
        "- AI4I 覆盖 TWF/HDF/PWF/OSF/RNF 五类官方规则；Hydraulic 覆盖 profile.txt 五列标签。",
        "- 所有答案要点均由 gold evidence chunk 中的原子事实直接生成，没有维修动作、设备根因或经验性建议。",
        "- 原 52 条候选及其第一轮审核结论未修改；每条新 case 都保存 rewritten_from_case_id。",
        "",
        "## 审核状态",
        "",
        "- `gold_status=source_verified`：主审核 agent 已逐答案点核对 UCI 官方说明。",
        "- `label_source=api_assisted`：明确保留 agent 辅助生成来源，不标为 manual。",
        "- `human_review_status=reviewed`：表示已通过阶段 8.5 当前审核门禁，不等同于领域专家背书。",
        "- `second_review_status=pending`：仍建议使用已有复审提示词让另一个 agent 独立检查。",
        "",
        "## 运行边界",
        "",
        "- `gold_evidence_documents.jsonl` 的状态是 `gold_evidence_ready_for_import`，不是已入库。",
        "- 在阶段 8.5.4 跑 Planner 检索评测前，需先导入两个 evidence document，并生成新的环境快照。",
        "- 20 条目前全部放入 train；同一 UCI 官方说明不拆到 dev/test，后续 held-out 应使用独立来源文档。",
        "",
        "## 逐条映射",
        "",
        "| 新 gold case | 原候选 case | 证据 chunk | 答案点数 |",
        "|---|---|---|---:|",
    ]
    for audit in audits:
        lines.append(
            f"| `{audit.case_id}` | `{audit.rewritten_from_case_id}` | `{audit.chunk_id}` | {len(audit.answer_evidence)} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """生成 20 条 gold case、证据 chunk、文档清单、审计映射和 Markdown 报告。"""

    args = _build_parser().parse_args(argv)
    evidence_chunks, cases, audits = _build_outputs()
    layout = stage85_layout(args.base_dir)
    write_jsonl(layout.curated_intermediate / "gold_evidence_chunks.jsonl", evidence_chunks)
    write_jsonl(
        layout.curated_intermediate / "gold_evidence_documents.jsonl",
        _document_manifest(evidence_chunks),
    )
    write_jsonl(layout.curated_intermediate / "gold_cases_authoring.jsonl", cases)
    write_jsonl(layout.curated_review / "gold_case_audit.jsonl", audits)
    _write_report(layout.curated_review / "gold_rewrite_report.md", cases=cases, audits=audits)
    print(
        f"gold_cases={len(cases)}, evidence_chunks={len(evidence_chunks)}, "
        f"audits={len(audits)}, version={GOLD_VERSION}"
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成阶段 8.5 第一批 source-grounded gold。")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_STAGE85_ROOT,
        help="阶段 8.5 根目录；测试可传临时目录，避免覆盖工作区产物。",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
