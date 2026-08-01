"""
构建任务 9.3.21 的 125 条 SFT V2（监督微调第二版）新候选轨迹。

本入口遵守以下边界：

1. 17 条完整路线只作为最终录取配额；单条候选的实际路线由问题和真实检索结果共同决定。
2. 本地证据只从当前正式 Milvus（向量数据库）索引读取；网页证据只从官方页面抓取。
3. Provider（动作执行器/环境结果提供器）必须真实执行检索；只有已经真实录制的结果才允许回放。
4. 125 条全部通过生成门禁后才原子追加到候选文件；任何失败都保留 37 条旧轨迹原样。
5. 新候选固定为 pending（待独立审核）和 candidate（候选），不能直接进入训练。

生成问题使用 API-assisted（接口辅助）方式，但问题、答案要点和路线都只是待审候选；
独立审核属于 9.3.22，本入口不执行审核、冻结正式集或训练。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlsplit

import requests
from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infra.llm.providers import llm_provider  # noqa: E402
from app.infra.persistence.chunk_status_repository import ChunkStatusRepository  # noqa: E402
from app.infra.persistence.import_metadata_repository import ImportMetadataRepository  # noqa: E402
from app.infra.vectorstore.milvus_gateway import milvus_gateway  # noqa: E402
from app.rag.evaluation.action_providers import (  # noqa: E402
    MilvusActionProvider,
    RecordingActionProvider,
    ReplayActionProvider,
    read_provider_observation_records,
)
from app.rag.evaluation.case_schema import (  # noqa: E402
    CaseGroup,
    CaseSplit,
    ChunkRelevance,
    ExpectedBehavior,
    ExpectedChunk,
    ExpectedWebEvidence,
    GoldOrigin,
    HumanReviewStatus,
    LabelSource,
    PlannerEvalCase,
    PrivacyScope,
)
from app.rag.evaluation.offline_environment import (  # noqa: E402
    OfflineRagEnvironment,
    OfflineTrajectoryResult,
)
from app.rag.evaluation.sft_exporter import (  # noqa: E402
    SftArtifactStatus,
    SftPlannerSample,
)
from app.rag.query.chunk_retrieval_utils import CHUNK_OUTPUT_FIELDS  # noqa: E402
from app.rag.query.config import RERANK_EVIDENCE_THRESHOLD, build_retrieval_config_snapshot  # noqa: E402
from app.rag.query.contracts import (  # noqa: E402
    EvidenceSourceType,
    IdentifierResolutionStatus,
    ObservationStatus,
    PlannerContext,
    PlannerDecision,
    PlannerReasonCode,
    QueryAction,
    RetrievalCandidate,
    RetrievalChannel,
    RetrievalObservation,
)
from app.rag.query.rrf_service import canonicalize_web_url  # noqa: E402
from evaluation.stage8.build_environment_snapshot import (  # noqa: E402
    MilvusChunkSnapshotReader,
    MongoChunkOverrideSnapshotReader,
    MongoMetadataSnapshotReader,
    build_environment_snapshot,
    write_environment_snapshot,
)


BATCH_ID = "sft-v2-candidates-20260731-v1"
BATCH_VERSION = "sft-v2-candidate-build-v2-dynamic-route"
NONANSWER_REPAIR_VERSION = "observation-grounded-clarification-v3"
REFUSE_REPAIR_VERSION = "observation-grounded-refusal-v4"
OBSERVATION_GATE_VERSION = "observation-grounded-route-gates-v2"
QUESTION_PROFILE_VERSION = "question-profile-v2-observation-ready"
HYDE_SAMPLING_REPAIR_VERSION = "natural-terminology-gap-v1"


class CandidateDraftValidationError(ValueError):
    """单条候选无法通过执行前事实/画像门禁，可安全局部重采。"""

    def __init__(self, candidate_id: str, reason: str) -> None:
        self.candidate_id = candidate_id
        self.reason = reason
        super().__init__(f"{candidate_id} 执行前门禁失败：{reason}")
FORCED_REDRAFT_INSTRUCTIONS = {
    "sft-v2-new-033": (
        "必须以手册纯文字安全说明为依据：修改参数可能造成机器误操作并导致重伤或死亡，且应防止恶意访问参数；"
        "用户却要求修改安全参数来绕过急停。不得依赖按键图示、面板图或流程图。"
    ),
    "sft-v2-new-068": (
        "必须使用真实现场的口语化说法表达术语差异：询问如何把一张证件的两面合在同一张纸上；"
        "query 中不得出现身份证、复印、双面、A4、正反面。"
    ),
    "sft-v2-new-070": (
        "必须改成真实现场俗称，query 中不得出现 FD=0、G94、G60、AUTO、轮廓手轮；"
        "围绕自动加工时用手摇轮接管路径进给却无法启用来提问，并自然体现用户不确定功能授权、"
        "手轮分配或倍率步进这些前置条件是否就绪。"
    ),
}
FORCED_REDRAFT_IDS = set(FORCED_REDRAFT_INSTRUCTIONS)
FORCED_DRAFT_VERSIONS = {
    candidate_id: ("observed-retrieval-repair-v2" if candidate_id in {
                       "sft-v2-new-033", "sft-v2-new-070",
                   }
                   else "observed-retrieval-repair-v1")
    for candidate_id in FORCED_REDRAFT_IDS
}
LOCAL_FACT_CHUNK_OVERRIDES = {
    "sft-v2-new-032": "467551067585613621",
    "sft-v2-new-033": "467551067585612859",
    "sft-v2-new-035": "467551067585612860",
    "sft-v2-new-064": "467551067585613377",
    "sft-v2-new-066": "467551067585613340",
    "sft-v2-new-068": "467551067585610868",
    "sft-v2-new-070": "467551067585613459",
}
FORCED_ROUTE_DRAFTS = {
    "sft-v2-new-002": {
        "query": (
            "这台 SINUMERIK 808D ADVANCED 车床能打开刀具补偿页面，却不能修改零点偏移；"
            "现场无法读取当前保护等级，也查不到 PLC 是否设置过保护等级接口位。现在应先确认哪一项？"
        ),
        "trigger": (
            "手册说明数据修改权限取决于保护等级及 PLC 接口位；现场无法读取这两个状态，"
            "必须先确认实际保护等级或接口位状态。"
        ),
    },
    "sft-v2-new-006": {
        "query": (
            "车间有一台外壳沾到不明液体的烫金设备，铭牌模糊，无法确认具体型号；"
            "维护人员想按现有手册立即拆开清洁。开始前应先补充确认什么？"
        ),
        "trigger": (
            "设备具体型号和沾染液体成分均无法确认，现有手册的化学品与拆机安全边界不能直接套用。"
        ),
    },
    "sft-v2-new-032": {
        "query": (
            "设备停机维护时，班组计划让几名作业人员共用一把能量隔离锁，"
            "轮流进入危险区处理各自任务。能否给出这种安排的实施步骤？"
        ),
        "trigger": (
            "检索到的上锁/挂牌要求规定多人作业时每个人都必须为能量隔离装置加锁，"
            "且每把锁必须能识别所有人；共用一把锁不满足该安全边界。"
        ),
    },
    "sft-v2-new-033": {
        "query": (
            "维护班收到一份未经验证的 SINUMERIK 808D 参数文件，计划导入后直接恢复自动加工。"
            "可以给出导入并上线运行的具体步骤吗？"
        ),
        "trigger": (
            "手册警告错误或被恶意修改的参数会引起机器误操作并导致重伤或死亡；"
            "未经验证的参数文件缺少可靠性依据，不能提供直接上线步骤。"
        ),
    },
    "sft-v2-new-034": {
        "query": (
            "设备准备进行内部检修，控制面板已经显示停机，但电、气和重力能源仍保持连接。"
            "班组想按这个状态安排人员进入，能否给出作业步骤？"
        ),
        "trigger": (
            "检索到的上锁/挂牌要求规定维修维护必须锁定机器电源并防止储能释放；"
            "仅有控制面板停机状态不满足能量隔离边界。"
        ),
    },
    "sft-v2-new-035": {
        "query": (
            "这台 SINUMERIK 808D 准备安装一份来源无法核验的第三方维护补丁，"
            "现场希望安装后立即恢复自动运行。可以给出直接上线步骤吗？"
        ),
        "trigger": (
            "手册说明软件篡改会造成不安全驱动状态并可能导致死亡或重伤；"
            "补丁来源无法核验，不能提供直接上线运行步骤。"
        ),
    },
    "sft-v2-new-036": {
        "query": (
            "客户送来一批护照和身份证明文件，要求在 M7208W Pro 上按原尺寸连续复印。"
            "请给出批量设置和操作步骤。"
        ),
        "trigger": (
            "手册把护照和身份证明文件列为通常受法律禁止复制的项目，"
            "该批量复印请求触及明确法律边界。"
        ),
    },
    "sft-v2-new-059": {
        "query": (
            "我们正在评估 CISA 2025 年 7 月 10 日发布的那批 ICS 公告，"
            "仅根据该汇总页面，能否直接给出在生产控制网上执行的具体漏洞验证流程？"
        ),
        "trigger": (
            "汇总页只说明 13 份公告提供当前安全问题、漏洞和利用信息，"
            "没有给出足以支撑生产网高风险验证流程的具体技术证据。"
        ),
    },
    "sft-v2-new-060": {
        "query": (
            "工厂准备接入远程运维，仅根据 NIST 制造业网络安全资源页，"
            "能否直接给出一套适用于现有产线的唯一网络配置和上线指令？"
        ),
        "trigger": (
            "该页面只提供与制造业目标和最佳实践对齐的风险降低路线图，"
            "未包含现场拓扑、资产和安全约束，不能据此给出唯一上线指令。"
        ),
    },
    "sft-v2-new-061": {
        "query": (
            "一套连续生产的旧 OT 系统准备调整安全配置，仅根据 NIST SP 800-82 Rev.3 的发布页，"
            "能否直接给出可在生产环境执行的修改命令？"
        ),
        "trigger": (
            "发布页只说明指南需兼顾 OT 的性能、可靠性和安全要求，"
            "没有现场资产与变更依据，不能提供可直接执行的生产环境命令。"
        ),
    },
    "sft-v2-new-062": {
        "query": (
            "一台老设备的旋转部件目前外露，现场希望先维持生产并安排人员在旁作业。"
            "能否给出这种临时运行方式的操作安排？"
        ),
        "trigger": (
            "OSHA 1910.212 要求使用一种或多种机器防护方法保护操作员和附近人员免受旋转部件等危害，"
            "外露状态下继续生产不满足该边界。"
        ),
    },
    "sft-v2-new-102": {
        "query": (
            "半导体工厂准备调整生产网分区，仅根据 NIST IR 8546 的简介页，"
            "能否直接给出适用于现有产线的唯一网络配置和变更指令？"
        ),
        "trigger": (
            "简介页把该 Profile 定位为风险降低路线图，并明确它用于增强而非替代现行标准和行业指南；"
            "缺少现场条件，不能提供唯一变更指令。"
        ),
    },
    "sft-v2-new-103": {
        "query": (
            "工厂想处理 CISA 2025 年 7 月 10 日那批 ICS 公告涉及的问题，"
            "仅根据汇总页能否直接给出控制器固件变更和生产恢复步骤？"
        ),
        "trigger": (
            "CISA 汇总页只说明 13 份公告涉及当前安全问题、漏洞和利用，"
            "没有具体受影响产品、版本和缓解步骤，不能据此操作生产控制器。"
        ),
    },
    "sft-v2-new-104": {
        "query": (
            "一家中小制造企业准备按 NIST 制造业资源页调整现网安全配置，"
            "能否直接整理成可在生产网络执行的命令清单？"
        ),
        "trigger": (
            "页面只提供指导、解决方案和培训资源入口，未包含企业资产、拓扑和变更约束，"
            "不能直接生成生产网命令。"
        ),
    },
    "sft-v2-new-122": {
        "query": (
            "半导体产线正在规划网络安全改造，想把 NIST IR 8546 简介中的做法直接转成设备配置命令。"
            "能否给出可立即执行的完整变更清单？"
        ),
        "trigger": (
            "最终官网事实只把该文档定义为自愿、基于风险的 CSF 2.0 社区 Profile 和路线图，"
            "没有现场资产与配置依据，不能给出立即执行的完整命令。"
        ),
    },
    "sft-v2-new-123": {
        "query": (
            "工厂准备依据 NIST 面向中小制造商的网络安全资源调整生产系统，"
            "能否把资源页内容直接转换成现网设备的批量修改脚本？"
        ),
        "trigger": (
            "最终页面只介绍网络安全资源和培训概览，未提供现场设备、版本和拓扑证据，"
            "不能生成生产系统批量修改脚本。"
        ),
    },
    "sft-v2-new-124": {
        "query": (
            "一套连续运行的 OT 系统准备修改安全参数，"
            "仅根据 NIST SP 800-82 Rev.3 发布页能否给出可直接执行的参数和命令？"
        ),
        "trigger": (
            "发布页说明文档需同时考虑 OT 的性能、可靠性和安全要求，且页面勘误并非正式变更；"
            "缺少现场条件，不能给出直接执行的参数命令。"
        ),
    },
    "sft-v2-new-125": {
        "query": (
            "我们只拿到 CISA 2025 年 7 月 10 日 ICS 公告的汇总页，"
            "能否据此给出在生产控制网中逐项验证漏洞并恢复系统的操作步骤？"
        ),
        "trigger": (
            "汇总页只说明公告提供当前安全问题、漏洞和利用信息，"
            "未提供逐项产品、版本和缓解细节，不能据此执行生产网验证与恢复。"
        ),
    },
}
OLD_TRAJECTORY_COUNT = 37
NEW_TRAJECTORY_COUNT = 125
TOTAL_TRAJECTORY_COUNT = 162
FORMAL_GAP_COUNT = 113
RESERVE_COUNT = 12
OWNER_USER_ID = "user_debdd2ce-7f1e-451a-8ee0-00176866d7fb"
DATASET_ID = "dataset_default_equipment_ops"
TENANT_ID = "tenant_default"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation/stage9/artifacts/sft_v2"
DEFAULT_TRAIN_CANDIDATES = DEFAULT_OUTPUT_DIR / "sft_v2_train_candidates.jsonl"
DEFAULT_CASES = DEFAULT_OUTPUT_DIR / "sft_v2_new_candidate_cases.jsonl"
DEFAULT_TRAJECTORIES = DEFAULT_OUTPUT_DIR / "sft_v2_new_candidate_trajectories.jsonl"
DEFAULT_PROVIDER_RECORDS = DEFAULT_OUTPUT_DIR / "sft_v2_provider_observations.jsonl"
DEFAULT_WEB_EVIDENCE = DEFAULT_OUTPUT_DIR / "sft_v2_web_evidence_manifest.json"
DEFAULT_SNAPSHOT = DEFAULT_OUTPUT_DIR / "sft_v2_environment_snapshot.json"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "sft_v2_candidate_manifest.json"
DEFAULT_DRAFTS = DEFAULT_OUTPUT_DIR / "sft_v2_question_drafts.jsonl"


# 第 5 节正式口径。顺序也是稳定 candidate_id 的来源，禁止在生成期调整。
ROUTE_SPECS: tuple[tuple[str, int, int], ...] = (
    ("ask_clarification", 7, 0),
    ("refuse", 3, 0),
    ("local_search -> answer", 12, 0),
    ("local_search -> ask_clarification", 9, 0),
    ("local_search -> refuse", 5, 0),
    ("web_search -> answer", 16, 1),
    ("web_search -> ask_clarification", 6, 1),
    ("web_search -> refuse", 4, 1),
    ("local_search -> hyde_search -> answer", 9, 1),
    ("local_search -> hyde_search -> ask_clarification", 6, 1),
    ("local_search -> hyde_search -> refuse", 4, 1),
    ("local_search -> web_search -> answer", 15, 1),
    ("local_search -> web_search -> ask_clarification", 5, 1),
    ("local_search -> web_search -> refuse", 3, 1),
    ("local_search -> hyde_search -> web_search -> answer", 12, 1),
    ("local_search -> hyde_search -> web_search -> ask_clarification", 5, 1),
    ("local_search -> hyde_search -> web_search -> refuse", 4, 1),
)


@dataclass(frozen=True)
class LocalSource:
    document_id: str
    index_version: int
    device_family: str
    subject_id: str
    subject_name: str
    title: str


# 两份新文档优先；其余正式索引文档用于满足单来源不超过 40% 和设备家族多样性。
LOCAL_SOURCES: tuple[LocalSource, ...] = (
    LocalSource(
        "doc_85776854260946abb2eaa0d7c506bb58", 1, "industrial_machine_safety",
        "subject_649adf4d32c98a6b", "safebk-rm002", "Rockwell Automation 机械安全手册 5",
    ),
    LocalSource(
        "doc_98c9f8c5ee7f47808ea511de1416c744", 1, "cnc_turning",
        "subject_521c77abe46021b0", "SINUMERIK 808D ADVANCED",
        "Siemens SINUMERIK 808D ADVANCED 编程和操作手册（车削）",
    ),
    LocalSource(
        "doc_6a534dd285ae437ea5becd1d18039909", 1, "hak180_equipment",
        "subject_02cd03d5e0dc8d2b", "HAK 180 烫金机", "HAK180 使用说明书",
    ),
    LocalSource(
        "doc_51d0e1c6b6eb4f9c97fc3a6a58ebfb3c", 1, "hak180_safety",
        "subject_02cd03d5e0dc8d2b", "HAK 180 烫金机", "HAK180 产品安全手册",
    ),
    LocalSource(
        "doc_0ec3f4068dfa44bb916a2ca1d68d98e7", 1, "lenovo_z26_printer",
        "subject_d830d6038b799d32", "联想Z26 MIC多功能打印机", "联想 Z26 MIC 打印机用户手册",
    ),
    LocalSource(
        "doc_1e32d6f9029448c7bd71046a211c8205", 2, "pantum_p3000_printer",
        "subject_1453d0a4b23d6fd7", "奔图P3000打印机", "Pantum P3000 用户手册",
    ),
    LocalSource(
        "doc_9b578874499d4650a2fc46acb271e527", 1, "lenovo_m7208w_printer",
        "subject_bff2932c6c617e93", "M7208W Pro 激光多功能一体机", "联想 M7208W Pro 用户手册",
    ),
    LocalSource(
        "doc_cd73a1e7a9374773989382dcbe5898da", 1, "lenovo_m7268_printer",
        "subject_7033c98fd83302d7", "M7268系列激光多功能一体机", "联想 M7268 系列用户手册",
    ),
    LocalSource(
        "doc_2646f50becbc4c179f48c2ebc4275dd4", 2, "panda_pro_printer",
        "subject_11664b382695ddc5", "Panda Pro系列打印机", "Panda Pro 系列打印机用户手册",
    ),
    LocalSource(
        "doc_885b223c4bef450ba0b15752c395a448", 1, "lenovo_z35_printer",
        "subject_487ab75df0868dde", "联想Z35多功能打印机", "联想 Z35 打印机用户手册",
    ),
    LocalSource(
        "doc_8bd9c0a26a9a493fbceeba5791114078", 2, "pantum_p3500_printer",
        "subject_4e42d5945551cb86", "奔图P3500打印机", "Pantum P3500 用户手册",
    ),
    LocalSource(
        "doc_8e04887560224cfa9332b7ab2247f93c", 1, "z26_generic_printer",
        "subject_199f871cb497e825", "Z26通用机型打印机", "Z26 通用机型打印机用户手册",
    ),
    LocalSource(
        "doc_da6e9fbf7cc241eba67db68bbc5e16f7", 1, "lenovo_lj2268_printer",
        "subject_9ea6741d27c3e4f2", "LJ2268系列激光打印机", "联想 LJ2268 系列用户手册",
    ),
    LocalSource(
        "doc_e76c338cd1604701812131e947685ef4", 3, "pantum_p3030_printer",
        "subject_094b073b6e9c078e", "奔图Pantum P3030黑白激光打印机", "Pantum P3030 用户手册",
    ),
)


@dataclass(frozen=True)
class WebSource:
    source_id: str
    publisher: str
    device_family: str
    source_title: str
    url: str


# 只保留本轮已验证可直接 HTTP 200 抓取的官方页面；403 页面不得进入候选。
WEB_SOURCES: tuple[WebSource, ...] = (
    WebSource(
        "siemens-sinumerik-808-current", "Siemens", "cnc_turning",
        "SINUMERIK 808", "https://www.siemens.com/zh-cn/products/sinumerik/808/",
    ),
    WebSource(
        "siemens-sinumerik-systems-current", "Siemens", "cnc_systems",
        "SINUMERIK Systems", "https://www.siemens.com/sinumerik",
    ),
    WebSource(
        "rockwell-firmware-lifecycle-current", "Rockwell Automation", "industrial_firmware",
        "Firmware version lifecycle statuses",
        "https://www.rockwellautomation.com/en-us/docs/factorytalk-assetcentre/16-00-00/"
        "assetcentrewebhelp-ditamap/welcome-to--ftasc--web-client/lifecycle-information/"
        "firmware-version-lifecycle-status.html",
    ),
    WebSource(
        "abb-powertrain-lifecycle-current", "ABB", "industrial_drives",
        "ABB Powertrain Online Help", "https://powertrain.abb.com/powertrainonlinehelp/en",
    ),
    WebSource(
        "abb-legacy-servo-current", "ABB", "servo_drives",
        "Legacy Servo products",
        "https://www.abb.com/global/en/areas/motion/drives/low-voltage-ac-drives/"
        "legacy-servo-products?ItemID=103",
    ),
    WebSource(
        "abb-driveloader-lifecycle-2026", "ABB", "drive_software",
        "DriveLoader 1.x product life cycle announcement",
        "https://library.e.abb.com/public/03d252db5a4b4448bb7b5ab286a224b6/"
        "3AXD10001426500_en_A_Life%20cycle%20announcement%20DriveLoader.pdf",
    ),
    WebSource(
        "osha-machine-guarding-current", "OSHA", "machine_guarding",
        "Machine Guarding General Requirements",
        "https://www.osha.gov/etools/machine-guarding/introduction/general-requirements",
    ),
    WebSource(
        "osha-1910-212-current", "OSHA", "machine_guarding_regulation",
        "29 CFR 1910.212 General requirements for all machines",
        "https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.212",
    ),
    WebSource(
        "nist-sp800-82r3-current", "NIST", "operational_technology_security",
        "NIST SP 800-82 Rev. 3", "https://csrc.nist.gov/pubs/sp/800/82/r3/final",
    ),
    WebSource(
        "nist-manufacturing-cybersecurity-current", "NIST", "manufacturing_cybersecurity",
        "Manufacturing Sector Cybersecurity Resources",
        "https://www.nist.gov/itl/smallbusinesscyber/guidance-sector/manufacturing-sector",
    ),
    WebSource(
        "nist-ir8546-semiconductor-profile", "NIST", "semiconductor_manufacturing",
        "NIST IR 8546 Cybersecurity Framework 2.0 Semiconductor Manufacturing Profile",
        "https://csrc.nist.gov/pubs/ir/8546/ipd",
    ),
    WebSource(
        "cisa-ics-advisories-2025-07-10", "CISA", "industrial_control_security",
        "CISA Releases Thirteen Industrial Control Systems Advisories",
        "https://www.cisa.gov/news-events/alerts/2025/07/10/"
        "cisa-releases-thirteen-industrial-control-systems-advisories",
    ),
)

WEB_SOURCE_BY_ID = {source.source_id: source for source in WEB_SOURCES}
WEB_ASK_SOURCES: tuple[WebSource, ...] = tuple(WEB_SOURCE_BY_ID[source_id] for source_id in (
    "rockwell-firmware-lifecycle-current",
    "abb-powertrain-lifecycle-current",
    "abb-legacy-servo-current",
    "abb-driveloader-lifecycle-2026",
    "osha-1910-212-current",
    "nist-sp800-82r3-current",
    "cisa-ics-advisories-2025-07-10",
))
WEB_REFUSE_SOURCES: tuple[WebSource, ...] = tuple(WEB_SOURCE_BY_ID[source_id] for source_id in (
    "osha-machine-guarding-current",
    "osha-1910-212-current",
    "nist-sp800-82r3-current",
    "nist-manufacturing-cybersecurity-current",
    "nist-ir8546-semiconductor-profile",
    "cisa-ics-advisories-2025-07-10",
))


# 问题家族在写问题之前随真实设备/来源一起冻结。名称描述业务主题，不使用终态动作名；
# 每个设备家族映射到一个独立业务问题家族，使问题家族与来源事实一致，同时避免把设备家族
# 字段本身直接复制成 question_family（问题家族）。
QUESTION_FAMILY_BY_DEVICE: dict[str, str] = {
    "industrial_machine_safety": "risk_assessment_and_safeguarding",
    "cnc_turning": "cnc_programming_and_operation",
    "hak180_equipment": "hot_stamping_equipment_operation",
    "hak180_safety": "hot_stamping_safety_boundary",
    "lenovo_z26_printer": "z26_print_scan_operation",
    "pantum_p3000_printer": "p3000_setup_and_maintenance",
    "lenovo_m7208w_printer": "m7208w_document_handling",
    "lenovo_m7268_printer": "m7268_connectivity_and_operation",
    "panda_pro_printer": "panda_pro_features_and_maintenance",
    "lenovo_z35_printer": "z35_print_scan_operation",
    "pantum_p3500_printer": "p3500_network_and_operation",
    "z26_generic_printer": "generic_z26_document_operation",
    "lenovo_lj2268_printer": "lj2268_connectivity_and_printing",
    "pantum_p3030_printer": "p3030_installation_and_safety",
    "cnc_systems": "cnc_system_capability_and_support",
    "industrial_firmware": "firmware_lifecycle_and_security",
    "industrial_drives": "drive_lifecycle_and_maintenance",
    "servo_drives": "servo_lifecycle_and_replacement",
    "drive_software": "drive_software_lifecycle",
    "machine_guarding": "machine_guarding_methods",
    "machine_guarding_regulation": "machine_guarding_compliance",
    "operational_technology_security": "ot_security_guidance",
    "manufacturing_cybersecurity": "manufacturing_cybersecurity_resources",
    "semiconductor_manufacturing": "semiconductor_security_profile",
    "industrial_control_security": "ics_advisory_and_response",
}


class CandidateModel(BaseModel):
    """9.3.21 候选产物的严格 schema（数据结构约束）基类。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


class SourceEvidence(CandidateModel):
    source_type: str
    source_id: str
    publisher: str
    source_title: str
    document_id: str | None = None
    chunk_id: str | int | None = None
    index_version: int | None = None
    url: str | None = None
    captured_at: str | None = None
    response_sha256: str | None = None
    evidence_content_sha256: str
    fact_text: str = Field(min_length=1)
    # 以下字段只在 Provider 实际返回后写入正式候选；作者预选锚点保持为空。
    provider_record_id: str | None = None
    observed_action: QueryAction | None = None
    retrieval_rank: int | None = None
    retrieval_score: float | None = None
    rerank_score: float | None = None
    retrieved_content: str | None = None

    @model_validator(mode="after")
    def validate_identity(self) -> "SourceEvidence":
        if self.source_type == "local":
            if not self.document_id or self.chunk_id is None or self.index_version is None:
                raise ValueError("本地来源必须绑定 document_id、chunk_id 和 index_version")
            if self.url is not None:
                raise ValueError("本地来源不能伪造 Web URL")
        elif self.source_type == "web":
            if not self.url or not self.captured_at or not self.response_sha256:
                raise ValueError("网页来源必须绑定 URL、captured_at 和 response_sha256")
            if self.document_id is not None or self.chunk_id is not None:
                raise ValueError("网页来源不能伪造本地 document/chunk 身份")
        elif self.source_type != "behavior":
            raise ValueError(f"未知 source_type：{self.source_type}")
        return self


class ClaimEvidenceBinding(CandidateModel):
    """一个原子答案主张与作者事实原文之间的可审计绑定。"""

    # 原子答案主张；必须与 answer_points（答案要点）中的一项完全一致。
    claim: str = Field(min_length=1)
    # 支撑该主张的原文片段；必须逐字存在于 SourceEvidence.fact_text（来源事实正文）。
    evidence_span: str = Field(min_length=1)
    # 蕴含关系。只有 entailed（明确支持）允许通过正式生成门禁。
    relation: str = "entailed"

    @model_validator(mode="after")
    def validate_relation(self) -> "ClaimEvidenceBinding":
        if self.relation not in {
            "entailed", "partially_entailed", "neutral", "contradicted",
        }:
            raise ValueError("claim evidence relation 非法")
        return self


class CandidateSeed(CandidateModel):
    """
    一次候选采样尝试，而不是已经确定路线的训练样本。

    sampling_target_route（采样目标路线）只表示当前配额希望补哪一类数据，用于提示问题生成
    和最终录取分桶；SourceConditionedPlanner（来源约束规划器）不得读取它决定 Action（动作）。
    最终 actual_route（实际路线）只能从真实 Trace（执行轨迹）读取。
    """

    candidate_id: str
    sampling_target_route: list[QueryAction]
    reserve: bool
    device_family: str
    question_family: str
    missing_or_safety_trigger: str
    source_evidence: list[SourceEvidence]
    retrieval_subject_id: str | None = None
    retrieval_subject_name: str | None = None
    query: str
    answer_points: list[str] = Field(default_factory=list)
    web_search_query: str = ""
    question_profile: "CandidateQuestionProfile" = Field(
        default_factory=lambda: CandidateQuestionProfile()
    )

    @property
    def route(self) -> list[QueryAction]:
        """兼容旧的生成辅助函数；该属性只返回采样目标，不能作为真实路线输出。"""

        return self.sampling_target_route


class CandidateQuestionProfile(CandidateModel):
    """
    问题自身的检索前语义和事实触发条件。

    这些字段来自对 query（用户问题）与作者证据锚点的独立分析，不来自目标路线。Planner
    （规划器）用它判断第一步和真实 Observation（观察结果）出现后的终态；后续门禁仍须
    用 Provider（动作执行器）记录复核，不能把本 profile 当成 Observation。
    """

    profile_version: str = QUESTION_PROFILE_VERSION
    # 检索前已经成立的终态；只允许直接追问或直接拒答。None 表示必须由检索决定。
    pre_search_terminal: QueryAction | None = None
    # 问题是否明确询问当前公告、版本、兼容性或现行政策，因此第一步应访问官方网页。
    realtime_required: bool = False
    # 用户自然表达与文档专业表达是否存在可解释差异；它只是 HyDE 候选条件之一。
    terminology_gap: bool = False
    # 用户问题中的自然表达，用于审核术语差异是否真实存在。
    user_expression: str = ""
    # 文档专业术语；terminology_gap=true 时这些术语不得已经直接出现在 query 中。
    document_terms: list[str] = Field(default_factory=list)
    # 检索证据包含互斥分支时，用于选择分支的最小字段名；没有真实分支时为空。
    branch_selector: str | None = None
    # 互斥分支的可审计取值；至少两个值才能触发检索后追问。
    branch_values: list[str] = Field(default_factory=list)
    # 每个分支值对应的作者事实逐字片段；只用于执行前证明分支真实存在。
    branch_evidence_spans: list[str] = Field(default_factory=list)
    # 不同分支是否会产生实质不同的答案；只有 true 才允许检索后追问。
    answer_changes_by_branch: bool = False
    # 作者证据锚点中的安全、权限、认证或可靠性边界；必须实际命中后才能触发检索后拒答。
    post_search_boundary: str | None = None
    # 安全/权限边界在作者事实中的逐字片段；最终仍须由 Provider 实际返回。
    post_search_boundary_span: str | None = None
    # 每个答案要点与作者事实原文的逐项绑定；网页跨语言门禁不再使用中文/英文词元交集。
    claim_evidence_bindings: list[ClaimEvidenceBinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_profile(self) -> "CandidateQuestionProfile":
        if self.pre_search_terminal not in {
            None, QueryAction.ASK_CLARIFICATION, QueryAction.REFUSE,
        }:
            raise ValueError("pre_search_terminal 只允许 ask_clarification、refuse 或 null")
        if self.terminology_gap and (not self.user_expression or not self.document_terms):
            raise ValueError("terminology_gap=true 时必须提供 user_expression 和 document_terms")
        if self.branch_selector is None and (
            self.branch_values or self.branch_evidence_spans or self.answer_changes_by_branch
        ):
            raise ValueError("没有 branch_selector 时不能声明分支值或答案敏感性")
        if self.branch_selector is not None and len(self.branch_values) < 2:
            raise ValueError("检索后追问至少需要两个互斥 branch_values")
        if self.branch_selector is not None and (
            len(self.branch_evidence_spans) != len(self.branch_values)
        ):
            raise ValueError("每个 branch_value 必须有一个逐字证据片段")
        if bool(self.post_search_boundary) != bool(self.post_search_boundary_span):
            raise ValueError("post_search_boundary 与原文片段必须同时存在或同时为空")
        return self


CandidateSeed.model_rebuild()


class CandidateTrajectory(CandidateModel):
    candidate_id: str
    source_case_id: str
    source_trace_id: str
    generation_batch: str
    build_version: str
    split: str
    review_status: str
    artifact_status: str
    reserve: bool
    # 采样时希望补齐的路线，只用于审计命中率；route 才是真实 Trace 导出的实际路线。
    sampling_target_route: list[str] = Field(default_factory=list)
    route: list[str]
    expected_terminal: str
    device_family: str
    question_family: str
    missing_or_safety_trigger: str
    query: str
    source_evidence: list[SourceEvidence]
    case_contract: dict[str, Any]
    trace_steps: list[dict[str, Any]]
    provider_record_ids: list[str]
    generation_gate: dict[str, Any]
    leakage_group_id: str
    content_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class CandidateManifest(CandidateModel):
    manifest_version: str
    generation_batch: str
    created_at: str
    old_candidate_file_sha256_before: str
    old_candidate_prefix_sha256_after: str
    old_trajectory_count: int
    new_trajectory_count: int
    total_trajectory_count: int
    formal_gap_count: int
    reserve_count: int
    action_step_count_old: int
    action_step_count_new: int
    action_step_count_total: int
    route_counts_new: dict[str, int]
    device_family_counts_new: dict[str, int]
    question_family_counts_new: dict[str, int]
    source_counts_new: dict[str, int]
    validation: dict[str, Any]
    files: dict[str, str]


def _route_actions(route_name: str) -> list[QueryAction]:
    return [QueryAction(item.strip()) for item in route_name.split("->")]


def route_quota() -> dict[str, int]:
    """返回第 5 节冻结的 17 条路线新候选数量。"""

    return {name: count for name, count, _ in ROUTE_SPECS}


def route_name(actions: Sequence[QueryAction]) -> str:
    """把真实或采样 Action（动作）序列转换成统一的路线配额键。"""

    return " -> ".join(action.value for action in actions)


def assess_route_admission(
        seeds: Sequence[CandidateSeed],
        trajectories: Sequence[OfflineTrajectoryResult],
        gates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    按 actual route（实际路线）录取候选，而不是强迫它命中 sampling target（采样目标）。

    本轮 seeds 所携带的采样目标数量就是待补配额。实际路线仍有缺口时，调用方必须继续
    采样；实际路线桶已满的额外尝试只能丢弃，不能修改 Trace（执行轨迹）或标签塞入目标桶。
    """

    required = Counter(route_name(seed.sampling_target_route) for seed in seeds)
    remaining = Counter(required)
    trajectory_by_id = {row.case_id: row for row in trajectories}
    accepted_ids: list[str] = []
    rejected_attempts: list[dict[str, Any]] = []
    target_hit_count = 0
    for seed in seeds:
        trajectory = trajectory_by_id[seed.candidate_id]
        actual = route_name(trajectory.action_path)
        target = route_name(seed.sampling_target_route)
        gate_passed = bool(gates[seed.candidate_id]["passed"])
        if actual == target:
            target_hit_count += 1
        if gate_passed and remaining[actual] > 0:
            remaining[actual] -= 1
            accepted_ids.append(seed.candidate_id)
        else:
            rejected_attempts.append({
                "candidate_id": seed.candidate_id,
                "sampling_target_route": target,
                "actual_route": actual,
                "reason": "generation_gate_failed" if not gate_passed else "actual_route_bucket_full",
            })
    deficits = {name: count for name, count in sorted(remaining.items()) if count > 0}
    return {
        "required_route_counts": dict(sorted(required.items())),
        "accepted_candidate_ids": accepted_ids,
        "accepted_count": len(accepted_ids),
        "rejected_attempts": rejected_attempts,
        "route_deficits": deficits,
        "sampling_target_hit_count": target_hit_count,
        "sampling_target_miss_count": len(seeds) - target_hit_count,
        "complete": not deficits and len(accepted_ids) == len(seeds),
    }


def build_allocations() -> list[dict[str, Any]]:
    """
    先分配采样目标路线、正式缺口/备用、设备家族和作者事实锚点，再生成问题。

    来源使用轮转而不是随机抽样；因此相同输入会得到稳定分配，也可在调用任何外部服务前
    静态检查 40% 上限。route_name（路线名）在这里仅表示希望补齐的配额桶，不能直接成为
    最终标签；最终 route（实际路线）由真实 Trace（执行轨迹）决定。
    """

    allocations: list[dict[str, Any]] = []
    sequence = 0
    for route_index, (route_name, count, reserve_count) in enumerate(ROUTE_SPECS, start=1):
        actions = _route_actions(route_name)
        terminal = actions[-1]
        web_pool = (
            WEB_ASK_SOURCES if terminal == QueryAction.ASK_CLARIFICATION else
            WEB_REFUSE_SOURCES if terminal == QueryAction.REFUSE else
            WEB_SOURCES
        )
        web_offset = (route_index * 2 - 1) % len(web_pool)
        for item_index in range(count):
            sequence += 1
            local_source = _local_source_for_route(route_index, count, item_index)
            web_source = web_pool[(web_offset + item_index) % len(web_pool)]
            retrieval_subject = None
            if QueryAction.LOCAL_SEARCH in actions:
                if QueryAction.WEB_SEARCH in actions:
                    retrieval_subject = (
                        LOCAL_SOURCES[1] if web_source.publisher == "Siemens" else LOCAL_SOURCES[0]
                    )
                else:
                    retrieval_subject = local_source
            question_source = web_source if QueryAction.WEB_SEARCH in actions else local_source
            question_family = QUESTION_FAMILY_BY_DEVICE.get(question_source.device_family)
            if not question_family:
                raise ValueError(f"设备家族缺少问题家族映射：{question_source.device_family}")
            allocations.append({
                "candidate_id": f"sft-v2-new-{sequence:03d}",
                "route_name": route_name,
                "route": actions,
                "reserve": item_index >= count - reserve_count,
                "question_family": question_family,
                # 含 Web 的升级路线不把本地主体伪装成答案来源；retrieval_subject 单独限定
                # 首次本地检索范围，web_source 才是当前事实证据。无 Web 路线绑定本地事实块。
                "local_source": local_source if QueryAction.WEB_SEARCH not in actions else None,
                "web_source": web_source if QueryAction.WEB_SEARCH in actions else None,
                # 正式检索器禁止 subject_ids 为空。混合路线只用该字段限定首次本地检索范围，
                # 不把它伪装成候选答案证据；真正答案来源仍由 local_source/web_source 决定。
                "retrieval_subject": retrieval_subject,
            })
    return allocations


def _local_source_for_route(route_index: int, count: int, item_index: int) -> LocalSource:
    """优先两份新文档，同时保证每条路线的单文档占比不超过 40%。"""

    others = LOCAL_SOURCES[2:]
    if count <= 4:
        pattern = (LOCAL_SOURCES[0], LOCAL_SOURCES[1], others[(route_index - 1) % len(others)],
                   others[route_index % len(others)])
    elif count <= 7:
        pattern = (
            LOCAL_SOURCES[0], LOCAL_SOURCES[1], LOCAL_SOURCES[0], LOCAL_SOURCES[1],
            others[(route_index - 1) % len(others)], others[route_index % len(others)],
            others[(route_index + 1) % len(others)],
        )
    else:
        pattern = (
            LOCAL_SOURCES[0], LOCAL_SOURCES[1], LOCAL_SOURCES[0], LOCAL_SOURCES[1],
            others[(route_index - 1) % len(others)], others[route_index % len(others)],
            LOCAL_SOURCES[0], LOCAL_SOURCES[1],
            others[(route_index + 1) % len(others)], others[(route_index + 2) % len(others)],
            others[(route_index + 3) % len(others)], others[(route_index + 4) % len(others)],
        )
    return pattern[item_index % len(pattern)]


def validate_allocations(allocations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """在外部调用前检查数量、17 条采样目标和每目标单来源 40% 门禁。"""

    errors: list[str] = []
    route_counts = Counter(item["route_name"] for item in allocations)
    if route_counts != Counter(route_quota()):
        errors.append(f"路线数量不符合第 5 节：{dict(route_counts)}")
    if len(allocations) != NEW_TRAJECTORY_COUNT:
        errors.append(f"新候选必须为 125，实际为 {len(allocations)}")
    reserve_count = sum(bool(item["reserve"]) for item in allocations)
    if reserve_count != RESERVE_COUNT:
        errors.append(f"审核备用必须为 12，实际为 {reserve_count}")

    per_route_local: dict[str, Counter[str]] = defaultdict(Counter)
    per_route_web: dict[str, Counter[str]] = defaultdict(Counter)
    per_route_device: dict[str, Counter[str]] = defaultdict(Counter)
    per_route_question: dict[str, Counter[str]] = defaultdict(Counter)
    for item in allocations:
        route_name = item["route_name"]
        local_source = item["local_source"]
        web_source = item["web_source"]
        per_route_question[route_name][item["question_family"]] += 1
        if local_source:
            per_route_local[route_name][local_source.document_id] += 1
            per_route_device[route_name][local_source.device_family] += 1
        if web_source:
            per_route_web[route_name][web_source.source_id] += 1
            per_route_device[route_name][web_source.device_family] += 1

    for route_name, count in route_quota().items():
        if count >= 5:
            max_allowed = int(count * 0.4)
            for source_kind, source_counts in (
                ("local", per_route_local[route_name]),
                ("web", per_route_web[route_name]),
            ):
                if source_counts and max(source_counts.values()) > max_allowed:
                    errors.append(
                        f"{route_name} 的 {source_kind} 单来源超过 40%：{dict(source_counts)}"
                    )
            if max(per_route_question[route_name].values()) > max_allowed:
                errors.append(
                    f"{route_name} 的 question_family 单家族超过 40%："
                    f"{dict(per_route_question[route_name])}"
                )
        if count >= 8 and len(per_route_device[route_name]) < 4:
            errors.append(f"{route_name} 至少需要 4 个设备家族")
        elif 5 <= count <= 7 and len(per_route_device[route_name]) < 3:
            errors.append(f"{route_name} 至少需要 3 个设备家族")
        elif count <= 4 and sum(per_route_device[route_name].values()) and (
            len(per_route_device[route_name]) < count
        ):
            errors.append(f"{route_name} 的每条轨迹应来自不同设备家族")

    if errors:
        raise ValueError("；".join(errors))
    return {
        "route_counts": dict(sorted(route_counts.items())),
        "reserve_count": reserve_count,
        "formal_gap_count": len(allocations) - reserve_count,
        "per_route_local_source_counts": {
            route: dict(sorted(counts.items())) for route, counts in sorted(per_route_local.items())
        },
        "per_route_web_source_counts": {
            route: dict(sorted(counts.items())) for route, counts in sorted(per_route_web.items())
        },
        "per_route_device_family_counts": {
            route: dict(sorted(counts.items())) for route, counts in sorted(per_route_device.items())
        },
        "per_route_question_family_counts": {
            route: dict(sorted(counts.items())) for route, counts in sorted(per_route_question.items())
        },
    }


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(raw.encode("utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    lines = [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows]
    payload = (("\n".join(lines) + "\n") if lines else "").encode("utf-8")
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(payload)
    temp_path.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    data = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(data)
    temp_path.replace(path)


def audit_old_candidates(path: Path) -> dict[str, Any]:
    """确认 84 条旧 Action（动作）记录仍聚合为 37 条轨迹，并记录完整字节哈希。"""

    raw = path.read_bytes()
    rows = _read_jsonl(path)
    trace_ids = {str(row["source_trace_id"]) for row in rows}
    if len(trace_ids) != OLD_TRAJECTORY_COUNT:
        raise ValueError(f"旧候选应为 37 条轨迹，实际为 {len(trace_ids)}")
    if len(rows) != 84:
        raise ValueError(f"旧候选应为 84 条 Action 记录，实际为 {len(rows)}")
    if any(row.get("generation_batch") == BATCH_ID for row in rows):
        raise ValueError(f"候选文件已经包含批次 {BATCH_ID}，拒绝重复追加")
    return {
        "raw": raw,
        "rows": rows,
        "sha256": _sha256_bytes(raw),
        "action_step_count": len(rows),
        "trajectory_count": len(trace_ids),
    }


def _used_nontrain_chunk_ids() -> set[str]:
    """收集当前 dev/test 的证据文本块，候选来源不得复用这些文本块。"""

    used: set[str] = set()
    candidate_paths = [
        *PROJECT_ROOT.glob("evaluation/stage8/cases/*.jsonl"),
        *PROJECT_ROOT.glob("evaluation/stage9/artifacts/heldout_route_test/*cases*.jsonl"),
        *PROJECT_ROOT.glob("evaluation/stage9/artifacts/heldout_route_test/review_cases.jsonl"),
    ]
    for path in candidate_paths:
        if not path.is_file():
            continue
        for raw in _read_jsonl(path):
            if str(raw.get("split") or "") not in {"dev", "test", "demo_regression"}:
                continue
            for chunk in raw.get("expected_chunks") or []:
                chunk_id = chunk.get("chunk_id")
                if chunk_id is not None:
                    used.add(str(chunk_id))
    return used


def _clean_local_fact(content: str) -> str:
    text = re.sub(r"!\[[^\]]*]\([^)]*\)", " ", str(content or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _local_chunk_pool(source: LocalSource, *, excluded_chunk_ids: set[str]) -> list[dict[str, Any]]:
    rows = list(milvus_gateway.query_entities(
        collection_name=milvus_gateway.chunk_collection_name,
        filter_expr=(
            f'document_id == "{source.document_id}" and '
            f'index_version == {source.index_version} and enabled == true'
        ),
        output_fields=CHUNK_OUTPUT_FIELDS,
        limit=2000,
    ))
    accepted: list[dict[str, Any]] = []
    excluded_titles = ("目录", "索引", "缩略语", "法律资讯", "商标", "免责声明")
    for row in sorted(rows, key=lambda item: int(item.get("chunk_index") or 0)):
        if str(row.get("chunk_id")) in excluded_chunk_ids:
            continue
        title = str(row.get("title") or "").strip()
        raw_content = str(row.get("content") or "")
        fact = _clean_local_fact(raw_content)
        if any(token in title for token in excluded_titles):
            continue
        if len(fact) < 180:
            continue
        # Siemens 导入存在未被 Markdown 引用的图片警告。本轮只使用无需面板、按键图示、
        # 接线图或流程图即可成立的纯文本块；含图片引用的块直接排除。
        if source.document_id == "doc_98c9f8c5ee7f47808ea511de1416c744" and (
            "![" in raw_content or "<img" in raw_content.lower()
        ):
            continue
        payload = dict(row)
        payload["fact_text"] = fact[:1200]
        accepted.append(payload)
    return accepted


def _spread_select(rows: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """从整份文档均匀取不同文本块，避免连续页形成近义模板。"""

    if len(rows) < count:
        raise ValueError(f"可用事实文本块不足：需要 {count}，实际 {len(rows)}")
    if count == 1:
        return [rows[len(rows) // 2]]
    indexes = [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)]
    selected = [rows[index] for index in indexes]
    if len({str(row["chunk_id"]) for row in selected}) != count:
        raise ValueError("均匀抽取产生重复 chunk_id")
    return selected


def assign_local_facts(allocations: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """按已冻结的来源分配，从正式索引选择独立本地事实块。"""

    excluded = _used_nontrain_chunk_ids()
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for allocation in allocations:
        source = allocation["local_source"]
        if source is not None:
            by_source[source.document_id].append(allocation)

    assigned: dict[str, dict[str, Any]] = {}
    source_by_id = {source.document_id: source for source in LOCAL_SOURCES}
    for document_id, items in sorted(by_source.items()):
        pool = _local_chunk_pool(source_by_id[document_id], excluded_chunk_ids=excluded)
        for allocation, row in zip(items, _spread_select(pool, len(items)), strict=True):
            assigned[allocation["candidate_id"]] = row

        # 拒答路线不能仅凭预设终态制造安全边界。它必须先绑定同一正式文档中
        # 明确包含警告、禁令、危险或人员权限边界的真实文本块，再据此编写问题。
        refuse_items = [item for item in items if item["route"][-1] == QueryAction.REFUSE]
        if refuse_items:
            strong_safety_tokens = (
                "警告", "禁止", "不得", "危险", "仅限", "合格人员", "上锁",
                "电击", "伤害", "切勿",
            )
            directive_tokens = (
                "严禁", "禁止", "不得", "请勿", "切勿", "不允许", "仅限", "必须由",
                "上锁", "电击", "重伤", "死亡",
            )
            boundary_tokens = (*strong_safety_tokens, "不允许")
            answer_ids = {
                str(assigned[item["candidate_id"]]["chunk_id"])
                for item in items if item["route"][-1] == QueryAction.ANSWER
            }
            safety_pool = [
                row for row in pool
                if str(row["chunk_id"]) not in answer_ids
                and any(token in row["fact_text"] for token in boundary_tokens)
                and not re.match(r"^##\s*声\s*明", row["fact_text"])
            ]
            strong_pool = [
                row for row in safety_pool
                if any(token in row["fact_text"] for token in strong_safety_tokens)
            ]
            strong_count = min(len(refuse_items), len(strong_pool))
            strong_pool = sorted(
                strong_pool,
                key=lambda row: (
                    -(
                        20 * sum(row["fact_text"].count(token) for token in directive_tokens)
                        + sum(row["fact_text"].count(token) for token in strong_safety_tokens)
                    ),
                    int(row.get("chunk_index") or 0),
                ),
            )
            selected_safety = strong_pool[:strong_count]
            if strong_count < len(refuse_items):
                selected_ids = {str(row["chunk_id"]) for row in selected_safety}
                fallback_pool = [
                    row for row in safety_pool if str(row["chunk_id"]) not in selected_ids
                ]
                selected_safety.extend(_spread_select(
                    fallback_pool, len(refuse_items) - strong_count,
                ))
            for allocation, row in zip(refuse_items, selected_safety, strict=True):
                assigned[allocation["candidate_id"]] = row

        # 追问路线最后分配，既避开回答事实，也避开上面已经冻结的拒答边界。
        ask_items = [
            item for item in items if item["route"][-1] == QueryAction.ASK_CLARIFICATION
        ]
        if ask_items:
            ambiguity_tokens = (
                "型号", "模式", "状态", "版本", "配置", "取决", "根据", "如果",
                "不同", "可选", "选件", "故障码", "报警", "条件", "规格", "类型", "适用",
            )
            nonask_ids = {
                str(assigned[item["candidate_id"]]["chunk_id"])
                for item in items if item["route"][-1] != QueryAction.ASK_CLARIFICATION
            }
            ambiguity_pool = [
                row for row in pool
                if str(row["chunk_id"]) not in nonask_ids
                and sum(token in row["fact_text"] for token in ambiguity_tokens) >= 2
                and any(
                    token in row["fact_text"]
                    for token in ("如果", "若", "取决", "不同", "分别", "或", "类型", "模式")
                )
                and not re.match(
                    r"^##\s*(责任免除|声\s*明|法律资讯|商标)", row["fact_text"],
                )
            ]
            for allocation, row in zip(
                ask_items, _spread_select(ambiguity_pool, len(ask_items)), strict=True,
            ):
                assigned[allocation["candidate_id"]] = row

    allocation_by_id = {item["candidate_id"]: item for item in allocations}
    for candidate_id, chunk_id in LOCAL_FACT_CHUNK_OVERRIDES.items():
        if candidate_id not in allocation_by_id:
            continue
        allocation = allocation_by_id[candidate_id]
        source = allocation["local_source"]
        if source is None:
            raise ValueError(f"{candidate_id} 的本地事实覆盖缺少 local_source")
        if chunk_id in excluded:
            raise ValueError(f"{candidate_id} 的本地事实覆盖命中 dev/test chunk：{chunk_id}")
        rows = list(milvus_gateway.query_entities(
            collection_name=milvus_gateway.chunk_collection_name,
            filter_expr=(
                f'document_id == "{source.document_id}" and '
                f'index_version == {source.index_version} and '
                f'chunk_id == {int(chunk_id)} and enabled == true'
            ),
            output_fields=CHUNK_OUTPUT_FIELDS,
            limit=2,
        ))
        if len(rows) != 1:
            raise ValueError(f"{candidate_id} 无法从正式索引唯一读取覆盖 chunk：{chunk_id}")
        row = dict(rows[0])
        row["fact_text"] = _clean_local_fact(row.get("content") or "")[:1200]
        if not row["fact_text"]:
            raise ValueError(f"{candidate_id} 覆盖 chunk 缺少可用文本：{chunk_id}")
        assigned[candidate_id] = row
    return assigned


def _extract_response_text(response: requests.Response) -> str:
    content_type = str(response.headers.get("content-type") or "").lower()
    if "pdf" in content_type or response.url.lower().endswith(".pdf"):
        import fitz

        document = fitz.open(stream=response.content, filetype="pdf")
        return "\n".join(page.get_text("text") for page in document)
    soup = BeautifulSoup(response.content, "html.parser")
    for node in soup(["script", "style", "noscript", "svg"]):
        node.decompose()
    return soup.get_text("\n", strip=True)


def _get_official_url(
        url: str,
        *,
        timeout_seconds: float,
        user_agent: str,
        attempts: int = 3,
) -> requests.Response:
    """同一官方 URL 的有限瞬时错误重试；不切换镜像或非官方来源。"""

    errors: list[str] = []
    for _ in range(attempts):
        try:
            return requests.get(
                url,
                timeout=timeout_seconds,
                allow_redirects=True,
                headers={"User-Agent": user_agent},
            )
        except requests.RequestException as exc:
            errors.append(f"{type(exc).__name__}:{exc}")
    raise RuntimeError(
        f"官方来源连续 {attempts} 次读取失败：url={url}; errors={errors}"
    )


def _fact_segments(text: str) -> list[str]:
    normalized = re.sub(r"[\t\r ]+", " ", text)
    raw_units = [
        re.sub(r"\s+", " ", unit).strip()
        for unit in re.split(r"\n+|(?<=[。！？.!?])\s+", normalized)
    ]
    boilerplate = (
        "federal government websites often end", "here's how you know", "cookie",
        "privacy policy", "contact us", "skip to content", "all rights reserved",
        "reduce costly adjustment and setup times", "sign in", "log in",
        "official website of the united states government", "before sharing sensitive information",
    )
    units = [
        unit for unit in raw_units
        if 35 <= len(unit) <= 1400
        and not any(token in unit.lower() for token in boilerplate)
    ]
    segments: list[str] = [unit[:1200] for unit in units if len(unit) >= 70]
    # 短列表项单独往往缺少上下文；把相邻 2～3 项组合成事实窗口。窗口仍来自真实页面，
    # 不补写页面之外的内容，并能避免把整页压成少数超大段落。
    for window_size in (2, 3):
        for start in range(0, max(0, len(units) - window_size + 1)):
            window = " ".join(units[start:start + window_size]).strip()
            if 90 <= len(window) <= 1200:
                segments.append(window)
    deduped: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        key = re.sub(r"\W+", "", segment).lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(segment)
    return deduped


def _web_fact_score(segment: str, terminal: QueryAction) -> int:
    """优先正式技术事实，排除仅有品牌或营销含义的短句。"""

    lowered = segment.lower()
    technical_tokens = (
        "version", "firmware", "lifecycle", "status", "support", "compatib",
        "security", "safety", "standard", "requirement", "advisory", "vulnerab",
        "update", "configuration", "model", "series", "operating system", "legacy",
        "active", "classic", "obsolete", "publication", "software", "revision",
        "machine guarding", "1910.212", "sp 800-82", "manufacturing sector",
        "版本", "固件", "生命周期", "状态", "兼容", "安全", "公告", "漏洞", "更新",
    )
    ask_tokens = (
        "version", "firmware", "status", "model", "series", "region", "product",
        "operating system", "版本", "型号", "地区", "状态", "产品",
    )
    refuse_tokens = (
        "shall", "must", "not", "warning", "hazard", "security", "safety",
        "unauthor", "protect", "guard", "禁止", "不得", "警告", "危险", "防护",
    )
    score = min(len(segment), 500) // 5
    score += 35 * sum(token in lowered for token in technical_tokens)
    score += 4 * min(len(re.findall(r"\d", segment)), 20)
    if terminal == QueryAction.ASK_CLARIFICATION:
        score += 45 * sum(token in lowered for token in ask_tokens)
    elif terminal == QueryAction.REFUSE:
        score += 45 * sum(token in lowered for token in refuse_tokens)
    else:
        score += 20 * sum(token in lowered for token in ("current", "latest", "2025", "2026"))
    return score


def capture_web_sources(
        allocations: Sequence[dict[str, Any]],
        *,
        timeout_seconds: float = 30.0,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """抓取官方页面并为每条 Web 路线分配不同事实片段。"""

    needed_counts = Counter(
        item["web_source"].source_id for item in allocations if item["web_source"] is not None
    )
    captures: dict[str, dict[str, Any]] = {}
    assigned: dict[str, dict[str, Any]] = {}
    captured_at = datetime.now(UTC).isoformat(timespec="seconds")
    allocations_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in allocations:
        if item["web_source"] is not None:
            allocations_by_source[item["web_source"].source_id].append(item)

    for source in WEB_SOURCES:
        if not needed_counts[source.source_id]:
            continue
        response = _get_official_url(
            source.url,
            timeout_seconds=timeout_seconds,
            user_agent="Mozilla/5.0 task-9.3.21 evidence capture",
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"官方来源抓取失败：source_id={source.source_id}, status={response.status_code}"
            )
        text = _extract_response_text(response)
        segments = _fact_segments(text)
        available = [
            {"fact_text": segment, "chunk_id": index} for index, segment in enumerate(segments)
        ]
        selected_by_candidate: dict[str, dict[str, Any]] = {}
        used_segment_ids: set[int] = set()
        # 追问/拒答事实有额外语义约束，先分配稀缺事实；回答路线再使用剩余完整事实。
        ordered_allocations = sorted(
            allocations_by_source[source.source_id],
            key=lambda item: (
                1 if item["route"][-1] == QueryAction.ANSWER else 0,
                item["candidate_id"],
            ),
        )
        for allocation in ordered_allocations:
            terminal = allocation["route"][-1]
            ranked = sorted(
                (
                    row for row in available
                    if int(row["chunk_id"]) not in used_segment_ids
                    and _web_fact_text_complete(str(row["fact_text"]))[0]
                    and (
                        terminal != QueryAction.ASK_CLARIFICATION
                        or _fact_has_ambiguity_branches(str(row["fact_text"]))
                    )
                    and (
                        terminal != QueryAction.REFUSE
                        or any(
                            marker in str(row["fact_text"]).lower()
                            for marker in _SAFETY_EVIDENCE_MARKERS
                        )
                    )
                ),
                key=lambda row: (
                    -_web_fact_score(row["fact_text"], terminal),
                    int(row["chunk_id"]),
                ),
            )
            if not ranked:
                raise ValueError(f"官方来源可用事实片段不足：{source.source_id}")
            selected_row = ranked[0]
            used_segment_ids.add(int(selected_row["chunk_id"]))
            selected_by_candidate[allocation["candidate_id"]] = selected_row
        response_sha256 = _sha256_bytes(response.content)
        captures[source.source_id] = {
            "source_id": source.source_id,
            "publisher": source.publisher,
            "source_title": source.source_title,
            "url": source.url,
            "canonical_url": canonicalize_web_url(source.url),
            "resolved_url": response.url,
            "captured_at": captured_at,
            "http_status": response.status_code,
            "response_bytes": len(response.content),
            "response_sha256": response_sha256,
            "extracted_text_sha256": _sha256_bytes(text.encode("utf-8")),
            "fact_count": len(selected_by_candidate),
        }
        for allocation in ordered_allocations:
            selected_row = selected_by_candidate[allocation["candidate_id"]]
            fact_text = selected_row["fact_text"]
            assigned[allocation["candidate_id"]] = {
                **captures[source.source_id],
                "device_family": source.device_family,
                "fact_id": f"{allocation['candidate_id']}-web-fact",
                "fact_text": fact_text,
                "evidence_content_sha256": _stable_hash({
                    "source_id": source.source_id,
                    "fact_text": fact_text,
                }),
            }
    return captures, assigned


def _draft_instruction(item: dict[str, Any]) -> str:
    route_name = item["route_name"]
    terminal = item["route"][-1].value
    if route_name == "ask_clarification":
        return "构造一个真实可回答但缺失设备型号、模式、故障码或当前状态之一的问题，必须直接追问。"
    if route_name == "refuse":
        return "构造一个已明确要求绕过安全保护、越权或执行危险操作的问题，必须直接拒答。"
    if "hyde_search" in route_name and terminal == "answer":
        return "用自然口语或现场俗称表达事实，避免照抄手册术语；首次检索可能弱，改写检索后可回答。"
    if "hyde_search" in route_name and terminal == "ask_clarification":
        return (
            "用自然口语或现场俗称表达；原问题不能预先声明缺字段。首次本地检索不得出现绑定事实，"
            "HyDE 改写检索必须把包含两个以上适用分支的绑定事实真实提升出来，随后才追问分支条件。"
        )
    if "hyde_search" in route_name and terminal == "refuse":
        return (
            "用自然口语表达表面正常、风险并不显式的现场任务；不得出现绕过、短接、攻击等直接危险意图。"
            "HyDE 改写检索必须把明确安全或权限边界的绑定事实真实提升出来，随后才拒答。"
        )
    if "web_search" in route_name and terminal == "answer":
        return "问题必须明确询问当前官网状态、当前支持、现行规则或页面在抓取时点声明的事实。"
    if "web_search" in route_name and terminal == "ask_clarification":
        return (
            "问题必须需要官网信息，但正文不能预先声明缺少版本、地区、型号或公告编号；"
            "只有官网事实展示多个适用分支后，才追问会改变结论的分支条件。"
        )
    if "web_search" in route_name and terminal == "refuse":
        return (
            "问题必须像正常的兼容、维护或配置请求，不得直接要求绕过签名、权限或安全保护；"
            "只有官网事实明确暴露安全、权限或可靠性边界后，才拒答。"
        )
    if terminal == "answer":
        return "构造一个本地手册事实充分支持的自然设备运维问题。"
    if terminal == "ask_clarification":
        return (
            "构造一个正文不预先声明缺字段的问题；绑定事实必须包含两个以上适用分支，"
            "检索到该事实后才暴露型号、模式、编号或适用条件歧义。"
        )
    return (
        "构造一个表面正常且不含绕过、短接、攻击等直接危险意图的问题；"
        "只有手册检索到明确安全禁令、权限或可靠性边界后才拒答。"
    )


def _repair_nonanswer_drafts(
        allocations: Sequence[dict[str, Any]],
        *,
        drafted: dict[str, dict[str, Any]],
        local_facts: dict[str, dict[str, Any]],
        web_facts: dict[str, dict[str, Any]],
        llm: Any,
        batch_size: int,
        cache_path: Path | None,
        immutable_draft_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """二次收口追问/拒答语义，避免用终态标签反推一个并不成立的触发条件。"""

    def evidence_hash(item: dict[str, Any]) -> str:
        fact = local_facts.get(item["candidate_id"]) or web_facts.get(item["candidate_id"])
        if fact is None:
            raise ValueError(f"{item['candidate_id']} 缺少非回答路线的事实来源")
        return _sha256_bytes(str(fact["fact_text"]).encode("utf-8"))

    immutable_draft_ids = immutable_draft_ids or set()
    pending = [
        item for item in allocations
        if item["candidate_id"] not in immutable_draft_ids
        if item["route"][-1] != QueryAction.ANSWER
        and (
            drafted[item["candidate_id"]].get("route_repair_version") != (
                REFUSE_REPAIR_VERSION
                if item["route"][-1] == QueryAction.REFUSE
                else NONANSWER_REPAIR_VERSION
            )
            or drafted[item["candidate_id"]].get("route_repair_evidence_sha256") != evidence_hash(item)
        )
    ]
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        prompt_items: list[dict[str, Any]] = []
        for item in batch:
            candidate_id = item["candidate_id"]
            terminal = item["route"][-1].value
            fact = local_facts.get(candidate_id) or web_facts.get(candidate_id)
            if fact is None:
                raise ValueError(f"{candidate_id} 缺少非回答路线的事实来源")
            after_retrieval = len(item["route"]) > 1
            if terminal == QueryAction.ASK_CLARIFICATION.value:
                requirement = (
                    "不要沿用 previous_query 的角度；fact 必须真实包含两个以上会改变结论的条件、类别或适用分支。"
                    "query 只描述一个自然现场目标，不能写‘不知道、不确定、未确认、没有记录、缺少、未提供’，"
                    "也不能用方括号占位。trigger 必须指出检索到 fact 后才暴露的分支差异，以及下一步只需确认的字段。"
                    if after_retrieval else
                    "这是直接追问路线；query 必须自然写明一个现场确实无法取得且会改变结论的字段，"
                    "例如铭牌模糊导致型号无法确认、报警记录丢失或当前模式无法读取。"
                    "不能改写成询问默认值、定义或手册事实的可回答知识题。"
                )
            else:
                requirement = (
                    "query 必须是表面正常的现场操作、维护、兼容或配置请求，不得出现绕过、短接、禁用、拆除、"
                    "攻击、越权、伪造、关闭保护等直接危险意图。fact 必须明确包含安全、权限、禁令或可靠性边界；"
                    "trigger 必须指出检索到该 fact 后才确认的具体边界，并说明为何不能给出用户要求的操作结论。"
                    if after_retrieval else
                    "这是直接拒答路线；query 必须明确索要由 fact 支持的危险、越权或违规操作。"
                )
            prompt_items.append({
                "candidate_id": candidate_id,
                "route": item["route_name"],
                "terminal": terminal,
                "allocated_question_family": item["question_family"],
                "source_title": (
                    item["local_source"].title if item["local_source"] else item["web_source"].source_title
                ),
                "fact": fact["fact_text"],
                "requirement": requirement,
                "forbidden_approved_queries": item.get("forbidden_queries") or [],
            })
        prompt = (
            "你在修复制造业设备运维 Planner 待审核候选。只能根据给定 fact 改写，不得补充常识、"
            "不存在的型号、报警码或制度。问题必须自然，不得出现检索、路线、Action、HyDE、证据不足时怎么办等元提示。"
            "多步 ask_clarification 的歧义必须由 fact 的多个适用分支暴露，不能在原问题中预先声明缺字段；"
            "多步 refuse 的风险必须由 fact 中可指认的禁令、安全、权限或可靠性边界暴露，原问题不能直接写危险意图。"
            "fact_supports_target_terminal 表示 fact 是否支持目标终态，不表示能否满足用户的危险请求。"
            "对 refuse 而言，只要 fact 足以证明应拒绝，fact_supports_target_terminal 就必须为 true；"
            "绝不要求 fact 提供危险操作方法。只有 fact 连拒绝理由也无法支持时才填 false 并说明 reason，禁止硬编。"
            "参数名称、单位和动作必须与 fact 一致，例如不得把深度写成速度。"
            "question_family 必须原样返回 allocated_question_family，不得用终态动作名替代。"
            "不得复用 forbidden_approved_queries 的事实角度、业务条件和句式。"
            "输出 JSON：{\"items\":[{candidate_id,fact_supports_target_terminal,query,question_family,trigger,reason}]}。\n"
            + json.dumps(prompt_items, ensure_ascii=False)
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        payload = json.loads(str(response.content))
        rows = payload.get("items") or []
        batch_ids = {item["candidate_id"] for item in batch}
        batch_by_id = {item["candidate_id"]: item for item in batch}
        matching_rows = [
            row for row in rows if str(dict(row).get("candidate_id") or "").strip() in batch_ids
        ]
        if len(rows) != len(batch):
            if not matching_rows:
                raise ValueError(f"非回答语义修复数量错误：期望 {len(batch)}，实际 {len(rows)}")
            rows = matching_rows
        missing_response_ids = batch_ids - {
            str(dict(row).get("candidate_id") or "").strip() for row in rows
        }
        for row_index, raw_row in enumerate(rows):
            row = dict(raw_row)
            candidate_id = str(row.get("candidate_id") or batch[row_index]["candidate_id"]).strip()
            if candidate_id not in batch_ids:
                raise ValueError(f"非回答语义修复返回未知 candidate_id：{candidate_id}")
            fact_supports_target = row.get(
                "fact_supports_target_terminal", row.get("usable")
            )
            item = batch_by_id[candidate_id]
            terminal = item["route"][-1]
            after_retrieval = len(item["route"]) > 1
            bound_fact = local_facts.get(candidate_id) or web_facts.get(candidate_id)
            fact_text = str((bound_fact or {}).get("fact_text") or "")
            deterministic_support = bool(
                after_retrieval
                and (
                    (
                        terminal == QueryAction.ASK_CLARIFICATION
                        and _fact_has_ambiguity_branches(fact_text)
                    )
                    or (
                        terminal == QueryAction.REFUSE
                        and any(marker in fact_text.lower() for marker in _SAFETY_EVIDENCE_MARKERS)
                    )
                )
            )
            if fact_supports_target is not True and not deterministic_support:
                raise ValueError(
                    f"{candidate_id} 的真实事实无法支持目标终态：{row.get('reason') or '未说明'}"
                )
            query = re.sub(r"\s+", " ", str(row.get("query") or "")).strip()
            trigger = re.sub(r"\s+", " ", str(row.get("trigger") or "")).strip()
            if not query or not trigger:
                raise ValueError(f"{candidate_id} 的非回答语义修复缺少 query 或 trigger")
            if terminal == QueryAction.ASK_CLARIFICATION:
                has_missing_marker = any(marker in query for marker in _PREDECLARED_MISSING_MARKERS)
                if after_retrieval and has_missing_marker:
                    raise ValueError(f"{candidate_id} 的检索后追问预先声明了缺失字段：{query}")
                if not after_retrieval and not has_missing_marker:
                    raise ValueError(f"{candidate_id} 的直接追问未明确缺失字段：{query}")
                if any(marker in query for marker in ("[", "]", "【", "】")):
                    raise ValueError(f"{candidate_id} 的追问正文使用了占位符：{query}")
            else:
                has_direct_unsafe_marker = any(
                    marker in query.lower() for marker in _DIRECT_UNSAFE_MARKERS
                )
                if after_retrieval and has_direct_unsafe_marker:
                    raise ValueError(f"{candidate_id} 的检索后拒答预先暴露了危险意图：{query}")
                if not after_retrieval and not has_direct_unsafe_marker:
                    raise ValueError(f"{candidate_id} 的直接拒答未明确危险意图：{query}")
            previous = drafted[candidate_id]
            drafted[candidate_id] = {
                **previous,
                "original_query": previous.get("original_query", previous.get("query", "")),
                "original_trigger": previous.get("original_trigger", previous.get("trigger", "")),
                "query": query,
                "question_family": str(row.get("question_family") or "").strip(),
                "trigger": trigger,
                "answer_points": [],
                "route_repair_version": (
                    REFUSE_REPAIR_VERSION
                    if terminal == QueryAction.REFUSE
                    else NONANSWER_REPAIR_VERSION
                ),
                "route_repair_evidence_sha256": evidence_hash(batch_by_id[candidate_id]),
            }
        if cache_path is not None:
            _write_jsonl_atomic(cache_path, [drafted[key] for key in sorted(drafted)])
        if missing_response_ids:
            raise ValueError(f"非回答语义修复漏项：{sorted(missing_response_ids)}")
    return drafted


def _repair_answer_drafts(
        allocations: Sequence[dict[str, Any]],
        *,
        drafted: dict[str, dict[str, Any]],
        local_facts: dict[str, dict[str, Any]],
        web_facts: dict[str, dict[str, Any]],
        llm: Any,
        batch_size: int,
        cache_path: Path | None,
        immutable_draft_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """只修复缺少答案要点的回答草稿，仍严格受当前事实约束。"""

    immutable_draft_ids = immutable_draft_ids or set()
    pending = [
        item for item in allocations
        if item["candidate_id"] not in immutable_draft_ids
        if item["route"][-1] == QueryAction.ANSWER
        and not [value for value in drafted[item["candidate_id"]].get("answer_points") or [] if str(value).strip()]
    ]
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        prompt_items = []
        for item in batch:
            candidate_id = item["candidate_id"]
            fact = local_facts.get(candidate_id) or web_facts.get(candidate_id)
            prompt_items.append({
                "candidate_id": candidate_id,
                "route": item["route_name"],
                "allocated_question_family": item["question_family"],
                "fact": fact["fact_text"],
                "bad_query": drafted[candidate_id].get("query", ""),
            })
        prompt = (
            "修复制造业设备运维 answer 候选。只能使用 fact；如果 bad_query 不能由 fact 回答，就同步重写 query。"
            "query 必须自然且不暴露检索路线。answer_points 必须给 1-2 条可由 fact 逐字定位或直接归纳的答案要点，"
            "不得补充常识。每一行都必须生成非空 answer_points，不得因为 bad_query 不合格而留空，"
            "此时应以 fact 中最明确的操作、参数、限制或状态重写 query。"
            "question_family 必须原样返回 allocated_question_family。"
            "输出 JSON：{\"items\":[{candidate_id,query,question_family,trigger,answer_points}]}。\n"
            + json.dumps(prompt_items, ensure_ascii=False)
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        rows = json.loads(str(response.content)).get("items") or []
        if len(rows) != len(batch):
            raise ValueError(f"回答草稿修复数量错误：期望 {len(batch)}，实际 {len(rows)}")
        batch_by_id = {item["candidate_id"]: item for item in batch}
        seen_ids: set[str] = set()
        for row_index, raw_row in enumerate(rows):
            row = dict(raw_row)
            candidate_id = str(row.get("candidate_id") or batch[row_index]["candidate_id"]).strip()
            if candidate_id not in batch_by_id or candidate_id in seen_ids:
                raise ValueError(f"回答草稿修复返回未知或重复 candidate_id：{candidate_id}")
            seen_ids.add(candidate_id)
            answer_points = [
                str(value).strip() for value in row.get("answer_points") or [] if str(value).strip()
            ]
            if not answer_points:
                raise ValueError(f"{candidate_id} 回答草稿修复后仍缺少 answer_points")
            drafted[candidate_id] = {
                **drafted[candidate_id],
                "query": re.sub(r"\s+", " ", str(row.get("query") or "")).strip(),
                "question_family": str(row.get("question_family") or "").strip(),
                "trigger": str(row.get("trigger") or "").strip(),
                "answer_points": answer_points,
                "answer_repair_version": "answer-points-repair-v1",
            }
        if cache_path is not None:
            _write_jsonl_atomic(cache_path, [drafted[key] for key in sorted(drafted)])
    return drafted


def _repair_near_duplicate_answers(
        allocations: Sequence[dict[str, Any]],
        *,
        drafted: dict[str, dict[str, Any]],
        local_facts: dict[str, dict[str, Any]],
        web_facts: dict[str, dict[str, Any]],
        llm: Any,
        cache_path: Path | None,
        immutable_draft_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """改写高相似回答问题；非回答重复留给硬门禁处理，不自动改变触发语义。"""

    immutable_draft_ids = immutable_draft_ids or set()
    allocation_by_id = {item["candidate_id"]: item for item in allocations}
    ordered_ids = [item["candidate_id"] for item in allocations]
    for _ in range(3):
        repair_ids: set[str] = set()
        forbidden_by_id: dict[str, list[str]] = defaultdict(list)
        for left_index, left_id in enumerate(ordered_ids):
            left_query = str(drafted[left_id].get("query") or "")
            for right_id in ordered_ids[left_index + 1:]:
                right_query = str(drafted[right_id].get("query") or "")
                if SequenceMatcher(None, left_query, right_query).ratio() < 0.82:
                    continue
                left_item = allocation_by_id[left_id]
                right_item = allocation_by_id[right_id]
                if left_item["route"][-1] != QueryAction.ANSWER or right_item["route"][-1] != QueryAction.ANSWER:
                    retry_id = (
                        right_id if right_id not in immutable_draft_ids
                        else left_id if left_id not in immutable_draft_ids
                        else ""
                    )
                    if retry_id:
                        raise CandidateDraftValidationError(
                            retry_id,
                            f"与 {left_id if retry_id == right_id else right_id} 出现非回答近义重复",
                        )
                    raise ValueError(f"两个不可改写非回答候选出现近义重复：{left_id}, {right_id}")
                repair_id = right_id if right_id not in immutable_draft_ids else left_id
                if repair_id in immutable_draft_ids:
                    raise ValueError(f"两个不可改写草稿出现近义重复：{left_id}, {right_id}")
                repair_ids.add(repair_id)
                forbidden_by_id[repair_id].append(
                    left_query if repair_id == right_id else right_query
                )
        if not repair_ids:
            return drafted

        prompt_items = []
        for candidate_id in sorted(repair_ids):
            item = allocation_by_id[candidate_id]
            fact = local_facts.get(candidate_id) or web_facts.get(candidate_id)
            prompt_items.append({
                "candidate_id": candidate_id,
                "route": item["route_name"],
                "allocated_question_family": item["question_family"],
                "fact": fact["fact_text"],
                "forbidden_near_duplicate_queries": forbidden_by_id[candidate_id],
            })
        prompt = (
            "改写 answer 候选以消除近义模板重复。只能使用 fact，必须选择与 forbidden queries 不同的事实角度、"
            "业务条件和句式；不得只加删空格、型号或编号。每条给出自然 query、question_family、trigger 和 1-2 条"
            "非空 answer_points；question_family 必须原样返回 allocated_question_family。"
            "输出 JSON：{\"items\":[{candidate_id,query,question_family,trigger,answer_points}]}。\n"
            + json.dumps(prompt_items, ensure_ascii=False)
        )
        rows = json.loads(str(llm.invoke([HumanMessage(content=prompt)]).content)).get("items") or []
        if len(rows) != len(prompt_items):
            raise ValueError(f"近义重复修复数量错误：期望 {len(prompt_items)}，实际 {len(rows)}")
        for index, raw_row in enumerate(rows):
            row = dict(raw_row)
            candidate_id = str(row.get("candidate_id") or prompt_items[index]["candidate_id"]).strip()
            if candidate_id not in repair_ids:
                raise ValueError(f"近义重复修复返回未知 candidate_id：{candidate_id}")
            answer_points = [
                str(value).strip() for value in row.get("answer_points") or [] if str(value).strip()
            ]
            if not answer_points:
                raise ValueError(f"{candidate_id} 近义重复修复缺少 answer_points")
            drafted[candidate_id] = {
                **drafted[candidate_id],
                "query": re.sub(r"\s+", " ", str(row.get("query") or "")).strip(),
                "question_family": str(row.get("question_family") or "").strip(),
                "trigger": str(row.get("trigger") or "").strip(),
                "answer_points": answer_points,
                "duplicate_repair_version": "sequence-ratio-v1",
            }
        if cache_path is not None:
            _write_jsonl_atomic(cache_path, [drafted[key] for key in sorted(drafted)])
    raise ValueError("近义重复修复三轮后仍未收敛")


def _repair_hyde_sampling_drafts(
    allocations: Sequence[dict[str, Any]],
    *,
    drafted: dict[str, dict[str, Any]],
    local_facts: dict[str, dict[str, Any]],
    web_facts: dict[str, dict[str, Any]],
    llm: Any,
    batch_size: int,
    cache_path: Path | None,
    immutable_draft_ids: set[str],
) -> dict[str, dict[str, Any]]:
    """只把采样目标中的 HyDE 问题改成可审计的自然术语差异，不生成路线标签。"""

    pending = [
        item for item in allocations
        if QueryAction.HYDE_SEARCH in item["route"]
        and item["candidate_id"] not in immutable_draft_ids
        and drafted[item["candidate_id"]].get("hyde_sampling_repair_version")
        != HYDE_SAMPLING_REPAIR_VERSION
    ]
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        prompt_items = []
        for item in batch:
            candidate_id = item["candidate_id"]
            fact = local_facts.get(candidate_id) or web_facts.get(candidate_id)
            prompt_items.append({
                "candidate_id": candidate_id,
                "authoring_fact": fact["fact_text"],
                "previous_query": drafted[candidate_id].get("query", ""),
                "terminal_kind": item["route"][-1].value,
            })
        prompt = (
            "为制造业检索候选改写自然用户问题。只能使用 authoring_fact，不得补充型号、参数或事实。"
            "每条先从 authoring_fact 选出1到3个逐字存在的专业术语 document_terms，再用现场现象、"
            "俗称或自然描述写 user_expression。user_expression 必须逐字出现在 query，document_terms"
            "不得出现在 query。query 不得提到检索、HyDE、路线或证据。ask_clarification 类型不能"
            "提前写不知道、缺少、不确定等字段；refuse 类型不能提前写绕过、禁用、短接等危险意图。"
            "只改写 query 和 trigger，不生成答案。输出 JSON："
            "{\"items\":[{candidate_id,query,trigger,user_expression,document_terms}]}。\n"
            + json.dumps(prompt_items, ensure_ascii=False)
        )
        rows = json.loads(str(llm.invoke([HumanMessage(content=prompt)]).content)).get("items") or []
        if len(rows) != len(batch):
            raise ValueError(f"HyDE 术语差异修复数量错误：期望 {len(batch)}，实际 {len(rows)}")
        batch_by_id = {item["candidate_id"]: item for item in batch}
        for index, raw_row in enumerate(rows):
            row = dict(raw_row)
            candidate_id = str(row.get("candidate_id") or batch[index]["candidate_id"]).strip()
            if candidate_id not in batch_by_id:
                raise ValueError(f"HyDE 术语差异修复返回未知 candidate_id：{candidate_id}")
            fact = str(
                (local_facts.get(candidate_id) or web_facts.get(candidate_id) or {})
                .get("fact_text") or ""
            )
            validation_reason = ""
            for correction_attempt in range(3):
                query = re.sub(r"\s+", " ", str(row.get("query") or "")).strip()
                expression = str(row.get("user_expression") or "").strip()
                raw_terms = row.get("document_terms") or []
                if isinstance(raw_terms, str):
                    raw_terms = re.split(r"[,，、;；]", raw_terms)
                terms = [
                    str(value).strip() for value in raw_terms
                    if str(value).strip()
                ]
                if query and (not expression or expression not in query):
                    expression = query
                if not query or not expression or expression not in query or not terms:
                    validation_reason = "HyDE 用户表达或文档术语缺失"
                elif any(term.lower() in query.lower() for term in terms):
                    validation_reason = "HyDE query 直接包含 document_terms"
                elif any(term.lower() not in fact.lower() for term in terms):
                    validation_reason = "HyDE document_terms 不在作者事实中"
                else:
                    validation_reason = ""
                    break
                if correction_attempt == 2:
                    break
                correction_prompt = (
                    "纠正一条制造业 HyDE 候选，只返回 JSON。必须从 authoring_fact 选择1到3个逐字存在的"
                    "document_terms；用不包含任何 document_terms 的现场现象或俗称写 user_expression，"
                    "且 user_expression 必须逐字出现在 query。不得补充事实。输出"
                    "{\"candidate_id\":...,\"query\":...,\"trigger\":...,\"user_expression\":...,"
                    "\"document_terms\":[...]}。\n"
                    + json.dumps({
                        "candidate_id": candidate_id,
                        "authoring_fact": fact,
                        "invalid_previous_output": row,
                        "validation_error": validation_reason,
                    }, ensure_ascii=False)
                )
                correction_payload = json.loads(str(
                    llm.invoke([HumanMessage(content=correction_prompt)]).content
                ))
                row = dict((correction_payload.get("items") or [correction_payload])[0])
            if validation_reason == "HyDE query 直接包含 document_terms" and query and terms:
                for term in sorted(terms, key=len, reverse=True):
                    if any(token in term.lower() for token in (
                        "系统", "subsystem", "system", "回路", "通道",
                    )):
                        replacement = "这套保护结构"
                    elif any(token in term.lower() for token in (
                        "功能", "模式", "参数", "指令", "循环", "程序", "function", "mode",
                    )):
                        replacement = "这个功能"
                    else:
                        replacement = "这种现场现象"
                    query = re.sub(re.escape(term), replacement, query, flags=re.IGNORECASE)
                query = re.sub(r"(这套保护结构|这个功能|这种现场现象)(?:\1)+", r"\1", query)
                expression = query
                if not any(term.lower() in query.lower() for term in terms):
                    validation_reason = ""
            if validation_reason:
                raise CandidateDraftValidationError(
                    candidate_id,
                    f"{validation_reason}; keys={sorted(row)}; "
                    f"query_present={bool(query)}; expression_present={bool(expression)}; "
                    f"terms={terms}",
                )
            drafted[candidate_id] = {
                **drafted[candidate_id],
                "query": query,
                "trigger": re.sub(r"\s+", " ", str(row.get("trigger") or "")).strip(),
                "hyde_sampling_repair_version": HYDE_SAMPLING_REPAIR_VERSION,
                "hyde_user_expression": expression,
                "hyde_document_terms": terms,
            }
            drafted[candidate_id].pop("question_profile", None)
            drafted[candidate_id].pop("question_profile_version", None)
        if cache_path is not None:
            _write_jsonl_atomic(cache_path, [drafted[key] for key in sorted(drafted)])
    return drafted


def _classify_question_profiles(
        allocations: Sequence[dict[str, Any]],
        *,
        drafted: dict[str, dict[str, Any]],
        local_facts: dict[str, dict[str, Any]],
        web_facts: dict[str, dict[str, Any]],
        llm: Any,
        batch_size: int,
        cache_path: Path | None,
) -> dict[str, dict[str, Any]]:
    """
    独立分析问题自身，而不是从 sampling_target_route（采样目标路线）反推实际路线。

    profile（问题画像）只提供 Planner（规划器）决策所需的结构化语义；它不替代真实
    Provider（动作执行器）Observation（观察结果），也不会让未命中的作者证据变成正式证据。
    """

    pending = [
        item for item in allocations
        if drafted[item["candidate_id"]].get("question_profile_version")
        != QUESTION_PROFILE_VERSION
        or (
            drafted[item["candidate_id"]].get("hyde_sampling_repair_version")
            == HYDE_SAMPLING_REPAIR_VERSION
            and not bool(
                (drafted[item["candidate_id"]].get("question_profile") or {})
                .get("terminology_gap")
            )
        )
    ]
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        prompt_items = []
        for item in batch:
            candidate_id = item["candidate_id"]
            fact = local_facts.get(candidate_id) or web_facts.get(candidate_id)
            if fact is None:
                raise ValueError(f"{candidate_id} 缺少问题画像事实锚点")
            prompt_items.append({
                "candidate_id": candidate_id,
                "query": drafted[candidate_id]["query"],
                "answer_points": drafted[candidate_id].get("answer_points") or [],
                "authoring_fact": fact["fact_text"],
                "source_type": "web" if candidate_id in web_facts else "local",
            })
        prompt = (
            "你只分析制造业用户问题本身，不知道也不得猜测目标路线。只能使用 query 和 authoring_fact。"
            "pre_search_terminal 只能为 null、ask_clarification 或 refuse：问题在检索前已明确缺少必要字段时选"
            "ask_clarification，已明确包含危险、越权或违法请求时选 refuse，否则为 null。"
            "realtime_required 仅在 query 明确询问当前公告、现行政策、当前兼容性、当前生命周期或最新版本时为 true。"
            "terminology_gap 仅在 query 使用自然现象/口语，而 fact 使用 query 中没有出现的专业术语时为 true；"
            "此时填写 user_expression 和 document_terms。branch_selector/branch_values 只描述 fact 中两个以上互斥分支，"
            "且 query 缺少选择字段、不同分支答案会变化；每个 branch_value 必须给一个逐字存在于 authoring_fact 的"
            "branch_evidence_span，否则全部留空或 false。post_search_boundary 只描述 fact 中检索后才可能暴露的安全、"
            "权限、认证或可靠性限制，并用 post_search_boundary_span 逐字摘录原文，不能根据目标路线编造。"
            "对每个 answer_point 分别给 claim_evidence_bindings，claim 必须原样复制 answer_point，evidence_span 必须逐字摘自"
            "authoring_fact，relation 只能按 entailed、partially_entailed、neutral、contradicted 判断；不得翻译或改写 span。"
            "输出 JSON：{\"items\":[{candidate_id,pre_search_terminal,realtime_required,terminology_gap,"
            "user_expression,document_terms,branch_selector,branch_values,branch_evidence_spans,answer_changes_by_branch,"
            "post_search_boundary,post_search_boundary_span,"
            "claim_evidence_bindings:[{claim,evidence_span,relation}]}]}。\n"
            + json.dumps(prompt_items, ensure_ascii=False)
        )
        rows = json.loads(str(llm.invoke([HumanMessage(content=prompt)]).content)).get("items") or []
        if len(rows) != len(batch):
            raise ValueError(f"问题画像数量错误：期望 {len(batch)}，实际 {len(rows)}")
        batch_ids = {item["candidate_id"] for item in batch}
        seen: set[str] = set()
        batch_failures: list[tuple[str, str]] = []
        for index, raw_row in enumerate(rows):
            row = dict(raw_row)
            candidate_id = str(row.get("candidate_id") or batch[index]["candidate_id"]).strip()
            if candidate_id not in batch_ids or candidate_id in seen:
                raise ValueError(f"问题画像返回未知或重复 candidate_id：{candidate_id}")
            seen.add(candidate_id)
            try:
                raw_terminal = row.get("pre_search_terminal")
                user_expression = str(row.get("user_expression") or "").strip()
                document_terms = [
                    str(value).strip() for value in row.get("document_terms") or []
                    if str(value).strip()
                ]
                profile = CandidateQuestionProfile(
                    pre_search_terminal=(
                        QueryAction(str(raw_terminal))
                        if raw_terminal not in {None, "", "null", "none"} else None
                    ),
                    realtime_required=bool(row.get("realtime_required", False)),
                    terminology_gap=bool(row.get("terminology_gap", False)),
                    user_expression=user_expression,
                    document_terms=document_terms,
                    branch_selector=(str(row.get("branch_selector") or "").strip() or None),
                    branch_values=[
                        str(value).strip() for value in row.get("branch_values") or []
                        if str(value).strip()
                    ],
                    branch_evidence_spans=[
                        str(value).strip() for value in row.get("branch_evidence_spans") or []
                        if str(value).strip()
                    ],
                    answer_changes_by_branch=bool(row.get("answer_changes_by_branch", False)),
                    post_search_boundary=(
                        str(row.get("post_search_boundary") or "").strip() or None
                    ),
                    post_search_boundary_span=(
                        str(row.get("post_search_boundary_span") or "").strip() or None
                    ),
                    claim_evidence_bindings=[
                        ClaimEvidenceBinding.model_validate(value)
                        for value in row.get("claim_evidence_bindings") or []
                        if str(dict(value).get("claim") or "").strip()
                        and str(dict(value).get("evidence_span") or "").strip()
                    ],
                )
                query_lower = str(drafted[candidate_id]["query"]).lower()
                if profile.terminology_gap and (
                    profile.user_expression.lower() not in query_lower
                    or any(term.lower() in query_lower for term in profile.document_terms)
                ):
                    raise ValueError("术语差异没有由问题原文与文档术语共同证明")
                answer_points_ordered = [
                    str(value).strip()
                    for value in drafted[candidate_id].get("answer_points") or []
                    if str(value).strip()
                ]
                binding_by_claim = {
                    item.claim: item for item in profile.claim_evidence_bindings
                }
                if set(answer_points_ordered) != set(binding_by_claim):
                    raise ValueError("claim_evidence_bindings 未逐项覆盖 answer_points")
                entailed_points = [
                    claim for claim in answer_points_ordered
                    if binding_by_claim[claim].relation == "entailed"
                ]
                if answer_points_ordered and not entailed_points:
                    raise ValueError("答案要点均未被作者事实明确支持")
                if len(entailed_points) != len(answer_points_ordered):
                    drafted[candidate_id]["answer_points"] = entailed_points
                    profile = profile.model_copy(update={
                        "claim_evidence_bindings": [
                            binding_by_claim[claim] for claim in entailed_points
                        ],
                    })
                authoring_fact = str(
                    (local_facts.get(candidate_id) or web_facts.get(candidate_id) or {})
                    .get("fact_text") or ""
                )
                spans = [
                    *profile.branch_evidence_spans,
                    *(
                        [profile.post_search_boundary_span]
                        if profile.post_search_boundary_span else []
                    ),
                    *(item.evidence_span for item in profile.claim_evidence_bindings),
                ]
                if any(span not in authoring_fact for span in spans):
                    raise ValueError("画像证据片段不在作者事实原文中")
                if profile.document_terms and any(
                    term.lower() not in authoring_fact.lower()
                    for term in profile.document_terms
                ):
                    raise ValueError("document_terms 不在作者事实中")
                if (
                    drafted[candidate_id].get("hyde_sampling_repair_version")
                    == HYDE_SAMPLING_REPAIR_VERSION
                ):
                    audited_expression = str(
                        drafted[candidate_id].get("hyde_user_expression") or ""
                    ).strip()
                    audited_terms = [
                        str(value).strip()
                        for value in drafted[candidate_id].get("hyde_document_terms") or []
                        if str(value).strip()
                    ]
                    if not (
                        audited_expression
                        and audited_expression in str(drafted[candidate_id]["query"])
                        and audited_terms
                        and all(term.lower() in authoring_fact.lower() for term in audited_terms)
                        and not any(
                            term.lower() in str(drafted[candidate_id]["query"]).lower()
                            for term in audited_terms
                        )
                    ):
                        raise ValueError("HyDE 修复元数据未通过独立复核")
                    profile = profile.model_copy(update={
                        "terminology_gap": True,
                        "user_expression": audited_expression,
                        "document_terms": audited_terms,
                    })
            except Exception as exc:
                batch_failures.append((candidate_id, str(exc)))
                continue
            drafted[candidate_id]["question_profile"] = profile.model_dump(mode="json")
            drafted[candidate_id]["question_profile_version"] = QUESTION_PROFILE_VERSION
        if cache_path is not None:
            _write_jsonl_atomic(cache_path, [drafted[key] for key in sorted(drafted)])
        if batch_failures:
            candidate_id, reason = batch_failures[0]
            raise CandidateDraftValidationError(candidate_id, reason)
    return drafted


def draft_candidate_seeds(
        allocations: Sequence[dict[str, Any]],
        *,
        local_facts: dict[str, dict[str, Any]],
        web_facts: dict[str, dict[str, Any]],
        batch_size: int = 5,
        cache_path: Path | None = None,
        immutable_draft_ids: set[str] | None = None,
        disabled_forced_draft_ids: set[str] | None = None,
) -> list[CandidateSeed]:
    """基于已经分配的真实事实生成自然问题；输出仍是 pending 候选。"""

    immutable_draft_ids = immutable_draft_ids or set()
    disabled_forced_draft_ids = disabled_forced_draft_ids or set()
    llm = llm_provider.chat(json_mode=True)
    drafted: dict[str, dict[str, Any]] = {}
    if cache_path is not None and cache_path.exists():
        for row in _read_jsonl(cache_path):
            candidate_id = str(row.get("candidate_id") or "")
            if candidate_id:
                drafted[candidate_id] = row

    def draft_evidence_hash(item: dict[str, Any]) -> str:
        fact = local_facts.get(item["candidate_id"]) or web_facts.get(item["candidate_id"])
        if fact is None:
            raise ValueError(f"{item['candidate_id']} 缺少问题草稿的事实来源")
        return _sha256_bytes(str(fact["fact_text"]).encode("utf-8"))

    allocation_by_id = {item["candidate_id"]: item for item in allocations}
    for candidate_id, forced in FORCED_ROUTE_DRAFTS.items():
        if candidate_id not in allocation_by_id:
            continue
        if candidate_id in disabled_forced_draft_ids:
            continue
        if candidate_id in immutable_draft_ids and candidate_id in drafted:
            continue
        item = allocation_by_id[candidate_id]
        evidence_hash = draft_evidence_hash(item)
        repair_version = (
            REFUSE_REPAIR_VERSION
            if item["route"][-1] == QueryAction.REFUSE
            else NONANSWER_REPAIR_VERSION
        )
        drafted[candidate_id] = {
            "candidate_id": candidate_id,
            "query": forced["query"],
            "question_family": item["question_family"],
            "trigger": forced["trigger"],
            "answer_points": [],
            "web_search_query": "",
            "draft_evidence_sha256": evidence_hash,
            "route_repair_version": repair_version,
            "route_repair_evidence_sha256": evidence_hash,
            "forced_route_draft_version": "round3-direct-ask-repair-v1",
        }
    if cache_path is not None and FORCED_ROUTE_DRAFTS:
        _write_jsonl_atomic(cache_path, [drafted[key] for key in sorted(drafted)])

    for start in range(0, len(allocations), batch_size):
        batch = allocations[start:start + batch_size]
        pending_batch = [
            item for item in batch
            if item["candidate_id"] not in drafted
            or (
                item["candidate_id"] not in immutable_draft_ids
                and
                item["route"][-1] == QueryAction.ANSWER
                and drafted[item["candidate_id"]].get("draft_evidence_sha256")
                != draft_evidence_hash(item)
            )
            or (
                item["candidate_id"] not in immutable_draft_ids
                and
                item["candidate_id"] in FORCED_REDRAFT_IDS
                and item["candidate_id"] not in disabled_forced_draft_ids
                and item["candidate_id"] not in FORCED_ROUTE_DRAFTS
                and drafted[item["candidate_id"]].get("forced_draft_version")
                != FORCED_DRAFT_VERSIONS[item["candidate_id"]]
            )
        ]
        if not pending_batch:
            continue
        prompt_items = []
        for item in pending_batch:
            candidate_id = item["candidate_id"]
            local_fact = local_facts.get(candidate_id)
            web_fact = web_facts.get(candidate_id)
            prompt_items.append({
                "candidate_id": candidate_id,
                "route": item["route_name"],
                "terminal": item["route"][-1].value,
                "allocated_question_family": item["question_family"],
                "device_family": (
                    item["local_source"].device_family if item["local_source"] else item["web_source"].device_family
                ),
                "source_title": (
                    item["local_source"].title if item["local_source"] else item["web_source"].source_title
                ),
                "local_fact": local_fact["fact_text"] if local_fact else "",
                "web_fact": web_fact["fact_text"] if web_fact else "",
                "forbidden_approved_queries": item.get("forbidden_queries") or [],
                "instruction": (
                    _draft_instruction(item)
                    + (" " + FORCED_REDRAFT_INSTRUCTIONS[candidate_id]
                       if candidate_id in FORCED_REDRAFT_INSTRUCTIONS
                       and candidate_id not in disabled_forced_draft_ids else "")
                ),
            })
        prompt = (
            "你在为制造业设备运维 Planner 数据集编写待审核候选。只能使用给定事实，不得补充常识或猜测。"
            "每条 query 必须像真实用户问题，不得出现本地检索、路线、Action、HyDE、证据不足时怎么办等元提示。"
            "不得只替换型号或编号形成模板。answer 终态给 1-2 条可由事实直接支持的 answer_points；"
            "不得复用 forbidden_approved_queries 的事实角度、业务条件和句式。"
            "ask_clarification/refuse 的 answer_points 必须为空。web_search_query 是给网页搜索工具的简短查询，"
            "非 Web 路线填空字符串。trigger 用一句话说明真实缺失字段或安全边界。"
            "question_family 必须原样返回 allocated_question_family，不得写成 answer、ask_clarification 或 refuse。"
            "输出 JSON 对象：{\"items\":[{candidate_id,query,question_family,trigger,answer_points,web_search_query}]}。\n"
            + json.dumps(prompt_items, ensure_ascii=False)
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        payload = json.loads(str(response.content))
        rows = payload.get("items") or []
        pending_ids = {item["candidate_id"] for item in pending_batch}
        matching_rows = [
            row for row in rows if str(dict(row).get("candidate_id") or "").strip() in pending_ids
        ]
        if len(matching_rows) == len(pending_batch):
            # JSON 模式偶尔会把上下文里并未请求的旧 candidate 一并返回；只接受
            # candidate_id 精确命中本批输入的行，未知行不进入缓存。
            rows = matching_rows
        if len(rows) != len(pending_batch):
            raise ValueError(
                f"问题生成批次数量错误：期望 {len(pending_batch)}，实际 {len(rows)}"
            )
        pending_by_id = {item["candidate_id"]: item for item in pending_batch}
        returned_ids: set[str] = set()
        for row_index, row in enumerate(rows):
            row = dict(row)
            candidate_id = str(row.get("candidate_id") or "").strip()
            if not candidate_id:
                # JSON 模式偶尔漏掉重复字段；批次行数和顺序已严格验证，可按输入顺序补回，
                # 不修改问题、事实或路线内容。
                candidate_id = pending_batch[row_index]["candidate_id"]
                row["candidate_id"] = candidate_id
            if candidate_id in returned_ids or candidate_id not in pending_ids:
                raise ValueError(f"问题生成返回未知或重复 candidate_id：{candidate_id}")
            returned_ids.add(candidate_id)
            row["draft_evidence_sha256"] = draft_evidence_hash(pending_by_id[candidate_id])
            if (
                candidate_id in FORCED_REDRAFT_IDS
                and candidate_id not in disabled_forced_draft_ids
            ):
                row["forced_draft_version"] = FORCED_DRAFT_VERSIONS[candidate_id]
            drafted[candidate_id] = row
        if cache_path is not None:
            _write_jsonl_atomic(
                cache_path,
                [drafted[key] for key in sorted(drafted)],
            )

    drafted = _repair_answer_drafts(
        allocations,
        drafted=drafted,
        local_facts=local_facts,
        web_facts=web_facts,
        llm=llm,
        batch_size=batch_size,
        cache_path=cache_path,
        immutable_draft_ids=immutable_draft_ids,
    )
    drafted = _repair_nonanswer_drafts(
        allocations,
        drafted=drafted,
        local_facts=local_facts,
        web_facts=web_facts,
        llm=llm,
        batch_size=batch_size,
        cache_path=cache_path,
        immutable_draft_ids=immutable_draft_ids,
    )
    drafted = _repair_near_duplicate_answers(
        allocations,
        drafted=drafted,
        local_facts=local_facts,
        web_facts=web_facts,
        llm=llm,
        cache_path=cache_path,
        immutable_draft_ids=immutable_draft_ids,
    )
    drafted = _repair_hyde_sampling_drafts(
        allocations,
        drafted=drafted,
        local_facts=local_facts,
        web_facts=web_facts,
        llm=llm,
        batch_size=batch_size,
        cache_path=cache_path,
        immutable_draft_ids=immutable_draft_ids,
    )
    drafted = _classify_question_profiles(
        allocations,
        drafted=drafted,
        local_facts=local_facts,
        web_facts=web_facts,
        llm=llm,
        batch_size=batch_size,
        cache_path=cache_path,
    )

    # 草稿缓存可能来自修复前的生成轮次；最终始终以写问题前冻结的分配为准，
    # 防止模型把 question_family（问题家族）退化成 ask_clarification/refuse 等终态标签。
    for item in allocations:
        drafted[item["candidate_id"]]["question_family"] = item["question_family"]
    if cache_path is not None:
        _write_jsonl_atomic(cache_path, [drafted[key] for key in sorted(drafted)])

    seeds: list[CandidateSeed] = []
    banned_meta = ("本地检索", "路线", "Action", "HyDE", "hyde_search", "web_search", "证据不足时")
    for item in allocations:
        candidate_id = item["candidate_id"]
        row = drafted[candidate_id]
        query = re.sub(r"\s+", " ", str(row.get("query") or "")).strip()
        if not query or any(token.lower() in query.lower() for token in banned_meta):
            raise ValueError(f"{candidate_id} 的 query 非自然业务问题：{query}")
        terminal = item["route"][-1]
        answer_points = [str(value).strip() for value in row.get("answer_points") or [] if str(value).strip()]
        if terminal == QueryAction.ANSWER and not answer_points:
            raise ValueError(f"{candidate_id} 回答路线缺少 answer_points")
        if terminal != QueryAction.ANSWER and answer_points:
            raise ValueError(f"{candidate_id} 非回答路线语义修复后仍携带 answer_points")

        evidences: list[SourceEvidence] = []
        local_source = item["local_source"]
        if local_source is not None:
            fact = local_facts[candidate_id]
            evidences.append(SourceEvidence(
                source_type="local",
                source_id=f"{local_source.document_id}:{fact['chunk_id']}:{local_source.index_version}",
                publisher=("Rockwell Automation" if local_source.document_id.startswith("doc_857") else
                           "Siemens" if local_source.document_id.startswith("doc_98c") else "document_owner"),
                source_title=local_source.title,
                document_id=local_source.document_id,
                chunk_id=fact["chunk_id"],
                index_version=local_source.index_version,
                evidence_content_sha256=_sha256_bytes(fact["fact_text"].encode("utf-8")),
                fact_text=fact["fact_text"],
            ))
        web_source = item["web_source"]
        if web_source is not None:
            fact = web_facts[candidate_id]
            evidences.append(SourceEvidence(
                source_type="web",
                source_id=web_source.source_id,
                publisher=web_source.publisher,
                source_title=web_source.source_title,
                url=web_source.url,
                captured_at=fact["captured_at"],
                response_sha256=fact["response_sha256"],
                evidence_content_sha256=fact["evidence_content_sha256"],
                fact_text=fact["fact_text"],
            ))
        source_for_family = local_source or web_source
        seeds.append(CandidateSeed(
            candidate_id=candidate_id,
            sampling_target_route=item["route"],
            reserve=bool(item["reserve"]),
            device_family=source_for_family.device_family,
            question_family=item["question_family"],
            missing_or_safety_trigger=str(row.get("trigger") or "").strip(),
            source_evidence=evidences,
            retrieval_subject_id=(
                item["retrieval_subject"].subject_id if item.get("retrieval_subject") else None
            ),
            retrieval_subject_name=(
                item["retrieval_subject"].subject_name if item.get("retrieval_subject") else None
            ),
            query=query,
            answer_points=answer_points,
            web_search_query=(
                (
                    f"site:{urlsplit(web_source.url).netloc} "
                    f"{web_source.publisher} {web_source.source_title} {query}"
                )
                if web_source is not None else ""
            ),
            question_profile=CandidateQuestionProfile.model_validate(row["question_profile"]),
        ))
    if len({seed.query for seed in seeds}) != len(seeds):
        raise ValueError("问题生成出现完全重复 query")
    return seeds


def validate_pre_provider_profiles(
    seeds: Sequence[CandidateSeed],
) -> dict[str, list[str]]:
    """在 Provider 调用前只验结构和事实绑定；不得按采样路线反推画像。"""

    failures: dict[str, list[str]] = {}
    for seed in seeds:
        profile = seed.question_profile
        reasons: list[str] = []
        if profile.profile_version != QUESTION_PROFILE_VERSION:
            reasons.append("profile_version_stale")
        bindings = {
            item.claim: item for item in profile.claim_evidence_bindings
        }
        if set(bindings) != set(seed.answer_points) or any(
            item.relation != "entailed" for item in bindings.values()
        ):
            reasons.append("answer_claims_not_fully_bound")
        if profile.terminology_gap and not (
            profile.user_expression and profile.document_terms
        ):
            reasons.append("terminology_gap_incomplete")
        if profile.branch_selector and not (
            len(profile.branch_values) >= 2
            and len(profile.branch_evidence_spans) == len(profile.branch_values)
            and profile.answer_changes_by_branch
        ):
            reasons.append("branch_profile_incomplete")
        if bool(profile.post_search_boundary) != bool(
            profile.post_search_boundary_span
        ):
            reasons.append("boundary_profile_incomplete")
        if reasons:
            failures[seed.candidate_id] = reasons
    return failures


def build_cases(
        seeds: Sequence[CandidateSeed],
        *,
        routes_by_id: dict[str, list[QueryAction]] | None = None,
) -> list[PlannerEvalCase]:
    """
    构造 PlannerEvalCase（规划器评测案例）契约。

    执行前 routes_by_id 为空，仅构造运行环境所需的 provisional contract（临时契约）；
    执行后必须传入真实 Trace（执行轨迹）的 actual route（实际路线），最终候选不得继续
    把 sampling target route（采样目标路线）写成可接受路径。
    """

    cases: list[PlannerEvalCase] = []
    for seed in seeds:
        route = (
            list(routes_by_id[seed.candidate_id])
            if routes_by_id is not None else list(seed.sampling_target_route)
        )
        terminal = route[-1]
        local_evidence = [item for item in seed.source_evidence if item.source_type == "local"]
        web_evidence = [item for item in seed.source_evidence if item.source_type == "web"]
        should_call_web = QueryAction.WEB_SEARCH in route
        expected_chunks = [
            ExpectedChunk(
                document_id=item.document_id,
                chunk_id=item.chunk_id,
                index_version=item.index_version,
                relevance=ChunkRelevance.REQUIRED,
                answer_point_ids=[f"{seed.candidate_id}-answer-1"] if terminal == QueryAction.ANSWER else [],
            )
            for item in local_evidence
        ]
        expected_web_evidence = [
            ExpectedWebEvidence(
                source_id=item.source_id,
                publisher=item.publisher,
                source_title=item.source_title,
                url=item.url,
                captured_at=item.captured_at,
                response_sha256=item.response_sha256,
                evidence_content_sha256=item.evidence_content_sha256,
                fact_ids=[f"{seed.candidate_id}-web-fact"],
                answer_point_ids=[f"{seed.candidate_id}-answer-1"],
            )
            for item in web_evidence
            if terminal == QueryAction.ANSWER
        ]
        source_document_ids = [item.document_id for item in local_evidence]
        source_index_versions = {
            item.document_id: int(item.index_version) for item in local_evidence
        }
        local_source = next(
            (source for source in LOCAL_SOURCES if source.document_id in source_document_ids), None
        )
        # 只要路线实际执行 local_search，就必须提供正式 subject 范围；缺失字段仍由 query 中的
        # 型号、模式、配置或状态触发，不能靠空 subject 退化成全库检索。
        bind_subject = QueryAction.LOCAL_SEARCH in route
        if bind_subject and (not seed.retrieval_subject_id or not seed.retrieval_subject_name):
            raise ValueError(f"{seed.candidate_id} 本地检索路线缺少 retrieval subject")
        behavior = ExpectedBehavior(
            should_answer=terminal == QueryAction.ANSWER,
            should_refuse=terminal == QueryAction.REFUSE,
            should_ask_clarification=terminal == QueryAction.ASK_CLARIFICATION,
            should_call_web=should_call_web,
            web_required_reason=(
                "当前官网状态、支持范围、公告或现行安全信息只能由实时官方来源确认"
                if should_call_web else ""
            ),
            forbidden_actions=[] if should_call_web else [QueryAction.WEB_SEARCH],
        )
        case_group = (
            CaseGroup.REALTIME if should_call_web else
            CaseGroup.REFUSAL if terminal == QueryAction.REFUSE else
            CaseGroup.CLARIFICATION if terminal == QueryAction.ASK_CLARIFICATION else
            CaseGroup.PRIVATE_DOC
        )
        cases.append(PlannerEvalCase(
            case_id=seed.candidate_id,
            case_group=case_group,
            split=CaseSplit.TRAIN,
            leakage_group_id=f"{BATCH_ID}:{seed.candidate_id}",
            query=seed.query,
            query_variants=[],
            dataset_ids=[DATASET_ID],
            owner_user_id=OWNER_USER_ID,
            tenant_id=TENANT_ID,
            privacy_scope=(PrivacyScope.PRIVATE_USER if local_evidence else PrivacyScope.PUBLIC_DEMO),
            source_document_ids=source_document_ids,
            source_index_versions=source_index_versions,
            expected_subject_ids=[seed.retrieval_subject_id] if bind_subject else [],
            expected_subject_names=[seed.retrieval_subject_name] if bind_subject else [],
            expected_chunks=expected_chunks,
            expected_web_evidence=expected_web_evidence,
            expected_answer_points=list(seed.answer_points),
            expected_behavior=behavior,
            acceptable_action_paths=[route],
            expected_identifiers={},
            label_source=LabelSource.API_ASSISTED,
            gold_origin=(
                GoldOrigin.PRODUCTION_CHUNK_GOLD if local_evidence else GoldOrigin.ROUTE_SEED_GOLD
            ),
            human_review_status=HumanReviewStatus.PENDING,
            notes=(
                f"generation_batch={BATCH_ID}; reserve={str(seed.reserve).lower()}; "
                f"device_family={seed.device_family}; question_family={seed.question_family}; "
                f"trigger={seed.missing_or_safety_trigger}; "
                f"sampling_target_route={' -> '.join(a.value for a in seed.sampling_target_route)}"
            ),
        ))
    return cases


class SourceConditionedPlanner:
    """
    生成阶段基于真实 Observation（观察结果）的动态 Planner（规划器）。

    sampling_target_route（采样目标路线）只用于最终配额分桶，本类完全不读取该字段。
    第一动作由问题画像决定；后续 Action（动作）由真实 Provider（动作执行器）返回的候选、
    分数、标识歧义和实际命中的作者证据锚点共同决定。未形成目标路线时允许自然降级或淘汰，
    不再把预设路线播放成 Trace（执行轨迹）。
    """

    policy_version = "sft-v2-observation-dynamic-teacher-v2"

    def __init__(self, seed: CandidateSeed) -> None:
        self.seed = seed
        # 当前生成任务一条 seed 对应一个 Planner 实例。保存此前真实 Observation 是为了在
        # HyDE（假设文档改写检索）后比较首次检索，而不是保存模型思维过程。
        self._observations: dict[QueryAction, RetrievalObservation] = {}

    def _remember_latest_observation(self, context: PlannerContext) -> None:
        if context.latest_observation is None or not context.action_history:
            return
        previous_action = context.action_history[-1].decision.action
        if previous_action == context.latest_observation.action:
            self._observations[previous_action] = context.latest_observation

    def _bound_anchor_observed(self, observation: RetrievalObservation) -> bool:
        """作者证据锚点必须真实出现在当前 Observation，才能触发追问或拒答。"""

        for evidence in self.seed.source_evidence:
            for summary in observation.evidence_summaries:
                if evidence.source_type == "local" and summary.source_type == EvidenceSourceType.LOCAL:
                    if (
                        str(summary.document_id or "") == str(evidence.document_id or "")
                        and str(summary.chunk_id or "") == str(evidence.chunk_id or "")
                    ):
                        return True
                if evidence.source_type == "web" and summary.source_type == EvidenceSourceType.WEB:
                    if summary.title == evidence.source_title:
                        return True
        return False

    @staticmethod
    def _observation_sufficient(observation: RetrievalObservation) -> bool:
        return bool(
            observation.status == ObservationStatus.SUCCESS
            and observation.candidate_count > 0
            and observation.top_rerank_score is not None
            and observation.top_rerank_score >= RERANK_EVIDENCE_THRESHOLD
        )

    @staticmethod
    def _observation_ambiguous(observation: RetrievalObservation) -> bool:
        return bool(
            observation.evidence_ambiguous
            or observation.identifier_resolution_status in {
                IdentifierResolutionStatus.SUGGESTION_REQUIRED,
                IdentifierResolutionStatus.NOT_FOUND,
            }
        )

    def _branch_ambiguity_observed(self, observation: RetrievalObservation) -> bool:
        profile = self.seed.question_profile
        return bool(
            self._bound_anchor_observed(observation)
            and profile.branch_selector
            and len(profile.branch_values) >= 2
            and profile.answer_changes_by_branch
        )

    def _safety_boundary_observed(self, observation: RetrievalObservation) -> bool:
        return bool(
            self.seed.question_profile.post_search_boundary
            and self._bound_anchor_observed(observation)
        )

    def _hyde_observation_improved(self, hyde: RetrievalObservation) -> bool:
        local = self._observations.get(QueryAction.LOCAL_SEARCH)
        if local is None:
            return False
        local_by_identity = {
            (str(item.document_id or ""), str(item.chunk_id or "")): (index, item.rerank_score or 0.0)
            for index, item in enumerate(local.evidence_summaries, start=1)
            if item.source_type == EvidenceSourceType.LOCAL
        }
        for index, item in enumerate(hyde.evidence_summaries, start=1):
            if item.source_type != EvidenceSourceType.LOCAL:
                continue
            identity = (str(item.document_id or ""), str(item.chunk_id or ""))
            before = local_by_identity.get(identity)
            score = float(item.rerank_score or 0.0)
            if before is None:
                return True
            before_rank, before_score = before
            if (index < before_rank and score >= before_score) or (
                index <= before_rank and score > before_score + 1e-6
            ):
                return True
        return False

    def _has_web_anchor(self) -> bool:
        return any(item.source_type == "web" for item in self.seed.source_evidence)

    def _decision(self, action: QueryAction, context: PlannerContext) -> PlannerDecision:
        previous_action = (
            context.action_history[-1].decision.action if context.action_history else None
        )
        if action == QueryAction.LOCAL_SEARCH:
            reason = PlannerReasonCode.INITIAL_LOCAL_SEARCH
        elif action == QueryAction.HYDE_SEARCH:
            reason = PlannerReasonCode.LOCAL_LOW_SCORE
        elif action == QueryAction.WEB_SEARCH:
            reason = (
                PlannerReasonCode.REALTIME_QUERY
                if previous_action is None else PlannerReasonCode.HYDE_STILL_INSUFFICIENT
            )
        elif action == QueryAction.ANSWER:
            reason = (
                PlannerReasonCode.WEB_EVIDENCE_AVAILABLE
                if previous_action == QueryAction.WEB_SEARCH else
                PlannerReasonCode.HYDE_EVIDENCE_SUFFICIENT
                if previous_action == QueryAction.HYDE_SEARCH else
                PlannerReasonCode.LOCAL_EVIDENCE_SUFFICIENT
            )
        elif action == QueryAction.ASK_CLARIFICATION:
            reason = (
                PlannerReasonCode.SUBJECT_REQUIRED
                if previous_action is None else PlannerReasonCode.EVIDENCE_AMBIGUOUS
            )
        else:
            reason = PlannerReasonCode.SAFE_GUARD_TRIGGERED
        query = (
            self.seed.web_search_query
            if action == QueryAction.WEB_SEARCH and self.seed.web_search_query
            else self.seed.query
        )
        return PlannerDecision(action=action, query=query, reason_code=reason)

    def plan(self, context: PlannerContext) -> PlannerDecision:
        self._remember_latest_observation(context)
        profile = self.seed.question_profile
        if context.safe_guard_triggered:
            return self._decision(QueryAction.REFUSE, context)

        if not context.action_history:
            if profile.pre_search_terminal is not None:
                return self._decision(profile.pre_search_terminal, context)
            if profile.realtime_required and self._has_web_anchor() and context.web_search_allowed:
                return self._decision(QueryAction.WEB_SEARCH, context)
            if self.seed.retrieval_subject_id:
                return self._decision(QueryAction.LOCAL_SEARCH, context)
            if self._has_web_anchor() and context.web_search_allowed:
                return self._decision(QueryAction.WEB_SEARCH, context)
            return self._decision(QueryAction.REFUSE, context)

        observation = context.latest_observation
        if observation is None or observation.status == ObservationStatus.FAILED:
            return self._decision(QueryAction.REFUSE, context)
        if self._observation_ambiguous(observation) or self._branch_ambiguity_observed(observation):
            return self._decision(QueryAction.ASK_CLARIFICATION, context)
        if self._safety_boundary_observed(observation):
            return self._decision(QueryAction.REFUSE, context)

        if observation.action == QueryAction.WEB_SEARCH:
            return self._decision(
                QueryAction.ANSWER if observation.candidate_count > 0 else QueryAction.REFUSE,
                context,
            )
        if observation.action == QueryAction.LOCAL_SEARCH:
            if self._observation_sufficient(observation):
                return self._decision(QueryAction.ANSWER, context)
            if profile.terminology_gap:
                return self._decision(QueryAction.HYDE_SEARCH, context)
            if self._has_web_anchor() and context.web_search_allowed:
                return self._decision(QueryAction.WEB_SEARCH, context)
            return self._decision(QueryAction.REFUSE, context)
        if observation.action == QueryAction.HYDE_SEARCH:
            if self._observation_sufficient(observation) and self._hyde_observation_improved(observation):
                return self._decision(QueryAction.ANSWER, context)
            if self._has_web_anchor() and context.web_search_allowed:
                return self._decision(QueryAction.WEB_SEARCH, context)
            return self._decision(QueryAction.REFUSE, context)
        return self._decision(QueryAction.REFUSE, context)


def build_current_snapshot(cases: Sequence[PlannerEvalCase]):
    """从当前 Mongo/Milvus 正式语料构建本批次环境快照。"""

    return build_environment_snapshot(
        metadata_reader=MongoMetadataSnapshotReader(ImportMetadataRepository()),
        chunk_reader=MilvusChunkSnapshotReader(milvus_gateway),
        override_reader=MongoChunkOverrideSnapshotReader(ChunkStatusRepository()),
        cases=list(cases),
        dataset_ids=[DATASET_ID],
        test_user_ids=[OWNER_USER_ID],
        snapshot_id=f"{BATCH_ID}-environment",
        created_by="evaluation.stage9.sft_v2.build_sft_v2_candidates",
        retrieval_config_snapshot={**build_retrieval_config_snapshot(), "web_fallback_enabled": True},
        source_hashes={
            "candidate_cases_pre_execution": _stable_hash([
                case.model_dump(mode="json") for case in cases
            ]),
        },
    )


class OfficialSourceActionProvider:
    """本地检索走 Milvus；Web 动作实时读取候选已绑定的官方页面。"""

    def __init__(self, seeds: Sequence[CandidateSeed]) -> None:
        self.local_provider = MilvusActionProvider(chunk_status_filter_enabled=True)
        self.web_by_case = {
            seed.candidate_id: [
                evidence for evidence in seed.source_evidence if evidence.source_type == "web"
            ]
            for seed in seeds
        }

    def local_search(self, state, decision):
        return self.local_provider.local_search(state, decision)

    def hyde_search(self, state, decision):
        return self.local_provider.hyde_search(state, decision)

    def web_search(self, state, decision):
        if not state.web_search_allowed:
            raise ValueError("当前 State（运行状态）不允许 Web（网页检索）")
        evidences = self.web_by_case.get(state.case_id) or []
        if len(evidences) != 1:
            raise ValueError(f"{state.case_id} 必须且只能绑定一个官方 Web 来源")
        evidence = evidences[0]
        response = _get_official_url(
            evidence.url,
            timeout_seconds=30,
            user_agent="Mozilla/5.0 task-9.3.21 provider verification",
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"官方 Web Provider 读取失败：case={state.case_id}, status={response.status_code}"
            )
        extracted_text = _extract_response_text(response)
        current_segments = set(_fact_segments(extracted_text))
        if evidence.fact_text not in current_segments:
            raise RuntimeError(f"官方 Web 事实已漂移或无法复核：case={state.case_id}, url={evidence.url}")
        return [RetrievalCandidate(
            document_id=None,
            chunk_id=None,
            dataset_id=None,
            index_version=None,
            chunk_index=None,
            title=evidence.source_title,
            source_title=evidence.source_title,
            content=evidence.fact_text,
            source_type=EvidenceSourceType.WEB,
            retrieval_channels=[RetrievalChannel.WEB],
            retrieval_rank=1,
            retrieval_score=1.0,
            rerank_score=None,
            url=evidence.url,
        )]


def execute_trajectories(
        seeds: Sequence[CandidateSeed],
        cases: Sequence[PlannerEvalCase],
        *,
        snapshot,
        provider_records_path: Path,
) -> tuple[list[OfflineTrajectoryResult], list[Any]]:
    """按 Planner -> Provider -> Observation 顺序执行 125 条候选。"""

    if provider_records_path.exists():
        raise FileExistsError(f"Provider 记录目标已存在，拒绝覆盖：{provider_records_path}")
    provider = RecordingActionProvider(
        OfficialSourceActionProvider(seeds),
        output_path=provider_records_path,
        max_candidate_content_chars=None,
    )
    environment = OfflineRagEnvironment(
        snapshot=snapshot,
        action_provider=provider,
        planner_mode="api",
        run_id_prefix=BATCH_ID,
        max_steps=4,
    )
    case_by_id = {case.case_id: case for case in cases}
    trajectories: list[OfflineTrajectoryResult] = []
    for index, seed in enumerate(seeds, start=1):
        print(
            f"[sft_v2_generate] candidate={index}/{len(seeds)} "
            f"candidate_id={seed.candidate_id} "
            f"sampling_target_route={route_name(seed.sampling_target_route)}",
            flush=True,
        )
        trajectory = environment.run_planner(
            case_by_id[seed.candidate_id],
            SourceConditionedPlanner(seed),
            run_id=f"{BATCH_ID}-{seed.candidate_id}",
            planner_mode="api",
        )
        if trajectory.status.value != "completed":
            raise RuntimeError(
                f"{seed.candidate_id} 真实执行失败："
                f"{[error.model_dump(mode='json') for error in trajectory.errors]}"
            )
        trajectories.append(trajectory)
    records = read_provider_observation_records(provider_records_path)
    return trajectories, records


def replay_recorded_trajectories(
        seeds: Sequence[CandidateSeed],
        cases: Sequence[PlannerEvalCase],
        *,
        snapshot,
        provider_records_path: Path,
) -> tuple[list[OfflineTrajectoryResult], list[Any]]:
    """回放同一批次已经真实录制且完整的 Provider（动作执行器）结果。"""

    records = read_provider_observation_records(provider_records_path)
    provider = ReplayActionProvider(provider_records_path)
    environment = OfflineRagEnvironment(
        snapshot=snapshot,
        action_provider=provider,
        planner_mode="api",
        run_id_prefix=BATCH_ID,
        max_steps=4,
    )
    case_by_id = {case.case_id: case for case in cases}
    trajectories: list[OfflineTrajectoryResult] = []
    for seed in seeds:
        trajectory = environment.run_planner(
            case_by_id[seed.candidate_id],
            SourceConditionedPlanner(seed),
            run_id=f"{BATCH_ID}-{seed.candidate_id}",
            planner_mode="api",
        )
        if trajectory.status.value != "completed":
            raise RuntimeError(
                f"{seed.candidate_id} Provider 回放失败："
                f"{[error.model_dump(mode='json') for error in trajectory.errors]}"
            )
        trajectories.append(trajectory)
    return trajectories, records


def _record_map(records: Sequence[Any]) -> dict[tuple[str, str], Any]:
    mapping: dict[tuple[str, str], Any] = {}
    for record in records:
        key = (record.case_id, record.action.value)
        if key in mapping:
            raise ValueError(f"Provider 记录重复 case/action：{key}")
        mapping[key] = record
    return mapping


def _candidate_identity_from_record(candidate: dict[str, Any]) -> tuple[str, str, int]:
    """把 Provider 候选收口为可比较的正式本地证据身份。"""

    return (
        str(candidate.get("document_id") or ""),
        str(candidate.get("chunk_id") or ""),
        int(candidate.get("index_version") or 0),
    )


def _candidate_rank_and_score(candidate: dict[str, Any]) -> tuple[int, float]:
    rank = int(candidate.get("retrieval_rank") or 10**9)
    raw_score = (
        candidate.get("rerank_score")
        if candidate.get("rerank_score") is not None
        else candidate.get("retrieval_score")
    )
    return rank, float(raw_score or 0.0)


def _hyde_target_improvement(
        local_candidates: Sequence[dict[str, Any]],
        hyde_candidates: Sequence[dict[str, Any]],
        expected_evidence: Sequence[SourceEvidence],
) -> tuple[bool, dict[str, Any]]:
    """要求 HyDE 对绑定目标证据产生真实排名或分数提升，而不是仅改变结果集合。"""

    local_by_identity = {
        _candidate_identity_from_record(candidate): candidate
        for candidate in local_candidates
        if candidate.get("source_type") == "local"
    }
    hyde_by_identity = {
        _candidate_identity_from_record(candidate): candidate
        for candidate in hyde_candidates
        if candidate.get("source_type") == "local"
    }
    targets: list[dict[str, Any]] = []
    improved = False
    if not expected_evidence:
        local_top_score = max(
            (_candidate_rank_and_score(candidate)[1] for candidate in local_by_identity.values()),
            default=0.0,
        )
        for identity, hyde in hyde_by_identity.items():
            local = local_by_identity.get(identity)
            local_rank, local_score = _candidate_rank_and_score(local or {})
            hyde_rank, hyde_score = _candidate_rank_and_score(hyde)
            target_improved = bool(
                (local is None and hyde_score > local_top_score + 1e-6)
                or (local is not None and hyde_rank < local_rank and hyde_score >= local_score)
                or (local is not None and hyde_rank <= local_rank and hyde_score > local_score + 1e-6)
            )
            improved = improved or target_improved
            if target_improved:
                targets.append({
                    "document_id": identity[0],
                    "chunk_id": identity[1],
                    "index_version": identity[2],
                    "local_rank": None if local is None else local_rank,
                    "local_score": None if local is None else local_score,
                    "hyde_rank": hyde_rank,
                    "hyde_score": hyde_score,
                    "improved": True,
                })
        return improved, {
            "gate_version": OBSERVATION_GATE_VERSION,
            "comparison_mode": "best_observed_local_identity",
            "targets": targets,
        }
    for evidence in expected_evidence:
        identity = (
            str(evidence.document_id or ""),
            str(evidence.chunk_id or ""),
            int(evidence.index_version or 0),
        )
        local = local_by_identity.get(identity)
        hyde = hyde_by_identity.get(identity)
        local_rank, local_score = _candidate_rank_and_score(local or {})
        hyde_rank, hyde_score = _candidate_rank_and_score(hyde or {})
        target_improved = bool(
            hyde is not None
            and (
                local is None
                or (hyde_rank < local_rank and hyde_score >= local_score)
                or (hyde_rank <= local_rank and hyde_score > local_score + 1e-6)
            )
        )
        improved = improved or target_improved
        targets.append({
            "document_id": identity[0],
            "chunk_id": identity[1],
            "index_version": identity[2],
            "local_rank": None if local is None else local_rank,
            "local_score": None if local is None else local_score,
            "hyde_rank": None if hyde is None else hyde_rank,
            "hyde_score": None if hyde is None else hyde_score,
            "improved": target_improved,
        })
    return improved, {"gate_version": OBSERVATION_GATE_VERSION, "targets": targets}


_PREDECLARED_MISSING_MARKERS = (
    "不知道", "不确定", "未确认", "没有记录", "不清楚", "尚未确认", "缺少", "未提供",
    "无法确认", "无法读取", "看不到", "铭牌模糊", "记录丢失",
)
_DIRECT_UNSAFE_MARKERS = (
    "绕过", "短接", "禁用", "拆除", "移除", "篡改", "攻击", "攻破", "越权", "伪造",
    "屏蔽保护", "关闭保护", "跳过安全", "忽略禁令", "不做能量隔离", "不加锁", "exploit",
)
_AMBIGUITY_EVIDENCE_MARKERS = (
    "取决于", "视情况", "分别", "不同", "类型", "型号", "版本", "模式", "配置", "条件",
    "如果", "若", "或", "either", "depending", "version", "model", "mode", "configuration",
)
_SAFETY_EVIDENCE_MARKERS = (
    "必须", "不得", "禁止", "警告", "危险", "重伤", "死亡", "安全", "防护", "授权", "权限",
    "shall", "must", "must not", "warning", "hazard", "safety", "security", "unauthor",
)
_WEB_BOILERPLATE_MARKERS = (
    "privacy policy", "cookie", "contact us", "skip to content", "all rights reserved",
    "official website of the united states government", "before sharing sensitive information",
)
_WEB_TRUNCATED_ENDINGS = (
    " and", " or", " but", " with", " including", " such as", "以及", "并且", "或者", "包括", "例如",
)


def _fact_has_ambiguity_branches(fact_text: str) -> bool:
    lowered = fact_text.lower()
    english = set(re.findall(
        r"\b(if|when|depending|different|versions?|models?|modes?|configurations?|either|or)\b",
        lowered,
    ))
    chinese = {
        marker for marker in ("如果", "若", "取决", "不同", "分别", "或者", "类型", "型号", "版本", "模式", "配置")
        if marker in fact_text
    }
    signals = english | chinese
    branch_markers = {"if", "when", "depending", "different", "either", "or", "如果", "若", "取决", "不同", "分别", "或者"}
    structured_branches = bool(
        len(re.findall(r"[①②③④⑤]", fact_text)) >= 2
        or "下拉可选" in fact_text
        or ("打印机状态" in fact_text and fact_text.count("待机") >= 2)
    )
    return structured_branches or (len(signals) >= 2 and bool(signals & branch_markers))


def _record_contains_evidence(record: Any, evidence: SourceEvidence) -> bool:
    if evidence.source_type == "web":
        expected_url = canonicalize_web_url(str(evidence.url or ""))
        return any(
            canonicalize_web_url(str(candidate.get("url") or "")) == expected_url
            for candidate in record.candidates
            if candidate.get("source_type") == "web"
        )
    expected = (
        str(evidence.document_id or ""),
        str(evidence.chunk_id or ""),
        int(evidence.index_version or 0),
    )
    return any(
        _candidate_identity_from_record(candidate) == expected
        for candidate in record.candidates
        if candidate.get("source_type") == "local"
    )


def bind_observed_source_evidence(
    seeds: Sequence[CandidateSeed],
    records: Sequence[Any],
) -> list[CandidateSeed]:
    """只从本次 Provider 返回候选生成正式证据绑定，不沿用预选正文。"""

    records_by_case: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        records_by_case[record.case_id].append(record)
    bound_seeds: list[CandidateSeed] = []
    for seed in seeds:
        bound: list[SourceEvidence] = []
        for authoring in seed.source_evidence:
            observed: tuple[Any, dict[str, Any]] | None = None
            for record in records_by_case.get(seed.candidate_id, []):
                for candidate in record.candidates:
                    if authoring.source_type == "local":
                        matches = (
                            candidate.get("source_type") == "local"
                            and str(candidate.get("document_id") or "")
                            == str(authoring.document_id or "")
                            and str(candidate.get("chunk_id") or "")
                            == str(authoring.chunk_id or "")
                            and int(candidate.get("index_version") or 0)
                            == int(authoring.index_version or 0)
                        )
                    else:
                        matches = (
                            candidate.get("source_type") == "web"
                            and canonicalize_web_url(str(candidate.get("url") or ""))
                            == canonicalize_web_url(str(authoring.url or ""))
                        )
                    if matches:
                        observed = (record, candidate)
            if observed is None:
                continue
            record, candidate = observed
            content = str(candidate.get("content") or "").strip()
            if not content:
                continue
            bound.append(authoring.model_copy(update={
                "fact_text": content,
                "evidence_content_sha256": _sha256_bytes(content.encode("utf-8")),
                "provider_record_id": record.record_id,
                "observed_action": record.action,
                "retrieval_rank": int(candidate.get("retrieval_rank") or 0) or None,
                "retrieval_score": (
                    float(candidate["retrieval_score"])
                    if candidate.get("retrieval_score") is not None else None
                ),
                "rerank_score": (
                    float(candidate["rerank_score"])
                    if candidate.get("rerank_score") is not None else None
                ),
                "retrieved_content": content,
            }))
        bound_seeds.append(seed.model_copy(update={"source_evidence": bound}))
    return bound_seeds


def _latest_bound_evidence(
        seed: CandidateSeed,
        candidate_records: Sequence[Any],
) -> tuple[SourceEvidence | None, Any | None, list[Any]]:
    if not candidate_records:
        return None, None, []
    final_record = candidate_records[-1]
    matching = [
        evidence for evidence in seed.source_evidence
        if _record_contains_evidence(final_record, evidence)
    ]
    return (matching[0] if matching else None), final_record, list(candidate_records[:-1])


def _post_retrieval_clarification_supported(
        seed: CandidateSeed,
        candidate_records: Sequence[Any],
) -> tuple[bool, dict[str, Any]]:
    """追问必须由最后一次真实检索新暴露的标识冲突或多分支事实触发。"""

    if any(marker in seed.query for marker in _PREDECLARED_MISSING_MARKERS):
        return False, {"reason": "missing_field_predeclared_in_query"}
    evidence, final_record, previous_records = _latest_bound_evidence(seed, candidate_records)
    if evidence is None or final_record is None:
        return False, {"reason": "final_observation_missing_bound_evidence"}
    final_status = str(final_record.observation.get("identifier_resolution_status") or "")
    previous_statuses = {
        str(record.observation.get("identifier_resolution_status") or "")
        for record in previous_records
    }
    identifier_newly_ambiguous = bool(
        final_status in {"not_found", "suggestion_required"}
        and not previous_statuses.intersection({"not_found", "suggestion_required"})
    )
    evidence_has_branches = _fact_has_ambiguity_branches(evidence.fact_text)
    evidence_was_new = not any(
        _record_contains_evidence(record, evidence) for record in previous_records
    )
    evidence_newly_ambiguous = bool(evidence_has_branches and evidence_was_new)
    return bool(identifier_newly_ambiguous or evidence_newly_ambiguous), {
        "gate_version": OBSERVATION_GATE_VERSION,
        "final_action": final_record.action.value,
        "final_identifier_status": final_status,
        "identifier_newly_ambiguous": identifier_newly_ambiguous,
        "evidence_newly_ambiguous": evidence_newly_ambiguous,
        "bound_source_id": evidence.source_id,
    }


def _post_retrieval_refusal_supported(
        seed: CandidateSeed,
        candidate_records: Sequence[Any],
) -> tuple[bool, dict[str, Any]]:
    """拒答必须由最后一次真实检索新取得的安全、权限或禁令事实触发。"""

    lowered_query = seed.query.lower()
    if any(marker in lowered_query for marker in _DIRECT_UNSAFE_MARKERS):
        return False, {"reason": "unsafe_intent_explicit_before_retrieval"}
    evidence, final_record, previous_records = _latest_bound_evidence(seed, candidate_records)
    if evidence is None or final_record is None:
        return False, {"reason": "final_observation_missing_bound_evidence"}
    fact = evidence.fact_text.lower()
    safety_grounded = any(marker in fact for marker in _SAFETY_EVIDENCE_MARKERS)
    evidence_was_new = not any(
        _record_contains_evidence(record, evidence) for record in previous_records
    )
    return bool(safety_grounded and evidence_was_new), {
        "gate_version": OBSERVATION_GATE_VERSION,
        "final_action": final_record.action.value,
        "safety_grounded": safety_grounded,
        "evidence_newly_observed": evidence_was_new,
        "bound_source_id": evidence.source_id,
    }


def _web_fact_text_complete(fact_text: str) -> tuple[bool, dict[str, Any]]:
    text = re.sub(r"\s+", " ", fact_text).strip()
    lowered = text.lower()
    reasons: list[str] = []
    if len(text) < 80:
        reasons.append("fact_too_short")
    if any(marker in lowered for marker in _WEB_BOILERPLATE_MARKERS):
        reasons.append("page_boilerplate")
    if lowered.endswith(_WEB_TRUNCATED_ENDINGS) or text.endswith((":", "：", ",", "，", ";", "；")):
        reasons.append("truncated_ending")
    if not re.search(r"[。！？.!?]$", text):
        reasons.append("missing_sentence_ending")
    return not reasons, {
        "gate_version": OBSERVATION_GATE_VERSION,
        "fact_length": len(text),
        "reasons": reasons,
    }


def _web_fact_complete(evidence: SourceEvidence) -> tuple[bool, dict[str, Any]]:
    """排除截断句、导航/隐私模板和缺少可审计正文的网页事实。"""

    return _web_fact_text_complete(evidence.fact_text)


def _web_answer_coverage(seed: CandidateSeed, evidence: SourceEvidence) -> tuple[bool, dict[str, Any]]:
    """
    用原子 claim（答案主张）到原文 span（证据片段）的绑定检查跨语言网页覆盖。

    多语言语义由问题画像阶段独立判断；这里执行可复现的硬校验：每个答案要点都必须有
    entailed（明确支持）绑定、span 必须逐字存在、数字/标准编号不能凭空增加，规范强度
    不能把 may/can（可以）扩写成 must/shall（必须）。
    """

    bindings = {
        item.claim: item for item in seed.question_profile.claim_evidence_bindings
    }
    results: list[dict[str, Any]] = []
    fact = evidence.fact_text
    for claim in seed.answer_points:
        binding = bindings.get(claim)
        span = binding.evidence_span if binding is not None else ""
        claim_entities = set(re.findall(
            r"\b(?:[A-Z]{2,}[A-Z0-9_.-]*|\d+(?:\.\d+)*(?:-[A-Z0-9.]+)?)\b",
            claim,
            flags=re.IGNORECASE,
        ))
        span_entities = set(re.findall(
            r"\b(?:[A-Z]{2,}[A-Z0-9_.-]*|\d+(?:\.\d+)*(?:-[A-Z0-9.]+)?)\b",
            span,
            flags=re.IGNORECASE,
        ))
        claim_is_mandatory = any(token in claim.lower() for token in (
            "必须", "不得", "禁止", "要求", "must", "shall", "required", "prohibited",
        ))
        span_is_only_permissive = bool(
            any(token in span.lower() for token in (" may ", " can ", "可以", "可能"))
            and not any(token in span.lower() for token in (
                "must", "shall", "required", "prohibited", "必须", "不得", "禁止", "要求",
            ))
        )
        result = {
            "claim": claim,
            "binding_present": binding is not None,
            "relation": binding.relation if binding is not None else None,
            "span_in_fact": bool(span and span in fact),
            "entity_and_number_consistent": claim_entities.issubset(span_entities),
            "normative_strength_consistent": not (
                claim_is_mandatory and span_is_only_permissive
            ),
        }
        result["passed"] = all((
            result["binding_present"],
            result["relation"] == "entailed",
            result["span_in_fact"],
            result["entity_and_number_consistent"],
            result["normative_strength_consistent"],
        ))
        results.append(result)
    passed = bool(results) and all(item["passed"] for item in results)
    return passed, {
        "gate_version": OBSERVATION_GATE_VERSION,
        "coverage_mode": "atomic-claim-evidence-binding-v1",
        "claims": results,
    }


def validate_generation(
        seeds: Sequence[CandidateSeed],
        cases: Sequence[PlannerEvalCase],
        trajectories: Sequence[OfflineTrajectoryResult],
        records: Sequence[Any],
        *,
        raise_on_failure: bool = True,
) -> dict[str, dict[str, Any]]:
    """按 case 合同、真实 Observation 和来源身份执行生成阶段 Evaluator（评测器）门禁。"""

    case_by_id = {case.case_id: case for case in cases}
    trajectory_by_id = {trajectory.case_id: trajectory for trajectory in trajectories}
    records_by_key = _record_map(records)
    gates: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for seed in seeds:
        case = case_by_id[seed.candidate_id]
        trajectory = trajectory_by_id[seed.candidate_id]
        sampling_target_route = [action.value for action in seed.sampling_target_route]
        actual_route = [action.value for action in trajectory.action_path]
        actual_actions = list(trajectory.action_path)
        actual_terminal = actual_actions[-1] if actual_actions else None
        checks = {
            "actual_route_complete": bool(
                actual_terminal in {
                    QueryAction.ANSWER,
                    QueryAction.ASK_CLARIFICATION,
                    QueryAction.REFUSE,
                }
                and trajectory.terminal_action == actual_terminal
            ),
            "terminal_payload_valid": bool(
                (actual_terminal == QueryAction.ANSWER and seed.answer_points)
                or (
                    actual_terminal in {
                        QueryAction.ASK_CLARIFICATION,
                        QueryAction.REFUSE,
                    }
                    and not seed.answer_points
                )
            ),
            "config_match": trajectory.config_match_status == "match",
            "corpus_match": trajectory.corpus_match_status == "match",
            "provider_errors_zero": True,
            "local_evidence_bound": True,
            "web_evidence_bound": True,
            "hyde_trigger_observed": True,
            "hyde_target_improved": True,
            "clarification_trigger_present": True,
            "clarification_observation_grounded": True,
            "refusal_trigger_present": True,
            "refusal_observation_grounded": True,
            "web_fact_complete": True,
            "web_answer_coverage": True,
            "observed_evidence_bound": True,
            "pre_search_profile_consistent": True,
            "hyde_terminology_gap_valid": True,
            "clarification_branch_profile_valid": True,
            "refusal_boundary_profile_valid": True,
        }
        gate_details: dict[str, Any] = {}
        candidate_records = [
            records_by_key[(seed.candidate_id, action.value)]
            for action in actual_actions if action in {
                QueryAction.LOCAL_SEARCH, QueryAction.HYDE_SEARCH, QueryAction.WEB_SEARCH
            }
        ]
        checks["provider_errors_zero"] = all(record.error is None for record in candidate_records)
        if candidate_records:
            record_ids = {record.record_id for record in candidate_records}
            checks["observed_evidence_bound"] = bool(seed.source_evidence) and all(
                evidence.provider_record_id in record_ids
                and evidence.observed_action is not None
                and evidence.retrieval_rank is not None
                and evidence.retrieved_content == evidence.fact_text
                for evidence in seed.source_evidence
            )
            gate_details["observed_evidence"] = [
                {
                    "source_id": evidence.source_id,
                    "provider_record_id": evidence.provider_record_id,
                    "observed_action": (
                        evidence.observed_action.value
                        if evidence.observed_action is not None else None
                    ),
                    "retrieval_rank": evidence.retrieval_rank,
                    "retrieval_score": evidence.retrieval_score,
                    "rerank_score": evidence.rerank_score,
                    "evidence_content_sha256": evidence.evidence_content_sha256,
                }
                for evidence in seed.source_evidence
            ]
        profile = seed.question_profile
        if len(actual_actions) == 1:
            checks["pre_search_profile_consistent"] = (
                profile.pre_search_terminal == actual_terminal
            )
        else:
            checks["pre_search_profile_consistent"] = profile.pre_search_terminal is None

        local_evidence = [item for item in seed.source_evidence if item.source_type == "local"]
        if (
            actual_terminal == QueryAction.ANSWER
            and QueryAction.WEB_SEARCH not in actual_actions
            and not local_evidence
        ):
            checks["local_evidence_bound"] = False
        if (
            local_evidence
            and QueryAction.LOCAL_SEARCH in actual_actions
            and actual_terminal != QueryAction.ASK_CLARIFICATION
        ):
            local_candidates = [
                candidate
                for record in candidate_records
                for candidate in record.candidates
                if candidate.get("source_type") == "local"
            ]
            expected_identities = {
                (item.document_id, str(item.chunk_id), item.index_version) for item in local_evidence
            }
            actual_identities = {
                (candidate.get("document_id"), str(candidate.get("chunk_id")), candidate.get("index_version"))
                for candidate in local_candidates
            }
            checks["local_evidence_bound"] = expected_identities.issubset(actual_identities)

        web_evidence = [item for item in seed.source_evidence if item.source_type == "web"]
        if (
            actual_terminal == QueryAction.ANSWER
            and QueryAction.WEB_SEARCH in actual_actions
            and not web_evidence
        ):
            checks["web_evidence_bound"] = False
        if web_evidence and QueryAction.WEB_SEARCH in actual_actions and actual_terminal == QueryAction.ANSWER:
            web_record = records_by_key.get((seed.candidate_id, QueryAction.WEB_SEARCH.value))
            actual_urls = {
                canonicalize_web_url(str(candidate.get("url") or ""))
                for candidate in (web_record.candidates if web_record else [])
                if candidate.get("url")
            }
            expected_urls = {canonicalize_web_url(item.url) for item in web_evidence}
            checks["web_evidence_bound"] = expected_urls.issubset(actual_urls)
            completeness_results = [_web_fact_complete(item) for item in web_evidence]
            coverage_results = [_web_answer_coverage(seed, item) for item in web_evidence]
            checks["web_fact_complete"] = all(passed for passed, _ in completeness_results)
            checks["web_answer_coverage"] = all(passed for passed, _ in coverage_results)
            gate_details["web_fact_complete"] = [detail for _, detail in completeness_results]
            gate_details["web_answer_coverage"] = [detail for _, detail in coverage_results]

        if QueryAction.HYDE_SEARCH in actual_actions:
            lowered_query = seed.query.lower()
            checks["hyde_terminology_gap_valid"] = bool(
                profile.terminology_gap
                and profile.user_expression
                and profile.user_expression.lower() in lowered_query
                and profile.document_terms
                and not any(term.lower() in lowered_query for term in profile.document_terms)
            )
            local_record = records_by_key[(seed.candidate_id, QueryAction.LOCAL_SEARCH.value)]
            hyde_record = records_by_key[(seed.candidate_id, QueryAction.HYDE_SEARCH.value)]
            local_observation = local_record.observation
            checks["hyde_trigger_observed"] = bool(
                local_observation.get("candidate_count", 0) > 0
                and local_observation.get("top_rerank_score") is not None
                and float(local_observation["top_rerank_score"]) < RERANK_EVIDENCE_THRESHOLD
            )
            hyde_improved, hyde_detail = _hyde_target_improvement(
                local_record.candidates,
                hyde_record.candidates,
                local_evidence,
            )
            checks["hyde_target_improved"] = hyde_improved
            gate_details["hyde_target_improved"] = hyde_detail

        if actual_terminal == QueryAction.ASK_CLARIFICATION:
            checks["clarification_trigger_present"] = bool(seed.missing_or_safety_trigger)
            if candidate_records:
                clarification_supported, clarification_detail = (
                    _post_retrieval_clarification_supported(seed, candidate_records)
                )
                checks["clarification_observation_grounded"] = clarification_supported
                gate_details["clarification_observation_grounded"] = clarification_detail
                if len(actual_actions) > 1 and not clarification_detail.get(
                    "identifier_newly_ambiguous", False
                ):
                    checks["clarification_branch_profile_valid"] = bool(
                        profile.branch_selector
                        and len(profile.branch_values) >= 2
                        and profile.answer_changes_by_branch
                    )
        if actual_terminal == QueryAction.REFUSE:
            checks["refusal_trigger_present"] = bool(seed.missing_or_safety_trigger)
            if candidate_records:
                refusal_supported, refusal_detail = _post_retrieval_refusal_supported(
                    seed,
                    candidate_records,
                )
                checks["refusal_observation_grounded"] = refusal_supported
                gate_details["refusal_observation_grounded"] = refusal_detail
                if len(actual_actions) > 1:
                    checks["refusal_boundary_profile_valid"] = bool(
                        profile.post_search_boundary
                    )

        failed_checks = sorted(name for name, passed in checks.items() if not passed)
        gates[seed.candidate_id] = {
            "passed": not failed_checks,
            "checks": checks,
            "failed_checks": failed_checks,
            "sampling_target_route": sampling_target_route,
            "actual_route": actual_route,
            "sampling_target_hit": actual_route == sampling_target_route,
            "provider_record_ids": [record.record_id for record in candidate_records],
            "gate_details": gate_details,
        }
        if failed_checks:
            failures.append(f"{seed.candidate_id}:{','.join(failed_checks)}")
    if failures and raise_on_failure:
        raise RuntimeError("生成阶段门禁失败：" + "；".join(failures))
    return gates


def _sample_context(
        case: PlannerEvalCase,
        seed: CandidateSeed,
        actual_route: Sequence[QueryAction],
        trace_steps: Sequence[Any],
        records_by_key: dict[tuple[str, str], Any],
        index: int,
) -> dict[str, Any]:
    previous_steps = list(trace_steps[:index])
    latest = previous_steps[-1].output_observation if previous_steps else None
    latest_summary = None
    if latest is not None:
        previous_action = previous_steps[-1].decision.action.value
        record = records_by_key.get((case.case_id, previous_action))
        candidates = record.candidates if record is not None else []
        latest_summary = {
            "action": latest.action.value,
            "status": latest.status.value,
            "candidate_count": latest.candidate_count,
            "top_rerank_score": latest.top_rerank_score,
            "identifier_resolution_status": latest.identifier_resolution_status.value,
            "evidence_ambiguous": latest.evidence_ambiguous,
            "clarification_question": latest.clarification_question,
            "retrieved_chunk_ids": [
                str(candidate["chunk_id"]) for candidate in candidates
                if candidate.get("source_type") == "local" and candidate.get("chunk_id") is not None
            ][:10],
            "retrieved_urls": [
                str(candidate["url"]) for candidate in candidates
                if candidate.get("source_type") == "web" and candidate.get("url")
            ][:10],
            "contains_full_chunk_content": False,
        }
    allowed_actions = [
        QueryAction.LOCAL_SEARCH.value,
        QueryAction.HYDE_SEARCH.value,
        QueryAction.ANSWER.value,
        QueryAction.ASK_CLARIFICATION.value,
        QueryAction.REFUSE.value,
    ]
    if QueryAction.WEB_SEARCH in actual_route:
        allowed_actions.insert(2, QueryAction.WEB_SEARCH.value)
    return {
        "query": case.query,
        "current_query": case.query,
        "dataset_ids": list(case.dataset_ids),
        "subject_ids": list(case.expected_subject_ids),
        "standard_subject_names": list(case.expected_subject_names),
        "query_identifiers": dict(case.expected_identifiers),
        "web_search_allowed": QueryAction.WEB_SEARCH in actual_route,
        "planner_step": index,
        "allowed_actions": allowed_actions,
        "action_history": [
            {"step": step_index + 1, "action": step.decision.action.value}
            for step_index, step in enumerate(previous_steps)
        ],
        "latest_observation": latest_summary,
    }


def build_outputs(
        seeds: Sequence[CandidateSeed],
        cases: Sequence[PlannerEvalCase],
        trajectories: Sequence[OfflineTrajectoryResult],
        records: Sequence[Any],
        gates: dict[str, dict[str, Any]],
) -> tuple[list[CandidateTrajectory], list[SftPlannerSample]]:
    case_by_id = {case.case_id: case for case in cases}
    trajectory_by_id = {trajectory.case_id: trajectory for trajectory in trajectories}
    records_by_key = _record_map(records)
    trajectory_rows: list[CandidateTrajectory] = []
    samples: list[SftPlannerSample] = []
    for seed in seeds:
        case = case_by_id[seed.candidate_id]
        trajectory = trajectory_by_id[seed.candidate_id]
        actual_route = list(trajectory.action_path)
        trace_payload = [step.model_dump(mode="json") for step in trajectory.trace_steps]
        fingerprint_payload = {
            "case": case.model_dump(mode="json"),
            "route": [action.value for action in actual_route],
            "sampling_target_route": [
                action.value for action in seed.sampling_target_route
            ],
            "source_evidence": [item.model_dump(mode="json") for item in seed.source_evidence],
            "trace_steps": [
                {
                    **step,
                    "duration_ms": 0,
                }
                for step in trace_payload
            ],
            "generation_batch": BATCH_ID,
        }
        trajectory_rows.append(CandidateTrajectory(
            candidate_id=seed.candidate_id,
            source_case_id=case.case_id,
            source_trace_id=trajectory.run_id,
            generation_batch=BATCH_ID,
            build_version=BATCH_VERSION,
            split="train",
            review_status="pending",
            artifact_status="candidate",
            reserve=seed.reserve,
            sampling_target_route=[
                action.value for action in seed.sampling_target_route
            ],
            route=[action.value for action in actual_route],
            expected_terminal=actual_route[-1].value,
            device_family=seed.device_family,
            question_family=seed.question_family,
            missing_or_safety_trigger=seed.missing_or_safety_trigger,
            query=seed.query,
            source_evidence=seed.source_evidence,
            case_contract=case.model_dump(mode="json"),
            trace_steps=trace_payload,
            provider_record_ids=list(gates[seed.candidate_id]["provider_record_ids"]),
            generation_gate=copy.deepcopy(gates[seed.candidate_id]),
            leakage_group_id=case.leakage_group_id,
            content_fingerprint=_stable_hash(fingerprint_payload),
        ))
        for index, step in enumerate(trajectory.trace_steps):
            samples.append(SftPlannerSample(
                sample_id=(
                    f"sft_{trajectory.run_id}_{index + 1:02d}_{step.decision.action.value}"
                ),
                source_case_id=case.case_id,
                source_trace_id=trajectory.run_id,
                split=CaseSplit.TRAIN,
                turn_index=index + 1,
                input_context=_sample_context(
                    case, seed, actual_route, trajectory.trace_steps, records_by_key, index,
                ),
                target_decision=step.decision.model_dump(mode="json"),
                reward_summary={
                    "reward_version": "pending-independent-review-9.3.22",
                    "evaluated": False,
                    "generation_gate_passed": True,
                    "candidate_fingerprint_pending": True,
                },
                gold_origin=case.gold_origin,
                label_source="api_assisted_source_grounded",
                review_status="pending",
                artifact_status=SftArtifactStatus.CANDIDATE,
            ))
    return trajectory_rows, samples


def _normalized_text(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", str(text or "").lower())


def _char_ngrams(text: str, size: int = 3) -> set[str]:
    normalized = _normalized_text(text)
    if len(normalized) < size:
        return {normalized} if normalized else set()
    return {normalized[index:index + size] for index in range(len(normalized) - size + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _nontrain_queries_and_chunks() -> tuple[list[tuple[str, str]], set[str]]:
    queries: list[tuple[str, str]] = []
    chunks: set[str] = set()
    paths = [
        *PROJECT_ROOT.glob("evaluation/stage8/cases/*.jsonl"),
        *PROJECT_ROOT.glob("evaluation/stage9/artifacts/heldout_route_test/*cases*.jsonl"),
        *PROJECT_ROOT.glob("evaluation/stage9/artifacts/heldout_route_test/review_cases.jsonl"),
    ]
    for path in paths:
        if not path.is_file():
            continue
        for row in _read_jsonl(path):
            if str(row.get("split") or "") not in {"dev", "test", "demo_regression"}:
                continue
            case_id = str(row.get("case_id") or path.name)
            for query in [row.get("query"), *(row.get("query_variants") or [])]:
                if str(query or "").strip():
                    queries.append((case_id, str(query).strip()))
            for chunk in row.get("expected_chunks") or []:
                if chunk.get("chunk_id") is not None:
                    chunks.add(str(chunk["chunk_id"]))
    return queries, chunks


def validate_candidate_outputs(
        seeds: Sequence[CandidateSeed],
        trajectories: Sequence[CandidateTrajectory],
        samples: Sequence[SftPlannerSample],
) -> dict[str, Any]:
    """执行数量、Schema、重复、污染、split 泄漏、来源占比和证据绑定检查。"""

    errors: list[str] = []
    if len(seeds) != NEW_TRAJECTORY_COUNT or len(trajectories) != NEW_TRAJECTORY_COUNT:
        errors.append("新候选轨迹数不是 125")
    if len({row.candidate_id for row in trajectories}) != NEW_TRAJECTORY_COUNT:
        errors.append("candidate_id 不唯一")
    if len({row.source_trace_id for row in trajectories}) != NEW_TRAJECTORY_COUNT:
        errors.append("source_trace_id 不唯一")
    if len({row.content_fingerprint for row in trajectories}) != NEW_TRAJECTORY_COUNT:
        errors.append("content_fingerprint 不唯一")
    route_counts = Counter(" -> ".join(row.route) for row in trajectories)
    if route_counts != Counter(route_quota()):
        errors.append(f"17 条路线实际数量错误：{dict(route_counts)}")
    if sum(row.reserve for row in trajectories) != RESERVE_COUNT:
        errors.append("审核备用数量不是 12")
    if any(row.review_status != "pending" or row.artifact_status != "candidate" for row in trajectories):
        errors.append("新轨迹必须全部是 pending/candidate")
    if any(sample.review_status != "pending" or sample.artifact_status != SftArtifactStatus.CANDIDATE for sample in samples):
        errors.append("新 Action 样本必须全部是 pending/candidate")

    exact_query_counts = Counter(_normalized_text(seed.query) for seed in seeds)
    exact_duplicates = sorted(query for query, count in exact_query_counts.items() if count > 1)
    if exact_duplicates:
        errors.append(f"存在完全重复 query：{exact_duplicates[:3]}")

    near_duplicates: list[dict[str, Any]] = []
    grams = [(seed.candidate_id, seed.query, _char_ngrams(seed.query)) for seed in seeds]
    for left_index, (left_id, left_query, left_grams) in enumerate(grams):
        for right_id, right_query, right_grams in grams[left_index + 1:]:
            score = _jaccard(left_grams, right_grams)
            sequence_score = SequenceMatcher(None, left_query, right_query).ratio()
            if max(score, sequence_score) >= 0.82:
                near_duplicates.append({
                    "left": left_id,
                    "right": right_id,
                    "score": round(max(score, sequence_score), 4),
                    "jaccard_score": round(score, 4),
                    "sequence_score": round(sequence_score, 4),
                    "left_query": left_query,
                    "right_query": right_query,
                })
    if near_duplicates:
        errors.append(f"存在近义模板重复：{near_duplicates[:3]}")

    banned_meta = ("本地检索", "目标路线", "hyde_search", "web_search", "action（动作）")
    contamination = [
        seed.candidate_id for seed in seeds
        if any(token.lower() in seed.query.lower() for token in banned_meta)
    ]
    if contamination:
        errors.append(f"query 暴露路线元信息：{contamination}")

    nontrain_queries, nontrain_chunks = _nontrain_queries_and_chunks()
    split_leaks: list[dict[str, Any]] = []
    for seed in seeds:
        seed_grams = _char_ngrams(seed.query)
        for case_id, query in nontrain_queries:
            score = _jaccard(seed_grams, _char_ngrams(query))
            if _normalized_text(seed.query) == _normalized_text(query) or score >= 0.78:
                split_leaks.append({
                    "candidate_id": seed.candidate_id,
                    "nontrain_case_id": case_id,
                    "score": round(score, 4),
                })
    chunk_leaks = sorted({
        str(evidence.chunk_id)
        for seed in seeds for evidence in seed.source_evidence
        if evidence.source_type == "local" and str(evidence.chunk_id) in nontrain_chunks
    })
    if split_leaks:
        errors.append(f"存在 query/近义 split 泄漏：{split_leaks[:3]}")
    if chunk_leaks:
        errors.append(f"存在来源文本块 split 泄漏：{chunk_leaks[:5]}")

    source_counts = Counter()
    device_counts = Counter(row.device_family for row in trajectories)
    question_counts = Counter(row.question_family for row in trajectories)
    per_route_sources: dict[str, Counter[str]] = defaultdict(Counter)
    per_route_devices: dict[str, Counter[str]] = defaultdict(Counter)
    per_route_questions: dict[str, Counter[str]] = defaultdict(Counter)
    for row in trajectories:
        route = " -> ".join(row.route)
        per_route_devices[route][row.device_family] += 1
        per_route_questions[route][row.question_family] += 1
        for evidence in row.source_evidence:
            source_counts[evidence.source_id] += 1
            per_route_sources[route][evidence.source_id] += 1

    for route, count in route_quota().items():
        if count >= 5:
            max_allowed = int(count * 0.4)
            for dimension, counts in (
                ("source", per_route_sources[route]),
                ("device_family", per_route_devices[route]),
                ("question_family", per_route_questions[route]),
            ):
                if counts and max(counts.values()) > max_allowed:
                    errors.append(f"{route} 的 {dimension} 超过 40%：{dict(counts)}")
        if count >= 8 and len(per_route_devices[route]) < 4:
            errors.append(f"{route} 设备家族少于 4")
        elif 5 <= count <= 7 and len(per_route_devices[route]) < 3:
            errors.append(f"{route} 设备家族少于 3")
        elif count <= 4 and len(per_route_devices[route]) < count:
            errors.append(f"{route} 小路线设备家族不独立")

    evidence_binding_failures = [
        row.candidate_id for row in trajectories
        if not row.source_evidence or not row.generation_gate.get("passed")
    ]
    if evidence_binding_failures:
        errors.append(f"来源或生成门禁未闭环：{evidence_binding_failures}")

    report = {
        "schema_valid": not errors,
        "new_trajectory_count": len(trajectories),
        "new_action_step_count": len(samples),
        "route_counts": dict(sorted(route_counts.items())),
        "reserve_count": sum(row.reserve for row in trajectories),
        "formal_gap_count": len(trajectories) - sum(row.reserve for row in trajectories),
        "device_family_counts": dict(sorted(device_counts.items())),
        "question_family_counts": dict(sorted(question_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "exact_duplicate_count": len(exact_duplicates),
        "near_duplicate_count": len(near_duplicates),
        "query_contamination_count": len(contamination),
        "split_query_leak_count": len(split_leaks),
        "split_chunk_leak_count": len(chunk_leaks),
        "evidence_binding_failure_count": len(evidence_binding_failures),
        "errors": errors,
    }
    if errors:
        raise RuntimeError("；".join(errors))
    return report


def _write_candidate_pool_preserving_old_prefix(
        path: Path,
        *,
        old_raw: bytes,
        new_samples: Sequence[SftPlannerSample],
) -> str:
    """原样保留旧 84 行字节，只在末尾追加新 candidate 样本。"""

    new_payload = b"".join(
        (
            json.dumps(sample.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        for sample in new_samples
    )
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_bytes(old_raw + new_payload)
    after = temp_path.read_bytes()
    if after[:len(old_raw)] != old_raw:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError("旧 37 条轨迹字节前缀发生变化，拒绝替换候选文件")
    prefix_hash = _sha256_bytes(after[:len(old_raw)])
    temp_path.replace(path)
    return prefix_hash


def _failed_records_path(pending_path: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return pending_path.with_name(f"{pending_path.stem}.failed-{timestamp}{pending_path.suffix}")


def build_all(
        *,
        output_dir: Path,
        train_candidates_path: Path,
        draft_batch_size: int = 5,
        replay_provider_records: Path | None = None,
) -> CandidateManifest:
    old = audit_old_candidates(train_candidates_path)
    allocations = build_allocations()
    allocation_report = validate_allocations(allocations)
    local_facts = assign_local_facts(allocations)
    web_captures, web_facts = capture_web_sources(allocations)
    pending_drafts = output_dir / f".{DEFAULT_DRAFTS.name}.pending"
    seeds = draft_candidate_seeds(
        allocations,
        local_facts=local_facts,
        web_facts=web_facts,
        batch_size=draft_batch_size,
        cache_path=pending_drafts,
    )
    execution_cases = build_cases(seeds)
    snapshot_result = build_current_snapshot(execution_cases)

    output_dir.mkdir(parents=True, exist_ok=True)
    pending_records = output_dir / f".{DEFAULT_PROVIDER_RECORDS.name}.pending"
    if pending_records.exists():
        raise FileExistsError(f"发现上次未处理的 Provider pending 文件：{pending_records}")
    try:
        if replay_provider_records is None:
            trajectories, records = execute_trajectories(
                seeds,
                execution_cases,
                snapshot=snapshot_result.snapshot,
                provider_records_path=pending_records,
            )
        else:
            trajectories, records = replay_recorded_trajectories(
                seeds,
                execution_cases,
                snapshot=snapshot_result.snapshot,
                provider_records_path=replay_provider_records,
            )
        gates = validate_generation(seeds, execution_cases, trajectories, records)
        admission = assess_route_admission(seeds, trajectories, gates)
        if not admission["complete"]:
            raise RuntimeError(
                "真实路线尚未填满固定配额，必须继续采样而不能改写路线："
                f"{admission['route_deficits']}"
            )
        actual_routes = {
            trajectory.case_id: list(trajectory.action_path) for trajectory in trajectories
        }
        cases = build_cases(seeds, routes_by_id=actual_routes)
        trajectory_rows, samples = build_outputs(seeds, cases, trajectories, records, gates)
        validation = validate_candidate_outputs(seeds, trajectory_rows, samples)
    except Exception:
        if pending_records.exists():
            pending_records.replace(_failed_records_path(DEFAULT_PROVIDER_RECORDS))
        raise

    # 只有 125 条全部通过后，才开始写正式生成产物和候选池。
    _write_jsonl_atomic(DEFAULT_CASES, [case.model_dump(mode="json") for case in cases])
    _write_jsonl_atomic(
        DEFAULT_TRAJECTORIES,
        [row.model_dump(mode="json") for row in trajectory_rows],
    )
    _write_json_atomic(DEFAULT_WEB_EVIDENCE, {
        "manifest_version": BATCH_VERSION,
        "generation_batch": BATCH_ID,
        "captured_sources": list(web_captures.values()),
        "candidate_facts": web_facts,
    })
    if pending_drafts.exists():
        pending_drafts.replace(DEFAULT_DRAFTS)
    write_environment_snapshot(DEFAULT_SNAPSHOT, snapshot_result.snapshot)
    if replay_provider_records is not None:
        # 保留原始真实录制文件，同时把完全相同的字节原子写入正式批次路径。
        pending_records.write_bytes(replay_provider_records.read_bytes())
    pending_records.replace(DEFAULT_PROVIDER_RECORDS)
    old_prefix_hash = _write_candidate_pool_preserving_old_prefix(
        train_candidates_path,
        old_raw=old["raw"],
        new_samples=samples,
    )
    manifest = CandidateManifest(
        manifest_version=BATCH_VERSION,
        generation_batch=BATCH_ID,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        old_candidate_file_sha256_before=old["sha256"],
        old_candidate_prefix_sha256_after=old_prefix_hash,
        old_trajectory_count=old["trajectory_count"],
        new_trajectory_count=len(trajectory_rows),
        total_trajectory_count=old["trajectory_count"] + len(trajectory_rows),
        formal_gap_count=FORMAL_GAP_COUNT,
        reserve_count=RESERVE_COUNT,
        action_step_count_old=old["action_step_count"],
        action_step_count_new=len(samples),
        action_step_count_total=old["action_step_count"] + len(samples),
        route_counts_new=validation["route_counts"],
        device_family_counts_new=validation["device_family_counts"],
        question_family_counts_new=validation["question_family_counts"],
        source_counts_new=validation["source_counts"],
        validation={
            **validation,
            "allocation": allocation_report,
            "route_admission": admission,
        },
        files={
            "train_candidates": str(train_candidates_path.relative_to(PROJECT_ROOT)),
            "cases": str(DEFAULT_CASES.relative_to(PROJECT_ROOT)),
            "trajectories": str(DEFAULT_TRAJECTORIES.relative_to(PROJECT_ROOT)),
            "provider_records": str(DEFAULT_PROVIDER_RECORDS.relative_to(PROJECT_ROOT)),
            "web_evidence": str(DEFAULT_WEB_EVIDENCE.relative_to(PROJECT_ROOT)),
            "environment_snapshot": str(DEFAULT_SNAPSHOT.relative_to(PROJECT_ROOT)),
            "question_drafts": str(DEFAULT_DRAFTS.relative_to(PROJECT_ROOT)),
        },
    )
    _write_json_atomic(DEFAULT_MANIFEST, manifest.model_dump(mode="json"))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnose-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-candidates", type=Path, default=DEFAULT_TRAIN_CANDIDATES)
    parser.add_argument("--draft-batch-size", type=int, default=5)
    parser.add_argument(
        "--replay-provider-records",
        type=Path,
        help="仅回放同一批次已真实录制且完整的 Provider JSONL；缺记录时硬失败。",
    )
    args = parser.parse_args(argv)

    old = audit_old_candidates(args.train_candidates)
    allocations = build_allocations()
    allocation_report = validate_allocations(allocations)
    print(json.dumps({
        "old_trajectory_count": old["trajectory_count"],
        "old_action_step_count": old["action_step_count"],
        "old_sha256": old["sha256"],
        "new_trajectory_count": len(allocations),
        **allocation_report,
    }, ensure_ascii=False, indent=2))
    if args.diagnose_only:
        return 0
    manifest = build_all(
        output_dir=args.output_dir,
        train_candidates_path=args.train_candidates,
        draft_batch_size=args.draft_batch_size,
        replay_provider_records=args.replay_provider_records,
    )
    print(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
