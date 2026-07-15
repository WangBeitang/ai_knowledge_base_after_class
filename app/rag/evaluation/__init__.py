"""
阶段 8 离线评测契约与工具包。

本包只暴露“数据契约”和“边界校验”相关对象，不执行真实检索、Reward 计算或模型推理。
这样做是为了让后续脚本、测试和训练导出都复用同一套 schema（数据形状约束），避免
JSONL 样本、环境快照、评测结果和 SFT 导出之间各自定义一套字段。

关键术语：
- Planner：规划器，只决定下一步 Action，不直接访问 Milvus/Mongo/Web。
- Action：Planner 输出的受限动作，例如 local_search、answer、refuse。
- Trace：一次查询的结构化追踪记录，后续评测可用它复盘轨迹，但不保存私有思维链。
- Reward：阶段 9 用来评分轨迹的分项奖励，本包只定义承载 Reward 结果的字段边界。
"""

from app.rag.evaluation.case_schema import (
    # 样本分组、用途拆分和来源枚举。枚举值会稳定写入 JSONL/报告，不能随意重命名。
    CaseGroup,
    CaseSplit,
    ChunkRelevance,
    # 环境快照与评测结果。它们是阶段 8 一键评测和阶段 9 对照实验的输入/输出边界。
    EnvironmentSnapshot,
    ExpectedBehavior,
    ExpectedChunk,
    HumanReviewStatus,
    LabelSource,
    PlannerEvalCase,
    PlannerEvalResult,
    PrivacyScope,
    SplitManifest,
    # 跨样本校验函数。单条 Pydantic schema 无法检查重复 case_id 或 split 泄漏。
    validate_case_collection,
    validate_cases_for_sft_export,
)


__all__ = [
    # 评测样本的枚举和结构。
    "CaseGroup",
    "CaseSplit",
    "ChunkRelevance",
    "EnvironmentSnapshot",
    "ExpectedBehavior",
    "ExpectedChunk",
    "HumanReviewStatus",
    "LabelSource",
    "PlannerEvalCase",
    "PlannerEvalResult",
    "PrivacyScope",
    "SplitManifest",
    # 训练导出前必须调用的边界校验函数。
    "validate_case_collection",
    "validate_cases_for_sft_export",
]
