"""
阶段 8 离线评测契约与工具包。

本包暴露阶段 8 离线评测的稳定对象：数据契约、边界校验、离线 Environment 和
Reward v1 评分入口。它不执行真实检索或模型推理；Reward 计算只读取已经产生的
OfflineTrajectoryResult，不连接 Mongo/Milvus/Web。这样做是为了让后续脚本、测试和
训练导出都复用同一套 schema（数据形状约束），避免 JSONL 样本、环境快照、评测结果
和 SFT/GRPO 导出之间各自定义一套字段。

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
    GoldOrigin,
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
from app.rag.evaluation.offline_environment import (
    # 离线 Environment。它把固定 snapshot 转成可执行 State/Trace，不写聊天历史。
    EmptyOfflineActionProvider,
    OfflineActionProvider,
    OfflineError,
    OfflineRagEnvironment,
    OfflineState,
    OfflineStepResult,
    OfflineTraceStep,
    OfflineTrajectoryResult,
    OfflineTrajectoryStatus,
)
from app.rag.evaluation.reward import (
    # Reward v1 评分入口和结果 schema。Reward 的中文含义是“奖励/评分信号”，阶段 8
    # 用于 baseline 评测，阶段 9 会复用 total_reward 作为 GRPO 训练信号。
    REWARD_VERSION,
    RewardComponent,
    RewardConfig,
    RewardWeights,
    TrajectoryReward,
    score_answer,
    score_behavior,
    score_citation,
    score_cost,
    score_format,
    score_retrieval,
    score_trajectory,
)
from app.rag.evaluation.baseline_runner import (
    # 阶段 8.6 baseline runner。它固定 case/snapshot/reward 跑 Planner 对照评测，
    # 不训练模型，也不计算 GRPO 组内 advantage。
    BaselineEvalOutput,
    BaselinePlannerSummary,
    SnapshotExpectedChunkActionProvider,
    load_environment_snapshot,
    parse_planner_modes,
    run_baseline_evaluation,
    run_baseline_evaluation_from_files,
    write_baseline_eval_output,
)
from app.rag.evaluation.sft_exporter import (
    # 阶段 8.7 SFT 数据导出器。它把已评分轨迹筛成单步 PlannerDecision 监督样本，
    # 不保存完整 chunk 正文、答案 Prompt 或模型私有思维链。
    SFT_EXPORT_VERSION,
    SftArtifactStatus,
    SftExportConfig,
    SftExportManifest,
    SftExportResult,
    SftPlannerSample,
    export_sft_samples,
    export_sft_samples_from_files,
    load_baseline_eval_output,
    parse_allowed_splits,
    write_sft_manifest,
    write_sft_samples,
)


__all__ = [
    # 评测样本的枚举和结构。
    "CaseGroup",
    "CaseSplit",
    "ChunkRelevance",
    "EnvironmentSnapshot",
    "ExpectedBehavior",
    "ExpectedChunk",
    "GoldOrigin",
    "HumanReviewStatus",
    "LabelSource",
    "PlannerEvalCase",
    "PlannerEvalResult",
    "PrivacyScope",
    "SplitManifest",
    # 阶段 8.4 离线执行器。
    "EmptyOfflineActionProvider",
    "OfflineActionProvider",
    "OfflineError",
    "OfflineRagEnvironment",
    "OfflineState",
    "OfflineStepResult",
    "OfflineTraceStep",
    "OfflineTrajectoryResult",
    "OfflineTrajectoryStatus",
    # 阶段 8.5 Reward v1 评分器。
    "REWARD_VERSION",
    "RewardComponent",
    "RewardConfig",
    "RewardWeights",
    "TrajectoryReward",
    "score_answer",
    "score_behavior",
    "score_citation",
    "score_cost",
    "score_format",
    "score_retrieval",
    "score_trajectory",
    # 阶段 8.6 Planner baseline 跑批器。
    "BaselineEvalOutput",
    "BaselinePlannerSummary",
    "SnapshotExpectedChunkActionProvider",
    "load_environment_snapshot",
    "parse_planner_modes",
    "run_baseline_evaluation",
    "run_baseline_evaluation_from_files",
    "write_baseline_eval_output",
    # 阶段 8.7 Planner SFT 数据导出。
    "SFT_EXPORT_VERSION",
    "SftArtifactStatus",
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
    # 训练导出前必须调用的边界校验函数。
    "validate_case_collection",
    "validate_cases_for_sft_export",
]
