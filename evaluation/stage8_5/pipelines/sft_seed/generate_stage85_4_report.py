"""校验阶段 8.5.4 产物并生成评测与 SFT 训练种子报告。

这个脚本不是简单地把 JSON 转成 Markdown。它先对 baseline、SFT manifest 和 SFT
JSONL 做交叉校验，确认三者使用同一 ``run_id/snapshot_id/reward_version``，并检查正式
训练种子只能来自 reviewed train Gold。任一边界不满足时脚本直接失败，避免生成一份
看起来正常、实际混入错误 split 或未审核数据的报告。

当前 baseline 使用 ``snapshot_expected_chunks``：它根据 case 中的 expected_chunks 构造
确定性离线候选，用于验证 Planner 路由、Reward 和导出链路；它不调用真实 Milvus，也不
运行答案模型。因此报告必须把 ``answer=0`` 解释为“未评测”，不能解释为答案质量为零。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.evaluation.baseline_runner import (  # noqa: E402
    SNAPSHOT_EXPECTED_PROVIDER_NAME,
    BaselineEvalOutput,
)
from app.rag.evaluation.case_schema import CaseSplit, GoldOrigin  # noqa: E402
from app.rag.evaluation.sft_exporter import (  # noqa: E402
    SftArtifactStatus,
    SftExportManifest,
    SftPlannerSample,
)
from evaluation.stage8_5.pipelines.common.paths import stage85_layout  # noqa: E402


_LAYOUT = stage85_layout()
DEFAULT_BASELINE = _LAYOUT.sft_intermediate / "reward_v1_1_baseline_train.json"
DEFAULT_SFT = _LAYOUT.final / "sft_curated_seed_train.jsonl"
DEFAULT_MANIFEST = _LAYOUT.final / "sft_curated_seed_manifest.json"
DEFAULT_OUTPUT = _LAYOUT.reports / "阶段8.5.4评测与SFT训练种子报告.md"
EXPECTED_CASE_COUNT = 20


def load_sft_samples(path: Path) -> list[SftPlannerSample]:
    """逐行读取 SFT JSONL，并让 Pydantic 拒绝未知字段和非法枚举值。"""

    samples: list[SftPlannerSample] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            samples.append(SftPlannerSample.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"SFT JSONL 第 {line_number} 行校验失败：{exc}") from exc
    return samples


def validate_stage85_4_artifacts(
        *,
        baseline: BaselineEvalOutput,
        manifest: SftExportManifest,
        samples: list[SftPlannerSample],
) -> dict[str, object]:
    """执行阶段 8.5.4 的硬门禁，并返回报告需要的统计。"""

    if baseline.split != CaseSplit.TRAIN:
        raise ValueError("阶段 8.5.4 curated seed baseline 必须只运行 train split")
    if baseline.reward_version != "reward-v1.1":
        raise ValueError("阶段 8.5.4 必须使用 reward-v1.1")
    if baseline.action_provider != SNAPSHOT_EXPECTED_PROVIDER_NAME:
        raise ValueError("当前报告只适用于 snapshot_expected_chunks 离线 provider")
    if baseline.case_count != EXPECTED_CASE_COUNT or len(baseline.results) != EXPECTED_CASE_COUNT:
        raise ValueError(f"baseline 应包含 {EXPECTED_CASE_COUNT} 条 case 和结果")
    if any(result.errors for result in baseline.results):
        raise ValueError("baseline 存在执行或 Reward 错误，不能导出正式训练种子")
    expected_path = ["local_search", "answer"]
    if any([action.value for action in result.action_path] != expected_path for result in baseline.results):
        raise ValueError("curated seed baseline 存在非 local_search -> answer 路线")

    if manifest.artifact_status != SftArtifactStatus.APPROVED_TRAINING_SEED:
        raise ValueError("SFT manifest 不是 approved_training_seed")
    if manifest.source_run_id != baseline.run_id:
        raise ValueError("SFT manifest 与 baseline run_id 不一致")
    if manifest.snapshot_id != baseline.snapshot_id:
        raise ValueError("SFT manifest 与 baseline snapshot_id 不一致")
    if manifest.reward_version != baseline.reward_version:
        raise ValueError("SFT manifest 与 baseline reward_version 不一致")
    if manifest.allowed_splits != [CaseSplit.TRAIN]:
        raise ValueError("正式 curated seed SFT 只能允许 train split")
    if manifest.exported_case_count != EXPECTED_CASE_COUNT:
        raise ValueError(f"SFT manifest 应导出 {EXPECTED_CASE_COUNT} 个 case")
    if manifest.exported_trajectory_count != EXPECTED_CASE_COUNT:
        raise ValueError(f"SFT manifest 应导出 {EXPECTED_CASE_COUNT} 条轨迹")
    if manifest.sample_count != len(samples):
        raise ValueError("SFT manifest sample_count 与 JSONL 行数不一致")
    if manifest.filter_counts:
        raise ValueError("正式 curated seed 导出存在被过滤轨迹，应先查明原因")

    case_ids = {sample.source_case_id for sample in samples}
    if len(case_ids) != EXPECTED_CASE_COUNT:
        raise ValueError(f"SFT JSONL 应覆盖 {EXPECTED_CASE_COUNT} 个唯一 case")
    if any(sample.split != CaseSplit.TRAIN for sample in samples):
        raise ValueError("SFT JSONL 混入非 train 样本")
    if any(sample.review_status != "reviewed" for sample in samples):
        raise ValueError("SFT JSONL 混入未复核样本")
    if any(sample.gold_origin != GoldOrigin.CURATED_SEED_GOLD for sample in samples):
        raise ValueError("SFT JSONL 混入非 curated_seed_gold 样本")
    if any(sample.artifact_status != SftArtifactStatus.APPROVED_TRAINING_SEED for sample in samples):
        raise ValueError("SFT JSONL 存在未批准的训练样本")

    action_counts = Counter(str(sample.target_decision.get("action", "")) for sample in samples)
    if action_counts != Counter({"local_search": EXPECTED_CASE_COUNT, "answer": EXPECTED_CASE_COUNT}):
        raise ValueError("每条 curated Gold 应导出一个 local_search 和一个 answer 决策样本")

    summary = baseline.planner_summaries[0]
    component_scores = summary.reward.get("component_average_scores", {})
    return {
        "run_id": baseline.run_id,
        "snapshot_id": baseline.snapshot_id,
        "reward_version": baseline.reward_version,
        "case_count": baseline.case_count,
        "sample_count": len(samples),
        "average_total_reward": summary.reward.get("average_total_reward"),
        "component_scores": component_scores,
        "action_counts": dict(sorted(action_counts.items())),
    }


def build_markdown_report(stats: dict[str, object]) -> str:
    """生成边界明确、可以直接提交审计的 Markdown 报告。"""

    components = stats["component_scores"]
    actions = stats["action_counts"]
    component_rows = "\n".join(
        f"| `{name}` | {score:.2f} |" for name, score in sorted(components.items())
    )
    action_rows = "\n".join(
        f"| `{name}` | {count} |" for name, count in sorted(actions.items())
    )
    return f"""# 阶段 8.5.4 评测与 SFT 训练种子报告

## 结论

- 20 条二审通过的 `curated_seed_gold` 已完成 train 冒烟评测，并作为正式 Planner SFT 训练种子导出。
- 共导出 40 条单步决策样本：每条 case 对应一次 `local_search` 决策和一次 `answer` 决策。
- 这些样本可以进入阶段 9 的 Planner SFT，但不能代表独立 dev/test，也不能证明真实 Milvus 检索或答案生成质量。

## 运行身份

| 字段 | 值 |
|---|---|
| baseline run | `{stats['run_id']}` |
| environment snapshot | `{stats['snapshot_id']}` |
| Reward | `{stats['reward_version']}` |
| action provider | `snapshot_expected_chunks` |
| case split | `train` |
| case 数 | {stats['case_count']} |
| SFT 样本数 | {stats['sample_count']} |
| 平均 Reward | {stats['average_total_reward']:.2f} |

## Reward 分项

| 分项 | 平均分 |
|---|---:|
{component_rows}

`answer=0` 不表示 Gold 答案错误。当前离线 provider 只构造 expected chunk 候选并返回占位答案，
没有调用答案模型，所以答案要点覆盖率未被评测。本次 0.85 只能证明规则 Planner 按预期完成
`local_search -> answer`、Reward v1.1 可计算且 SFT 导出链路有效。

## SFT 决策分布

| 目标 Action | 样本数 |
|---|---:|
{action_rows}

全部样本满足：`split=train`、`review_status=reviewed`、
`gold_origin=curated_seed_gold`、`artifact_status=approved_training_seed`。导出中不包含完整 chunk
正文、标准答案、答案 Prompt 或模型私有思维链。

## 使用边界

- **可以用于**：阶段 9 Planner SFT 的小规模高置信启动集、训练代码冒烟、路由格式和基本动作学习。
- **不能用于**：Reward 权重调参、模型优选、正式泛化结论、真实检索召回率结论、答案质量结论。
- 后续 dev/test 必须来自独立真实文档，并遵循“原始文档先生产入库、冻结真实 chunk、再生成 Gold”的流程。
"""


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    baseline = BaselineEvalOutput.model_validate_json(args.baseline.read_text(encoding="utf-8"))
    manifest = SftExportManifest.model_validate_json(args.manifest.read_text(encoding="utf-8"))
    samples = load_sft_samples(args.sft)
    stats = validate_stage85_4_artifacts(baseline=baseline, manifest=manifest, samples=samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_markdown_report(stats), encoding="utf-8")
    print(f"case_count={stats['case_count']}")
    print(f"sample_count={stats['sample_count']}")
    print(f"output={args.output}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="校验阶段 8.5.4 产物并生成 Markdown 报告。")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--sft", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
