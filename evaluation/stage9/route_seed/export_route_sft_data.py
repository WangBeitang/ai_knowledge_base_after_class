"""导出并合并阶段 9 route seed SFT 数据。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.evaluation.case_schema import CaseSplit, GoldOrigin, load_planner_cases  # noqa: E402
from app.rag.evaluation.sft_exporter import (  # noqa: E402
    SftArtifactStatus,
    SftExportConfig,
    SftPlannerSample,
    export_sft_samples_from_files,
    write_sft_manifest,
    write_sft_samples,
)
from evaluation.stage9.route_seed.build_route_seed_cases import DEFAULT_OUTPUT, DEFAULT_PATHS, read_route_seed_paths  # noqa: E402
from evaluation.stage9.route_seed.run_route_seed_paths import DEFAULT_BASELINE  # noqa: E402


STAGE9_SFT_MERGE_VERSION = "stage9-sft-merge-v1"
DEFAULT_ROUTE_SFT = PROJECT_ROOT / "evaluation/stage9/artifacts/route_seed/sft_route_seed_train.jsonl"
DEFAULT_ROUTE_MANIFEST = PROJECT_ROOT / "evaluation/stage9/artifacts/route_seed/sft_route_seed_manifest.json"
DEFAULT_BASE_SFT = PROJECT_ROOT / "evaluation/stage8_5/artifacts/final/sft_curated_seed_train.jsonl"
DEFAULT_BASE_MANIFEST = PROJECT_ROOT / "evaluation/stage8_5/artifacts/final/sft_curated_seed_manifest.json"
DEFAULT_MERGED_SFT = PROJECT_ROOT / "evaluation/stage9/artifacts/sft/sft_planner_stage9_train.jsonl"
DEFAULT_MERGED_MANIFEST = PROJECT_ROOT / "evaluation/stage9/artifacts/sft/sft_planner_stage9_manifest.json"


class Stage9SftMergeModel(BaseModel):
    """阶段 9 合并 SFT manifest schema 公共基类。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


class Stage9SftMergeManifest(Stage9SftMergeModel):
    """
    阶段 9 Planner SFT 合并清单。

    Manifest 的中文含义是“清单”。它记录 base curated seed 和 route seed 合并后的 Action
    分布、路线覆盖和来源边界，防止训练时只看到 JSONL 而不知道数据是否仍然单一路线。
    """

    manifest_id: str = Field(
        min_length=1,
        description="必填，合并清单 ID；一次导出生成一个，用于回溯具体 SFT 文件。",
    )
    merge_version: str = Field(
        default=STAGE9_SFT_MERGE_VERSION,
        description="合并逻辑版本，默认 stage9-sft-merge-v1；脚本行为变化时必须升级。",
    )
    created_at: str = Field(
        min_length=1,
        description="必填，UTC ISO 生成时间；表示当前 manifest 的落盘生命周期起点。",
    )
    artifact_status: SftArtifactStatus = Field(
        description="导出审批级别；阶段 9.1 只生成 approved_training_seed，candidate 不允许进入正式训练。",
    )
    reward_version: str = Field(
        min_length=1,
        description="必填，样本来源轨迹使用的 Reward 版本；当前冻结为 reward-v1.1。",
    )
    base_sft_path: str = Field(
        min_length=1,
        description="必填，阶段 8.5 curated seed SFT 输入路径；只读引用，不在本脚本中改写。",
    )
    base_manifest_path: str = Field(
        min_length=1,
        description="必填，阶段 8.5 curated seed manifest 路径；用于保留来源审计信息。",
    )
    route_sft_path: str = Field(
        min_length=1,
        description="必填，阶段 9 route seed SFT 输出路径；只包含 train-only 路线样本。",
    )
    route_manifest_path: str = Field(
        min_length=1,
        description="必填，阶段 9 route seed 导出清单路径；用于追踪过滤门禁和样本来源。",
    )
    sample_count: int = Field(
        ge=0,
        description="合并后单步 SFT 样本数；统计单位是 Planner 决策，不是 case。",
    )
    source_case_count: int = Field(
        ge=0,
        description="合并后来源 case 数；用于确认 route seed 没有覆盖或删除原 curated Gold。",
    )
    action_counts: dict[str, int] = Field(
        default_factory=dict,
        description="按目标 Action 统计的样本数；用于验收 Planner 是否见过追问、HyDE、Web、拒答等动作。",
    )
    route_family_counts: dict[str, int] = Field(
        default_factory=dict,
        description="按路线家族统计的样本数；统计单位是单步决策，非 case。",
    )
    terminal_action_counts: dict[str, int] = Field(
        default_factory=dict,
        description="按终止 Action 统计的样本数；用于确认 answer/refuse/ask_clarification 终态都有覆盖。",
    )
    label_source_counts: dict[str, int] = Field(
        default_factory=dict,
        description="按标签来源统计的样本数；区分 rule 和 manual_route_seed，避免来源混淆。",
    )
    gold_origin_counts: dict[str, int] = Field(
        default_factory=dict,
        description="按 Gold 生产方式统计的样本数；route_seed_gold 只允许 train，不代表 held-out test。",
    )
    review_status_counts: dict[str, int] = Field(
        default_factory=dict,
        description="按审核状态统计的样本数；阶段 9.1 合并产物必须全为 reviewed。",
    )
    split_counts: dict[str, int] = Field(
        default_factory=dict,
        description="按 split 统计的样本数；阶段 9.1 合并产物必须全为 train，禁止 dev/test 污染。",
    )
    source_manifests: list[dict[str, Any]] = Field(
        default_factory=list,
        description="来源 manifest 快照；保存 curated_seed 和 route_seed 的原始清单，便于审计。",
    )
    excluded_payloads: list[str] = Field(
        default_factory=list,
        description="明确不写入训练样本的字段；用于防止完整 chunk 正文、答案 Prompt 或私有思维链泄漏。",
    )
    notes: str = Field(
        default="",
        description="中文说明；记录本批合并的阶段边界和不可当作 held-out test 的限制。",
    )


def export_and_merge_route_sft(
        *,
        eval_result_path: str | Path,
        cases_path: str | Path,
        paths_path: str | Path,
        route_output_path: str | Path,
        route_manifest_path: str | Path,
        base_sft_path: str | Path,
        base_manifest_path: str | Path,
        merged_output_path: str | Path,
        merged_manifest_path: str | Path,
) -> Stage9SftMergeManifest:
    """导出 route seed SFT，并和阶段 8.5 curated seed 合并。"""
    route_export = export_sft_samples_from_files(
        eval_result_path=eval_result_path,
        cases_path=cases_path,
        output_path=route_output_path,
        manifest_path=route_manifest_path,
        config=SftExportConfig(
            reward_threshold=0.80,
            allowed_splits=(CaseSplit.TRAIN,),
            artifact_status=SftArtifactStatus.APPROVED_TRAINING_SEED,
        ),
    )
    # 确保写入的是经过当前 schema 序列化的 manifest，避免调用方传入旧文件后残留旧字段。
    write_sft_manifest(route_export.manifest, route_manifest_path)

    base_samples = _load_sft_samples(base_sft_path)
    route_samples = route_export.samples
    merged_samples = base_samples + route_samples
    write_sft_samples(merged_samples, merged_output_path)

    route_family_by_case = {
        path.case_id: path.route_family
        for path in read_route_seed_paths(paths_path)
    }
    base_manifest = json.loads(Path(base_manifest_path).read_text(encoding="utf-8"))
    route_manifest = route_export.manifest.model_dump(mode="json")
    merged_manifest = _build_merged_manifest(
        samples=merged_samples,
        route_family_by_case=route_family_by_case,
        base_sft_path=base_sft_path,
        base_manifest_path=base_manifest_path,
        route_sft_path=route_output_path,
        route_manifest_path=route_manifest_path,
        base_manifest=base_manifest,
        route_manifest=route_manifest,
    )
    _write_json(merged_manifest_path, merged_manifest.model_dump(mode="json"))
    return merged_manifest


def _build_merged_manifest(
        *,
        samples: list[SftPlannerSample],
        route_family_by_case: dict[str, str],
        base_sft_path: str | Path,
        base_manifest_path: str | Path,
        route_sft_path: str | Path,
        route_manifest_path: str | Path,
        base_manifest: dict[str, Any],
        route_manifest: dict[str, Any],
) -> Stage9SftMergeManifest:
    action_counts = Counter(sample.target_decision["action"] for sample in samples)
    route_family_counts = Counter(
        route_family_by_case.get(sample.source_case_id, "stop_when_enough")
        for sample in samples
    )
    terminal_action_counts = Counter(
        sample.target_decision["action"]
        for sample in samples
        if _is_terminal_sample(sample)
    )
    return Stage9SftMergeManifest(
        manifest_id=f"stage9_sft_manifest_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        artifact_status=SftArtifactStatus.APPROVED_TRAINING_SEED,
        reward_version="reward-v1.1",
        base_sft_path=str(base_sft_path),
        base_manifest_path=str(base_manifest_path),
        route_sft_path=str(route_sft_path),
        route_manifest_path=str(route_manifest_path),
        sample_count=len(samples),
        source_case_count=len({sample.source_case_id for sample in samples}),
        action_counts=dict(sorted(action_counts.items())),
        route_family_counts=dict(sorted(route_family_counts.items())),
        terminal_action_counts=dict(sorted(terminal_action_counts.items())),
        label_source_counts=dict(sorted(Counter(sample.label_source for sample in samples).items())),
        gold_origin_counts=dict(sorted(Counter(sample.gold_origin.value for sample in samples).items())),
        review_status_counts=dict(sorted(Counter(sample.review_status for sample in samples).items())),
        split_counts=dict(sorted(Counter(sample.split.value for sample in samples).items())),
        source_manifests=[
            {"kind": "curated_seed", "manifest": base_manifest},
            {"kind": "route_seed", "manifest": route_manifest},
        ],
        excluded_payloads=[
            "full_chunk_content",
            "answer_prompt",
            "private_chain_of_thought",
            "model_reasoning_text",
        ],
        notes=(
            "阶段 9 初始 SFT 数据：20 条 curated seed Gold + 50 条 route_seed_gold。"
            "route_seed_gold 只用于 train，不代表 held-out test。"
        ),
    )


def _is_terminal_sample(sample: SftPlannerSample) -> bool:
    return sample.target_decision["action"] in {"answer", "ask_clarification", "refuse"}


def _load_sft_samples(path: str | Path) -> list[SftPlannerSample]:
    samples: list[SftPlannerSample] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                samples.append(SftPlannerSample.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number} SFT 样本非法：{exc}") from exc
    if not samples:
        raise ValueError(f"{path} 没有 SFT 样本")
    return samples


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    # 提前校验 case 和 path 文件，错误直接暴露，避免导出空 route seed。
    load_planner_cases(args.cases)
    read_route_seed_paths(args.paths)
    manifest = export_and_merge_route_sft(
        eval_result_path=args.eval_result,
        cases_path=args.cases,
        paths_path=args.paths,
        route_output_path=args.output,
        route_manifest_path=args.manifest,
        base_sft_path=args.base_sft,
        base_manifest_path=args.base_manifest,
        merged_output_path=args.merged_output,
        merged_manifest_path=args.merged_manifest,
    )
    print(f"route_output={args.output}")
    print(f"route_manifest={args.manifest}")
    print(f"merged_output={args.merged_output}")
    print(f"merged_manifest={args.merged_manifest}")
    print(f"sample_count={manifest.sample_count}")
    print(f"action_counts={json.dumps(manifest.action_counts, ensure_ascii=False, sort_keys=True)}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="导出阶段 9 route seed SFT 并合并 curated seed。")
    parser.add_argument("--eval-result", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--cases", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--paths", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROUTE_SFT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_ROUTE_MANIFEST)
    parser.add_argument("--base-sft", type=Path, default=DEFAULT_BASE_SFT)
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--merged-output", type=Path, default=DEFAULT_MERGED_SFT)
    parser.add_argument("--merged-manifest", type=Path, default=DEFAULT_MERGED_MANIFEST)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
