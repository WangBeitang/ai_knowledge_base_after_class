"""阶段 8.8 评测报告生成脚本。

这个脚本只做“读取已固化结果 -> 汇总成 Markdown 报告”，不重新跑 Planner、不改
Reward、不写训练数据。这样报告可以反复生成，且不会因为生成报告本身改变评测证据。
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEV_RESULT = PROJECT_ROOT / "evaluation/stage8/results/planner_eval_dev.json"
DEFAULT_TRAIN_RESULT = PROJECT_ROOT / "evaluation/stage8/results/planner_eval_train.json"
DEFAULT_DEMO_RESULT = PROJECT_ROOT / "evaluation/stage8/results/planner_eval_demo.json"
DEFAULT_SFT_MANIFEST = PROJECT_ROOT / "evaluation/stage8/results/sft_manifest.json"
DEFAULT_STAGE5_RESULTS = PROJECT_ROOT / "evaluation/stage5"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation/stage8/reports/阶段8评测报告.md"


# 报告中固定展示的 Reward 分项顺序，和 reward.py 的实现思路保持一致。
REWARD_COMPONENT_ORDER = ("format", "retrieval", "citation", "answer", "behavior", "cost")

# 阶段 5 三种检索模式的中文名。报告读者通常关心 A/B/C 含义，不直接关心枚举名。
STAGE5_MODE_LABELS = {
    "dense_learned_sparse": "A：dense + learned sparse",
    "dense_bm25": "B：dense + BM25",
    "dense_learned_sparse_bm25": "C：dense + learned sparse + BM25",
}


def main(argv: list[str] | None = None) -> int:
    """命令行入口：解析文件路径，生成阶段 8 Markdown 报告。"""
    args = _build_parser().parse_args(argv)
    dev_result = _load_json(args.dev_result)
    train_result = _load_json(args.train_result)
    demo_result = _load_json(args.demo_result)
    sft_manifest = _load_json(args.sft_manifest)
    stage5_rows = _load_stage5_quality_rows(args.stage5_results)

    report = build_report(
        dev_result=dev_result,
        train_result=train_result,
        demo_result=demo_result,
        sft_manifest=sft_manifest,
        stage5_rows=stage5_rows,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"wrote={output_path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数；默认值指向阶段 8 当前固化产物。"""
    parser = argparse.ArgumentParser(description="生成阶段 8 评测报告 Markdown。")
    parser.add_argument("--dev-result", type=Path, default=DEFAULT_DEV_RESULT, help="dev 评测 JSON。")
    parser.add_argument("--train-result", type=Path, default=DEFAULT_TRAIN_RESULT, help="train 评测 JSON。")
    parser.add_argument("--demo-result", type=Path, default=DEFAULT_DEMO_RESULT, help="Demo 回归评测 JSON。")
    parser.add_argument("--sft-manifest", type=Path, default=DEFAULT_SFT_MANIFEST, help="SFT 导出 manifest。")
    parser.add_argument("--stage5-results", type=Path, default=DEFAULT_STAGE5_RESULTS, help="阶段 5 评测结果目录。")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Markdown 报告输出路径。")
    return parser


def build_report(
        *,
        dev_result: dict[str, Any],
        train_result: dict[str, Any],
        demo_result: dict[str, Any],
        sft_manifest: dict[str, Any],
        stage5_rows: list[dict[str, Any]],
) -> str:
    """把 8.6/8.7 的 JSON 产物汇总成可读报告。"""
    eval_outputs = [dev_result, train_result, demo_result]
    snapshot_ids = sorted({output["snapshot_id"] for output in eval_outputs})
    reward_versions = sorted({output["reward_version"] for output in eval_outputs})
    providers = sorted({output["action_provider"] for output in eval_outputs})
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")

    lines = [
        "# 阶段 8 评测报告",
        "",
        "> 本报告由 `evaluation/stage8/generate_eval_report.py` 生成初稿；结论段保留人工复核口径。",
        "",
        "## 运行摘要",
        "",
        f"- 生成时间：`{generated_at}`。",
        f"- EnvironmentSnapshot：`{', '.join(snapshot_ids)}`。",
        f"- Reward 版本：`{', '.join(reward_versions)}`。",
        f"- ActionProvider：`{', '.join(providers)}`。",
        "- 当前报告覆盖：dev baseline、train SFT 候选输入、Demo 回归集、SFT 导出 manifest。",
        "- 当前 `api` 和 `local_base` 只注册名称但未加载 provider，因此报告中保留 skipped 原因，不伪造分数。",
        "",
        "## 阶段 9 前置结论",
        "",
        (
            "阶段 8 已经具备进入阶段 9 的基本条件：case schema、固定环境快照、离线执行器、"
            "Reward v1、baseline runner、SFT 数据导出和评测报告都能形成闭环。"
        ),
        "",
        (
            "建议阶段 9 先进入 SFT，不直接宣称 GRPO 增益。当前 `snapshot_expected_chunks` "
            "provider 证明的是离线管线、Reward 和数据边界可用，不代表真实 Milvus 召回已经由阶段 8 再次提升。"
        ),
        "",
        (
            "GRPO 是否值得继续，应在阶段 9 使用同一 held-out test、同一 snapshot 和同一 Reward 版本，"
            "对比 rule、SFT、GRPO 三组轨迹后决定。"
        ),
        "",
        "## 重构前后召回和引用质量",
        "",
        (
            "这里分两层看：阶段 5 是真实检索评测，阶段 8 当前是离线 Planner 管线自检。"
            "两者口径不同，不能把阶段 8 的 `snapshot_expected_chunks` 当作新的线上检索质量。"
        ),
        "",
        _stage5_table(stage5_rows),
        "",
        _stage8_retrieval_table(eval_outputs),
        "",
        (
            "结论：阶段 5 的真实检索已经把核心集 C 模式 Recall/引用命中率稳定到 1.0000；"
            "阶段 8 没有重新优化检索，而是把 expected chunk 固化进环境快照，用来验证 Planner 轨迹、Citation、"
            "Reward 和训练数据导出的可复现性。"
        ),
        "",
        "## rule/api/local_base baseline 对比",
        "",
        _baseline_table(eval_outputs),
        "",
        "## Reward 分项",
        "",
        _reward_component_table(eval_outputs),
        "",
        (
            "解读：`retrieval` 和 `citation` 在当前 provider 下为 1.0000，说明管线能稳定取到已标注证据；"
            "`answer` 偏低主要来自离线规则 Planner 的答案文本是简化终态文本，不是答案模型生成的完整说明，"
            "因此它更适合作为阶段 9 前的弱点定位，而不是否定检索链路。"
        ),
        "",
        "## Demo 回归集结果",
        "",
        _demo_summary(demo_result),
        "",
        "## SFT 数据导出",
        "",
        _sft_manifest_table(sft_manifest),
        "",
        (
            "导出边界：不导出 test/demo，不导出格式非法轨迹，不导出低 Reward API teacher 轨迹，"
            "导出字段不包含完整 chunk 正文、答案 Prompt 或模型私有思维链。"
        ),
        "",
        "## 限制和下一步",
        "",
        "- `snapshot_expected_chunks` 是阶段 8 第一版离线 provider，只用于闭环自检，不代表真实 Milvus/Web provider。",
        "- `api` 和 `local_base` 当前因 provider/模型未配置跳过；阶段 9 接入后应复用同一 runner 追加对比。",
        "- held-out test 当前未用于调参、SFT 导出或模型选择；阶段 9 评测必须继续保持这一边界。",
        "- 阶段 9 训练前建议冻结本报告对应的 snapshot、case 文件、reward_version 和导出 manifest。",
        "- 若要扩充制造业公开数据，应先进入候选池和人工/Reward 过滤，不应直接混入 SFT 或 GRPO 训练。",
        "",
    ]
    return "\n".join(lines)


def _load_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件；路径错误时让异常直接暴露，避免生成缺证据报告。"""
    return json.loads(path.read_text(encoding="utf-8"))


def _load_stage5_quality_rows(stage5_dir: Path) -> list[dict[str, Any]]:
    """读取阶段 5 四轮检索结果，计算核心集和口语集的质量均值。"""
    result_paths = sorted(stage5_dir.glob("retrieval_eval_results*.json"))
    if not result_paths:
        return []

    loaded_results = [_load_json(path) for path in result_paths]
    rows: list[dict[str, Any]] = []
    for group in ("core", "colloquial"):
        first_summary = loaded_results[0]["summary"][group]
        for mode in STAGE5_MODE_LABELS:
            mode_values = [result["summary"][group][mode] for result in loaded_results]
            rows.append({
                "group": "核心集" if group == "core" else "口语集",
                "mode": STAGE5_MODE_LABELS[mode],
                "case_count": first_summary[mode]["case_count"],
                "recall_at_k": mean(value["recall_at_k"] for value in mode_values),
                "mrr": mean(value["mrr"] for value in mode_values),
                "ndcg": mean(value["ndcg"] for value in mode_values),
                "citation_hit_rate": mean(value["citation_hit_rate"] for value in mode_values),
            })
    return rows


def _stage5_table(rows: list[dict[str, Any]]) -> str:
    """生成阶段 5 真实检索质量表。"""
    if not rows:
        return "阶段 5 评测结果文件缺失，无法生成真实检索对照表。"
    lines = [
        "阶段 5 真实检索评测四轮均值：",
        "",
        "| 分组 | 模式 | 样本数 | Recall@K | MRR | nDCG | 引用命中率 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['mode']} | {row['case_count']} | "
            f"{_fmt(row['recall_at_k'])} | {_fmt(row['mrr'])} | {_fmt(row['ndcg'])} | "
            f"{_fmt(row['citation_hit_rate'])} |"
        )
    return "\n".join(lines)


def _stage8_retrieval_table(eval_outputs: list[dict[str, Any]]) -> str:
    """生成阶段 8 离线 provider 下的召回/引用自检表。"""
    lines = [
        "阶段 8 离线管线自检：",
        "",
        "| split | provider | rule 可评分回答样本 | recall@k | citation_hit_rate | 口径 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for output in eval_outputs:
        values = _rule_metric_values(output)
        lines.append(
            f"| {output['split']} | `{output['action_provider']}` | {values['metric_case_count']} | "
            f"{_fmt_or_na(values['recall_at_k'])} | {_fmt_or_na(values['citation_hit_rate'])} | "
            "expected chunk 快照回放 |"
        )
    return "\n".join(lines)


def _baseline_table(eval_outputs: list[dict[str, Any]]) -> str:
    """生成 rule/api/local_base baseline 对比表。"""
    lines = [
        "| split | planner | 状态 | case 数 | 失败数 | 平均 Reward | 跳过原因 |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for output in eval_outputs:
        for summary in output["planner_summaries"]:
            reward = summary["reward"].get("average_total_reward")
            skip_reason = summary.get("skip_reason") or ""
            lines.append(
                f"| {output['split']} | `{summary['planner_mode']}` | {summary['status']} | "
                f"{summary['case_count']} | {summary['failed_case_count']} | "
                f"{_fmt_or_na(reward)} | {skip_reason or '-'} |"
            )
    return "\n".join(lines)


def _reward_component_table(eval_outputs: list[dict[str, Any]]) -> str:
    """生成 Reward 分项均值表；只展示已完成的 rule 结果。"""
    lines = [
        "| split | total | format | retrieval | citation | answer | behavior | cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for output in eval_outputs:
        summary = _summary_for_planner(output, "rule")
        components = summary["reward"].get("component_average_scores", {})
        lines.append(
            f"| {output['split']} | {_fmt(summary['reward']['average_total_reward'])} | "
            + " | ".join(_fmt(components.get(component)) for component in REWARD_COMPONENT_ORDER)
            + " |"
        )
    return "\n".join(lines)


def _demo_summary(demo_result: dict[str, Any]) -> str:
    """生成 Demo 回归集摘要。"""
    summary = _summary_for_planner(demo_result, "rule")
    terminal_counts: dict[str, int] = {}
    for result in _rule_results(demo_result):
        terminal = result.get("terminal_action") or "unknown"
        terminal_counts[terminal] = terminal_counts.get(terminal, 0) + 1
    terminal_text = ", ".join(f"`{key}`={value}" for key, value in sorted(terminal_counts.items()))
    return "\n".join([
        f"- Demo 回归集 case 数：`{demo_result['case_count']}`。",
        f"- `rule` 完成 case 数：`{summary['completed_case_count']}`，失败数：`{summary['failed_case_count']}`。",
        f"- `rule` 平均 Reward：`{_fmt(summary['reward']['average_total_reward'])}`。",
        f"- 终态分布：{terminal_text or '无'}。",
        "- 这组数据用于面试 Demo 和快速回归，不进入 SFT/GRPO 训练导出。",
    ])


def _sft_manifest_table(manifest: dict[str, Any]) -> str:
    """生成 SFT 导出 manifest 摘要。"""
    lines = [
        "| 字段 | 值 |",
        "|---|---|",
        f"| manifest_id | `{manifest['manifest_id']}` |",
        f"| export_version | `{manifest['export_version']}` |",
        f"| source_run_id | `{manifest['source_run_id']}` |",
        f"| snapshot_id | `{manifest['snapshot_id']}` |",
        f"| reward_threshold | `{manifest['reward_threshold']}` |",
        f"| allowed_splits | `{', '.join(manifest['allowed_splits'])}` |",
        f"| exported_case_count | `{manifest['exported_case_count']}` |",
        f"| exported_trajectory_count | `{manifest['exported_trajectory_count']}` |",
        f"| sample_count | `{manifest['sample_count']}` |",
        f"| source_counts | `{json.dumps(manifest['source_counts'], ensure_ascii=False)}` |",
        f"| review_status_counts | `{json.dumps(manifest['review_status_counts'], ensure_ascii=False)}` |",
        f"| filter_counts | `{json.dumps(manifest['filter_counts'], ensure_ascii=False)}` |",
    ]
    return "\n".join(lines)


def _rule_metric_values(output: dict[str, Any]) -> dict[str, float | int | None]:
    """计算 rule 结果里非空 recall/citation 指标的均值。"""
    recalls: list[float] = []
    citation_rates: list[float] = []
    for result in _rule_results(output):
        recall = result["metrics"].get("recall_at_k")
        citation = result["metrics"].get("citation_hit_rate")
        if recall is not None:
            recalls.append(float(recall))
        if citation is not None:
            citation_rates.append(float(citation))
    return {
        "metric_case_count": len(recalls),
        "recall_at_k": mean(recalls) if recalls else None,
        "citation_hit_rate": mean(citation_rates) if citation_rates else None,
    }


def _rule_results(output: dict[str, Any]) -> list[dict[str, Any]]:
    """返回某个输出文件里的 rule 逐 case 结果。"""
    return [result for result in output["results"] if result["planner_mode"] == "rule"]


def _summary_for_planner(output: dict[str, Any], planner_mode: str) -> dict[str, Any]:
    """按 planner_mode 取聚合摘要；缺失时直接报错，避免报告静默漏列。"""
    for summary in output["planner_summaries"]:
        if summary["planner_mode"] == planner_mode:
            return summary
    raise ValueError(f"评测结果缺少 planner 摘要：{planner_mode}")


def _fmt(value: float | int | None) -> str:
    """统一四位小数格式；None 按 n/a 展示。"""
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _fmt_or_na(value: float | int | None) -> str:
    """表格中需要保留 n/a 的场景使用。"""
    return "n/a" if value is None else _fmt(value)


if __name__ == "__main__":
    raise SystemExit(main())
