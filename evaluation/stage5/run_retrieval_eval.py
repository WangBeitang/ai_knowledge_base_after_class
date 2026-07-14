"""
在同一份 Milvus chunk 快照上公平比较阶段 5 的三种本地召回组合。

运行示例：

    uv run python evaluation/stage5/run_retrieval_eval.py \
      --cases evaluation/stage5/retrieval_cases.jsonl \
      --output evaluation/stage5/retrieval_eval_results.json

脚本只比较本地检索 schema，不执行 Web，也不训练或修改 Planner。评测样本必须由真实
导入文档标注 expected_chunk_ids；缺少标注时拒绝运行，避免用虚构 ID 得出 schema 结论。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import BaseModel, Field, model_validator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.process.query.agent.state import create_query_default_state  # noqa: E402
from app.rag.query.citation_service import build_citations  # noqa: E402
from app.rag.query.contracts import (  # noqa: E402
    IdentifierResolutionStatus,
    RetrievalMode,
    SubjectResolutionStatus,
)
from app.rag.query.embedding_search_service import search_by_embedding  # noqa: E402
from app.rag.query.query_identifier_service import (  # noqa: E402
    extract_query_identifiers,
    identifier_requires_clarification,
)
from app.rag.query.rerank_service import rerank_documents  # noqa: E402


EVALUATION_MODES = (
    RetrievalMode.DENSE_LEARNED_SPARSE,
    RetrievalMode.DENSE_BM25,
    RetrievalMode.DENSE_LEARNED_SPARSE_BM25,
)


class RetrievalEvalCase(BaseModel):
    """一条真实设备运维检索评测样本。"""

    query_id: str = Field(min_length=1)  # 稳定样本 ID，开发集和 held-out 集不能重复。
    split: str = Field(pattern="^(dev|held_out)$")  # dev 可调参；held_out 只做最终确认。
    query: str = Field(min_length=1)  # 原始用户问题，三种模式必须完全一致。
    dataset_id: str = Field(min_length=1)  # 固定知识库范围。
    owner_user_id: str = Field(min_length=1)  # 固定用户权限上下文。
    tenant_id: str = "tenant_default"  # shared 数据的租户边界。
    expected_subject_id: str = Field(min_length=1)  # 已人工确认的主体 ID，跳过主体 LLM 变量。
    expected_chunk_ids: list[str | int]  # 与问题相关的人工标注 chunk ID。
    expected_identifiers: dict[str, list[str]] = Field(default_factory=dict)  # 期望提取的型号/报警码。
    expected_identifier_resolution_status: IdentifierResolutionStatus  # 标识确认状态的人工期望值。
    expected_suggested_identifiers: dict[str, list[str]] = Field(default_factory=dict)
    should_answer: bool  # 完整 Planner 是否应该允许回答；供报告分组，不在本脚本中改策略。
    should_use_web: bool  # 是否属于必须联网场景；本地 schema 评测中应单独观察。
    # 仅口语样本填写。value 指向同证据的标准问法，用于比较表达方式造成的排名变化；
    # 它不是线上查询字段，也不会进入检索 State。
    paired_query_id: str | None = None

    @model_validator(mode="after")
    def require_relevance_labels(self):
        if self.should_answer and not self.expected_chunk_ids:
            raise ValueError("should_answer=true 的评测样本必须标注 expected_chunk_ids")
        if "请替换" in self.expected_subject_id or any(
            "请替换" in str(chunk_id) for chunk_id in self.expected_chunk_ids
        ):
            raise ValueError("评测模板中的 subject_id/chunk_id 占位符必须替换为真实标注")
        extracted_identifiers = extract_query_identifiers(self.query)
        if extracted_identifiers != self.expected_identifiers:
            raise ValueError(
                "expected_identifiers 与当前确定性提取结果不一致："
                f"expected={self.expected_identifiers}, actual={extracted_identifiers}"
            )
        return self

    @property
    def is_colloquial(self) -> bool:
        """是否属于口语鲁棒性子集；统一用稳定 query_id 前缀识别。"""
        return self.query_id.startswith("held-colloquial-")


def load_cases(path: Path, *, allow_small_set: bool = False) -> list[RetrievalEvalCase]:
    """读取 JSONL 并强制当前阶段 5 正式评测集为 60 条。"""
    cases = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                cases.append(RetrievalEvalCase.model_validate_json(line))
            except Exception as error:
                raise ValueError(f"{path}:{line_number} 评测标注非法：{error}") from error
    if not cases:
        raise ValueError("评测集不能为空")
    if not allow_small_set and len(cases) != 60:
        raise ValueError("当前阶段 5 正式专项评测集必须包含 60 条评测样本")
    query_ids = [case.query_id for case in cases]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("query_id 不能重复")
    case_by_id = {case.query_id: case for case in cases}
    for case in cases:
        if case.is_colloquial:
            if case.split != "held_out":
                raise ValueError(f"口语样本 {case.query_id} 必须属于 held_out")
            if not case.paired_query_id:
                raise ValueError(f"口语样本 {case.query_id} 必须填写 paired_query_id")
            paired_case = case_by_id.get(case.paired_query_id)
            if paired_case is None:
                raise ValueError(
                    f"口语样本 {case.query_id} 指向不存在的 paired_query_id={case.paired_query_id}"
                )
            if (
                case.expected_subject_id != paired_case.expected_subject_id
                or {str(value) for value in case.expected_chunk_ids}
                != {str(value) for value in paired_case.expected_chunk_ids}
            ):
                raise ValueError(f"口语样本 {case.query_id} 必须与对照样本使用相同 subject/chunk 标注")
        elif case.paired_query_id:
            raise ValueError(f"非口语样本 {case.query_id} 不能填写 paired_query_id")
    return cases


def _dcg(relevance: list[int]) -> float:
    return sum(value / math.log2(rank + 1) for rank, value in enumerate(relevance, start=1))


def metrics_for_ranking(
    expected_chunk_ids: list[str | int],
    retrieved_chunk_ids: list[str | int],
) -> dict[str, float | int | None]:
    """计算单条 query 的二值 relevance recall、MRR 和 nDCG。"""
    expected = {str(chunk_id) for chunk_id in expected_chunk_ids}
    retrieved = [str(chunk_id) for chunk_id in retrieved_chunk_ids]
    if not expected:
        return {
            "recall_at_k": 1.0 if not retrieved else 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "first_relevant_rank": None,
        }
    relevance = [1 if chunk_id in expected else 0 for chunk_id in retrieved]
    hits = len(set(retrieved).intersection(expected))
    first_relevant_rank = next((rank for rank, value in enumerate(relevance, start=1) if value), None)
    ideal_relevance = [1] * min(len(expected), len(retrieved))
    ideal_dcg = _dcg(ideal_relevance)
    return {
        "recall_at_k": hits / len(expected),
        "mrr": 1 / first_relevant_rank if first_relevant_rank else 0.0,
        "ndcg": _dcg(relevance) / ideal_dcg if ideal_dcg else 0.0,
        "first_relevant_rank": first_relevant_rank,
    }


def _percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return float(ordered[index])


def evaluate_case(case: RetrievalEvalCase, mode: RetrievalMode) -> dict[str, Any]:
    """使用相同权限/主体/问题执行一个模式的本地检索与统一 rerank。"""
    state = create_query_default_state(
        session_id=f"eval-{case.query_id}-{mode.value}",
        original_query=case.query,
        rewritten_query=case.query,
        owner_user_id=case.owner_user_id,
        tenant_id=case.tenant_id,
        dataset_ids=[case.dataset_id],
        subject_ids=[case.expected_subject_id],
        standard_subject_names=[case.expected_subject_id],
        subject_resolution_status=SubjectResolutionStatus.CONFIRMED,
        query_identifiers=extract_query_identifiers(case.query),
        retrieval_mode=mode.value,
        trace_persistence_enabled=False,
    )
    started_at = perf_counter()
    state = search_by_embedding(state)
    observation = state["retrieval_observation"]
    identifier_expectation_match = (
        observation.identifier_resolution_status == case.expected_identifier_resolution_status
        and observation.suggested_identifiers == case.expected_suggested_identifiers
    )
    if identifier_requires_clarification(observation):
        state["reranked_docs"] = []
    else:
        # 本脚本比较单个 local Action 内的三种模式，因此直接把该 Action 原始排名交给
        # 同一 reranker；不加入 HyDE/Web，避免跨 Action 策略差异污染 schema 对比。
        state["rrf_chunks"] = list(state.get("embedding_chunks") or [])
        state = rerank_documents(state)
    latency_ms = (perf_counter() - started_at) * 1000
    citations = build_citations(state.get("reranked_docs") or [])
    retrieved_chunk_ids = [
        document["chunk_id"] for document in state.get("reranked_docs") or []
    ]
    ranking_metrics = metrics_for_ranking(case.expected_chunk_ids, retrieved_chunk_ids)
    expected_ids = {str(value) for value in case.expected_chunk_ids}
    citation_hit = int(any(
        citation.chunk_id is not None and str(citation.chunk_id) in expected_ids
        for citation in citations
    ))
    return {
        "query_id": case.query_id,
        "split": case.split,
        "case_group": "colloquial" if case.is_colloquial else "core",
        "paired_query_id": case.paired_query_id,
        "mode": mode.value,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "citation_count": len(citations),
        "citation_hit": citation_hit,
        "identifier_resolution_status": observation.identifier_resolution_status.value,
        "suggested_identifiers": observation.suggested_identifiers,
        "identifier_expectation_match": identifier_expectation_match,
        "latency_ms": latency_ms,
        **ranking_metrics,
    }


def _summarize_group(results: list[dict[str, Any]]) -> dict[str, Any]:
    """按 retrieval mode 汇总一个固定评测样本分组。"""
    summary: dict[str, Any] = {}
    for mode in EVALUATION_MODES:
        mode_results = [result for result in results if result["mode"] == mode.value]
        if not mode_results:
            continue
        summary[mode.value] = {
            "case_count": len(mode_results),
            "recall_at_k": statistics.fmean(result["recall_at_k"] for result in mode_results),
            "mrr": statistics.fmean(result["mrr"] for result in mode_results),
            "ndcg": statistics.fmean(result["ndcg"] for result in mode_results),
            "citation_hit_rate": statistics.fmean(result["citation_hit"] for result in mode_results),
            "identifier_expectation_match_rate": statistics.fmean(
                result.get("identifier_expectation_match", True) for result in mode_results
            ),
            "average_latency_ms": statistics.fmean(result["latency_ms"] for result in mode_results),
            "p95_latency_ms": _percentile_95([result["latency_ms"] for result in mode_results]),
            # 索引大小和入库耗时必须从同一轮真实重建记录，脚本不能根据字段数猜测。
            "index_size_bytes": None,
            "ingestion_duration_ms": None,
        }
    return summary


def _summarize_colloquial_pairs(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """逐模式比较标准问法与同证据口语问法，正 delta 表示口语问法指标更高。"""
    result_by_key = {
        (result["query_id"], result["mode"]): result
        for result in results
    }
    comparisons = []
    for colloquial in results:
        paired_query_id = colloquial.get("paired_query_id")
        if not paired_query_id:
            continue
        core = result_by_key.get((paired_query_id, colloquial["mode"]))
        if core is None:
            continue
        comparisons.append({
            "colloquial_query_id": colloquial["query_id"],
            "core_query_id": paired_query_id,
            "mode": colloquial["mode"],
            "core_first_relevant_rank": core.get("first_relevant_rank"),
            "colloquial_first_relevant_rank": colloquial.get("first_relevant_rank"),
            "recall_at_k_delta": colloquial["recall_at_k"] - core["recall_at_k"],
            "mrr_delta": colloquial["mrr"] - core["mrr"],
            "ndcg_delta": colloquial["ndcg"] - core["ndcg"],
        })
    return comparisons


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """分别汇总核心集、口语集和切分，避免复用证据的口语样本改变主结论权重。"""
    core_results = [result for result in results if result.get("case_group", "core") == "core"]
    colloquial_results = [result for result in results if result.get("case_group") == "colloquial"]
    return {
        "all_cases": _summarize_group(results),
        "core": _summarize_group(core_results),
        "colloquial": _summarize_group(colloquial_results),
        "by_split": {
            split: _summarize_group([result for result in results if result["split"] == split])
            for split in sorted({result["split"] for result in results})
        },
        "colloquial_pairs": _summarize_colloquial_pairs(results),
    }


def run(cases: list[RetrievalEvalCase], *, warmup: bool = True) -> dict[str, Any]:
    # embedding 和 reranker 都采用延迟加载。三种模式分别预热后再计时，避免第一种模式
    # 独自承担模型初始化成本，导致平均/P95 延迟失去可比性。
    if warmup:
        for mode in EVALUATION_MODES:
            evaluate_case(cases[0], mode)
    results = [
        evaluate_case(case, mode)
        for case in cases
        for mode in EVALUATION_MODES
    ]
    return {"summary": summarize_results(results), "cases": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段 5 三种本地检索 schema 公平对比")
    parser.add_argument("--cases", type=Path, required=True, help="已完成真实 chunk 标注的 JSONL")
    parser.add_argument("--output", type=Path, required=True, help="评测结果 JSON 输出路径")
    parser.add_argument("--allow-small-set", action="store_true", help="仅用于开发调试，正式报告禁止使用")
    args = parser.parse_args()
    cases = load_cases(args.cases, allow_small_set=args.allow_small_set)
    report = run(cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
