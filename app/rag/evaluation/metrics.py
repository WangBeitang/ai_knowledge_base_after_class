"""
阶段 8.5 Reward v1 的纯指标函数。

metrics 的中文含义是“指标”。本文件只做确定性计算：把 expected chunk、实际候选、
最终答案和 Action 路径转成可复现的数字。它不读取数据库、不调用模型、不决定权重；
权重和总分聚合放在 reward.py，避免指标层和训练目标耦合。
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from typing import Any

from app.rag.evaluation.case_schema import ExpectedChunk
from app.rag.query.contracts import QueryAction, RetrievalCandidate

# 第一部分：统一证据身份。Reward 先把不同来源里的 chunk 表达统一成同一个 key。
# ChunkKey = (document_id, chunk_id, index_version)：三者合起来才能证明“命中的是哪一版证据”。
ChunkKey = tuple[str, str, int]

def expected_chunk_keys(expected_chunks: Sequence[ExpectedChunk]) -> list[ChunkKey]:
    """从人工标注的 expected_chunks 中提取期望证据身份。"""
    return _compact_keys(  # 复用去重逻辑，避免同一个 chunk 被重复标注后重复计分。
        _to_chunk_key(chunk.document_id, chunk.chunk_id, chunk.index_version)  # 每个 ExpectedChunk 都带版本号。
        for chunk in expected_chunks  # 保留人工标注顺序，后续报告更容易对照 case 文件。
    )

def candidate_chunk_keys(candidates: Sequence[RetrievalCandidate]) -> list[ChunkKey]:
    """从实际检索候选中提取本地 chunk 身份，Web 候选会被自动忽略。"""
    return _compact_keys(  # 去重时保留第一次出现的位置，因为 MRR/nDCG 依赖排名。
        _to_chunk_key(candidate.document_id, candidate.chunk_id, candidate.index_version)  # Web 缺少本地身份时返回 None。
        for candidate in candidates  # candidates 通常来自 OfflineTrajectoryResult.retrieved_candidates。
    )

# 第二部分：检索质量指标。score_retrieval 会按这个顺序计算 recall、MRR、nDCG 和标识命中。
def recall_at_k(retrieved_keys: Sequence[ChunkKey], expected_keys: Sequence[ChunkKey], *, k: int) -> float:
    """计算 recall@k：前 k 个候选覆盖了多少期望 chunk。"""
    if not expected_keys:  # 没有 expected chunk 的场景在上层通常是非回答型样本，这里按满分兜底。
        return 1.0
    top_keys = set(retrieved_keys[:max(0, k)])  # k 小于 0 时按 0 处理，避免切片语义产生误会。
    hit_count = sum(1 for key in expected_keys if key in top_keys)  # 逐个 expected chunk 统计是否命中。
    return hit_count / len(expected_keys)  # recall 的分母是期望证据数量，不是检索返回数量。



# 第四部分：内部小工具。放在文件末尾，避免打断上面的实际调用阅读顺序。
def normalize_text(value: Any) -> str:
    """把文本转成适合保守包含匹配的形式。"""
    text = str(value or "").strip().lower()  # None、数字等输入统一转字符串，避免指标函数抛异常。
    return re.sub(r"\s+", "", text)  # 去掉连续空白，让 “HAK 180” 和 “hak180” 可以对齐。


def _to_chunk_key(document_id: str | None, chunk_id: str | int | None, index_version: int | None) -> ChunkKey | None:
    """把本地证据身份转成 ChunkKey；Web 或缺版本证据返回 None。"""
    if not document_id or chunk_id is None or index_version is None:  # 三个字段缺任意一个都不能证明版本化 chunk 身份。
        return None
    return str(document_id), str(chunk_id), int(index_version)  # chunk_id 统一成 str，消除 123 和 "123" 的差异。


def _compact_keys(keys: Iterable[ChunkKey | None]) -> list[ChunkKey]:
    """去掉 None 和重复 key，同时保留首次出现顺序。"""
    compacted: list[ChunkKey] = []  # 输出列表用于保留排名顺序。
    seen: set[ChunkKey] = set()  # seen 用于去重，避免同一 chunk 重复计分。
    for key in keys:  # keys 可能是生成器，所以只能遍历一次。
        if key is None or key in seen:  # None 表示 Web/无效本地身份；重复 key 不再加入。
            continue
        compacted.append(key)  # 第一次出现的位置就是后续排名指标使用的位置。
        seen.add(key)  # 记录已出现，保证去重。
    return compacted


def _identifier_pairs(expected_identifiers: dict[str, list[str]]) -> list[tuple[str, str]]:
    """把 expected_identifiers 展平成可逐个评分的标识对。"""
    return [
        (identifier_type, normalized_value)  # 保留字段名，Reward 明细能说明是哪个标识没命中。
        for identifier_type, values in expected_identifiers.items()  # 例如 equipment_model、alarm_code、part_name。
        for normalized_value in [normalize_text(value) for value in values]  # 每个值都使用同一套文本归一化。
        if normalized_value  # 空值不参与评分，schema 正常情况下也会清理空值。
    ]


def _candidate_identifier_text(candidate: RetrievalCandidate, identifier_type: str) -> str:
    """拼出候选证据中可用于标识匹配的文本。"""
    metadata_value = getattr(candidate, identifier_type, None)  # 优先使用候选 metadata 中的结构化字段。
    return normalize_text(f"{metadata_value or ''} {candidate.title} {candidate.content}")  # 标题和正文作为兜底匹配来源。
