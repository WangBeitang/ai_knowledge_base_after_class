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
from urllib.parse import urlsplit

from app.rag.evaluation.case_schema import ExpectedChunk, ExpectedWebEvidence
from app.rag.query.contracts import EvidenceSourceType, QueryAction, RetrievalCandidate
from app.rag.query.rrf_service import canonicalize_web_url

# 第一部分：统一证据身份。Reward 先把不同来源里的 chunk 表达统一成同一个 key。
# ChunkKey = (document_id, chunk_id, index_version)：三者合起来才能证明“命中的是哪一版证据”。
ChunkKey = tuple[str, str, int]
# EvidenceKey（证据身份）统一承载本地 chunk 和 Web URL。首项显式写来源类型，
# 避免网页 URL 与本地字段字符串碰巧相同时发生身份冲突。
EvidenceKey = tuple[str, ...]

def expected_chunk_keys(expected_chunks: Sequence[ExpectedChunk]) -> list[ChunkKey]:
    """从人工标注的 expected_chunks 中提取期望证据身份。"""
    return _compact_keys(  # 复用去重逻辑，避免同一个 chunk 被重复标注后重复计分。
        _to_chunk_key(chunk.document_id, chunk.chunk_id, chunk.index_version)  # 每个 ExpectedChunk 都带版本号。
        for chunk in expected_chunks  # 保留人工标注顺序，后续报告更容易对照 case 文件。
    )

# candidate:候选
def candidate_chunk_keys(candidates: Sequence[RetrievalCandidate]) -> list[ChunkKey]:
    """从实际检索候选中提取本地 chunk 身份，Web 候选会被自动忽略。"""
    return _compact_keys(  # 去重时保留第一次出现的位置，因为 MRR/nDCG 依赖排名。
        _to_chunk_key(candidate.document_id, candidate.chunk_id, candidate.index_version)  # Web 缺少本地身份时返回 None。
        for candidate in candidates  # candidates 通常来自 OfflineTrajectoryResult.retrieved_candidates。
    )


def expected_web_evidence_keys(
    expected_evidence: Sequence[ExpectedWebEvidence],
) -> list[EvidenceKey]:
    """从人工冻结的 Web Gold 中提取规范化 URL 身份。"""
    return _compact_evidence_keys(
        ("web", evidence.canonical_url) for evidence in expected_evidence
    )


def candidate_web_evidence_keys(
    candidates: Sequence[RetrievalCandidate],
) -> list[EvidenceKey]:
    """从实际候选中提取 Web URL 身份，本地候选自动忽略。"""
    keys: list[EvidenceKey | None] = []
    for candidate in candidates:
        if candidate.source_type != EvidenceSourceType.WEB:
            continue
        canonical_url = _canonical_http_url(candidate.url)
        keys.append(None if canonical_url is None else ("web", canonical_url))
    return _compact_evidence_keys(keys)


def expected_evidence_keys(
    expected_chunks: Sequence[ExpectedChunk],
    expected_web_evidence: Sequence[ExpectedWebEvidence],
) -> list[EvidenceKey]:
    """合并本地 chunk 与 Web URL，作为回答型 case 的统一期望证据身份。"""
    local_keys: list[EvidenceKey] = [
        ("local", document_id, chunk_id, str(index_version))
        for document_id, chunk_id, index_version in expected_chunk_keys(expected_chunks)
    ]
    return _compact_evidence_keys(
        [*local_keys, *expected_web_evidence_keys(expected_web_evidence)]
    )


def candidate_evidence_keys(
    candidates: Sequence[RetrievalCandidate],
) -> list[EvidenceKey]:
    """按实际候选排名合并本地 chunk 与 Web URL 身份。"""
    keys: list[EvidenceKey | None] = []
    for candidate in candidates:
        if candidate.source_type == EvidenceSourceType.WEB:
            canonical_url = _canonical_http_url(candidate.url)
            keys.append(None if canonical_url is None else ("web", canonical_url))
            continue
        local_key = _to_chunk_key(
            candidate.document_id,
            candidate.chunk_id,
            candidate.index_version,
        )
        keys.append(
            None
            if local_key is None
            else ("local", local_key[0], local_key[1], str(local_key[2]))
        )
    return _compact_evidence_keys(keys)


# 第二部分：检索质量指标。score_retrieval 会按这个顺序计算 recall、MRR、nDCG 和标识命中。
def recall_at_k(
    retrieved_keys: Sequence[EvidenceKey | ChunkKey],
    expected_keys: Sequence[EvidenceKey | ChunkKey],
    *,
    k: int,
) -> float:
    """计算 recall@k：前 k 个候选覆盖了多少期望 chunk。"""
    if not expected_keys:  # 没有 expected chunk 的场景在上层通常是非回答型样本，这里按满分兜底。
        return 1.0
    top_keys = set(retrieved_keys[:max(0, k)])  # k 小于 0 时按 0 处理，避免切片语义产生误会。
    hit_count = sum(1 for key in expected_keys if key in top_keys)  # 逐个 expected chunk 统计是否命中。
    return hit_count / len(expected_keys)  # recall 的分母是期望证据数量，不是检索返回数量。

def reciprocal_rank(
    retrieved_keys: Sequence[EvidenceKey | ChunkKey],
    expected_keys: Sequence[EvidenceKey | ChunkKey],
) -> float:
    """计算 MRR 的单样本 reciprocal rank：第一个期望 chunk 越靠前越高。"""
    expected_set = set(expected_keys)  # set 查询是 O(1)，避免每个候选都线性扫描 expected_keys。
    if not expected_set:  # 没有 expected chunk 时不惩罚检索层。
        return 1.0
    for rank, key in enumerate(retrieved_keys, start=1):  # rank 从 1 开始，符合 MRR 公式 1/rank。
        if key in expected_set:  # 只看第一个命中的期望证据。
            return 1.0 / rank
    return 0.0  # 完全没有命中期望证据时 MRR 为 0。

def ndcg_at_k(
    retrieved_keys: Sequence[EvidenceKey | ChunkKey],
    expected_keys: Sequence[EvidenceKey | ChunkKey],
    *,
    k: int,
) -> float:
    """计算二值相关性的 nDCG@k：同样命中数量下，越靠前分越高。"""
    if not expected_keys:  # 非回答型或无期望证据样本不在指标层扣分。
        return 1.0
    expected_set = set(expected_keys)  # 第一版只做二值相关性：命中 expected chunk 就是 1，否则 0。
    ranked_keys = retrieved_keys[:max(0, k)]  # 只评价前 k 个候选，和 recall@k 的窗口保持一致。
    dcg = sum(  # DCG：实际排序下的折损累计收益。
        1.0 / math.log2(rank + 1)  # rank 越靠后，log 折损越大。
        for rank, key in enumerate(ranked_keys, start=1)  # rank 从 1 开始，公式分母使用 log2(rank + 1)。
        if key in expected_set  # 非期望 chunk 的相关性为 0，不贡献 DCG。
    )
    ideal_hits = min(len(expected_set), max(0, k))  # 理想排序最多只能命中 min(expected_count, k) 个。
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))  # IDCG 是最佳排序分。
    return 0.0 if ideal_dcg == 0 else dcg / ideal_dcg  # 避免 k=0 时除以 0。

def identifier_hit_rate(
        candidates: Sequence[RetrievalCandidate],
        expected_identifiers: dict[str, list[str]],
) -> tuple[float, dict[str, list[str]], dict[str, list[str]]]:
    """计算设备型号、报警码、部件名等结构化标识的命中率。"""
    expected_pairs = _identifier_pairs(expected_identifiers)  # 展平成 [(字段名, 规范化值)]，方便逐个判断。
    if not expected_pairs:  # case 没有标注标识时，不让这个补充指标拖低检索分。
        return 1.0, {}, {}

    hits: dict[str, list[str]] = {}  # 保存已命中的标识，写入 Reward 明细供报告解释。
    misses: dict[str, list[str]] = {}  # 保存未命中的标识，便于定位是设备型号错还是报警码错。
    for identifier_type, expected_value in expected_pairs:  # 每个标识值单独计分，避免一个字段多个值时被合并。
        matched = any(  # 任意候选命中该标识就算这个标识值命中。
            expected_value in _candidate_identifier_text(candidate, identifier_type)  # 在 metadata、标题和正文中查找。
            for candidate in candidates
        )
        target = hits if matched else misses  # 命中和未命中分开放，Reward details 更直观。
        target.setdefault(identifier_type, []).append(expected_value)  # 同一字段下可能有多个期望值。

    hit_count = sum(len(values) for values in hits.values())  # 命中的标识值总数。
    return hit_count / len(expected_pairs), hits, misses  # 返回命中率、命中明细、缺失明细。


# 第三部分：答案和 Action 路径指标。score_answer、score_behavior、score_cost 会调用这里。
def answer_point_coverage(answer: str, expected_answer_points: Sequence[str]) -> tuple[float, list[str], list[str]]:
    """计算答案要点覆盖率，第一版使用可复现的文本包含判断。"""
    if not expected_answer_points:  # 拒答/追问样本没有答案要点，不应在指标层被误扣分。
        return 1.0, [], []

    normalized_answer = normalize_text(answer)  # 统一大小写和空白，减少表面格式差异。
    hit_points: list[str] = []  # 保存已覆盖的人工要点。
    missing_points: list[str] = []  # 保存缺失的人工要点。
    for point in expected_answer_points:  # 每个答案要点独立判断，后续可定位漏了哪一点。
        normalized_point = normalize_text(point)  # 要点同样归一化，和 answer 使用同一规则。
        if normalized_point and normalized_point in normalized_answer:  # 命中时归入 hit_points。
            target = hit_points
        else:  # 未命中或空要点归入 missing_points，便于报告提示缺失项。
            target = missing_points
        target.append(point)  # details 保留原始中文要点，报告更可读。

    return len(hit_points) / len(expected_answer_points), hit_points, missing_points  # 覆盖率 = 命中要点数 / 总要点数。

def action_values(actions: Iterable[QueryAction | str]) -> list[str]:
    """把 QueryAction 或字符串统一成持久化 value，供路径比较和报告输出使用。"""
    return [action.value if isinstance(action, QueryAction) else str(action) for action in actions]  # 保持原顺序，不排序。

def matches_any_action_path(
        actual_path: Sequence[QueryAction | str],
        acceptable_paths: Sequence[Sequence[QueryAction | str]],
) -> bool:
    """判断实际 Action 路径是否完整匹配任意一条人工可接受路径。"""
    actual_values = action_values(actual_path)  # 先把实际路径统一成字符串列表。
    return any(actual_values == action_values(path) for path in acceptable_paths)  # 必须完整相等，不做前缀匹配。

def shortest_acceptable_path_length(acceptable_paths: Sequence[Sequence[QueryAction | str]]) -> int:
    """返回最短可接受路径长度，用于估计额外步骤成本。"""
    lengths = [len(path) for path in acceptable_paths if path]  # 空路径在 schema 层会拒绝，这里仍防御处理。
    return min(lengths) if lengths else 0  # 没有可接受路径时返回 0，让上层自行决定是否扣分。



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


def _compact_evidence_keys(
    keys: Iterable[EvidenceKey | None],
) -> list[EvidenceKey]:
    """去掉空值和重复的统一证据身份，并保留首次出现的排名。"""
    compacted: list[EvidenceKey] = []
    seen: set[EvidenceKey] = set()
    for key in keys:
        if key is None or key in seen:
            continue
        compacted.append(key)
        seen.add(key)
    return compacted


def _canonical_http_url(url: str | None) -> str | None:
    """只把 HTTP(S) URL 作为可评分的 Web 证据身份。"""
    normalized = str(url or "").strip()
    parsed = urlsplit(normalized)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None
    return canonicalize_web_url(normalized)


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
