"""所有已执行检索 Action 的候选去重与外层 RRF 融合。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import RETRIEVAL_DEFAULT_LIMIT, RETRIEVAL_RRF_K
from app.rag.query.contracts import (
    EvidenceSourceType,
    RetrievalCandidate,
    RetrievalChannel,
)
from app.rag.query.query_identifier_service import identifier_requires_clarification
from app.shared.runtime.logger import logger, step_log


# 通道输出顺序不能依赖 set 或 Action 到达顺序。固定顺序既便于人阅读，也保证相同输入
# 在 Trace、测试和后续训练样本中得到相同 JSON。
RETRIEVAL_CHANNEL_ORDER = {
    channel: index for index, channel in enumerate(RetrievalChannel)
}


def canonicalize_web_url(url: str) -> str:
    """
    生成 Web 去重使用的规范化 URL。

    scheme/host 不区分大小写，fragment（# 后页面锚点）不代表另一份网页内容，因此移除；
    query string 可能影响页面内容，第一版保留。该值只用于候选身份和稳定排序，不修改
    最终 Citation 展示的原始 URL。
    """
    normalized = str(url or "").strip()
    if not normalized:
        raise ValueError("Web 候选 url 不能为空")
    parsed = urlsplit(normalized)
    if not parsed.scheme or not parsed.netloc:
        # 某些受控搜索工具可能返回非标准但可访问的 URI。保守保留，只统一尾部斜杠；
        # RetrievalCandidate 已保证它非空，后续 Citation 层再决定允许的协议白名单。
        return normalized.rstrip("/")
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, parsed.query, ""))


def candidate_identity(candidate: RetrievalCandidate) -> str:
    """返回跨 Action 去重键：本地使用 chunk_id，Web 使用规范化 URL。"""
    if candidate.source_type == EvidenceSourceType.LOCAL:
        return f"local:{candidate.chunk_id}"
    return f"web:{canonicalize_web_url(candidate.url or '')}"


def _candidate_richness(candidate: RetrievalCandidate) -> int:
    """统计非空元数据数量，用于重复候选出现字段差异时稳定选择更完整记录。"""
    ignored_fields = {
        "enabled",
        "retrieval_channels",
        "retrieval_rank",
        "retrieval_score",
        "rerank_score",
    }
    return sum(
        value not in (None, "", [], {})
        for field_name, value in candidate.model_dump(mode="json").items()
        if field_name not in ignored_fields
    )


def _choose_stable_candidate(
        left: RetrievalCandidate,
        right: RetrievalCandidate,
) -> RetrievalCandidate:
    """
    重复候选元数据不完全一致时，选择更完整且与列表到达顺序无关的一份。

    正常情况下同一 chunk 的身份字段和正文应一致；该规则主要防止 original/HyDE 的
    output_fields 投影差异导致“后到者覆盖先到者”。完整度相同则按稳定 JSON 字典序选择。
    """
    left_richness = _candidate_richness(left)
    right_richness = _candidate_richness(right)
    if left_richness != right_richness:
        return left if left_richness > right_richness else right
    left_json = json.dumps(left.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    right_json = json.dumps(right.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    return left if left_json <= right_json else right


def _normalize_candidate(candidate: RetrievalCandidate | Mapping[str, object]) -> RetrievalCandidate:
    """在 RRF 边界重新执行统一契约校验，拒绝 rerank 前已经丢失身份的候选。"""
    if isinstance(candidate, RetrievalCandidate):
        return candidate.model_copy(deep=True)
    return RetrievalCandidate.model_validate(candidate)


def fuse_ranked_candidate_lists(
        candidate_lists: Sequence[Sequence[RetrievalCandidate | Mapping[str, object]]],
        *,
        limit: int = RETRIEVAL_DEFAULT_LIMIT,
        k: int = RETRIEVAL_RRF_K,
) -> list[dict]:
    """
    从所有已执行 Action 的原始排名列表一次性计算外层 RRF。

    每个子列表代表一个真实执行过的检索 Action，例如 original、HyDE 或 Web。函数不接收
    上一轮 RRF 结果，因此不会形成 RRF 套 RRF；新增 Action 后调用方应把所有原始列表
    再传入一次。不同 Action 的原始分数不直接相加，只按 ``1 / (k + rank)`` 累计名次票。
    """
    if limit <= 0:
        raise ValueError("RRF limit 必须大于 0")
    if k <= 0:
        raise ValueError("RRF k 必须大于 0")

    score_by_identity: dict[str, float] = {}
    candidate_by_identity: dict[str, RetrievalCandidate] = {}
    channels_by_identity: dict[str, set[RetrievalChannel]] = {}

    for candidate_list in candidate_lists:
        seen_in_current_action: set[str] = set()
        for rank, raw_candidate in enumerate(candidate_list, start=1):
            candidate = _normalize_candidate(raw_candidate)
            identity = candidate_identity(candidate)
            channels_by_identity.setdefault(identity, set()).update(candidate.retrieval_channels)

            if identity in candidate_by_identity:
                candidate_by_identity[identity] = _choose_stable_candidate(
                    candidate_by_identity[identity],
                    candidate,
                )
            else:
                candidate_by_identity[identity] = candidate

            # 同一个 Action 列表意外重复同一 chunk/URL 时只投一票，避免脏数据放大排名。
            if identity in seen_in_current_action:
                continue
            seen_in_current_action.add(identity)
            score_by_identity[identity] = score_by_identity.get(identity, 0.0) + 1 / (k + rank)

    ranked_identities = sorted(
        score_by_identity,
        key=lambda identity: (-score_by_identity[identity], identity),
    )[:limit]

    fused_candidates = []
    for final_rank, identity in enumerate(ranked_identities, start=1):
        candidate = candidate_by_identity[identity]
        sorted_channels = sorted(
            channels_by_identity[identity],
            key=lambda channel: RETRIEVAL_CHANNEL_ORDER[channel],
        )
        # Web 的规范化 URL 仅用于 identity 和次级排序，结果继续保留搜索工具返回的原 URL。
        fused_candidate = candidate.model_copy(update={
            "retrieval_channels": sorted_channels,
            "retrieval_rank": final_rank,
            "retrieval_score": score_by_identity[identity],
            "rerank_score": None,
        })
        fused_candidates.append(fused_candidate.model_dump(mode="json"))
    return fused_candidates


def check_params(state):
    """读取当前图中已经有结果的 original、HyDE、Web 排名列表。"""
    embedding_chunks = state.get("embedding_chunks") or []
    hyde_embedding_chunks = state.get("hyde_embedding_chunks") or []
    web_search_docs = state.get("web_search_docs") or []
    if not any((embedding_chunks, hyde_embedding_chunks, web_search_docs)):
        # 阶段 9 起“检索成功但零候选”是正常 Observation，而不是编程异常。返回三个空列表
        # 让外层 RRF 产生 []，随后 Planner 可以根据 LOCAL_EMPTY/WEB_EMPTY 确定性 fallback。
        logger.info("所有已执行检索 Action 的候选都为空，交由 Planner 判断下一步")
    return embedding_chunks, hyde_embedding_chunks, web_search_docs


@step_log()
def fuse_by_rrf(state: QueryGraphState) -> QueryGraphState:
    """
    汇总当前已执行的 original、HyDE、Web 原始排名并生成累计候选。

    阶段 9 起只执行 Planner 选择的 Action；未执行 Action 的 State 字段保持空列表，
    自然不会参与本函数。真正执行但返回空的 Action 由其 Observation/Trace
    记录，空列表本身不伪造候选或影响其他列表继续融合。
    """
    embedding_chunks, hyde_embedding_chunks, web_search_docs = check_params(state)
    action_candidate_lists = [
        candidates
        for candidates in (embedding_chunks, hyde_embedding_chunks, web_search_docs)
        if candidates
    ]
    state["rrf_chunks"] = fuse_ranked_candidate_lists(action_candidate_lists)
    return state
