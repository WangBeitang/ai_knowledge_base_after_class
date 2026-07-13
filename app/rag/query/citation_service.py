"""从最终 reranked evidence 确定性生成结构化 Citation。"""

from collections.abc import Mapping, Sequence

from app.rag.query.contracts import Citation, EvidenceSourceType, RetrievalCandidate
from app.rag.query.rrf_service import canonicalize_web_url


def build_citations(
        candidates: Sequence[RetrievalCandidate | Mapping[str, object]],
) -> list[Citation]:
    """
    按最终答案上下文顺序生成引用并去重。

    Citation（引用）不是让 LLM 编写的 JSON。只有真正送进答案 Prompt 的最终候选才进入
    本函数：本地使用 document_id + chunk_id 去重，Web 使用规范化 URL 去重；展示仍保留
    搜索工具返回的原始 URL。
    """
    citations: list[Citation] = []
    seen: set[tuple[str, str, str]] = set()

    for raw_candidate in candidates:
        candidate = (
            raw_candidate
            if isinstance(raw_candidate, RetrievalCandidate)
            else RetrievalCandidate.model_validate(raw_candidate)
        )
        if candidate.rerank_score is None:
            raise ValueError("生成 Citation 前候选必须已经完成 rerank")

        if candidate.source_type == EvidenceSourceType.LOCAL:
            identity = ("local", str(candidate.document_id), str(candidate.chunk_id))
            source = candidate.source_title or candidate.title
            citation = Citation(
                document_id=candidate.document_id,
                chunk_id=candidate.chunk_id,
                title=candidate.title or candidate.source_title,
                source=source,
                score=candidate.rerank_score,
                source_type=EvidenceSourceType.LOCAL,
            )
        else:
            normalized_url = canonicalize_web_url(candidate.url or "")
            identity = ("web", normalized_url, "")
            citation = Citation(
                document_id=None,
                chunk_id=None,
                title=candidate.title,
                source=candidate.url or normalized_url,
                score=candidate.rerank_score,
                source_type=EvidenceSourceType.WEB,
            )

        if identity in seen:
            continue
        seen.add(identity)
        citations.append(citation)

    return citations

