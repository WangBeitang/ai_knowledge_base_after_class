from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter

from app.infra.llm.providers import llm_provider
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.query.agent.state import QueryGraphState
from app.rag.query.chunk_retrieval_utils import (
    CHUNK_OUTPUT_FIELDS,
    build_chunk_retrieval_filter,
    format_chunk_search_item,
    select_structured_query_identifiers,
)
from app.rag.query.chunk_status_filter_service import get_disabled_chunk_ids_for_query
from app.rag.query.config import (
    BM25_SPARSE_FIELD,
    LEARNED_SPARSE_FIELD,
    RETRIEVAL_DEFAULT_LIMIT,
    RETRIEVAL_RRF_K,
    channels_for_retrieval_mode,
    normalize_retrieval_mode,
)
from app.rag.query.contracts import (
    IdentifierResolutionStatus,
    ObservationStatus,
    QueryAction,
    RetrievalChannel,
    RetrievalMode,
    RetrievalObservation,
)
from app.rag.query.query_identifier_service import (
    IDENTIFIER_DICTIONARY_OUTPUT_FIELDS,
    append_identifiers_to_query,
    build_not_found_question,
    build_suggestion_question,
    extract_query_identifiers,
    filter_records_matching_requested_identifiers,
    normalize_identifier_mapping,
    rank_identifier_suggestions,
    requested_identifiers_are_covered,
)
from app.shared.runtime.logger import logger, step_log


def _build_filter(
        state: Mapping[str, object],
        *,
        query_identifiers,
        disabled_chunk_ids=None,
) -> str:
    """
    使用同一份权限上下文构建检索表达式。

    第一段传入可映射的结构化标识；第二段传入空字典。两段都从同一个 State 读取
    dataset、subject、owner、tenant，因此 fallback（降级召回）只会移除编号条件，绝不
    会移除权限、dataset、subject 或 ``enabled == true``。
    """
    return build_chunk_retrieval_filter(
        dataset_ids=state.get("dataset_ids"),
        subject_ids=state.get("subject_ids"),
        owner_user_id=state.get("owner_user_id"),
        tenant_id=state.get("tenant_id"),
        query_identifiers=query_identifiers,
        disabled_chunk_ids=disabled_chunk_ids,
    )


def check_params(state):
    """校验检索文本和安全范围，并返回两阶段检索需要的稳定输入。"""
    rewritten_query = state.get("rewritten_query")
    if not rewritten_query:
        logger.error("请输入问题")
        raise ValueError("请输入问题")

    # 正常 HTTP 查询会在进入图前从 original_query 提取标识。这里仍提供服务级兜底，
    # 方便测试、离线评测和未来非 HTTP 调用复用；优先使用 original_query，避免主体改写
    # 自动补入的设备型号被误写成“用户亲自输入的编号”。
    raw_identifiers = state.get("query_identifiers")
    # 空字典是查询入口“已检查原问题但没有发现编号”的有效结果，不能用 ``or`` 把它当
    # 成缺失值后再从 rewritten_query 提取。否则主体改写补入的 HAK 180 会被误写成用户
    # 本轮亲自输入的型号。只有旧调用方完全没有该字段（值为 None）时才启用兼容兜底。
    if raw_identifiers is None:
        raw_identifiers = extract_query_identifiers(state.get("original_query") or rewritten_query)
    query_identifiers = normalize_identifier_mapping(raw_identifiers)
    structured_identifiers = select_structured_query_identifiers(query_identifiers)

    # 即使没有任何编号，也先构建基础表达式完成空 dataset/subject/owner/tenant 校验。
    # 这保证参数错误发生在 embedding 和 Milvus 调用之前，不会退化为全库搜索。
    disabled_chunk_ids = get_disabled_chunk_ids_for_query(state)
    state["disabled_chunk_ids"] = disabled_chunk_ids
    base_filter_expr = _build_filter(
        state,
        query_identifiers={},
        disabled_chunk_ids=disabled_chunk_ids,
    )
    logger.warning(f"{rewritten_query},类型{type(rewritten_query)}")
    return rewritten_query, query_identifiers, structured_identifiers, base_filter_expr, disabled_chunk_ids


def _embed_retrieval_query(retrieval_query: str) -> tuple[list[float], dict[int, float]]:
    """一次性生成 dense（稠密）和 learned sparse（学习式稀疏）向量，供两段复用。"""
    embedding_result = llm_provider.embed_documents([retrieval_query])
    return embedding_result["dense"][0], embedding_result["sparse"][0]


def query_chunk_by_milvus(
        rewritten_query: str,
        filter_expr: str,
        *,
        query_vectors: tuple[list[float], dict[int, float]] | None = None,
        retrieval_mode: RetrievalMode | str = RetrievalMode.DENSE_LEARNED_SPARSE,
        strict_errors: bool = False,
) -> list[dict]:
    """
    使用同一检索文本和指定 expr 执行 Milvus 混合搜索。

    ``query_vectors`` 允许第一段和第二段复用同一组向量，避免精确过滤零命中时重复调用
    embedding 服务。两段的区别只有 filter，不会因为降级而改变用户原问题。
    """
    dense_vector, sparse_vector = query_vectors or _embed_retrieval_query(rewritten_query)
    normalized_mode = normalize_retrieval_mode(retrieval_mode)

    reqs = milvus_gateway.create_requests(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        expr=filter_expr,
        retrieval_mode=normalized_mode.value,
        query_text=rewritten_query,
        learned_sparse_field=LEARNED_SPARSE_FIELD,
        bm25_sparse_field=BM25_SPARSE_FIELD,
    )
    hybrid_result = milvus_gateway.hybrid_search(
        collection_name=milvus_gateway.chunk_collection_name,
        reqs=reqs,
        ranker_type="rrf",
        rrf_k=RETRIEVAL_RRF_K,
        limit=RETRIEVAL_DEFAULT_LIMIT,
        output_fields=CHUNK_OUTPUT_FIELDS,
        raise_on_error=strict_errors,
    )

    if hybrid_result and hybrid_result[0]:
        retrieval_channels = [
            *channels_for_retrieval_mode(normalized_mode),
            RetrievalChannel.ORIGINAL,
        ]
        return [
            format_chunk_search_item(
                item,
                retrieval_channels=retrieval_channels,
                retrieval_rank=rank,
            )
            for rank, item in enumerate(hybrid_result[0], start=1)
        ]
    return []


def _records_with_structured_filter_guarantee(
        records: list[dict],
        structured_identifiers: Mapping[str, list[str]],
) -> list[dict]:
    """
    为第一段结果补充 expr 已经保证命中的字段，主要服务于兼容和可测试性。

    真实 Milvus output_fields 会返回这些 metadata；若旧测试桩或历史数据投影漏了字段，
    只要该记录确实来自对应 ``field in [...]`` 的查询，仍可安全认为它命中了该结构化值。
    该保证只适用于第一段结果，绝不会用于第二段宽松召回。
    """
    guaranteed_records = []
    for record in records:
        enriched = dict(record)
        for field_name, values in structured_identifiers.items():
            if not enriched.get(field_name):
                # 一个字段的多个请求值在 Milvus ``in`` 条件里是 OR。结果若缺少回传字段，
                # 无法知道具体命中哪一个；只有单值时才能安全补齐，多值仍要求实际 metadata。
                if len(values) == 1:
                    enriched[field_name] = values[0]
        guaranteed_records.append(enriched)
    return guaranteed_records


def _load_authorized_identifier_records(base_filter_expr: str) -> list[dict]:
    """
    读取当前权限范围内真实存在的结构化编号，作为纠错候选词典。

    词典查询仍携带完整基础 filter。读取失败不会扩大权限或自动采用 embedding 猜测值；
    服务会退回仅使用第二段已经召回的有权限 chunk 生成候选，并记录异常日志。
    """
    try:
        return milvus_gateway.query_entities(
            collection_name=milvus_gateway.chunk_collection_name,
            filter_expr=base_filter_expr,
            output_fields=IDENTIFIER_DICTIONARY_OUTPUT_FIELDS,
            limit=200,
        ) or []
    except Exception:
        logger.exception("读取有权限的设备标识候选词典失败，将仅使用宽松召回结果生成候选")
        return []


def _build_observation(
        *,
        chunks: list[dict],
        requested_identifiers: dict[str, list[str]],
        matched_identifiers: dict[str, list[str]],
        resolution_status: IdentifierResolutionStatus,
        suggested_identifiers: dict[str, list[str]] | None,
        clarification_question: str | None,
        used_structured_filter: bool,
        filter_fallback: bool,
        duration_ms: int,
        retrieval_mode: RetrievalMode,
) -> RetrievalObservation:
    """把两阶段检索事实收口成强校验 Observation，不在这里生成答案或 Citation。"""
    suggested_identifiers = suggested_identifiers or {}
    candidate_count = max(
        len(chunks),
        sum(len(values) for values in suggested_identifiers.values()),
    )
    return RetrievalObservation(
        action=QueryAction.LOCAL_SEARCH,
        status=ObservationStatus.SUCCESS if candidate_count else ObservationStatus.EMPTY,
        # Milvus hybrid_search 只返回模式内 RRF 后的统一列表，当前拿不到每个底层请求的
        # 独立命中数，因此按实际 retrieval_mode 记录组合结果数量，不伪造逐通道计数。
        channel_counts={retrieval_mode.value: len(chunks)},
        candidate_count=candidate_count,
        reranked_count=0,
        top_rerank_score=None,
        requested_identifiers=requested_identifiers,
        matched_identifiers=matched_identifiers,
        identifier_resolution_status=resolution_status,
        suggested_identifiers=suggested_identifiers,
        # Citation（引用）只能由最终进入答案的证据产生。任务 5 还在 rerank 之前，因此
        # 即使精确命中也保持 0；尤其 suggestion_required 必须严格为 0。
        citation_count=0,
        evidence_summaries=[],
        evidence_ambiguous=False,
        clarification_question=clarification_question,
        duration_ms=max(0, duration_ms),
        error_code=None,
        used_structured_filter=used_structured_filter,
        filter_fallback=filter_fallback,
    )


@step_log()
def search_by_embedding(state: QueryGraphState) -> QueryGraphState:
    """
    执行设备标识感知的两阶段本地混合检索。

    第一段使用“基础权限 + subject + 可映射结构化编号”精确过滤；若没有结果，或结果不
    能覆盖 SOP/零件编号等词法标识，第二段只移除结构化编号条件，并把规范化标识追加到
    query 后用 dense + learned sparse 召回。第二段仍找到相同编号时可作为同码证据；只
    找到 E021 这类相近但不同编号时，只生成追问候选，不能进入答案生成。
    """
    started_at = perf_counter()
    (
        rewritten_query,
        query_identifiers,
        structured_identifiers,
        base_filter_expr,
        disabled_chunk_ids,
    ) = check_params(state)

    state["query_identifiers"] = query_identifiers
    retrieval_mode = normalize_retrieval_mode(state.get("retrieval_mode"))
    state["retrieval_mode"] = retrieval_mode.value
    retrieval_query = append_identifiers_to_query(rewritten_query, query_identifiers)
    query_vectors = _embed_retrieval_query(retrieval_query)
    used_structured_filter = bool(structured_identifiers)
    filter_fallback = False

    # 没有设备标识时保持原有单段行为，只额外产出 NOT_APPLICABLE Observation。
    if not query_identifiers:
        chunks = query_chunk_by_milvus(
            retrieval_query,
            base_filter_expr,
            query_vectors=query_vectors,
            retrieval_mode=retrieval_mode,
            strict_errors=bool(state.get("provider_strict_errors", False)),
        )
        observation = _build_observation(
            chunks=chunks,
            requested_identifiers={},
            matched_identifiers={},
            resolution_status=IdentifierResolutionStatus.NOT_APPLICABLE,
            suggested_identifiers={},
            clarification_question=None,
            used_structured_filter=False,
            filter_fallback=False,
            duration_ms=int((perf_counter() - started_at) * 1000),
            retrieval_mode=retrieval_mode,
        )
        state["embedding_chunks"] = chunks
        state["retrieval_observation"] = observation
        return state

    # 第一段：只有 schema 已存在的标识才进入精确 expr。SOP/零件编号等 lexical-only
    # 标识仍会追加到 retrieval_query，并通过结果正文做同码核验。
    if used_structured_filter:
        exact_filter_expr = _build_filter(
            state,
            query_identifiers=structured_identifiers,
            disabled_chunk_ids=disabled_chunk_ids,
        )
        exact_chunks = query_chunk_by_milvus(
            retrieval_query,
            exact_filter_expr,
            query_vectors=query_vectors,
            retrieval_mode=retrieval_mode,
            strict_errors=bool(state.get("provider_strict_errors", False)),
        )
        guaranteed_chunks = _records_with_structured_filter_guarantee(
            exact_chunks,
            structured_identifiers,
        )
        exact_evidence, matched_identifiers = filter_records_matching_requested_identifiers(
            guaranteed_chunks,
            query_identifiers,
        )
        if requested_identifiers_are_covered(query_identifiers, matched_identifiers):
            observation = _build_observation(
                chunks=list(exact_evidence),
                requested_identifiers=query_identifiers,
                matched_identifiers=matched_identifiers,
                resolution_status=IdentifierResolutionStatus.EXACT_MATCH,
                suggested_identifiers={},
                clarification_question=None,
                used_structured_filter=True,
                filter_fallback=False,
                duration_ms=int((perf_counter() - started_at) * 1000),
                retrieval_mode=retrieval_mode,
            )
            state["embedding_chunks"] = list(exact_evidence)
            state["retrieval_observation"] = observation
            return state
        filter_fallback = True

    # 第二段：无可映射字段时，这是首次基础召回；尝试过结构化 filter 时，这是 fallback。
    # 两种情况都使用完整 base_filter_expr，绝不删除权限、dataset、subject 或 enabled。
    broad_chunks = query_chunk_by_milvus(
        retrieval_query,
        base_filter_expr,
        query_vectors=query_vectors,
        retrieval_mode=retrieval_mode,
        strict_errors=bool(state.get("provider_strict_errors", False)),
    )
    exact_evidence, matched_identifiers = filter_records_matching_requested_identifiers(
        broad_chunks,
        query_identifiers,
    )
    if requested_identifiers_are_covered(query_identifiers, matched_identifiers):
        resolution_status = (
            IdentifierResolutionStatus.FALLBACK_EXACT_MATCH
            if filter_fallback
            else IdentifierResolutionStatus.EXACT_MATCH
        )
        observation = _build_observation(
            chunks=list(exact_evidence),
            requested_identifiers=query_identifiers,
            matched_identifiers=matched_identifiers,
            resolution_status=resolution_status,
            suggested_identifiers={},
            clarification_question=None,
            used_structured_filter=used_structured_filter,
            filter_fallback=filter_fallback,
            duration_ms=int((perf_counter() - started_at) * 1000),
            retrieval_mode=retrieval_mode,
        )
        state["embedding_chunks"] = list(exact_evidence)
        state["retrieval_observation"] = observation
        return state

    # 没有同码证据时，候选只能来自当前基础 filter 下的词典和宽松召回 chunk。即使候选
    # 编辑距离很近，也只写 suggested_identifiers，不能覆盖 query_identifiers。
    authorized_records = [*_load_authorized_identifier_records(base_filter_expr), *broad_chunks]
    suggested_identifiers = rank_identifier_suggestions(query_identifiers, authorized_records)
    if suggested_identifiers:
        resolution_status = IdentifierResolutionStatus.SUGGESTION_REQUIRED
        clarification_question = build_suggestion_question(query_identifiers, suggested_identifiers)
    else:
        resolution_status = IdentifierResolutionStatus.NOT_FOUND
        clarification_question = build_not_found_question(query_identifiers)

    observation = _build_observation(
        chunks=broad_chunks,
        requested_identifiers=query_identifiers,
        # 部分编号命中但整体不兼容时也不能作为答案证据，Observation 不把它声明为 matched。
        matched_identifiers={},
        resolution_status=resolution_status,
        suggested_identifiers=suggested_identifiers,
        clarification_question=clarification_question,
        used_structured_filter=used_structured_filter,
        filter_fallback=filter_fallback,
        duration_ms=int((perf_counter() - started_at) * 1000),
        retrieval_mode=retrieval_mode,
    )
    state["embedding_chunks"] = broad_chunks
    state["retrieval_observation"] = observation
    state["clarification_question"] = clarification_question
    return state
