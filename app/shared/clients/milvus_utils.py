"""
工具模块，负责提供 milvus 相关的辅助能力。
"""
from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker, WeightedRanker
from app.shared.config.milvus_config import milvus_config
from app.shared.runtime.logger import logger

# 全局Milvus客户端实例，实现单例复用
_milvus_client: MilvusClient | None = None


def get_milvus_client() -> MilvusClient | None:
    """
    Milvus客户端单例获取方法
    实现客户端连接复用，避免重复创建连接消耗资源
    :return: MilvusClient实例，连接失败返回None
    """
    try:
        global _milvus_client
        # 单例判断：未初始化则创建新连接
        if _milvus_client is None:
            milvus_uri = milvus_config.milvus_url
            # 校验Milvus连接地址配置
            if not milvus_uri:
                logger.error("Milvus客户端连接失败：缺少MILVUS_URL环境变量配置")
                return None
            # 初始化Milvus客户端
            _milvus_client = MilvusClient(uri=milvus_uri,token=milvus_config.milvus_token)
            logger.info("Milvus客户端连接成功")
        return _milvus_client
    except Exception as e:
        logger.error(f"Milvus客户端连接异常：{str(e)}", exc_info=True)
        return None


def create_hybrid_search_requests(
        dense_vector,
        sparse_vector,
        dense_params=None,
        sparse_params=None,
        expr=None,
        limit=5,
        *,
        retrieval_mode="dense_learned_sparse",
        query_text=None,
        # 共享工具还服务于标准主题/别名 collection，因此默认字段继续叫 sparse_vector；
        # chunk 检索会显式传入 learned_sparse_vector，不能在这里全局改名。
        learned_sparse_field="sparse_vector",
        bm25_sparse_field="bm25_sparse_vector",
        bm25_params=None,
):
    """
    构建Milvus混合搜索请求对象
    分别创建稠密/稀疏向量的搜索请求，用于后续混合搜索融合
    :param dense_vector: 文本生成的稠密向量
    :param sparse_vector: 文本生成的稀疏向量
    :param dense_params: 稠密向量搜索参数，默认使用余弦相似度
    :param sparse_params: 稀疏向量搜索参数，默认使用内积相似度
    :param expr: 搜索过滤表达式，用于精准筛选数据
    :param limit: 单向量搜索返回结果数量，默认5
    :param retrieval_mode: 本次本地 Action 的召回组合，固定为三种关闭模式之一
    :param query_text: BM25 请求的原始/增强查询文本；只有包含 BM25 的模式必填
    :param learned_sparse_field: BGE-M3 学习式稀疏向量字段名
    :param bm25_sparse_field: Milvus BM25 Function 输出稀疏字段名
    :return: 按 dense、learned sparse、BM25 固定顺序生成的 2 或 3 个请求
    """
    # 稠密向量默认搜索参数：余弦相似度（COSINE），适配BGE-M3稠密向量并与建库参数保持一致
    if dense_params is None:
        dense_params = {"metric_type": "COSINE"}
    # 稀疏向量默认搜索参数：内积（IP），适配BGE-M3稀疏向量
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}
    if bm25_params is None:
        bm25_params = {"metric_type": "BM25"}

    supported_modes = {
        "dense_learned_sparse",
        "dense_bm25",
        "dense_learned_sparse_bm25",
    }
    if retrieval_mode not in supported_modes:
        raise ValueError(
            f"不支持的 retrieval_mode={retrieval_mode!r}，可选值：{', '.join(sorted(supported_modes))}"
        )

    # 构建稠密向量搜索请求，关联Milvus的dense_vector字段 近似最近邻（ANN）检索请求的核心类
    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        expr=expr, # 混合搜索的过滤条件   # 单列搜索 过滤条件 filter =
        limit=limit
    )

    requests = [dense_req]

    if retrieval_mode in {"dense_learned_sparse", "dense_learned_sparse_bm25"}:
        if sparse_vector is None:
            raise ValueError("包含 learned sparse 的 retrieval_mode 必须提供 sparse_vector")
        # learned sparse 的中文含义是“模型学习式稀疏向量”。它不是 BM25，两者必须使用
        # 不同字段和 metric，避免把 BGE-M3 sparse 误写成传统关键词分数。
        requests.append(AnnSearchRequest(
            data=[sparse_vector],
            anns_field=learned_sparse_field,
            param=sparse_params,
            expr=expr,
            limit=limit,
        ))

    if retrieval_mode in {"dense_bm25", "dense_learned_sparse_bm25"}:
        normalized_query_text = str(query_text or "").strip()
        if not normalized_query_text:
            raise ValueError("包含 BM25 的 retrieval_mode 必须提供非空 query_text")
        # BM25 请求直接提交文本。Milvus Function 使用与入库 lexical_text 相同的 Analyzer
        # 生成查询稀疏表示，应用侧不能把 BGE-M3 sparse_vector 填入 BM25 字段。
        requests.append(AnnSearchRequest(
            data=[normalized_query_text],
            anns_field=bm25_sparse_field,
            param=bm25_params,
            expr=expr,
            limit=limit,
        ))

    return requests


def hybrid_search(
        client,
        collection_name,
        reqs,
        ranker_weights=(0.5, 0.5),
        norm_score=False,
        limit=5,
        output_fields=None,
        search_params=None,
        *,
        ranker_type="weighted",
        rrf_k=60,
        raise_on_error=False,
):
    """
    执行Milvus稠密+稀疏向量混合搜索
    基于WeightedRanker实现双向量搜索结果加权融合，提升检索准确性
    :param client: MilvusClient实例
    :param collection_name: 集合名称
    :param reqs: 搜索请求列表，固定为[dense_req, sparse_req]
    :param ranker_weights: 加权融合权重，默认(0.5,0.5)，依次对应稠密/稀疏向量
    :param norm_score: 是否归一化评分后再融合，避免评分量级差异导致权重失效
    :param limit: 混合搜索最终返回结果数量，默认5
    :param output_fields: 需要返回的字段列表，默认返回standard_subject_name
    :param search_params: 搜索参数，如ef/topk等，默认None
    :return: 混合搜索结果列表，搜索失败返回None
    """
    try:
        if ranker_type == "weighted":
            if len(ranker_weights) != len(reqs):
                raise ValueError("WeightedRanker 的权重数量必须与 AnnSearchRequest 数量一致")
            # 标准主题/别名 collection 继续使用历史加权策略，避免 chunk 改造破坏阶段 2。
            rerank = WeightedRanker(*ranker_weights, norm_score=norm_score)
        elif ranker_type == "rrf":
            if rrf_k <= 0:
                raise ValueError("RRFRanker 的 k 必须大于 0")
            # chunk 的三路原始分数量级不同，使用名次融合而不是直接相加原始分数。
            rerank = RRFRanker(k=rrf_k)
        else:
            raise ValueError("ranker_type 只支持 weighted 或 rrf")

        # 默认返回字段：文档标识字段
        if output_fields is None:
            output_fields = ["standard_subject_name"]

        # 执行混合搜索：融合稠密+稀疏向量结果，按权重重新排序
        res = client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params
        )
        # res [[{id:111,distance:0.9,entity:{standard_subject_name:烫金机}},{},{},{},{}]]
        logger.info(f"Milvus混合搜索完成，集合[{collection_name}]共检索到{len(res[0])}条结果")
        return res
    except Exception as e:
        logger.error(f"Milvus混合搜索执行失败，集合[{collection_name}]：{str(e)}", exc_info=True)
        # 正式 GRPO（群组相对策略优化）训练会开启 raise_on_error（失败时抛错），
        # 防止把基础设施异常伪装成“真实空召回”后继续计算 Reward（奖励分数）。
        if raise_on_error:
            raise
        return None
