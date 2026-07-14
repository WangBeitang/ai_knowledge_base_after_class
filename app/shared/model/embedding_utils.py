"""
工具模块，负责提供 embedding 相关的辅助能力。
"""
from threading import Lock

from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from app.shared.config.embedding_config import embedding_config
from app.shared.runtime.logger import logger

_DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
_DEFAULT_EMBEDDING_DEVICE = "cpu"
_bge_m3_ef: BGEM3EmbeddingFunction | None = None

# initialization 的中文含义是“初始化”。导入服务会用后台线程同时处理多份文档，
# 如果多个线程在模型单例仍为空时一起执行 Transformers.from_pretrained，可能让
# 某个线程拿到尚未装载真实权重的 meta tensor（只有形状、没有数据的占位张量）。
# 初始化锁保证同一 Python 进程内只有一个线程负责构造完整模型。
_bge_m3_initialization_lock = Lock()

# encoding 的中文含义是“编码/生成向量”。FlagEmbedding 每次编码内部会调用
# model.to(device)、model.eval() 等会接触模型状态的方法，因此同一进程内复用
# 一个模型时串行执行编码，避免多份导入任务同时操作同一个 PyTorch Module。
# 该锁不限制 MinerU、Markdown 处理等其它导入步骤，也不跨 Uvicorn worker 生效。
_bge_m3_encoding_lock = Lock()


def get_bge_m3_ef() -> BGEM3EmbeddingFunction:
    """
    获取BGE-M3模型单例对象，自动加载环境变量配置
    :return: 初始化完成的BGEM3EmbeddingFunction实例
    """
    global _bge_m3_ef
    # 单例模式：已初始化则直接返回，避免重复加载模型
    if _bge_m3_ef is not None:
        logger.debug("BGE-M3模型单例已存在，直接返回实例")
        return _bge_m3_ef

    # 锁内再次判空是“双重检查”：当前线程等待锁期间，前一个线程可能已经完成初始化。
    # 如果不做第二次判断，排队线程仍会重复加载一次大模型。
    with _bge_m3_initialization_lock:
        if _bge_m3_ef is not None:
            logger.debug("等待初始化锁期间BGE-M3模型已加载，直接复用实例")
            return _bge_m3_ef

        # 从环境变量加载配置，无配置则使用默认值。
        # 本地有模型时优先使用本地地址，否则使用模型名交给底层依赖下载。
        model_name = embedding_config.bge_m3_path or embedding_config.bge_m3 or _DEFAULT_EMBEDDING_MODEL
        device = embedding_config.bge_device or _DEFAULT_EMBEDDING_DEVICE
        use_fp16 = embedding_config.bge_fp16

        logger.info(
            "开始初始化BGE-M3模型",
            extra={
                "model_name": model_name,
                "device": device,
                "use_fp16": use_fp16,
                "normalize_embeddings": True
            }
        )

        try:
            # 先构造局部变量，确认完整成功后再发布到全局单例。
            # 如果构造期间异常，全局值仍为 None，后续重建任务可以重新尝试初始化，
            # 不会复用半初始化或仍含 meta tensor 的损坏模型。
            initialized_model = BGEM3EmbeddingFunction(
                model_name=model_name,
                device=device,
                use_fp16=use_fp16,
                normalize_embeddings=True  # 模型原生对稠密+稀疏向量做L2归一化
            )
        except Exception as e:
            logger.error(f"BGE-M3模型初始化失败：{str(e)}", exc_info=True)
            raise  # 向上抛出异常，由调用方处理

        _bge_m3_ef = initialized_model
        logger.success("BGE-M3模型初始化成功，已开启原生L2归一化")
        return _bge_m3_ef


def generate_embeddings(texts: list[str]) -> dict[str, list]:
    """
    为文本列表生成稠密+稀疏混合向量嵌入（模型原生L2归一化）
    :param texts: 要生成嵌入的文本列表，单文本也需封装为列表
    :return: 字典格式的向量结果，key为dense/sparse，对应嵌套列表/字典列表
    :raise: 向量生成过程中的异常，由调用方捕获处理
    """
    # 入参合法性校验
    if not isinstance(texts, list) or len(texts) == 0:
        logger.warning("生成向量入参不合法，texts必须为非空列表")
        raise ValueError("参数texts必须是包含文本的非空列表")
    if any(not isinstance(text, str) for text in texts):
        logger.warning("生成向量入参不合法，texts中存在非字符串内容")
        raise ValueError("参数texts必须是字符串列表")

    logger.info(f"开始为{len(texts)}条文本生成混合向量嵌入")
    try:
        # 加载BGE-M3模型单例
        model = get_bge_m3_ef()
        # 模型编码生成向量，返回 dense（稠密向量）+ sparse（CSR 格式稀疏向量）。
        # 同一个模型实例的编码过程使用独立锁；初始化锁只保护首次构造，二者职责分离。
        with _bge_m3_encoding_lock:
            embeddings = model.encode_documents(texts)
        logger.debug(f"模型编码完成，开始解析稀疏向量格式，共{len(texts)}条")

        # 初始化稀疏向量处理结果，解析为字典格式（适配序列化/存储）
        processed_sparse = []
        # # 把模型输出的 CSR 稀疏矩阵 ，按“每条文本一行”拆成 {特征索引: 权重} 字典
        # # - indices ：非零元素的“列号（特征ID）”
        # # - data ：对应列号的权重值
        # # - indptr ：每一行在 indices/data 里的起止位置指针
        # # 数据示例:
        # # indices = [3, 8, 20, 1, 9]
        # # data    = [0.7, 0.2, 0.1, 0.6, 0.4]  -> milvus -> 稠密向量 [1024] 稀疏向量 : {index:值,index:值}
        # # indptr  = [0, 3, 5]
        # # 获取对应的数据
        # # - 第0条文本用 0:3 => indices=[3,8,20] , data=[0.7,0.2,0.1]
        # # - 第1条文本用 3:5 => indices=[1,9] , data=[0.6,0.4]
        for i in range(len(texts)):
            # 提取第i个文本的稀疏向量索引：np.int64 → Python int（满足字典key可哈希要求）
            sparse_indices = embeddings["sparse"].indices[
                embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]
            ].tolist()
            # 提取第i个文本的稀疏向量权重：np.float32 → Python float（适配JSON序列化/接口返回）
            sparse_data = embeddings["sparse"].data[
                embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]
            ].tolist()
            # 构造{特征索引: 归一化权重}的稀疏向量字典
            sparse_dict = {k: v for k, v in zip(sparse_indices, sparse_data)}
            processed_sparse.append(sparse_dict)

        # 构造最终返回结果，稠密向量转列表（解决numpy数组不可序列化问题）
        result = {
            # embeddings["dense"] = [[1稠密向量],[2稠密向量],[...]  -> 1024]
            # embeddings["sparse"] = [[1稀疏向量],[2稠密向量],[...]  -> 1024]
            "dense": [emb.tolist() for emb in embeddings["dense"]],  # 嵌套列表，与输入文本一一对应
            "sparse": processed_sparse  # 字典列表，模型已做L2归一化
        }
        logger.success(f"{len(texts)}条文本向量生成完成，格式已适配工业级使用")
        return result

    except Exception as e:
        logger.error(f"文本向量生成失败：{str(e)}", exc_info=True)
        raise  # 不吞异常，向上传递让调用方做重试/降级处理

