from pymilvus import DataType

from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import MILVUS_DEFAULT_VARCHAR_MAX_LENGTH
from app.shared.runtime.logger import logger,step_log

@step_log()
def params_check(state):
    chunks = state.get("chunks")
    if not chunks:
        logger.error("chunks为空，无法进行向量化！")
        raise ValueError("chunks为空，无法进行向量化！")
    file_title = state.get("file_title")
    if not file_title:
        logger.error("file_title为空，无法进行向量化！")
        raise ValueError("file_title为空，无法进行向量化！")
    return chunks, file_title


@step_log()
def prepare_chunks_collection(state):
    # 1.获取Milvus客户端
    client = milvus_gateway.client

    # 2.检查集合是否已存在
    collection_name = milvus_gateway.chunk_collection_name
    if client.has_collection(collection_name=collection_name):
        logger.info(f"集合{collection_name}已存在，无需创建")
        return

    # 3.schema构建
    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    # 主键
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    # 内容
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
    # 原始文件标题
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    # 标题
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    # 父标题
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    # 分片顺序标记
    schema.add_field(field_name="part", datatype=DataType.INT8)
    # 所属主体名称
    schema.add_field(field_name="subject_name", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    # 稠密向量
    schema.add_field(field_name="dense_vector",datatype=DataType.FLOAT_VECTOR, dim=1024)
    # 稀疏向量
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    # 4.索引构建
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 64, "efConstruction": 100}
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
    )

    # 5.创建集合
    if not client.has_collection(collection_name):
        try:
            client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
            )
            logger.info(f"集合 {collection_name} 初始化成功")
        except Exception as e:
            # 如果是并发导致的 already exists，可以忽略
            if "already" in str(e).lower() and "exist" in str(e).lower():
                logger.info(f"集合 {collection_name} 已被其他任务创建")
            else:
                raise e

    logger.info(f"集合{collection_name}初始化成功！")

    # 6.让 Milvus 的 QueryNode 能够拿这个集合做 search/query
    client.load_collection(collection_name=collection_name)


@step_log()
def remove_old_chunks(file_title):
    client = milvus_gateway.client
    client.delete(
        collection_name=milvus_gateway.chunk_collection_name,
        filter=f"file_title=='{file_title}'"
    )


@step_log()
def insert_chunks(chunks):
    client = milvus_gateway.client
    result = client.insert(
        collection_name=milvus_gateway.chunk_collection_name,
        data=chunks,
    )
    logger.info(f"向集合 {milvus_gateway.chunk_collection_name} 中插入了 {result.get('insert_count', 0)} 条数据")

    # 回写 chunk_id
    ids = result.get("ids", [])
    if ids and len(ids) == len(chunks):
        for i, chunk in enumerate(chunks):
            chunk["chunk_id"] = ids[i]



@step_log()
def index_chunks(state: ImportGraphState) -> ImportGraphState:
    """
    入库服务：
    1. 准备集合 schema 和索引
    2. 根据 file_title 删除旧数据
    3. 批量插入新的 chunks
    4. 回写 chunk_id 等入库结果
    """
    # 1.参数校验
    chunks, file_title = params_check(state)

    # 2.准备集合 schema 和索引
    prepare_chunks_collection(state)

    # 3.根据file_title删除旧数据
    remove_old_chunks(file_title)

    # 4.插入新数据
    insert_chunks(chunks)

    return state
