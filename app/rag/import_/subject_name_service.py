from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import DataType

from app.infra.llm.providers import llm_provider
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import SUBJECT_NAME_CONTEXT_CHUNK_K, SUBJECT_NAME_CONTEXT_TOTAL_MAX_CHARS, \
    MILVUS_DEFAULT_VARCHAR_MAX_LENGTH, MILVUS_VECTOR_DIM
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log
from app.shared.utils.escape_milvus_string_utils import escape_milvus_string


def validate_chunks_and_title(state):
    # 1.参数获取
    chunks, file_title = state.get("chunks"), state.get("file_title")

    # 2.判空
    if not chunks:
        logger.error("chunks为空，无法进行主体识别！")
        raise ValueError("chunks为空，无法进行主体识别！")
    if not file_title:
        file_title = chunks[0].get("file_title") or chunks[0].get("title") or "default_file_title"
        state["file_title"] = file_title

    return chunks, file_title


def build_document_context(chunks, file_title):
    # 1.获取数据
    chunks = chunks[:SUBJECT_NAME_CONTEXT_CHUNK_K]

    # 2.拼接上下文
    context = "".join([chunk.get("content") for chunk in chunks])

    # 3.限制最大长度
    context = context[:SUBJECT_NAME_CONTEXT_TOTAL_MAX_CHARS]

    return context


def recognize_subject_name(context, file_title):
    # 1.获取llm
    llm = llm_provider.chat()

    # 2.llm请求参数组装
    # 2.1 加载提示词模板，组装提示词
    system_prompt = load_prompt("product_recognition_system")
    human_prompt = load_prompt("subject_name_recognition", file_title=file_title, context=context)

    # 2.2 构造消息列表
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]
    chains = llm | StrOutputParser()

    # 3.调用llm, 获取结果
    result = chains.invoke(messages)

    # 4.判空,file_title作为兜底
    if not result:
        result = file_title

    return result


@step_log()
def generate_embeddings(subject_name):
    vector_dict = llm_provider.embed_documents([subject_name])
    return vector_dict["dense"][0], vector_dict["sparse"][0]


@step_log()
def prepare_subject_name_collection(state):
    milvus_client = milvus_gateway.client
    # 集合名称
    collection_name = milvus_gateway.subject_name_collection
    # 判断集合是否存在
    if milvus_client.has_collection(collection_name=collection_name):
        return

    # 创建schema
    schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name="subject_name", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=MILVUS_VECTOR_DIM)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    # 准备索引参数
    index_params = milvus_client.prepare_index_params()
    # 为稠密向量创建索引
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 64, "efConstruction": 100}
    )
    # 为稀疏向量创建索引
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
    )

    # 创建集合
    milvus_client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    logger.info(f"集合{collection_name}初始化成功！")


@step_log()
def insert_subject_name(subject_name, file_title, dense_vector, sparse_vector):
    milvus_client = milvus_gateway.client
    # 数据转义处理
    subject_name = escape_milvus_string(subject_name)

    # 1.删除已有记录
    milvus_client.delete(
        collection_name=milvus_gateway.subject_name_collection,
        filter=f"file_title=='{file_title}'"
    )
    # 2.插入数据
    milvus_client.insert(
        collection_name=milvus_gateway.subject_name_collection,
        data=[
            {
                "file_title": file_title,
                "subject_name": subject_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector
            }
        ]
    )


@step_log()
def recognize_and_index_subject_name(state: ImportGraphState) -> ImportGraphState:
    """
    主体识别服务：
    1. 基于 chunks 构造上下文
    2. 调用 LLM 识别 subject_name
    3. 将 subject_name 回填到 state 和 chunks
    4. 同步写入主体名称索引
    """
    # 1.参数校验
    chunks, file_title = validate_chunks_and_title(state)

    # 2.基于chunks构造上下文
    context = build_document_context(chunks, file_title)

    # 3.调用LLM识别subject_name
    subject_name = recognize_subject_name(context, file_title)

    # 4.将subject_name回写到state和chunks
    state["subject_name"] = subject_name
    for chunk in chunks:
        chunk["subject_name"] = subject_name
    state["chunks"] = chunks

    # 5.生成主体名称的稠密向量和稀疏向量
    dense_vector, sparse_vector = generate_embeddings(subject_name)

    # 6.构建集合
    prepare_subject_name_collection(state)

    # 7.入库
    insert_subject_name(subject_name, file_title, dense_vector, sparse_vector)

    return state
