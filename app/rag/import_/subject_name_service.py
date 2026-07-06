import hashlib
import re

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

# 这些字段是阶段 2 引入的轻量领域标签。
# 它们不直接参与“标准主题”的唯一性判断，但会跟随 chunk 一起入库，
# 后续可以用于更细粒度的过滤，例如按设备型号、报警码、部件或维护阶段检索。
SUBJECT_DOMAIN_FIELDS = (
    "equipment_model",
    "alarm_code",
    "part_name",
    "sop_type",
    "safety_level",
    "maintenance_stage",
)


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


def normalize_subject_name(subject_name, fallback=""):
    """
    归一化主题名称。

    当前只做轻量处理：把连续空白压缩成单个空格，并去掉首尾空白。
    这里不做复杂同义词替换，避免在导入阶段误把两个不同设备合并。
    fallback 用于 LLM 没有识别出主题时兜底，例如使用文件名。
    """
    subject_name = re.sub(r"\s+", " ", str(subject_name or "")).strip()
    if subject_name:
        return subject_name
    return re.sub(r"\s+", " ", str(fallback or "")).strip()


def build_subject_id(standard_subject_name):
    """
    根据标准主题名生成稳定 subject_id。

    这里使用标准主题名归一化后的 sha1 摘要，而不是 Milvus 自增主键：
    1. 同一个标准主题重复导入时能得到同一个 subject_id，便于幂等覆盖。
    2. 后续 chunk、别名、标准主题都可以用这个 ID 关联。
    3. 即使 Milvus 记录被删除重建，业务侧关联 ID 也不会变化。

    注意：如果未来支持人工修改标准主题名，需要额外维护 subject_id 映射，
    否则标准名变化会导致生成新的 subject_id。
    """
    normalized_name = normalize_subject_name(standard_subject_name).lower()
    digest = hashlib.sha1(normalized_name.encode("utf-8")).hexdigest()[:16]
    return f"subject_{digest}"


def build_subject_aliases(standard_subject_name, file_title, llm_subject_name):
    """
    构造标准主题的别名列表。

    第一版只使用三个稳定来源：
    - standard_subject_name：系统最终采用的标准名。
    - file_title：文件名经常包含设备型号、手册名称，是很强的别名来源。
    - llm_subject_name：保留模型原始识别结果，便于兼容模型输出和标准名不完全一致的情况。

    去重时忽略大小写，并保留首次出现的写法。这样可以保证别名 collection
    中“一条别名一条数据”，同时避免同义重复导致召回结果膨胀。
    """
    aliases = []
    seen = set()
    for alias in (standard_subject_name, file_title, llm_subject_name):
        normalized_alias = normalize_subject_name(alias)
        if not normalized_alias:
            continue
        alias_key = normalized_alias.lower()
        if alias_key in seen:
            continue
        aliases.append(normalized_alias)
        seen.add(alias_key)
    return aliases


def recognize_standard_subject_name(context, file_title):
    """
    识别标准主题名。

    目前仍复用原来的 subject_name prompt 和 LLM 调用，先把模型输出作为标准名。
    后续如果增加人工主题库或标准化规则，可以在这里接入：
    - 先用 LLM 提取候选主体。
    - 再和标准主题 collection 做相似匹配。
    - 命中已有标准主题时复用已有 subject_id。
    """
    llm_subject_name = recognize_subject_name(context, file_title)
    standard_subject_name = normalize_subject_name(llm_subject_name, fallback=file_title)
    return standard_subject_name, llm_subject_name


def _build_hybrid_index_params(milvus_client):
    """
    构建稠密 + 稀疏混合检索索引。

    标准主题、别名、chunk 都使用同一套向量形态：
    - dense_vector：用于语义相似度召回。
    - sparse_vector：用于关键词和型号等字面匹配。

    抽成公共函数可以减少 collection schema 演进时的重复修改。
    """
    index_params = milvus_client.prepare_index_params()
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
    return index_params


def _require_collection_name(collection_name, config_name):
    """
    校验 Milvus collection 配置是否存在。

    新增的标准主题/别名集合是阶段 2 的关键索引，如果环境变量缺失，
    这里提前抛出明确错误，避免后面在 Milvus create/insert 阶段出现难定位的问题。
    """
    if not collection_name:
        raise ValueError(f"缺少{config_name}配置，无法初始化Milvus集合")
    return collection_name


@step_log()
def generate_embeddings(subject_name):
    vector_dict = llm_provider.embed_documents([subject_name])
    return vector_dict["dense"][0], vector_dict["sparse"][0]


@step_log()
def generate_batch_embeddings(text_list):
    """
    批量生成别名向量。

    别名 collection 是“一个别名一条数据”，因此需要为每个 alias 单独生成向量。
    查询侧用户输入的别名也会向量化后检索这个集合，从而找到对应 subject_id。
    """
    if not text_list:
        return []

    vector_dict = llm_provider.embed_documents(text_list)
    dense_vectors = vector_dict["dense"]
    sparse_vectors = vector_dict["sparse"]
    return [
        {
            "dense_vector": dense_vectors[index],
            "sparse_vector": sparse_vectors[index],
        }
        for index in range(len(text_list))
    ]


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
    index_params = _build_hybrid_index_params(milvus_client)

    # 创建集合
    milvus_client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    logger.info(f"集合{collection_name}初始化成功！")


@step_log()
def prepare_standard_subject_collection(state):
    """
    准备标准主题 collection。

    一条标准主题一条记录，主要职责是管理知识体系本身：
    - subject_id：业务关联主键，chunk 和别名都引用它。
    - standard_subject_name：对外展示和 prompt 使用的标准名。
    - subject_aliases_text：当前用字符串保存别名快照，便于排查和展示。
    - 轻量领域字段：给后续按设备型号、报警码、维护阶段等过滤预留字段。
    - dense/sparse vector：后续可支持“导入时先匹配已有标准主题”。
    """
    milvus_client = milvus_gateway.client
    collection_name = _require_collection_name(
        milvus_gateway.standard_subject_collection,
        "STANDARD_SUBJECT_COLLECTION",
    )
    if milvus_client.has_collection(collection_name=collection_name):
        return

    schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="subject_id", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    schema.add_field(
        field_name="standard_subject_name",
        datatype=DataType.VARCHAR,
        max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH,
    )
    schema.add_field(field_name="subject_name", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name="subject_aliases_text", datatype=DataType.VARCHAR, max_length=4096)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    for field_name in SUBJECT_DOMAIN_FIELDS:
        schema.add_field(field_name=field_name, datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=MILVUS_VECTOR_DIM)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    index_params = _build_hybrid_index_params(milvus_client)
    milvus_client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    logger.info(f"集合{collection_name}初始化成功！")


@step_log()
def prepare_subject_alias_collection(state):
    """
    准备别名 collection。

    这个集合专门服务“用户输入别名 -> 标准主题”的识别链路。
    设计为一条别名一条数据，而不是把别名数组塞进标准主题记录，是为了：
    1. 每个别名都有自己的向量，召回更准确。
    2. 可以记录 alias_type，区分标准名、文件名、LLM 原始识别名等来源。
    3. 后续人工增删别名时，只需要操作单条别名记录。
    """
    milvus_client = milvus_gateway.client
    collection_name = _require_collection_name(
        milvus_gateway.subject_alias_collection,
        "SUBJECT_ALIAS_COLLECTION",
    )
    if milvus_client.has_collection(collection_name=collection_name):
        return

    schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="alias", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name="alias_type", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="subject_id", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    schema.add_field(
        field_name="standard_subject_name",
        datatype=DataType.VARCHAR,
        max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH,
    )
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=MILVUS_VECTOR_DIM)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    index_params = _build_hybrid_index_params(milvus_client)
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


def _domain_field_payload(state):
    return {field_name: state.get(field_name, "") for field_name in SUBJECT_DOMAIN_FIELDS}


@step_log()
def insert_standard_subject(
        subject_id,
        standard_subject_name,
        subject_aliases,
        file_title,
        dense_vector,
        sparse_vector,
        state,
):
    """
    写入标准主题记录。

    当前按 subject_id 先删后插，保证同一个标准主题重复导入时只保留最新记录。
    这不是最终的文档级幂等方案；阶段 4 引入 document_id/index_version 后，
    chunk 幂等会迁移到文档维度。但标准主题本身天然适合按 subject_id 覆盖。
    """
    milvus_client = milvus_gateway.client
    collection_name = _require_collection_name(
        milvus_gateway.standard_subject_collection,
        "STANDARD_SUBJECT_COLLECTION",
    )

    escaped_subject_id = escape_milvus_string(subject_id)
    milvus_client.delete(
        collection_name=collection_name,
        filter=f'subject_id=="{escaped_subject_id}"',
    )

    milvus_client.insert(
        collection_name=collection_name,
        data=[
            {
                "subject_id": subject_id,
                "standard_subject_name": standard_subject_name,
                "subject_name": standard_subject_name,
                "subject_aliases_text": "|".join(subject_aliases),
                "file_title": file_title,
                **_domain_field_payload(state),
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector,
            }
        ],
    )


def _alias_type(alias, standard_subject_name, file_title, llm_subject_name):
    """
    标记别名来源。

    alias_type 不是检索必要字段，但对排查很有价值：
    - standard：标准主题名本身。
    - file_title：来自文件标题，通常覆盖手册名和设备型号。
    - llm：模型原始识别结果。
    - generated：后续如果增加规则生成别名，可以归到这里。
    """
    if alias == standard_subject_name:
        return "standard"
    if alias == file_title:
        return "file_title"
    if alias == llm_subject_name:
        return "llm"
    return "generated"


@step_log()
def insert_subject_aliases(subject_id, standard_subject_name, subject_aliases, file_title, llm_subject_name):
    """
    写入别名索引。

    别名索引的关键约束是“一条别名一条数据”。查询阶段只要用户问题命中
    任意 alias，就能通过该记录拿到 subject_id 和 standard_subject_name。

    这里按 subject_id + file_title 删除旧别名，是为了让同一份文件重复导入时
    不产生重复别名。后续阶段有 document_id 后，可以把删除条件收敛到 document_id。
    """
    if not subject_aliases:
        return

    milvus_client = milvus_gateway.client
    collection_name = _require_collection_name(
        milvus_gateway.subject_alias_collection,
        "SUBJECT_ALIAS_COLLECTION",
    )
    escaped_subject_id = escape_milvus_string(subject_id)
    escaped_file_title = escape_milvus_string(file_title)

    milvus_client.delete(
        collection_name=collection_name,
        filter=f'subject_id=="{escaped_subject_id}" and file_title=="{escaped_file_title}"',
    )

    embeddings = generate_batch_embeddings(subject_aliases)
    alias_rows = []
    for index, alias in enumerate(subject_aliases):
        alias_rows.append(
            {
                "alias": alias,
                "alias_type": _alias_type(alias, standard_subject_name, file_title, llm_subject_name),
                "subject_id": subject_id,
                "standard_subject_name": standard_subject_name,
                "file_title": file_title,
                "dense_vector": embeddings[index]["dense_vector"],
                "sparse_vector": embeddings[index]["sparse_vector"],
            }
        )

    milvus_client.insert(collection_name=collection_name, data=alias_rows)


def apply_subject_to_chunks(state, chunks, subject_id, standard_subject_name):
    """
    把标准主题信息回填到每个 chunk。

    查询链路最终检索的是 chunk collection，因此 chunk 必须直接携带 subject_id。
    后续查询可以优先用 subject_id 过滤，而不是用容易变化的 subject_name 文本过滤。
    同时保留 subject_name，是为了兼容旧查询链路和已有测试。
    """
    for chunk in chunks:
        chunk["subject_id"] = subject_id
        chunk["standard_subject_name"] = standard_subject_name
        chunk["subject_name"] = standard_subject_name
        for field_name in SUBJECT_DOMAIN_FIELDS:
            chunk[field_name] = state.get(field_name, "")
    return chunks


@step_log()
def recognize_and_index_subject_name(state: ImportGraphState) -> ImportGraphState:
    """
    主体识别服务：
    1. 基于 chunks 构造上下文
    2. 调用 LLM 识别标准主题名
    3. 生成稳定 subject_id 和别名列表
    4. 写入标准主题集合、别名集合，并保留旧主体集合写入
    5. 将 subject_id / standard_subject_name / subject_name 回填到 state 和 chunks

    这一版仍然沿用原来的图节点名称和旧字段 subject_name：
    - 对外新增标准主题体系字段，供阶段 2 后续查询侧改造使用。
    - 对内保留 subject_name_collection 写入，保证旧查询逻辑还能工作。
    - chunk 同时写入新旧字段，支持平滑迁移。
    """
    # 1.参数校验
    chunks, file_title = validate_chunks_and_title(state)

    # 2.基于chunks构造上下文
    context = build_document_context(chunks, file_title)

    # 3.调用LLM识别标准主题名
    standard_subject_name, llm_subject_name = recognize_standard_subject_name(context, file_title)
    subject_id = build_subject_id(standard_subject_name)
    subject_aliases = build_subject_aliases(standard_subject_name, file_title, llm_subject_name)

    # 4.将标准主题回写到state和chunks，subject_name作为旧流程兼容字段保留
    state["subject_id"] = subject_id
    state["standard_subject_name"] = standard_subject_name
    state["subject_name"] = standard_subject_name
    state["subject_aliases"] = subject_aliases
    chunks = apply_subject_to_chunks(state, chunks, subject_id, standard_subject_name)
    state["chunks"] = chunks

    # 5.生成标准主题名的稠密向量和稀疏向量
    dense_vector, sparse_vector = generate_embeddings(standard_subject_name)

    # 6.构建集合
    prepare_standard_subject_collection(state)
    prepare_subject_alias_collection(state)
    prepare_subject_name_collection(state)

    # 7.入库：新主题体系 + 旧主体集合兼容写入
    insert_standard_subject(
        subject_id=subject_id,
        standard_subject_name=standard_subject_name,
        subject_aliases=subject_aliases,
        file_title=file_title,
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        state=state,
    )
    insert_subject_aliases(subject_id, standard_subject_name, subject_aliases, file_title, llm_subject_name)
    insert_subject_name(standard_subject_name, file_title, dense_vector, sparse_vector)

    return state
