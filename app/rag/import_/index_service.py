from pymilvus import DataType

from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.infra.persistence.import_metadata_repository import DEFAULT_INDEX_VERSION, DEFAULT_TENANT_ID, DEFAULT_VISIBILITY
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import MILVUS_DEFAULT_VARCHAR_MAX_LENGTH
from app.rag.import_.subject_name_service import SUBJECT_DOMAIN_FIELD_DESCRIPTIONS
from app.shared.runtime.logger import logger,step_log
from app.shared.utils.escape_milvus_string_utils import escape_milvus_string

CHUNK_SUBJECT_FIELDS = (
    "subject_id",
    "standard_subject_name",
    "equipment_model",
    "alarm_code",
    "part_name",
    "sop_type",
    "safety_level",
    "maintenance_stage",
)

def _add_varchar_field(schema, field_name, description, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH):
    schema.add_field(
        field_name=field_name,
        datatype=DataType.VARCHAR,
        max_length=max_length,
        description=description,
    )


def _require_state_text(state: ImportGraphState, field_name: str) -> str:
    value = str(state.get(field_name) or "").strip()
    if not value:
        logger.error(f"{field_name}为空，无法补齐chunk元数据！")
        raise ValueError(f"{field_name}为空，无法补齐chunk元数据！")
    return value


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
    schema = client.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
        description="知识切片集合：存储问答检索用的 chunk 文本、标准主题关联、领域标签和混合检索向量。",
    )
    # 主键
    schema.add_field(
        field_name="chunk_id",
        datatype=DataType.INT64,
        is_primary=True,
        auto_id=True,
        description="Milvus 自动生成的 chunk 主键。用于返回检索命中的切片标识，不作为业务幂等键。",
    )
    # 阶段4 document 维度元数据。用于 document 级幂等、删除、重建索引和阶段5权限过滤。
    _add_varchar_field(schema, "dataset_id", "chunk 所属知识库 ID。阶段5可基于该字段做多知识库过滤。")
    _add_varchar_field(schema, "document_id", "chunk 所属 document ID。导入幂等、删除和重建索引以该字段为业务键。")
    _add_varchar_field(schema, "owner_user_id", "chunk 所属用户 ID。阶段5查询权限过滤使用。")
    _add_varchar_field(schema, "tenant_id", "chunk 所属租户 ID。当前默认 tenant_default，为后续多租户预留。")
    _add_varchar_field(schema, "visibility", "chunk 可见性。当前默认 private，后续支持 shared/public。")
    schema.add_field(
        field_name="index_version",
        datatype=DataType.INT64,
        description="document 级检索索引产物版本。用于追踪该 chunk 属于 document 的第几版索引结果。",
    )
    schema.add_field(
        field_name="chunk_index",
        datatype=DataType.INT64,
        description="chunk 在当前 document 最终切片结果中的顺序，从 0 递增。",
    )
    schema.add_field(
        field_name="enabled",
        datatype=DataType.BOOL,
        description="chunk 是否启用。当前默认 true，阶段6可用于 chunk 启停。",
    )
    _add_varchar_field(schema, "source_title", "来源标题。当前使用 file_title，作为展示和兼容字段。")
    # 内容
    _add_varchar_field(schema, "content", "切片正文内容。答案生成阶段主要依据该字段组织回答。", max_length=65535)
    # 原始文件标题
    _add_varchar_field(schema, "file_title", "来源文件标题。仅作为展示和兼容字段，不再作为导入幂等删除条件。")
    # 标题
    _add_varchar_field(schema, "title", "当前切片标题。通常来自 Markdown 标题，用于答案引用和上下文定位。")
    # 父标题
    _add_varchar_field(schema, "parent_title", "当前切片的父级标题。用于保留章节层级，帮助定位切片所属上下文。")
    # 分片顺序标记
    schema.add_field(
        field_name="part",
        datatype=DataType.INT8,
        description="切片顺序标记。表示该 chunk 在拆分结果或章节中的顺序，便于后续恢复邻近上下文。",
    )
    # 标准主题ID。阶段2后查询优先用这个字段过滤，避免主题展示名变化导致检索失效。
    _add_varchar_field(schema, "subject_id", "标准主题稳定业务 ID。查询阶段先通过 alias collection 确认该 ID，再用它过滤 chunk。")
    # 标准主题名称。保留可读名称，便于日志、引用展示和调试。
    schema.add_field(
        field_name="standard_subject_name",
        datatype=DataType.VARCHAR,
        max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH,
        description="标准主题名称。可读展示字段，用于日志、答案 Prompt 和引用展示；检索过滤不依赖该字段。",
    )
    # 轻量领域字段。当前只做元数据入库，后续可用于报警码、部件、SOP类型等精确过滤。
    for field_name in (
        "equipment_model",
        "alarm_code",
        "part_name",
        "sop_type",
        "safety_level",
        "maintenance_stage",
    ):
        _add_varchar_field(schema, field_name, SUBJECT_DOMAIN_FIELD_DESCRIPTIONS[field_name])
    # 稠密向量
    schema.add_field(
        field_name="dense_vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=1024,
        description="chunk 检索文本的稠密向量。用于语义相似度召回。",
    )
    # 稀疏向量
    schema.add_field(
        field_name="sparse_vector",
        datatype=DataType.SPARSE_FLOAT_VECTOR,
        description="chunk 检索文本的稀疏向量。用于型号、报警码、关键词等字面匹配召回。",
    )

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


def normalize_chunk_subject_fields(chunks):
    """
    插入 Milvus 前补齐阶段2新增的 chunk 主体字段。

    主体识别节点已经会写入 subject_id / standard_subject_name / 领域字段。
    这里再做一次兜底，是为了让测试桩或手动调用也能满足新 schema：
    - subject_id 缺失时置为空字符串，后续查询不会命中这类未确认主题的数据。
    - standard_subject_name 缺失时置为空字符串，只作为可读展示字段。
    - 轻量领域字段缺失时置为空字符串，避免显式 schema 插入时报字段缺失。
    """
    for chunk in chunks:
        for field_name in CHUNK_SUBJECT_FIELDS:
            chunk.setdefault(field_name, "")
    return chunks


def normalize_chunk_document_fields(chunks, state: ImportGraphState, file_title: str):
    """
    插入 Milvus 前补齐阶段4新增的 document 维度 chunk 元数据。

    这些字段不依赖切分节点产出，统一从导入图 state 回填，避免上游节点重复关心
    document/user/visibility 这类横切元数据。
    """
    dataset_id = _require_state_text(state, "dataset_id")
    document_id = _require_state_text(state, "document_id")
    owner_user_id = _require_state_text(state, "owner_user_id")
    tenant_id = str(state.get("tenant_id") or DEFAULT_TENANT_ID)
    visibility = str(state.get("visibility") or DEFAULT_VISIBILITY)
    index_version = int(state.get("index_version") or DEFAULT_INDEX_VERSION)

    for chunk_index, chunk in enumerate(chunks):
        chunk["dataset_id"] = dataset_id
        chunk["document_id"] = document_id
        chunk["owner_user_id"] = owner_user_id
        chunk["tenant_id"] = tenant_id
        chunk["visibility"] = visibility
        chunk["index_version"] = index_version
        chunk["chunk_index"] = chunk_index
        chunk["enabled"] = True
        chunk["source_title"] = file_title
        chunk.setdefault("file_title", file_title)
    return chunks


@step_log()
def remove_old_chunks(document_id):
    client = milvus_gateway.client
    client.delete(
        collection_name=milvus_gateway.chunk_collection_name,
        filter=f"document_id=='{escape_milvus_string(document_id)}'"
    )


@step_log()
def insert_chunks(chunks):
    client = milvus_gateway.client
    chunks = normalize_chunk_subject_fields(chunks)
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
    2. 根据 document_id 删除旧数据
    3. 补齐 chunk 元数据并批量插入新的 chunks
    4. 回写 chunk_id 等入库结果
    """
    # 1.参数校验
    chunks, file_title = params_check(state)

    # 2.准备集合 schema 和索引
    prepare_chunks_collection(state)

    # 3.根据 document_id 删除旧数据，file_title 只保留为展示和兼容字段
    document_id = _require_state_text(state, "document_id")
    remove_old_chunks(document_id)

    # 4.补齐阶段4元数据并插入新数据
    chunks = normalize_chunk_document_fields(chunks, state, file_title)
    insert_chunks(chunks)

    return state
