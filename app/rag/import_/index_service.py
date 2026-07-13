from pymilvus import DataType, Function, FunctionType

from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.infra.persistence.import_metadata_repository import DEFAULT_INDEX_VERSION, DEFAULT_TENANT_ID, DEFAULT_VISIBILITY
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import MILVUS_DEFAULT_VARCHAR_MAX_LENGTH
from app.rag.import_.lexical_text_service import (
    LEXICAL_ANALYZER_PARAMS,
    LEXICAL_TEXT_MAX_LENGTH,
    build_chunk_lexical_text,
    validate_lexical_analyzer,
)
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

# 阶段 5 第七部分重建后的 chunk collection 必须同时具备三种召回载体：
# dense_vector（稠密语义）、learned_sparse_vector（BGE-M3 学习式稀疏）和
# bm25_sparse_vector（Milvus BM25 Function 输出）。旧 collection 不做字段兼容；
# 如果检测到旧 schema，要求显式删库重建，避免把新代码静默跑在旧数据结构上。
CHUNK_SCHEMA_REQUIRED_FIELDS = {
    "dense_vector",
    "learned_sparse_vector",
    "lexical_text",
    "bm25_sparse_vector",
}
CHUNK_SCHEMA_REQUIRED_FUNCTIONS = {"chunk_lexical_text_bm25"}


def _validate_existing_chunk_collection_schema(client, collection_name: str) -> None:
    """已有 collection 必须是阶段 5 新 schema，否则给出明确的删库重建错误。"""
    description = client.describe_collection(collection_name=collection_name)
    actual_fields = {
        str(field.get("name") or field.get("field_name") or "")
        for field in description.get("fields", [])
    }
    missing_fields = sorted(CHUNK_SCHEMA_REQUIRED_FIELDS - actual_fields)
    actual_functions = {
        str(function.get("name") or "")
        for function in description.get("functions", [])
    }
    missing_functions = sorted(CHUNK_SCHEMA_REQUIRED_FUNCTIONS - actual_functions)
    if missing_fields or missing_functions:
        missing_parts = []
        if missing_fields:
            missing_parts.append("字段：" + ", ".join(missing_fields))
        if missing_functions:
            missing_parts.append("Function：" + ", ".join(missing_functions))
        raise RuntimeError(
            "chunk collection 仍是阶段 5 之前的旧 schema，缺少"
            + "；".join(missing_parts)
            + "。当前只有可丢弃测试数据，请删除该 collection 后重新导入；"
              "本阶段不保留 sparse_vector 兼容分支。"
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
        _validate_existing_chunk_collection_schema(client, collection_name)
        logger.info(f"集合{collection_name}已存在且符合阶段5 BM25 schema，无需创建")
        return

    # Analyzer 由 Milvus 服务端执行。先用真实 token 验证中文与设备编号，再创建 schema；
    # 这样错误配置会在空 collection 阶段失败，不会等全量导入后才暴露。
    validate_lexical_analyzer(client)

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

    # lexical_text（词法检索文本）是 BM25 唯一输入。应用负责按固定规则拼接纯文本；
    # enable_analyzer 表示由 Milvus 对插入文本和查询文本使用同一套分词规则。
    schema.add_field(
        field_name="lexical_text",
        datatype=DataType.VARCHAR,
        max_length=LEXICAL_TEXT_MAX_LENGTH,
        enable_analyzer=True,
        analyzer_params=LEXICAL_ANALYZER_PARAMS,
        description=(
            "BM25 词法检索输入。按固定顺序拼接主题、设备标识、标题和正文，并追加编号变体；"
            "应用写入该文本，Milvus Analyzer/Function 负责生成 BM25 稀疏向量。"
        ),
    )
    # 稠密向量
    schema.add_field(
        field_name="dense_vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=1024,
        description="chunk 检索文本的稠密向量。用于语义相似度召回。",
    )
    # learned sparse 的中文含义是“模型学习式稀疏向量”。它由 BGE-M3 生成，
    # 与依赖词频统计的 BM25 是两条不同召回通道，因此必须使用独立字段。
    schema.add_field(
        field_name="learned_sparse_vector",
        datatype=DataType.SPARSE_FLOAT_VECTOR,
        description="BGE-M3 生成的学习式稀疏向量。用于关键词与模型学习权重结合的召回。",
    )
    # bm25_sparse_vector 是 Function 输出字段。应用插入 chunk 时绝不能手动写该字段；
    # Milvus 会根据 lexical_text 和当前 collection 的词频统计自动生成。
    schema.add_field(
        field_name="bm25_sparse_vector",
        datatype=DataType.SPARSE_FLOAT_VECTOR,
        description="Milvus BM25 Function 自动生成的稀疏向量，应用侧禁止直接写入。",
    )

    bm25_function = Function(
        name="chunk_lexical_text_bm25",
        function_type=FunctionType.BM25,
        input_field_names=["lexical_text"],
        output_field_names=["bm25_sparse_vector"],
        description="把 chunk.lexical_text 转换为 BM25 稀疏表示。",
    )
    schema.add_function(bm25_function)

    # 4.索引构建
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 64, "efConstruction": 100}
    )
    index_params.add_index(
        field_name="learned_sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
    )
    index_params.add_index(
        field_name="bm25_sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
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
        # 必须在 source_title、权限字段和主体字段补齐后生成 lexical_text，确保重复导入
        # 的同一 chunk 得到稳定 BM25 输入。bm25_sparse_vector 由 Milvus Function 生成。
        chunk["lexical_text"] = build_chunk_lexical_text(chunk)
    return chunks


@step_log()
def remove_old_chunks(document_id):
    client = milvus_gateway.client
    collection_name = milvus_gateway.chunk_collection_name
    # 解析或向量化阶段失败的 document 可能从未创建 chunk collection。
    # 删除接口需要保持幂等，此时直接视为“没有旧 chunk 可清理”。
    if not client.has_collection(collection_name=collection_name):
        logger.info(f"集合{collection_name}不存在，无需删除document_id={document_id}的chunk")
        return
    client.delete(
        collection_name=collection_name,
        filter=f"document_id=='{escape_milvus_string(document_id)}'"
    )


@step_log()
def insert_chunks(chunks):
    client = milvus_gateway.client
    chunks = normalize_chunk_subject_fields(chunks)
    for chunk in chunks:
        if "sparse_vector" in chunk:
            raise ValueError(
                "chunk 仍包含旧字段 sparse_vector；阶段5已改为 learned_sparse_vector，"
                "请重新执行 BGE-M3 向量化，不保留旧 schema 兼容写入"
            )
        if "bm25_sparse_vector" in chunk:
            raise ValueError(
                "bm25_sparse_vector 是 Milvus Function 输出字段，应用侧只能写 lexical_text"
            )
        # 这三个字段分别归应用向量化、应用词法文本构造负责。与 Function 输出不同，
        # 任一缺失都表示导入节点顺序或 partial state 已损坏，应在调用 Milvus 前报清楚。
        for required_field in ("dense_vector", "learned_sparse_vector", "lexical_text"):
            if required_field not in chunk or chunk[required_field] in (None, ""):
                raise ValueError(f"chunk 缺少必填入库字段 {required_field}")
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

    # 4.先补齐主体字段，再补齐 document 元数据并构造 lexical_text。顺序不能颠倒，
    # 否则设备型号、报警码等 metadata 会缺席于 BM25 输入。
    chunks = normalize_chunk_subject_fields(chunks)
    chunks = normalize_chunk_document_fields(chunks, state, file_title)
    insert_chunks(chunks)

    return state
