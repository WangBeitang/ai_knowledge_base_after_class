from app.shared.utils.escape_milvus_string_utils import escape_milvus_string


CHUNK_OUTPUT_FIELDS = [
    "chunk_id",
    "subject_id",
    "standard_subject_name",
    "content",
    "title",
    "parent_title",
    "part",
    "file_title",
    "equipment_model",
    "alarm_code",
    "part_name",
    "sop_type",
    "safety_level",
    "maintenance_stage",
]


def _milvus_string_list(values):
    """
    把 Python 字符串列表转换成 Milvus filter 可用的字符串数组表达式。

    查询过滤会拼成 `field in ["a", "b"]` 形式。这里统一做去空、去重和转义，
    避免设备型号、主题名称里出现引号或换行时破坏 Milvus 表达式。
    """
    normalized_values = []
    seen = set()
    for value in values or []:
        normalized_value = str(value or "").strip()
        if not normalized_value or normalized_value in seen:
            continue
        normalized_values.append(normalized_value)
        seen.add(normalized_value)
    return "[" + ",".join(f'"{escape_milvus_string(value)}"' for value in normalized_values) + "]"


def build_subject_filter_expr(subject_ids):
    """
    构建 chunk collection 的主体过滤表达式。

    阶段 2 之后只使用 subject_id 过滤。subject_id 是稳定主键，
    不依赖标准主题展示名，避免主题重命名影响检索。
    """
    return f"subject_id in {_milvus_string_list(subject_ids)}"


def resolve_subject_filter_values(state):
    """
    从查询 state 中取检索主体。

    返回 subject_ids。查询侧主体确认必须先通过 alias collection 得到标准主题 ID，
    然后检索链路只按 subject_id 过滤 chunk。
    """
    return state.get("subject_ids") or []


def format_chunk_search_item(item, source_type):
    entity = item.get("entity", {})
    return {
        "chunk_id": item.get("id") or entity.get("chunk_id"),
        "subject_id": entity.get("subject_id"),
        "standard_subject_name": entity.get("standard_subject_name"),
        "content": entity.get("content"),
        "title": entity.get("title"),
        "parent_title": entity.get("parent_title"),
        "part": entity.get("part"),
        "file_title": entity.get("file_title"),
        "equipment_model": entity.get("equipment_model"),
        "alarm_code": entity.get("alarm_code"),
        "part_name": entity.get("part_name"),
        "sop_type": entity.get("sop_type"),
        "safety_level": entity.get("safety_level"),
        "maintenance_stage": entity.get("maintenance_stage"),
        "score": item.get("distance", 0.0),
        "type": source_type,
        "url": None,
    }
