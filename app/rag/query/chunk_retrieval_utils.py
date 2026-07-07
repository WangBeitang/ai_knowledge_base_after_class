from app.shared.utils.escape_milvus_string_utils import escape_milvus_string


CHUNK_OUTPUT_FIELDS = [
    "chunk_id",
    "subject_id",
    "standard_subject_name",
    "subject_name",
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


def build_subject_filter_expr(subject_ids=None, subject_names=None):
    """
    构建 chunk collection 的主体过滤表达式。

    阶段2开始查询应该优先使用 subject_id，因为它是稳定主键，不会因为标准主题展示名
    调整而失效。subject_names 只作为旧数据/旧链路兜底，等查询侧主体确认完全切到
    alias -> subject_id 后，可以逐步减少对 subject_name 的依赖。
    """
    if subject_ids:
        return f"subject_id in {_milvus_string_list(subject_ids)}"
    return f"subject_name in {_milvus_string_list(subject_names)}"


def build_subject_filter_expr_candidates(subject_ids=None, subject_names=None):
    """
    构建检索过滤表达式候选列表，顺序代表查询优先级。

    阶段2新数据应该走 subject_id 精确过滤；但旧 chunk 数据可能还没有 subject_id 字段
    或 subject_id 为空。为了让旧 item_name/subject_name 逻辑平滑迁移，这里在存在
    subject_names 时追加一条 subject_name 兜底表达式。

    调用方应按顺序执行：第一条有召回就停止；第一条无召回才尝试后续 fallback。
    """
    expr_candidates = []
    if subject_ids:
        expr_candidates.append(f"subject_id in {_milvus_string_list(subject_ids)}")
    if subject_names:
        fallback_expr = f"subject_name in {_milvus_string_list(subject_names)}"
        if fallback_expr not in expr_candidates:
            expr_candidates.append(fallback_expr)
    return expr_candidates


def resolve_subject_filter_values(state):
    """
    从查询 state 中取检索主体。

    返回 subject_ids、subject_names 两组值，调用方可以先校验是否至少有一组可用。
    当前保留 subject_names 是为了兼容旧主体确认服务还没有完全迁移到标准主题体系。
    """
    return state.get("subject_ids") or [], state.get("subject_names") or []


def format_chunk_search_item(item, source_type):
    entity = item.get("entity", {})
    return {
        "chunk_id": item.get("id") or entity.get("chunk_id"),
        "subject_id": entity.get("subject_id"),
        "standard_subject_name": entity.get("standard_subject_name"),
        "subject_name": entity.get("subject_name"),
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
