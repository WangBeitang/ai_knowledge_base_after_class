from collections.abc import Mapping

from app.rag.query.contracts import EvidenceSourceType, RetrievalCandidate, RetrievalChannel
from app.shared.utils.escape_milvus_string_utils import escape_milvus_string


CHUNK_OUTPUT_FIELDS = [
    "chunk_id",
    "dataset_id",
    "document_id",
    "owner_user_id",
    "tenant_id",
    "visibility",
    "index_version",
    "chunk_index",
    "enabled",
    "source_title",
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


# structured identifier 的中文含义是“结构化标识”。这里使用固定字段白名单，而不是把
# query_identifiers 的 key 直接拼进 Milvus 表达式，原因有两点：
# 1. 避免调用方传入任意字段名，形成表达式注入或查询不存在的 schema 字段；
# 2. 明确任务 4 只使用当前 chunk schema 已落地的字段。SOP 编号、零件编号等新字段
#    必须等后续 schema 任务正式定义后再加入，不能在这里提前假设字段名。
STRUCTURED_IDENTIFIER_FIELDS = (
    "equipment_model",
    "alarm_code",
    "part_name",
    "sop_type",
    "safety_level",
    "maintenance_stage",
)

# lexical-only identifier 的中文含义是“当前只能通过正文/词法通道查找的标识”。
# ``sop_code``（SOP 编号）和 ``part_number``（零件/备件编号）还没有独立的 Milvus
# schema 字段，因此它们可以保存在 QueryGraphState.query_identifiers 中，也可以追加到
# 检索文本中，但绝不能被拼成不存在的 ``sop_code in [...]`` 表达式。等后续评测证明有
# 必要扩 schema 后，再把对应字段迁入 STRUCTURED_IDENTIFIER_FIELDS。
LEXICAL_ONLY_IDENTIFIER_FIELDS = (
    "sop_code",
    "part_number",
)

# QueryGraphState 当前允许承载的全部查询标识。该白名单同时防止调用方使用任意 key
# 绕过字段契约；其中只有 STRUCTURED_IDENTIFIER_FIELDS 会真正进入 Milvus filter。
QUERY_IDENTIFIER_FIELDS = STRUCTURED_IDENTIFIER_FIELDS + LEXICAL_ONLY_IDENTIFIER_FIELDS


def _normalize_milvus_string_values(values, *, field_name: str) -> list[str]:
    """
    把待过滤的字符串列表整理成稳定顺序，并拒绝容易误用的单个字符串。

    ``dataset_ids``、``subject_ids`` 和每一种结构化标识在业务上都是“多值列表”。如果
    调用方误传 ``"dataset_a"``，Python 会把它当成字符序列；这里显式报错，避免最后
    生成 ``["d", "a", ...]`` 这种表面合法、实际完全错误的 Milvus filter。

    空白项会被移除，重复项按首次出现顺序去重。该函数只负责规范化，是否允许最终为空
    由上层函数根据字段语义决定。
    """
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} 必须是字符串列表，不能直接传入单个字符串")

    normalized_values = []
    seen = set()
    for value in values:
        normalized_value = str(value or "").strip()
        if not normalized_value or normalized_value in seen:
            continue
        normalized_values.append(normalized_value)
        seen.add(normalized_value)
    return normalized_values


def _require_milvus_text(value, *, field_name: str) -> str:
    """校验权限过滤所需的单值文本，防止空用户或空 tenant 进入表达式。"""
    normalized_value = str(value or "").strip()
    if not normalized_value:
        raise ValueError(f"{field_name} 不能为空，无法构建安全的 Milvus 检索过滤表达式")
    return normalized_value


def _milvus_string_literal(value: str) -> str:
    """
    生成一个 Milvus 字符串字面量。

    所有动态字符串都必须经过同一个转义入口，不能由调用方手写双引号。这样用户 ID、
    tenant、dataset、subject 或设备标识中出现引号、反斜杠和换行时，不会截断表达式。
    """
    return f'"{escape_milvus_string(value)}"'


def _milvus_string_list(values: list[str]) -> str:
    """
    把已经规范化的 Python 字符串列表转换成 Milvus 数组表达式。

    返回形态为 ``["a","b"]``。去空、去重由 ``_normalize_milvus_string_values``
    完成；这里仍逐值调用统一字面量函数，保证没有动态字符串绕过转义。
    """
    return "[" + ",".join(_milvus_string_literal(value) for value in values) + "]"


def _build_required_in_clause(field_name: str, values, *, value_label: str) -> str:
    """
    构建不允许为空的 ``field in [...]`` 条件。

    dataset 和 subject 一旦为空就必须立即失败。不能生成 ``in []`` 后期待 Milvus 给出
    某种隐式结果，更不能删除该条件后退化为全库搜索。
    """
    normalized_values = _normalize_milvus_string_values(values, field_name=value_label)
    if not normalized_values:
        raise ValueError(f"{value_label} 不能为空，禁止退化为全库检索")
    return f"{field_name} in {_milvus_string_list(normalized_values)}"


def _build_optional_text_eq_clause(field_name: str, value, *, value_label: str) -> str:
    """
    构建可选的 ``field == "value"`` 条件。

    management filter（管理过滤器）允许 document_id 为空，表示跨文档列表；但只要调用方
    显式传入字段，就必须是非空文本。这样可以避免前端传入空字符串后悄悄退化为更宽的
    查询范围。
    """
    if value is None:
        return ""
    normalized_value = _require_milvus_text(value, field_name=value_label)
    return f"{field_name} == {_milvus_string_literal(normalized_value)}"


def _build_optional_int_eq_clause(field_name: str, value, *, value_label: str) -> str:
    """
    构建可选的 ``field == number`` 条件。

    ``index_version`` 是 document 级索引产物版本，用数字比较；它不是 Milvus 物理索引
    版本。bool 在 Python 中是 int 的子类，这里显式拒绝，避免 True 被写成版本 1。
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        raise ValueError(f"{value_label} 必须是大于等于 0 的整数")
    try:
        normalized_value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{value_label} 必须是大于等于 0 的整数") from exc
    if normalized_value < 0:
        raise ValueError(f"{value_label} 必须是大于等于 0 的整数")
    return f"{field_name} == {normalized_value}"


def _build_optional_bool_eq_clause(field_name: str, value, *, value_label: str) -> str:
    """
    构建可选的 ``field == true/false`` 条件。

    enabled 为 None 表示管理列表查看全部 chunk；只有明确传 True/False 时才拼启停条件。
    这与查询检索过滤不同：查询检索必须始终强制 ``enabled == true``。
    """
    if value is None:
        return ""
    if not isinstance(value, bool):
        raise ValueError(f"{value_label} 必须是 bool 或 None")
    return f"{field_name} == {'true' if value else 'false'}"


def _build_chunk_visibility_clause(*, owner_user_id: str, tenant_id: str) -> str:
    """
    构建 public/shared/owner 三个可见性分支，并始终返回带括号的 OR 组合。

    visibility 的中文含义是“可见性”。括号是安全边界：调用方会把它与 dataset、document、
    enabled 等条件用 AND 连接；如果这里不整体加括号，owner 分支可能绕过前面的范围条件。
    """
    normalized_owner_user_id = _require_milvus_text(
        owner_user_id,
        field_name="owner_user_id",
    )
    normalized_tenant_id = _require_milvus_text(tenant_id, field_name="tenant_id")

    public_value = _milvus_string_literal("public")
    shared_value = _milvus_string_literal("shared")
    tenant_value = _milvus_string_literal(normalized_tenant_id)
    owner_value = _milvus_string_literal(normalized_owner_user_id)

    return (
        "("
        f"visibility == {public_value} "
        f"OR (visibility == {shared_value} AND tenant_id == {tenant_value}) "
        f"OR owner_user_id == {owner_value}"
        ")"
    )


def build_chunk_access_filter(*, dataset_ids, owner_user_id: str, tenant_id: str) -> str:
    """
    构建 chunk 的知识库范围、启用状态和用户可见性过滤条件。

    access 的中文含义是“访问权限”。最终规则是：
    - 只能检索本次请求明确指定的 dataset；
    - 只允许 ``enabled == true`` 的 chunk 参与召回；
    - public 对所有用户可见；
    - shared 只对同 tenant 用户可见；
    - 其他数据只有 owner 本人可见。

    可见性三个 OR 分支必须整体放在括号内，再与 dataset/enabled 使用 AND 连接。如果省略
    这层括号，Milvus 的 AND/OR 优先级可能让 owner 条件绕过 dataset 或 enabled 限制。
    """
    dataset_clause = _build_required_in_clause(
        "dataset_id",
        dataset_ids,
        value_label="dataset_ids",
    )
    visibility_clause = _build_chunk_visibility_clause(
        owner_user_id=owner_user_id,
        tenant_id=tenant_id,
    )

    return (
        f"{dataset_clause} "
        "AND enabled == true "
        f"AND {visibility_clause}"
    )


def build_chunk_management_filter(
        *,
        dataset_ids,
        owner_user_id: str,
        tenant_id: str,
        document_id: str | None = None,
        index_version: int | None = None,
        enabled: bool | None = None,
) -> str:
    """
    生成 chunk 管理列表/详情使用的 Milvus 过滤表达式。

    management 的中文含义是“管理场景”。它与检索过滤的边界不同：
    - dataset 和可见性仍然是硬权限条件；
    - document_id、index_version 和 enabled 是可选筛选条件；
    - enabled 为 None 时不拼 ``enabled == true``，这样管理页才能看到 disabled chunk 并恢复。

    public/shared/owner 的 OR 分支仍由 ``_build_chunk_visibility_clause`` 统一加括号，不能让
    owner 条件绕过 dataset 或 document 范围。
    """
    clauses = [
        _build_required_in_clause(
            "dataset_id",
            dataset_ids,
            value_label="dataset_ids",
        )
    ]

    document_clause = _build_optional_text_eq_clause(
        "document_id",
        document_id,
        value_label="document_id",
    )
    if document_clause:
        clauses.append(document_clause)

    index_version_clause = _build_optional_int_eq_clause(
        "index_version",
        index_version,
        value_label="index_version",
    )
    if index_version_clause:
        clauses.append(index_version_clause)

    enabled_clause = _build_optional_bool_eq_clause(
        "enabled",
        enabled,
        value_label="enabled",
    )
    if enabled_clause:
        clauses.append(enabled_clause)

    clauses.append(_build_chunk_visibility_clause(
        owner_user_id=owner_user_id,
        tenant_id=tenant_id,
    ))
    return " AND ".join(clauses)


def build_structured_identifier_filter(query_identifiers=None) -> str:
    """
    构建设备运维结构化标识过滤条件；没有标识时返回空字符串。

    同一字段的多个候选值使用 ``in``，表示“命中其中任意一个”；不同字段之间使用
    ``AND``，例如设备型号和报警码同时存在时必须同时满足。字段输出顺序固定为 schema
    顺序，保证相同输入可以生成完全一致的表达式，便于测试、Trace 和后续评测重放。

    默认空字典表示当前问题没有提取到结构化标识，这是合法情况。若调用方明确传入某个
    字段却只有空白值，则直接报错，避免 Trace 声称使用了结构化过滤但实际没有生效。
    """
    if query_identifiers is None:
        return ""
    if not isinstance(query_identifiers, Mapping):
        raise ValueError("query_identifiers 必须是字段到字符串列表的字典")
    if not query_identifiers:
        return ""

    unsupported_fields = sorted(
        str(field_name)
        for field_name in query_identifiers
        if field_name not in STRUCTURED_IDENTIFIER_FIELDS
    )
    if unsupported_fields:
        raise ValueError(
            "query_identifiers 包含当前 chunk schema 不支持的字段："
            + ", ".join(str(field_name) for field_name in unsupported_fields)
        )

    clauses = []
    for field_name in STRUCTURED_IDENTIFIER_FIELDS:
        if field_name not in query_identifiers:
            continue
        clauses.append(
            _build_required_in_clause(
                field_name,
                query_identifiers[field_name],
                value_label=f"query_identifiers.{field_name}",
            )
        )
    return " AND ".join(clauses)


def select_structured_query_identifiers(query_identifiers=None) -> dict[str, list[str]]:
    """
    从查询标识中选出当前 Milvus schema 可以精确过滤的部分。

    该函数与 ``build_structured_identifier_filter`` 的边界不同：后者是严格的底层构建器，
    直接传入 ``sop_code`` 仍会报错；本函数是 State 到 filter 的适配器，允许 State 同时
    保存结构化标识和 lexical-only 标识，再只把前者交给 Milvus。
    """
    if query_identifiers is None:
        return {}
    if not isinstance(query_identifiers, Mapping):
        raise ValueError("query_identifiers 必须是字段到字符串列表的字典")

    unsupported_fields = sorted(
        str(field_name)
        for field_name in query_identifiers
        if field_name not in QUERY_IDENTIFIER_FIELDS
    )
    if unsupported_fields:
        raise ValueError(
            "query_identifiers 包含当前查询契约不支持的字段："
            + ", ".join(unsupported_fields)
        )

    structured_identifiers: dict[str, list[str]] = {}
    for field_name in STRUCTURED_IDENTIFIER_FIELDS:
        if field_name not in query_identifiers:
            continue
        normalized_values = _normalize_milvus_string_values(
            query_identifiers[field_name],
            field_name=f"query_identifiers.{field_name}",
        )
        if not normalized_values:
            raise ValueError(f"query_identifiers.{field_name} 不能为空")
        structured_identifiers[field_name] = normalized_values

    # lexical-only 标识虽然不进入 expr，也必须在边界校验非空，避免 State/Trace 声称
    # 提取到一个编号，实际追加到查询时却什么都没有。
    for field_name in LEXICAL_ONLY_IDENTIFIER_FIELDS:
        if field_name not in query_identifiers:
            continue
        normalized_values = _normalize_milvus_string_values(
            query_identifiers[field_name],
            field_name=f"query_identifiers.{field_name}",
        )
        if not normalized_values:
            raise ValueError(f"query_identifiers.{field_name} 不能为空")

    return structured_identifiers


def build_chunk_retrieval_filter(
        *,
        dataset_ids,
        subject_ids,
        owner_user_id: str,
        tenant_id: str,
        query_identifiers=None,
) -> str:
    """
    生成所有本地召回通道必须复用的最终 Milvus ``expr``。

    retrieval 的中文含义是“检索/召回”。该函数把三类约束合成一份表达式：
    1. access filter：dataset、enabled、public/shared/owner 权限；
    2. subject filter：已确认的稳定 subject_id；
    3. structured identifier filter：可选的设备型号、报警码等精确标识。

    普通 dense/learned sparse 混合检索、HyDE 以及后续 BM25 都只能调用本函数，不能各自
    拼接 expr。这样新增权限规则时只修改一个位置，也不会出现某一条检索通道漏过滤。
    """
    access_clause = build_chunk_access_filter(
        dataset_ids=dataset_ids,
        owner_user_id=owner_user_id,
        tenant_id=tenant_id,
    )
    subject_clause = _build_required_in_clause(
        "subject_id",
        subject_ids,
        value_label="subject_ids",
    )
    # State 可以同时包含暂时只有词法能力的 SOP/零件编号。这里只选择 schema 已存在的
    # 字段进入 expr；未知字段和空值仍由适配器明确拒绝，不能静默忽略。
    structured_identifiers = select_structured_query_identifiers(query_identifiers)
    identifier_clause = build_structured_identifier_filter(structured_identifiers)

    clauses = [access_clause, subject_clause]
    if identifier_clause:
        clauses.append(identifier_clause)
    return " AND ".join(clauses)


def build_chunk_retrieval_filter_from_state(state: Mapping[str, object]) -> str:
    """
    从 ``QueryGraphState`` 读取过滤上下文并生成最终表达式。

    这是查询节点和共享过滤契约之间的唯一适配入口。字段缺失或只有空白时由上述构建器
    抛出明确中文错误；这里绝不回退 anonymous_user、默认 dataset 或无条件全库检索。
    """
    return build_chunk_retrieval_filter(
        dataset_ids=state.get("dataset_ids"),
        subject_ids=state.get("subject_ids"),
        owner_user_id=state.get("owner_user_id"),
        tenant_id=state.get("tenant_id"),
        query_identifiers=state.get("query_identifiers"),
    )


def format_chunk_search_item(
        item,
        *,
        retrieval_channels: list[RetrievalChannel | str],
        retrieval_rank: int,
):
    """
    把 Milvus 命中转换成统一 ``RetrievalCandidate``，并立即校验本地身份字段。

    格式转换是元数据最容易丢失的边界，因此这里不再只返回 title/content/score。缺少
    document_id、chunk_id 或 dataset_id 的本地结果会立刻失败，避免等到 Citation 阶段
    才发现无法追踪来源。retrieval_channels 中的底层通道来自本次启用的 mode；Milvus
    内部 RRF 不返回逐请求命中明细，所以这里不声称该 chunk 命中了每一条底层通道。
    """
    entity = item.get("entity", {})
    if not isinstance(entity.get("enabled"), bool):
        raise ValueError("本地 chunk 缺少 bool enabled 字段，无法写入可观测 Trace")
    source_title = str(entity.get("source_title") or entity.get("file_title") or "").strip()
    candidate = RetrievalCandidate(
        document_id=entity.get("document_id"),
        chunk_id=item.get("id") if item.get("id") is not None else entity.get("chunk_id"),
        dataset_id=entity.get("dataset_id"),
        index_version=entity.get("index_version"),
        chunk_index=entity.get("chunk_index"),
        enabled=entity.get("enabled"),
        title=str(entity.get("title") or source_title or "未命名知识切片").strip(),
        source_title=source_title,
        subject_id=entity.get("subject_id"),
        standard_subject_name=entity.get("standard_subject_name"),
        parent_title=entity.get("parent_title"),
        content=str(entity.get("content") or "").strip(),
        equipment_model=entity.get("equipment_model"),
        alarm_code=entity.get("alarm_code"),
        part_name=entity.get("part_name"),
        sop_type=entity.get("sop_type"),
        safety_level=entity.get("safety_level"),
        maintenance_stage=entity.get("maintenance_stage"),
        source_type=EvidenceSourceType.LOCAL,
        retrieval_channels=[RetrievalChannel(channel) for channel in retrieval_channels],
        retrieval_rank=retrieval_rank,
        retrieval_score=max(0.0, float(item.get("distance") or 0.0)),
        rerank_score=None,
        url=None,
    )
    return candidate.model_dump(mode="json")
