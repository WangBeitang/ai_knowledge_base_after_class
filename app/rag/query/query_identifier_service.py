"""
设备运维查询标识的确定性提取、归一化和候选判断。

identifier 的中文含义是“可精确核对的业务标识”，例如设备型号 HAK 180、报警码
E020、SOP 编号和零件编号。这些值与普通自然语言关键词不同：E020 和 E021 即使向量
非常相近，也仍是两个不同编号。因此本模块只使用可解释规则确认“相同编号”；编辑距离
只能产生待用户确认的 suggestion（建议候选），不能静默改写用户问题。

第一版不调用额外 LLM，保证相同输入可以稳定重放，也便于后续把提取结果写入 Trace。
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping

from app.rag.query.chunk_retrieval_utils import QUERY_IDENTIFIER_FIELDS
from app.rag.query.contracts import IdentifierResolutionStatus, RetrievalObservation


# 允许 E020、E-020、E 020 等展示变体，但至少要求三位数字，避免把自然语言里的单个
# 字母 E 或科学计数法误识别为报警码。归一后始终保存为 E020。
ALARM_CODE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])E[\s_-]*(\d{3,4})(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# SOP 编号必须显式包含 SOP 前缀。后半段至少包含一个数字，避免把“SOP 操作流程”中的
# 普通单词 SOP 当作具体编号。
SOP_CODE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])SOP[\s_-]*([A-Za-z]*\d+(?:[-_][A-Za-z0-9]+)*)",
    re.IGNORECASE,
)

# 零件编号比设备型号更难仅凭外形区分，所以第一版必须带“零件编号、备件号、料号、
# P/N、part number”等明确上下文。这样 AB-123 不会在任意句子中被武断认成零件编号。
PART_NUMBER_PATTERN = re.compile(
    r"(?:零件(?:编号|号)?|备件(?:编号|号)?|配件(?:编号|号)?|料号|物料编码|"
    r"P\s*/\s*N|PART\s*(?:NUMBER|NO\.?))\s*[:：#]?\s*"
    r"([A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+|[A-Za-z]+\s+\d+[A-Za-z0-9]*|"
    r"[A-Za-z]+\d+[A-Za-z0-9]*|\d+[A-Za-z]+[A-Za-z0-9]*)",
    re.IGNORECASE,
)

# 设备型号使用“至少两个字母 + 至少两位数字”的保守规则。单字母 E 开头的报警码不会
# 落入该规则；已经被 SOP/零件规则占用的文本区间也会在提取时排除。
EQUIPMENT_MODEL_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z]{2,})[\s_-]*(\d{2,}[A-Za-z0-9]*)(?![A-Za-z0-9])",
    re.IGNORECASE,
)


IDENTIFIER_TYPE_LABELS = {
    "equipment_model": "设备型号",
    "alarm_code": "报警码",
    "sop_code": "SOP 编号",
    "part_number": "零件编号",
    "part_name": "部件名称",
    "sop_type": "SOP 类型",
    "safety_level": "安全等级",
    "maintenance_stage": "维护阶段",
}

# 查询时追加到自然语言文本的顺序保持固定，使日志、测试和后续 Trace 对同一输入得到
# 完全相同的增强文本。
EXTRACTED_IDENTIFIER_FIELDS = (
    "equipment_model",
    "alarm_code",
    "sop_code",
    "part_number",
)

# 从合法 chunk 词典读取候选时只需要当前 schema 已有的字段。SOP 编号和零件编号尚无
# 独立字段，它们只能从第二段召回到的正文中提取，不能向 Milvus 请求不存在的字段。
IDENTIFIER_DICTIONARY_OUTPUT_FIELDS = [
    "chunk_id",
    "equipment_model",
    "alarm_code",
    "part_name",
    "sop_type",
    "safety_level",
    "maintenance_stage",
]


def _deduplicate(values: Iterable[str]) -> list[str]:
    """去空并按首次出现顺序去重，保证标识列表稳定、可重放。"""
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        result.append(normalized)
        seen.add(normalized)
    return result


def normalize_equipment_model(prefix: str, number: str | None = None) -> str:
    """
    把设备型号统一成 ``字母前缀 + 空格 + 数字后缀``，例如 ``HAK 180``。

    ``number`` 为空时会从完整文本重新解析，供 metadata 和测试直接调用；无法满足保守
    型号规则时返回空字符串，不擅自把普通单词或短数字当设备型号。
    """
    if number is None:
        match = EQUIPMENT_MODEL_PATTERN.fullmatch(str(prefix or "").strip())
        if not match:
            return ""
        prefix, number = match.group(1), match.group(2)
    normalized_prefix = re.sub(r"[^A-Za-z]", "", str(prefix or "")).upper()
    normalized_number = re.sub(r"[\s_-]+", "", str(number or "")).upper()
    if not normalized_prefix or not normalized_number:
        return ""
    return f"{normalized_prefix} {normalized_number}"


def normalize_alarm_code(value: str) -> str:
    """把报警码统一成大写 E + 数字，例如 E-020、e 020 都归一为 E020。"""
    match = ALARM_CODE_PATTERN.fullmatch(str(value or "").strip())
    return f"E{match.group(1)}" if match else ""


def normalize_sop_code(value: str) -> str:
    """把 SOP 编号统一为 ``SOP-编号``；只做格式归一，不推断编号含义。"""
    match = SOP_CODE_PATTERN.fullmatch(str(value or "").strip())
    if not match:
        return ""
    payload = re.sub(r"[\s_-]+", "-", match.group(1).upper()).strip("-")
    return f"SOP-{payload}" if payload else ""


def normalize_part_number(value: str) -> str:
    """统一零件编号的大小写和分隔符，不在没有上下文时判断它是不是零件编号。"""
    normalized = re.sub(r"[\s_-]+", "-", str(value or "").strip().upper()).strip("-")
    return normalized if any(character.isdigit() for character in normalized) else ""


def normalize_identifier_value(identifier_type: str, value: str) -> str:
    """按标识类型调用对应归一规则；中文部件名等文本字段只压缩空白。"""
    if identifier_type == "equipment_model":
        return normalize_equipment_model(value)
    if identifier_type == "alarm_code":
        return normalize_alarm_code(value)
    if identifier_type == "sop_code":
        return normalize_sop_code(value)
    if identifier_type == "part_number":
        return normalize_part_number(value)
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_identifier_mapping(query_identifiers: Mapping[str, Iterable[str]] | None) -> dict[str, list[str]]:
    """
    规范化 State/Observation 使用的标识字典，并拒绝契约之外的字段。

    每个 key 是标识类型，每个 value 是用户实际输入的一个或多个值。这里不会用相近候选
    覆盖用户原编号；建议值必须单独写入 RetrievalObservation.suggested_identifiers。
    """
    if not query_identifiers:
        return {}
    unsupported = sorted(set(query_identifiers) - set(QUERY_IDENTIFIER_FIELDS))
    if unsupported:
        raise ValueError("query_identifiers 包含不支持的字段：" + ", ".join(unsupported))

    result: dict[str, list[str]] = {}
    for identifier_type in QUERY_IDENTIFIER_FIELDS:
        if identifier_type not in query_identifiers:
            continue
        raw_values = query_identifiers[identifier_type]
        if isinstance(raw_values, (str, bytes)):
            raise ValueError(f"query_identifiers.{identifier_type} 必须是字符串列表")
        normalized_values = _deduplicate(
            normalize_identifier_value(identifier_type, value) for value in raw_values
        )
        if not normalized_values:
            raise ValueError(f"query_identifiers.{identifier_type} 没有可用的规范化值")
        result[identifier_type] = normalized_values
    return result


def _overlaps(span: tuple[int, int], occupied_spans: list[tuple[int, int]]) -> bool:
    """判断正则命中的文本区间是否已经被更明确的标识规则占用。"""
    return any(span[0] < end and start < span[1] for start, end in occupied_spans)


def extract_query_identifiers(query: str) -> dict[str, list[str]]:
    """
    从用户原始问题中提取设备型号、报警码、SOP 编号和明确零件编号。

    提取顺序体现规则可信度：先识别带明确前缀/上下文的报警码、SOP 和零件编号，再识别
    通用设备型号；后者不会重复占用前面已经确认的文本。E020/E021 会分别保留，绝不会
    因为只差一位就合并成一个编号。
    """
    text = str(query or "")
    if not text.strip():
        return {}

    found: dict[str, list[str]] = {field_name: [] for field_name in EXTRACTED_IDENTIFIER_FIELDS}
    occupied_spans: list[tuple[int, int]] = []

    for match in ALARM_CODE_PATTERN.finditer(text):
        found["alarm_code"].append(f"E{match.group(1)}")
        occupied_spans.append(match.span())

    for match in SOP_CODE_PATTERN.finditer(text):
        found["sop_code"].append(normalize_sop_code(match.group(0)))
        occupied_spans.append(match.span())

    for match in PART_NUMBER_PATTERN.finditer(text):
        found["part_number"].append(normalize_part_number(match.group(1)))
        occupied_spans.append(match.span())

    for match in EQUIPMENT_MODEL_PATTERN.finditer(text):
        if _overlaps(match.span(), occupied_spans):
            continue
        # SOP123 具有更明确的 SOP 语义；即使前面的模式因异常尾缀没有完整匹配，也不能
        # 再把它降级解释成设备型号。
        if match.group(1).upper() == "SOP":
            continue
        found["equipment_model"].append(
            normalize_equipment_model(match.group(1), match.group(2))
        )

    return {
        field_name: normalized_values
        for field_name in EXTRACTED_IDENTIFIER_FIELDS
        if (normalized_values := _deduplicate(found[field_name]))
    }


def append_identifiers_to_query(query: str, query_identifiers: Mapping[str, Iterable[str]] | None) -> str:
    """
    把规范化标识显式追加到检索文本，帮助 dense/learned sparse 通道识别编号变体。

    这一步只是增强召回文本，不代表系统已经确认某个相近编号。原始 query 保持不变，
    Observation 仍使用 requested/matched/suggested 三个独立字段记录真实关系。
    """
    normalized = normalize_identifier_mapping(query_identifiers)
    if not normalized:
        return str(query or "")

    parts: list[str] = []
    for identifier_type in QUERY_IDENTIFIER_FIELDS:
        for value in normalized.get(identifier_type, []):
            label = IDENTIFIER_TYPE_LABELS.get(identifier_type, identifier_type)
            parts.append(f"{label} {value}")
    return f"{str(query or '').strip()}\n规范化设备标识：{'；'.join(parts)}"


def extract_identifiers_from_record(record: Mapping[str, object]) -> dict[str, list[str]]:
    """从一个有权限的 chunk/词典记录中读取 metadata，并补充正文里的确定性标识。"""
    result: dict[str, list[str]] = {}
    for identifier_type in QUERY_IDENTIFIER_FIELDS:
        raw_value = record.get(identifier_type)
        raw_values = raw_value if isinstance(raw_value, list) else [raw_value]
        normalized_values = _deduplicate(
            normalize_identifier_value(identifier_type, value) for value in raw_values
        )
        if normalized_values:
            result[identifier_type] = normalized_values

    # metadata 可能因旧导入流程漏填，所以第二段还要检查正文和标题。只使用确定性规则，
    # 不把 embedding 相似的编号写入 matched_identifiers。
    text = "\n".join(
        str(record.get(field_name) or "")
        for field_name in ("content", "title", "source_title", "parent_title", "file_title")
    )
    for identifier_type, values in extract_query_identifiers(text).items():
        result[identifier_type] = _deduplicate([*result.get(identifier_type, []), *values])
    return result


def _record_text(record: Mapping[str, object]) -> str:
    """拼接可用于编号核验的标题与正文；不包含向量或其他不可读字段。"""
    return "\n".join(
        str(record.get(field_name) or "")
        for field_name in ("content", "title", "source_title", "parent_title", "file_title")
    )


def _record_contains_exact_identifier(
        record: Mapping[str, object],
        identifier_type: str,
        requested_value: str,
) -> bool:
    """
    在已知用户原编号的前提下，核验正文是否包含同一编号的分隔符变体。

    该逻辑与“从任意正文猜一个零件编号”不同：只有用户已明确输入 AB-123 后，才检查
    正文中的 AB123、AB-123、AB 123 是否为同一规范值。字母数字两侧使用边界，避免在
    XAB123Y 中误命中 AB123。中文部件名等非 ASCII 文本继续使用普通完整子串比较。
    """
    text = _record_text(record)
    if not text:
        return False
    compact_value = re.sub(r"[^A-Za-z0-9]", "", requested_value).upper()
    if compact_value and identifier_type in {
        "equipment_model",
        "alarm_code",
        "sop_code",
        "part_number",
    }:
        flexible_value = r"[\s_-]*".join(re.escape(character) for character in compact_value)
        return bool(re.search(rf"(?<![A-Za-z0-9]){flexible_value}(?![A-Za-z0-9])", text, re.IGNORECASE))
    return requested_value in text


def filter_records_matching_requested_identifiers(
        records: Iterable[Mapping[str, object]],
        requested_identifiers: Mapping[str, list[str]],
) -> tuple[list[Mapping[str, object]], dict[str, list[str]]]:
    """
    保留在同一条记录内兼容所有标识类型的证据，并汇总同码命中。

    例如用户同时指定 HAK 180 和 E020 时，一条只有 HAK 180、另一条只有 E020 不能拼成
    “同一设备已命中”。每条保留记录都必须在每个标识类型上至少命中一个用户值；同类型
    的多个值可以由多条兼容记录共同覆盖，以支持 E020/E021 对比类问题。
    """
    normalized_requested = normalize_identifier_mapping(requested_identifiers)
    matched_records: list[Mapping[str, object]] = []
    matched: dict[str, list[str]] = {}

    for record in records:
        record_identifiers = extract_identifiers_from_record(record)
        # SOP/零件编号当前没有独立 metadata。用户已经明确给出编号后，允许正文用不同
        # 分隔符书写同一个值；这里补入的仍是 requested_value 本身，不会产生相近候选。
        for identifier_type, requested_values in normalized_requested.items():
            exact_text_values = [
                value
                for value in requested_values
                if _record_contains_exact_identifier(record, identifier_type, value)
            ]
            if exact_text_values:
                record_identifiers[identifier_type] = _deduplicate(
                    [*record_identifiers.get(identifier_type, []), *exact_text_values]
                )
        if any(
            not set(record_identifiers.get(identifier_type, [])).intersection(requested_values)
            for identifier_type, requested_values in normalized_requested.items()
        ):
            continue
        matched_records.append(record)
        for identifier_type, requested_values in normalized_requested.items():
            values = set(record_identifiers.get(identifier_type, []))
            actual = [value for value in requested_values if value in values]
            matched[identifier_type] = _deduplicate([*matched.get(identifier_type, []), *actual])

    return matched_records, matched


def requested_identifiers_are_covered(
        requested_identifiers: Mapping[str, list[str]],
        matched_identifiers: Mapping[str, list[str]],
) -> bool:
    """判断 matched 是否完整覆盖用户请求；只比较规范化后的相同值。"""
    normalized_requested = normalize_identifier_mapping(requested_identifiers)
    return bool(normalized_requested) and all(
        set(requested_values).issubset(set(matched_identifiers.get(identifier_type, [])))
        for identifier_type, requested_values in normalized_requested.items()
    )


def _compact_identifier(value: str) -> str:
    """移除展示分隔符，仅用于编辑距离比较，不作为持久化编号。"""
    return re.sub(r"[^A-Za-z0-9]", "", str(value or "")).upper()


def _levenshtein_distance(left: str, right: str) -> int:
    """计算两个短编号的编辑距离；距离 1 表示一次增、删或改单字符。"""
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def _has_compatible_format(identifier_type: str, requested: str, candidate: str) -> bool:
    """候选必须先满足同类编号格式，再允许用编辑距离排序。"""
    if identifier_type == "alarm_code":
        return bool(re.fullmatch(r"E\d{3,4}", requested) and re.fullmatch(r"E\d{3,4}", candidate))
    if identifier_type == "equipment_model":
        requested_prefix = requested.split(" ", 1)[0]
        candidate_prefix = candidate.split(" ", 1)[0]
        return requested_prefix == candidate_prefix
    if identifier_type == "sop_code":
        return requested.startswith("SOP-") and candidate.startswith("SOP-")
    if identifier_type == "part_number":
        requested_prefix = re.match(r"[A-Z]+", requested)
        candidate_prefix = re.match(r"[A-Z]+", candidate)
        return bool(requested_prefix and candidate_prefix and requested_prefix.group() == candidate_prefix.group())
    return False


def rank_identifier_suggestions(
        requested_identifiers: Mapping[str, list[str]],
        authorized_records: Iterable[Mapping[str, object]],
        *,
        limit_per_type: int = 3,
) -> dict[str, list[str]]:
    """
    从“当前用户有权读取”的记录中生成相近编号候选，不执行自动纠错。

    候选按编辑距离、出现频次和字典序排序。若用户同时给了设备型号和报警码，候选记录
    还必须精确兼容其他标识类型；跨设备记录、无权限记录或只靠 embedding 猜出的值都
    不会进入建议列表。
    """
    requested = normalize_identifier_mapping(requested_identifiers)
    records_with_identifiers = [
        (record, extract_identifiers_from_record(record)) for record in authorized_records
    ]
    result: dict[str, list[str]] = {}

    for identifier_type, requested_values in requested.items():
        if identifier_type not in {"equipment_model", "alarm_code", "sop_code", "part_number"}:
            continue

        candidate_frequency: Counter[str] = Counter()
        candidate_distance: dict[str, int] = {}
        for _, record_identifiers in records_with_identifiers:
            # 其他用户已明确的类型必须在同一记录上精确兼容。字段缺失也按不兼容处理，
            # 宁可多问一次，也不能把另一个设备的报警码推荐成当前设备纠错结果。
            if any(
                other_type != identifier_type
                and not set(record_identifiers.get(other_type, [])).intersection(other_values)
                for other_type, other_values in requested.items()
            ):
                continue

            for candidate in record_identifiers.get(identifier_type, []):
                if candidate in requested_values:
                    continue
                best_distance = min(
                    _levenshtein_distance(_compact_identifier(requested_value), _compact_identifier(candidate))
                    for requested_value in requested_values
                    if _has_compatible_format(identifier_type, requested_value, candidate)
                ) if any(
                    _has_compatible_format(identifier_type, requested_value, candidate)
                    for requested_value in requested_values
                ) else None
                if best_distance is None or best_distance > 1:
                    continue
                candidate_frequency[candidate] += 1
                candidate_distance[candidate] = min(candidate_distance.get(candidate, best_distance), best_distance)

        ranked = sorted(
            candidate_frequency,
            key=lambda value: (candidate_distance[value], -candidate_frequency[value], value),
        )[:limit_per_type]
        if ranked:
            result[identifier_type] = ranked
    return result


def format_identifier_mapping(identifier_mapping: Mapping[str, list[str]]) -> str:
    """把机器字典转成简短中文文本，供确定性追问使用。"""
    parts = []
    for identifier_type in QUERY_IDENTIFIER_FIELDS:
        values = identifier_mapping.get(identifier_type, [])
        if not values:
            continue
        parts.append(f"{IDENTIFIER_TYPE_LABELS.get(identifier_type, identifier_type)} {'、'.join(values)}")
    return "；".join(parts)


def build_suggestion_question(
        requested_identifiers: Mapping[str, list[str]],
        suggested_identifiers: Mapping[str, list[str]],
) -> str:
    """生成“相近但不同编号”的确定性确认问题。"""
    requested_text = format_identifier_mapping(requested_identifiers)
    suggested_text = format_identifier_mapping(suggested_identifiers)
    return (
        f"当前知识库范围内没有找到您输入的{requested_text}，但找到了相近候选："
        f"{suggested_text}。请确认您要查询的是这些候选，还是仍要查询原编号？"
    )


def build_not_found_question(requested_identifiers: Mapping[str, list[str]]) -> str:
    """生成未找到同码证据或可靠候选时的核对提示。"""
    requested_text = format_identifier_mapping(requested_identifiers)
    return (
        f"当前知识库范围内没有找到您输入的{requested_text}，也没有可靠的相近候选。"
        "请核对设备屏幕、铭牌或文档上的完整编号，必要时上传清晰照片后再试。"
    )


def identifier_requires_clarification(observation: object) -> bool:
    """
    判断 Observation 是否要求在答案模型前终止并向用户确认。

    同时兼容 Pydantic 对象和将来从 Trace/JSON 恢复的字典，但只识别关闭枚举中的两个
    状态，其他未知字符串不会被当作合法终止理由。
    """
    if isinstance(observation, RetrievalObservation):
        status = observation.identifier_resolution_status
    elif isinstance(observation, Mapping):
        status = observation.get("identifier_resolution_status")
    else:
        return False
    return status in {
        IdentifierResolutionStatus.SUGGESTION_REQUIRED,
        IdentifierResolutionStatus.NOT_FOUND,
        IdentifierResolutionStatus.SUGGESTION_REQUIRED.value,
        IdentifierResolutionStatus.NOT_FOUND.value,
    }
