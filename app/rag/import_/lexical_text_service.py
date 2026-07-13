"""
为 chunk 构造 Milvus BM25 使用的 ``lexical_text``（词法检索文本）。

``lexical`` 的中文含义是“词法/字面”。它与 dense、BGE-M3 learned sparse 的
语义召回不同：BM25 更关心用户输入的词或编号是否真实出现在文档中，以及它们在当前
知识库中的稀有程度。因此这里必须生成一份稳定、可解释、可重建的纯文本，而不能把
模型向量或临时检索分数写进来。

本模块只负责两件事：
1. 按固定字段顺序拼接 chunk 的可检索文字；
2. 为 HAK180、HAK-180、HAK 180 这类技术编号补齐常见分隔符变体。

BM25 稀疏向量不在这里计算。应用只写 ``lexical_text``，Milvus 的 BM25 Function 会
在插入时使用 Analyzer（分词器）生成 ``bm25_sparse_vector``。
"""

from __future__ import annotations

import re
from collections.abc import Mapping


# lexical_text 的最大长度必须与 collection schema 中 VARCHAR.max_length 完全一致。
# 当前 chunk 正文通常远小于该值；保留 65535 是为了不因新增 BM25 实验提前缩短正文。
LEXICAL_TEXT_MAX_LENGTH = 65535

# 字段顺序属于实验契约：相同 chunk 在重复导入时必须生成完全相同的 lexical_text，
# 才能公平比较 learned sparse 与 BM25。字段含义：
# - standard_subject_name：标准主题名，例如“HAK 180 烫金机”；
# - equipment_model：设备型号，例如 HAK 180；
# - alarm_code：报警/故障码，例如 E021；
# - part_name：中文部件名称，例如温度传感器；
# - sop_type：规程类型，例如开机、点检、维修；
# - safety_level：安全等级或防护要求；
# - maintenance_stage：维护流程阶段，例如故障定位；
# - source_title/title/parent_title：文档和章节标题；
# - content：chunk 正文，是 lexical_text 的主要内容来源。
LEXICAL_TEXT_SOURCE_FIELDS = (
    "standard_subject_name",
    "equipment_model",
    "alarm_code",
    "part_name",
    "sop_type",
    "safety_level",
    "maintenance_stage",
    "source_title",
    "title",
    "parent_title",
    "content",
)

# Analyzer（文本分析器）会把 lexical_text 切成 BM25 使用的 token（词元）。
# - jieba：负责中文分词；
# - cnalphanumonly：移除标点 token，同时保留汉字、拉丁字母和数字；
# - lowercase：把 HAK180/E021 转为 hak180/e021，保证查询大小写不同仍能命中。
#
# 不直接使用只面向普通英文的 standard Analyzer，因为运维文档同时包含连续中文和
# 大量设备编号。这个配置已通过 Milvus run_analyzer 真实验证，但部署到不同 Milvus
# 版本后仍应先执行 ``validate_lexical_analyzer``，再重建 collection。
LEXICAL_ANALYZER_PARAMS = {
    "tokenizer": "jieba",
    "filter": ["cnalphanumonly", "lowercase"],
}

# 技术编号的保守结构：字母前缀 + 数字开头的后缀。它覆盖 HAK180、E-021、
# SOP 2024-01、AB_123_X；不会把纯中文、普通英文单词或没有数字的文本扩成编号。
TECHNICAL_IDENTIFIER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z]{1,12})[\s_-]*(\d[A-Za-z0-9]*(?:[-_]+[A-Za-z0-9]+)*)(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def _deduplicate_non_empty(values) -> list[str]:
    """去空并按第一次出现的顺序去重，保证入库结果稳定、可重放。"""
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = re.sub(r"\s+", " ", str(value or "")).strip()
        if not normalized or normalized in seen:
            continue
        result.append(normalized)
        seen.add(normalized)
    return result


def build_identifier_variants(text: str) -> list[str]:
    """
    从已确认 metadata、标题和正文中提取技术编号，并生成三种常见书写形式。

    例如 ``HAK-180`` 会得到 ``HAK180``、``HAK-180``、``HAK 180``；``E 021``
    同理得到 ``E021``、``E-021``、``E 021``。这些变体只扩大“同一编号”的字面召回，
    不会把 E020 改写成 E021，也不承担编号纠错或用户确认职责。
    """
    variants: list[str] = []
    for match in TECHNICAL_IDENTIFIER_PATTERN.finditer(str(text or "")):
        prefix = match.group(1).upper()
        payload = re.sub(r"[\s_-]+", "", match.group(2)).upper()
        if not prefix or not payload:
            continue
        variants.extend(
            (
                f"{prefix}{payload}",
                f"{prefix}-{payload}",
                f"{prefix} {payload}",
            )
        )
    return _deduplicate_non_empty(variants)


def build_chunk_lexical_text(chunk: Mapping[str, object]) -> str:
    """
    按固定字段顺序构造一个 chunk 的 BM25 输入文本。

    超长时优先保留编号变体，因为正文已由 ``content`` 单独保存，而 lexical_text 的
    额外价值之一正是让设备型号、报警码、SOP/零件编号在不同分隔符下仍可精确召回。
    这里只截断 BM25 输入副本，不修改原始 content。
    """
    source_values = _deduplicate_non_empty(
        chunk.get(field_name) for field_name in LEXICAL_TEXT_SOURCE_FIELDS
    )
    source_text = "\n".join(source_values)
    identifier_text = " ".join(build_identifier_variants(source_text))

    if not identifier_text:
        return source_text[:LEXICAL_TEXT_MAX_LENGTH].rstrip()

    separator = "\n"
    full_text = f"{source_text}{separator}{identifier_text}" if source_text else identifier_text
    if len(full_text) <= LEXICAL_TEXT_MAX_LENGTH:
        return full_text

    # 极端情况下变体自身也可能很长，先限制它，再用剩余空间保存正文和标题前缀。
    kept_identifier_text = identifier_text[-LEXICAL_TEXT_MAX_LENGTH:]
    available_source_length = LEXICAL_TEXT_MAX_LENGTH - len(kept_identifier_text) - len(separator)
    if available_source_length <= 0:
        return kept_identifier_text[-LEXICAL_TEXT_MAX_LENGTH:]
    return f"{source_text[:available_source_length].rstrip()}{separator}{kept_identifier_text}"


def validate_lexical_analyzer(client) -> list[list[str]]:
    """
    调用真实 Milvus Analyzer 验证中文和技术编号 token，再允许创建 collection。

    schema 配置“看起来正确”并不足够：Analyzer 由 Milvus 服务端执行，版本或插件差异
    可能导致 HAK180/E021 被错误拆分或丢弃。这里至少要求连续编号保留为独立 token，
    且中文产生非空 token；失败时立即中止建库，避免全量重建后才发现 BM25 不可用。

    返回 token 列表是为了让运维脚本和测试可以记录真实分词结果；业务入库只关心校验
    是否通过，不把 token 写入 chunk 或 Trace。
    """
    sample_text = "HAK180 报警 E021 温度传感器故障"
    raw_result = client.run_analyzer(
        [sample_text],
        analyzer_params=LEXICAL_ANALYZER_PARAMS,
    )
    if not raw_result:
        raise RuntimeError("Milvus Analyzer 未返回任何 token，禁止创建 BM25 collection")

    # PyMilvus 2.6 对单条文本可能返回 AnalyzeResult，对文本列表则返回
    # list[AnalyzeResult]；测试桩常直接返回 list[list[str]]。这里统一读取 ``tokens``，
    # 避免把 SDK 对象误当成可下标/可迭代列表。
    first_result = raw_result[0] if isinstance(raw_result, list) else raw_result
    raw_tokens = getattr(first_result, "tokens", first_result)
    if not raw_tokens:
        raise RuntimeError("Milvus Analyzer 未返回任何 token，禁止创建 BM25 collection")
    tokens = [str(token).lower() for token in raw_tokens]
    missing_identifiers = [token for token in ("hak180", "e021") if token not in tokens]
    has_chinese_token = any(re.search(r"[\u4e00-\u9fff]", token) for token in tokens)
    if missing_identifiers or not has_chinese_token:
        detail = ", ".join(missing_identifiers) or "中文 token"
        raise RuntimeError(
            f"Milvus Analyzer 未稳定保留关键 token（缺少：{detail}），禁止创建 BM25 collection"
        )
    return [tokens]
