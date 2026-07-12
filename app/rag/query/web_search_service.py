import asyncio
import json

from app.infra.config.providers import infra_config
from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import RETRIEVAL_DEFAULT_LIMIT
from app.rag.query.contracts import (
    EvidenceSourceType,
    RetrievalCandidate,
    RetrievalChannel,
)
from app.shared.runtime.logger import logger,step_log

def check_params(state):
    rewritten_query = state.get("rewritten_query")
    if not rewritten_query:
        logger.error("请输入有效的查询")
        raise ValueError("请输入有效的查询")
    return rewritten_query


async def web_search_func(rewritten_query):
    # MCP/OpenAI Agents 依赖较重，并且只有联网搜索 fallback 真正执行时才需要。
    # 放在函数内懒加载，避免查询图 import/compile 阶段为未执行的 Web Search 付出启动成本。
    from agents.mcp import MCPServerStreamableHttp

    # 1.mcp_server初始化
    mcp_server = MCPServerStreamableHttp(
        name="web_search_mcp",
        client_session_timeout_seconds=300,
        params={
            "url": infra_config.mcp.mcp_base_url,
            "headers": {"Authorization": f"Bearer {infra_config.mcp.api_key}"},
            "timeout": 300
        },
        cache_tools_list=True,
        max_retry_attempts=3,
    )
    # 2.调用百炼联网搜索接口
    try:
        await mcp_server.connect()
        mcp_result = await mcp_server.call_tool(tool_name="bailian_web_search",
                                                arguments={"query": rewritten_query, "count": RETRIEVAL_DEFAULT_LIMIT})
        return mcp_result
    except Exception as e:
        logger.error(f"百炼联网搜索接口调用失败: {e}")
        raise e
    finally:
        await mcp_server.cleanup()


def search_by_web(state: QueryGraphState) -> QueryGraphState:
    """
    网络搜索服务：
    1. 通过 MCP 协议异步调用百炼联网搜索接口
    2. 将用户的查询转化为实时的、结构化的网络搜索结果
    3. 包含标题、链接和摘要
    4. 回写 web_search_docs
    """
    # 1.参数校验
    rewritten_query = check_params(state)

    # 2.调用业务的网络搜索工具
    mcp_result = asyncio.run(web_search_func(rewritten_query))
    logger.info(f"查询到的结果: {mcp_result}")

    # 3.获取结果
    search_text = mcp_result.content[0].text
    raw_pages = json.loads(search_text).get("pages", [])
    web_search_docs = []
    for rank, page in enumerate(raw_pages, start=1):
        url = str(page.get("url") or "").strip()
        content = str(page.get("snippet") or page.get("content") or "").strip()
        if not url or not content:
            # Web 候选没有本地 ID，URL 和摘要就是后续去重、rerank、Citation 的最低身份
            # 契约。缺少任一项时跳过，不能伪造 chunk_id 或把空正文送进 reranker。
            logger.warning(f"忽略缺少 url/摘要的 Web 搜索结果，rank={rank}")
            continue
        title = str(page.get("title") or url).strip()
        raw_score = page.get("score")
        retrieval_score = float(raw_score) if isinstance(raw_score, (int, float)) else 0.0
        candidate = RetrievalCandidate(
            document_id=None,
            chunk_id=None,
            dataset_id=None,
            index_version=None,
            chunk_index=None,
            title=title,
            source_title=title,
            content=content,
            source_type=EvidenceSourceType.WEB,
            retrieval_channels=[RetrievalChannel.WEB],
            retrieval_rank=rank,
            retrieval_score=max(0.0, retrieval_score),
            rerank_score=None,
            url=url,
        )
        web_search_docs.append(candidate.model_dump(mode="json"))
    logger.info(f"{rewritten_query}联网搜索查询到的结果: {web_search_docs}")

    # 4.回写
    state["web_search_docs"] = web_search_docs

    return state
