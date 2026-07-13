"""阶段 5 查询图：由 PlannerDecision 驱动的闭环检索流程。"""

from langgraph.graph import END, StateGraph

from app.process.query.agent.nodes.node_query_planner import node_query_planner
from app.process.query.agent.nodes.node_rerank import node_rerank
from app.process.query.agent.nodes.node_retrieval_observation import (
    current_decision,
    node_retrieval_observation,
)
from app.process.query.agent.nodes.node_rrf import node_rrf
from app.process.query.agent.nodes.node_search_embedding import node_search_embedding
from app.process.query.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.process.query.agent.nodes.node_subject_name_confirm import node_subject_name_confirm
from app.process.query.agent.nodes.node_terminal_response import node_terminal_response
from app.process.query.agent.nodes.node_trace_finalize import node_trace_finalize
from app.process.query.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.process.query.agent.state import QueryGraphState
from app.rag.query.contracts import QueryAction


# 路由表只做 Action -> LangGraph 节点的机械映射，不重新判断主体、分数或 fallback。
# 所有业务规则都已经在 RuleBasedPlanner.plan() 中执行，避免“Planner 一套规则、路由又
# 偷偷实现另一套规则”造成 Trace 无法解释真实路径。
PLANNER_ACTION_NODE = {
    QueryAction.LOCAL_SEARCH: "node_search_embedding",
    QueryAction.HYDE_SEARCH: "node_search_embedding_hyde",
    QueryAction.WEB_SEARCH: "node_web_search_mcp",
    QueryAction.ANSWER: "node_terminal_response",
    QueryAction.ASK_CLARIFICATION: "node_terminal_response",
    QueryAction.REFUSE: "node_terminal_response",
}


def route_planner_decision(state: QueryGraphState) -> str:
    """只读取经过 Pydantic 校验的 decision.action，并返回对应节点名称。"""
    return PLANNER_ACTION_NODE[current_decision(state).action]


query_graph_builder = StateGraph(QueryGraphState)

# 主体确认只输出结构化事实，随后进入 Planner；它不再通过 answer 是否为空决定路由。
query_graph_builder.add_node("node_subject_name_confirm", node_subject_name_confirm)
query_graph_builder.add_node("node_query_planner", node_query_planner)

# 三个检索 Action 只会执行 Planner 本轮选择的一个，不再在查询开始时并发全部运行。
query_graph_builder.add_node("node_search_embedding", node_search_embedding)
query_graph_builder.add_node("node_search_embedding_hyde", node_search_embedding_hyde)
query_graph_builder.add_node("node_web_search_mcp", node_web_search_mcp)

# 每个检索 Action 结束后都用所有已执行 Action 的原始列表重算一次外层 RRF，再统一
# rerank 和生成 Observation。Observation 写入 Action history 后回到 Planner 继续决策。
query_graph_builder.add_node("node_rrf", node_rrf)
query_graph_builder.add_node("node_rerank", node_rerank)
query_graph_builder.add_node("node_retrieval_observation", node_retrieval_observation)

# 三种终止 Action 共用一个交付节点；最后收口内存 Action Trace，再结束本次查询。
query_graph_builder.add_node("node_terminal_response", node_terminal_response)
query_graph_builder.add_node("node_trace_finalize", node_trace_finalize)

query_graph_builder.set_entry_point("node_subject_name_confirm")
query_graph_builder.add_edge("node_subject_name_confirm", "node_query_planner")
query_graph_builder.add_conditional_edges(
    "node_query_planner",
    route_planner_decision,
    {node_name: node_name for node_name in set(PLANNER_ACTION_NODE.values())},
)

for action_node in (
    "node_search_embedding",
    "node_search_embedding_hyde",
    "node_web_search_mcp",
):
    query_graph_builder.add_edge(action_node, "node_rrf")

query_graph_builder.add_edge("node_rrf", "node_rerank")
query_graph_builder.add_edge("node_rerank", "node_retrieval_observation")
query_graph_builder.add_edge("node_retrieval_observation", "node_query_planner")
query_graph_builder.add_edge("node_terminal_response", "node_trace_finalize")
query_graph_builder.add_edge("node_trace_finalize", END)

query_graph_app = query_graph_builder.compile()
