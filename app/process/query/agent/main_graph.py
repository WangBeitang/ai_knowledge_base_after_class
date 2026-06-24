from langgraph.graph import StateGraph, END
from app.process.query.agent.nodes.node_answer_output import node_answer_output
from app.process.query.agent.nodes.node_subject_name_confirm import node_subject_name_confirm
from app.process.query.agent.nodes.node_rerank import node_rerank
from app.process.query.agent.nodes.node_rrf import node_rrf
from app.process.query.agent.nodes.node_search_embedding import node_search_embedding
from app.process.query.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.process.query.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger

query_graph_builder = StateGraph(QueryGraphState)
# 添加节点
query_graph_builder.add_node("node_subject_name_confirm", node_subject_name_confirm)
query_graph_builder.add_node("node_search_embedding",node_search_embedding)
query_graph_builder.add_node("node_search_embedding_hyde",node_search_embedding_hyde)
query_graph_builder.add_node("node_web_search_mcp",node_web_search_mcp)
query_graph_builder.add_node("node_rrf",node_rrf)
query_graph_builder.add_node("node_rerank",node_rerank)
query_graph_builder.add_node("node_answer_output",node_answer_output)

# 添加起始索引
query_graph_builder.set_entry_point("node_subject_name_confirm")

# 条件路由
def after_node_subject_name_confirm(state: QueryGraphState):
    if state.get("answer"):
        logger.info("本次没有明确的主体名称，结束流程，待用户确定")
        return "node_answer_output"
    else:
        logger.info(f"有明确的主体名称{state.get('subject_names')}，开始向知识图谱检索")
        return "node_search_embedding","node_search_embedding_hyde","node_web_search_mcp"
# 添加条件边
query_graph_builder.add_conditional_edges(
    "node_subject_name_confirm",
    after_node_subject_name_confirm,
    {
        "node_answer_output":"node_answer_output",
        "node_search_embedding":"node_search_embedding",
        "node_search_embedding_hyde":"node_search_embedding_hyde",
        "node_web_search_mcp":"node_web_search_mcp"
    }
)

# 添加静态边
query_graph_builder.add_edge("node_search_embedding","node_rrf")
query_graph_builder.add_edge("node_search_embedding_hyde","node_rrf")
query_graph_builder.add_edge("node_web_search_mcp","node_rrf")
query_graph_builder.add_edge("node_rrf","node_rerank")
query_graph_builder.add_edge("node_rerank","node_answer_output")
query_graph_builder.add_edge("node_answer_output",END)

# 图编译
query_graph_app = query_graph_builder.compile()
