from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from app.process.import_.agent.state import ImportGraphState
from app.process.import_.agent.nodes.node_entry import node_entry
from app.process.import_.agent.nodes.node_pdf_to_md import node_pdf_to_md
from app.process.import_.agent.nodes.node_md_img import node_md_img
from app.process.import_.agent.nodes.node_document_split import node_document_split
from app.process.import_.agent.nodes.node_item_name_recognition import node_item_name_recognition
from app.process.import_.agent.nodes.node_bge_embedding import node_bge_embedding
from app.process.import_.agent.nodes.node_import_milvus import node_import_milvus
from app.shared.runtime.logger import logger

load_dotenv()

# 1.初始化
main_graph_builder = StateGraph(ImportGraphState)

# 2.节点
main_graph_builder.add_node("node_entry", node_entry)
main_graph_builder.add_node("node_pdf_to_md", node_pdf_to_md)
main_graph_builder.add_node("node_md_img", node_md_img)
main_graph_builder.add_node("node_document_split", node_document_split)
main_graph_builder.add_node("node_item_name_recognition", node_item_name_recognition)
main_graph_builder.add_node("node_bge_embedding", node_bge_embedding)
main_graph_builder.add_node("node_import_milvus", node_import_milvus)

# 3.起始节点+条件边
main_graph_builder.set_entry_point("node_entry")

# 条件边函数
def node_entry_after(state: ImportGraphState):
    if state["is_md_read_enabled"]:
        logger.info(f"文件{state['local_file_path']}类型为markdown")
        return "node_md_img"
    elif state["is_pdf_read_enabled"]:
        logger.info(f"文件{state['local_file_path']}类型为pdf")
        return "node_pdf_to_md"
    else:
        logger.warning(f"文件{state['local_file_path']}类型非法")
        return END

# 添加条件边
main_graph_builder.add_conditional_edges("node_entry",
                                         node_entry_after,
                                         {
                                     "node_md_img":"node_md_img",
                                     "node_pdf_to_md":"node_pdf_to_md",
                                     END:END
                                 })

# 4.添加普通边
main_graph_builder.add_edge("node_pdf_to_md", "node_md_img")
main_graph_builder.add_edge("node_md_img", "node_document_split")
main_graph_builder.add_edge("node_document_split", "node_item_name_recognition")
main_graph_builder.add_edge("node_item_name_recognition", "node_bge_embedding")
main_graph_builder.add_edge("node_bge_embedding", "node_import_milvus")

# 5.编译
main_graph = main_graph_builder.compile()