from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.split_service import split_document
from app.infra.persistence.import_metadata_repository import STATUS_PROCESSING, safe_update_document

@node_log("node_document_split")
def node_document_split(state: ImportGraphState) -> dict:
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    """
    add_running_task(state["task_id"], "node_document_split")
    result_state = split_document(state)
    # chunk_count 不放进 LangGraph state 里继续传递，因为后续节点真正依赖的是 chunks。
    # 这里直接回写 document 元数据，供文档列表展示和失败排查使用。
    safe_update_document(
        state.get("document_id", ""),
        chunk_count=len(result_state.get("chunks", [])),
        index_status=STATUS_PROCESSING,
    )
    add_done_task(state["task_id"], "node_document_split")
    return {
        "chunks": result_state.get("chunks", []),
    }


if __name__ == '__main__':
    from app.shared.utils.path_util import PROJECT_ROOT
    from app.shared.runtime.logger import  logger
    import  os
    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    test_md_name = os.path.join(r"output/hak180使用说明书", "hak180使用说明书_new.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": "",
            "file_title": "",
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        result_state = node_document_split(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
