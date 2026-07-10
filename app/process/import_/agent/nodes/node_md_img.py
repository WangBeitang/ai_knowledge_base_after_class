from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.enrich_markdown_images import enrich_markdown_images
from app.infra.persistence.import_metadata_repository import STATUS_COMPLETED, safe_update_document

@node_log("node_md_img")
def node_md_img(state: ImportGraphState) -> dict:
    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
    """
    add_running_task(state["task_id"], "node_md_img")
    result_state = enrich_markdown_images(state)
    safe_update_document(
        state.get("document_id", ""),
        parse_status=STATUS_COMPLETED,
        md_path=result_state.get("md_path", ""),
        image_prefix=result_state.get("image_prefix", ""),
    )
    add_done_task(state["task_id"], "node_md_img")
    return {
        "md_path": result_state.get("md_path", ""),
        "md_content": result_state.get("md_content", ""),
    }


if __name__ == "__main__":
    from app.shared.utils.path_util import PROJECT_ROOT
    from app.shared.runtime.logger import  logger
    import  os
    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    test_md_name = os.path.join(r"output\hak180使用说明书", "hak180使用说明书.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "document_id": "doc_debug_md_img",
            "md_content": "",
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
