from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.subject_name_service import recognize_and_index_subject_name

@node_log("node_subject_name_recognition")
def node_subject_name_recognition(state: ImportGraphState) -> dict:
    """
    节点: 主体识别 (node_subject_name_recognition)
    为什么叫这个名字: 识别文档核心描述的主体名称。
    """
    add_running_task(state["task_id"], "node_subject_name_recognition")
    result_state = recognize_and_index_subject_name(state)
    add_done_task(state["task_id"], "node_subject_name_recognition")
    # LangGraph 节点统一返回 partial state。
    # 这里既返回阶段 2 新增的标准主题字段，也继续返回 subject_name 兼容旧流程。
    # chunks 已经在 service 层完成 subject_id / standard_subject_name 回填，
    # 后续 embedding 和 Milvus 入库节点只需要继续透传 chunks 即可。
    return {
        "subject_name": result_state.get("subject_name", ""),
        "subject_id": result_state.get("subject_id", ""),
        "standard_subject_name": result_state.get("standard_subject_name", ""),
        "subject_aliases": result_state.get("subject_aliases", []),
        "chunks": result_state.get("chunks", []),
    }



# ===================== 本地测试方法（直接运行调试，无需启动LangGraph） =====================
def test_node_subject_name_recognition():
    from app.shared.runtime.logger import logger
    """
    主体名称识别节点本地测试方法
    功能：模拟LangGraph流程输入，独立测试node_subject_name_recognition节点全链路逻辑
    适用场景：本地开发、调试、单节点功能验证，无需启动整个LangGraph流程
    测试前准备：
        1. 确保项目环境变量配置完成（MILVUS_URL/SUBJECT_NAME_COLLECTION等）
        2. 确保大模型、Milvus、BGE-M3服务均可正常访问
        3. 确保prompt模板（subject_name_recognition/product_recognition_system）已存在
    """
    logger.info("=== 开始执行主体名称识别节点本地测试 ===")
    try:
        # 1. 构造模拟的ImportGraphState状态（模拟上游节点产出数据）
        mock_state = ImportGraphState({
            "task_id": "test_task_123456",  # 测试任务ID
            "file_title": "华为Mate60 Pro手机使用说明书",  # 模拟文件标题
            "file_name": "华为Mate60Pro说明书.pdf",  # 模拟原始文件名（兜底用）
            # 模拟文本切片列表（上游切片节点产出，含title/content字段）
            "chunks": [
                {
                    "parent_title":"华为手机天下第一",
                    "title": "产品简介",
                    "content": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。"
                },
                {
                    "parent_title": "华为手机天下第一",
                    "title": "拍照功能",
                    "content": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。"
                },
                {
                    "parent_title": "华为手机天下第一",
                    "title": "电池参数",
                    "content": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。"
                }
            ]
        })

        # 2. 调用主体名称识别核心节点
        result_state = node_subject_name_recognition(mock_state)

        # 3. 打印测试结果（调试用）
        logger.info("=== 主体名称识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别主体名称：{result_state.get('subject_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(f"第一个切片主体名称：{result_state.get('chunks', [{}])[0].get('subject_name')}")

    except Exception as e:
        logger.error(f"主体名称识别节点本地测试失败，原因：{str(e)}", exc_info=True)


# 测试方法运行入口：直接执行该文件即可触发测试
if __name__ == "__main__":
    # 执行本地测试
    test_node_subject_name_recognition()
