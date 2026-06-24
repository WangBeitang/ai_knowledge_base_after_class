# 定义主图全局state
import copy
import json
from typing import TypedDict
from app.shared.runtime.logger import logger


class ImportGraphState(TypedDict):
    # 任务标识：贯穿一次导入流程，用于日志、任务状态和前端进度查询
    task_id: str

    # 输入参数：由 API 或调用方传入，不应由后续节点随意改写
    local_file_path: str  # 原始上传文件路径，支持 .pdf / .md
    local_dir: str  # 本次导入任务的工作目录，用于保存解析产物、中间文件

    # 路由状态：由入口节点识别文件类型后写入，用于 LangGraph 条件分支
    is_md_read_enabled: bool  # 是否走 Markdown 导入路径
    is_pdf_read_enabled: bool  # 是否走 PDF 解析路径

    # 文件解析产物：PDF 转 Markdown 或直接读取 Markdown 后产生
    pdf_path: str  # PDF 文件路径，仅 PDF 导入时有效
    md_path: str  # 最终进入切分流程的 Markdown 文件路径
    md_content: str  # Markdown 完整正文内容
    file_title: str  # 文件标题，通常来自文件名 stem，作为主体识别兜底

    # 主体识别结果：阶段 1 引入 subject 概念，后续可扩展为标准主题和别名体系
    subject_name: str  # 文档对应的通用主体名称，后续可升级为标准主题

    # 切片结果：由 split 节点生成，后续主体识别、向量化、入库都围绕 chunks 增量补字段
    chunks: list  # 文档切片列表，每个 chunk 至少包含 content/title/file_title 等字段

    # 向量化结果：当前实现把 dense_vector / sparse_vector 回写到 chunks 内
    embedding_content: list  # 如保留独立字段，统一使用单数 embedding_content；不要再用 embeddings_content


# 提供快速实例化state的方法
# 模板
default_state: ImportGraphState = {
    "task_id": "",
    "local_file_path": "",
    "local_dir": "",
    "is_md_read_enabled": False,
    "is_pdf_read_enabled": False,
    "pdf_path": "",
    "md_path": "",
    "md_content": "",
    "file_title": "",
    "subject_name": "",
    "chunks": [],
    "embedding_content": [],
}

def create_default_state(**overwrite) -> ImportGraphState:
    copy_state = copy.deepcopy(default_state)
    copy_state.update(overwrite)
    return copy_state

def get_default_state():
    return copy.deepcopy(default_state)

if __name__ == "__main__":
    state = create_default_state(task_id="001")
    logger.info(f"测试state实例化方法：\n {json.dumps(state,ensure_ascii=False,indent=4)}")

    state1 = get_default_state()
    logger.info(f"测试state实例化方法：\n {json.dumps(state1,ensure_ascii=False, indent=4)}")

