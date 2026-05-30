# 定义主图全局state
import copy
import json
from typing import TypedDict
from app.shared.runtime.logger import logger


class ImportGraphState(TypedDict):
    task_id: str # 调用流程标识

    # 流程控制标记（文件状态判断）
    is_md_read_enabled: bool
    is_pdf_read_enabled: bool

    # 地址路径内容
    local_file_path: str # 要解析的文件的地址（pdf、md）
    local_dir: str # 存储转化pdf生成的md文件的目录，是目录，是文件夹  本次任务的工作目录/输出目录，用来存放转换后的文件
    md_path: str # 专门存储md地址  最终要读取的 Markdown 文件完整路径
    pdf_path: str # 专门存储pdf地址 PDF文件完整路径
    file_title: str # 存储文件名，没有后缀  兜底

    # 文本和切块内容
    md_content: str # Markdown文件读取出来的完整文本内容
    item_name: str # 模型从文档中识别出的主体名称，例如企业名、项目名、合同名 file_title是item_name的兜底
    chunks: list # 文档切片后的文本块列表
    embedding_content: list # 每个文本块生成向量后的结果，通常包含 text、metadata、dense vector、sparse vector，也就是向量数据库需要的数据字段与格式

# 提供快速实例化state的方法
# 模板
default_state:ImportGraphState = {
    "task_id": "",
    "is_md_read_enabled": False,
    "is_pdf_read_enabled": False,
    "local_file_path": "",
    "local_dir": "",
    "md_path": "",
    "pdf_path": "",
    "file_title": "",
    "md_content": "",
    "item_name": "",
    "chunks": [],
    "embeddings_content": []
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


