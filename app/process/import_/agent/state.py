# 定义主图全局state
import copy
import json
from typing import TypedDict
from app.shared.runtime.logger import logger


class ImportGraphState(TypedDict):
    # 任务标识：贯穿一次导入流程，用于日志、任务状态和前端进度查询
    task_id: str
    # 知识库标识：阶段 3 引入的 dataset/document/task 管理元数据关联字段
    dataset_id: str
    # 文档标识：一个 document 可关联多次 task，当前导入任务写入 latest_task_id
    document_id: str
    # 文档级检索索引产物版本：表示当前 document 对应的 chunk/vector 入库版本
    index_version: int
    # 用户归属：阶段 3.5 引入，用于 document/task 导入历史隔离
    owner_user_id: str
    # 租户与可见性：当前阶段使用默认值，为后续多租户和共享能力预留
    tenant_id: str
    visibility: str

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
    parse_result_zip_path: str  # MinerU 解析结果 zip 本地路径，仅 PDF 导入时写入
    parse_result_dir: str  # MinerU 解析结果解压目录，仅 PDF 导入时写入
    image_prefix: str  # 当前 document 在 MinIO 中的图片对象前缀，图片增强时写入
    file_title: str  # 文件标题，通常来自文件名 stem，作为主体识别兜底

    # 主体识别结果：阶段 2 引入标准主题体系，统一用标准主题字段贯穿导入和查询
    subject_id: str  # 标准主题唯一标识，用于 chunk 关联和查询过滤
    standard_subject_name: str  # 标准主题名称，用于统一管理知识体系
    subject_aliases: list[str]  # 标准主题别名列表，用于导入和查询时的主体识别

    # 轻量领域字段：设备运维场景下的可选结构化标签
    equipment_model: str  # 设备型号
    alarm_code: str  # 报警码或故障码
    part_name: str  # 部件名称
    sop_type: str  # SOP 类型，如开机、停机、点检、维护
    safety_level: str  # 安全等级或风险级别
    maintenance_stage: str  # 维护阶段，如日常点检、故障排查、定期保养

    # 切片结果：由 split 节点生成，后续主体识别、向量化、入库都围绕 chunks 增量补字段
    chunks: list  # 文档切片列表，每个 chunk 至少包含 content/title/file_title 等字段

    # 向量化结果：当前实现把 dense_vector / sparse_vector 回写到 chunks 内
    embedding_content: list  # 如保留独立字段，统一使用单数 embedding_content；不要再用 embeddings_content


# 提供快速实例化state的方法
# 模板
default_state: ImportGraphState = {
    "task_id": "",
    "dataset_id": "",
    "document_id": "",
    "index_version": 0,
    "owner_user_id": "",
    "tenant_id": "",
    "visibility": "",
    "local_file_path": "",
    "local_dir": "",
    "is_md_read_enabled": False,
    "is_pdf_read_enabled": False,
    "pdf_path": "",
    "md_path": "",
    "md_content": "",
    "parse_result_zip_path": "",
    "parse_result_dir": "",
    "image_prefix": "",
    "file_title": "",
    "subject_id": "",
    "standard_subject_name": "",
    "subject_aliases": [],
    "equipment_model": "",
    "alarm_code": "",
    "part_name": "",
    "sop_type": "",
    "safety_level": "",
    "maintenance_stage": "",
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
