"""
工具模块，负责提供 task 相关的辅助能力。

这里仍然保留原来的进程内任务状态，用于前端 SSE/轮询快速展示实时进度。
阶段 3 只是在导入任务上增加一层 Mongo 同步：已经注册过 document_id/dataset_id
的导入 task 会把状态快照写入 Mongo，查询链路复用 task_utils 时不会写入导入元数据表。
"""
from typing import Any, Dict, List
from app.infra.persistence.import_metadata_repository import (
    safe_update_task_nodes,
    safe_update_task_status,
)
from .sse_utils import push_to_session

# ---------------------------
# 内存态任务追踪（单进程）
# ---------------------------
# key: task_id
# value: 节点名列表（原始英文/节点ID）
_tasks_running_list: Dict[str, List[str]] = {}
_tasks_done_list: Dict[str, List[str]] = {}

# key: task_id
# value: status 字符串（如 pending/processing/completed/failed）
_tasks_status: Dict[str, str] = {}

# key: task_id
# value: 任务结果（例如 query 的 answer）
_tasks_result: Dict[str, Dict[str, Any]] = {}

# key: task_id
# value: 持久化任务元数据。
#
# “注册”的含义是：这个 task_id 已经在 upload 阶段创建过对应的
# document_id/dataset_id/task 元数据，可以安全同步到 Mongo 的 tasks 表。
# 查询流程也会用 task_utils 记录临时进度，但它没有 document/dataset 归属，
# 因此不会注册，也就不会把查询 session 写入导入任务历史。
_persistent_task_metadata: Dict[str, Dict[str, str]] = {}

TASK_STATUS_PENDING = "pending"
TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

# 节点名 -> 中文名映射（用于前端展示）
# 说明：这里的 key 应与 LangGraph 的 add_node("xxx", ...) 中的节点名一致。
_NODE_NAME_TO_CN: Dict[str, str] = {
    "upload_file": "开始上传文件",
    "node_entry": "检查文件",
    "node_pdf_to_md": "PDF转Markdown",
    "node_md_img": "Markdown图片处理",
    "node_subject_name_recognition": "主体名称识别",
    "node_document_split": "文档切分",
    "node_bge_embedding": "向量生成",
    "node_import_kg": "导入知识图谱",
    "node_import_milvus": "导入向量库",
    "__end__": "处理完成",
    "END": "处理完成",
    # --- Query 流程节点（kb/process/query/main_graph.py）---
    "node_subject_name_confirm": "确认问题主体",
    "node_answer_output": "生成答案",
    "node_rerank": "重排序",
    "node_rrf": "倒排融合",
    "node_web_search_mcp": "网络搜索",
    "node_search_embedding": "切片搜索",
    "node_search_embedding_hyde": "切片搜索(假设性文档)",
    "node_multi_search": "多路搜索",
    "node_query_kg": "查询知识图谱",
    "node_join": "多路搜索合并",
}


def _ensure_task(task_id: str) -> None:
    """确保 task_id 对应的数据结构已初始化。"""
    if task_id not in _tasks_running_list:
        _tasks_running_list[task_id] = []
    if task_id not in _tasks_done_list:
        _tasks_done_list[task_id] = []
    if task_id not in _tasks_result:
        _tasks_result[task_id] = {}


def _to_cn(node_name: str) -> str:
    """将节点名转换为中文展示名；若无映射则返回原名。"""
    return _NODE_NAME_TO_CN.get(node_name, node_name)


def to_display_node_list(node_names: List[str]) -> List[str]:
    return [_to_cn(n) for n in node_names]


def register_persistent_task(task_id: str, document_id: str, dataset_id: str, owner_user_id: str) -> None:
    """
    注册需要同步到 Mongo 的导入任务。

    upload 接口会先在 Mongo 里创建 document/task 记录，再调用这里建立
    task_id -> document_id/dataset_id/owner_user_id 的内存映射。后续 add_running_task、
    add_done_task、update_task_status 仍然维护原来的内存状态，但会额外把
    已注册导入任务的最新快照同步到 Mongo。

    查询链路也复用 task_utils 做 SSE 进度，但查询 session 不属于导入任务，
    因此只有显式注册过的 task_id 才会写入导入元数据表。
    """
    _ensure_task(task_id)
    _persistent_task_metadata[task_id] = {
        "document_id": document_id,
        "dataset_id": dataset_id,
        "owner_user_id": owner_user_id,
    }


def get_persistent_task_metadata(task_id: str) -> Dict[str, str]:
    return _persistent_task_metadata.get(task_id, {})


def _sync_persistent_task_nodes(task_id: str) -> None:
    """
    将已注册导入任务的节点进度同步到 Mongo。

    这里故意只同步 registered task：task_utils 是通用工具，查询 session、
    临时调试 task 都可能调用 add_running_task/add_done_task。如果不做注册判断，
    Mongo 的 tasks 表会混入不属于导入流程的临时进度记录。
    """
    if task_id not in _persistent_task_metadata:
        return
    safe_update_task_nodes(
        task_id=task_id,
        running_nodes=list(_tasks_running_list.get(task_id, [])),
        done_nodes=list(_tasks_done_list.get(task_id, [])),
    )


def add_running_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    """
    添加“正在运行”的节点任务。

    参数：
    - task_id: 任务ID
    - node_name: 节点名称(节点ID)
    """
    _ensure_task(task_id)
    running = _tasks_running_list[task_id]
    # 避免重复追加
    if node_name not in running:
        running.append(node_name)

    # 保持原来的内存态进度作为实时状态源；如果是 upload 阶段注册过的导入任务，
    # 再把当前 running/done 快照同步到 Mongo，用于服务重启后的历史状态查询。
    _sync_persistent_task_nodes(task_id)

    if is_stream:
        task_push_queue(task_id)


def add_done_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    """
    添加“已完成”的节点任务。

    注意：添加已完成任务时，会把同名的“正在运行”任务删除。

    参数：
    - task_id: 任务ID
    - node_name: 节点名称(节点ID)
    """
    _ensure_task(task_id)

    # 1) 从 running 中移除同名节点（可能出现重复，移除所有）
    running = _tasks_running_list[task_id]
    _tasks_running_list[task_id] = [n for n in running if n != node_name]

    # 2) 追加到 done（保持完成顺序），避免重复
    done = _tasks_done_list[task_id]
    if node_name not in done:
        done.append(node_name)

    # 同步的是完整快照而不是增量事件，这样 Mongo 中的 running_nodes/done_nodes
    # 始终能直接表达当前任务进度，不需要回放历史事件。
    _sync_persistent_task_nodes(task_id)

    if is_stream:
        task_push_queue(task_id)


def set_task_result(task_id: str, key: str, value: Any) -> None:
    """
    存储任务结果字段（如 answer / error）。
    """
    _ensure_task(task_id)
    _tasks_result[task_id][key] = value


def get_task_result(task_id: str, key: str, default: Any = "") -> Any:
    """
    获取任务结果字段（如 answer / error）。
    """
    _ensure_task(task_id)
    return _tasks_result.get(task_id, {}).get(key, default)


def get_task_status(task_id: str) -> str:
    """
    获取当前任务状态。

    参数：
    - task_id: 任务ID

    返回：
    - str: 状态名称；如果未设置过则返回空字符串
    """
    return _tasks_status.get(task_id, "")


def get_done_task_list(task_id: str) -> List[str]:
    """
    获取已完成节点列表（中文展示）。


    """
    _ensure_task(task_id)
    done = _tasks_done_list.get(task_id, [])
    return to_display_node_list(done)


def get_running_task_list(task_id: str) -> List[str]:
    """
    获取正在运行节点列表（中文展示）。

    """
    _ensure_task(task_id)
    running = _tasks_running_list.get(task_id, [])
    return to_display_node_list(running)


def get_done_task_node_names(task_id: str) -> List[str]:
    _ensure_task(task_id)
    return list(_tasks_done_list.get(task_id, []))


def get_running_task_node_names(task_id: str) -> List[str]:
    _ensure_task(task_id)
    return list(_tasks_running_list.get(task_id, []))


def get_last_running_task_node_name(task_id: str) -> str:
    running = get_running_task_node_names(task_id)
    if not running:
        return ""
    return running[-1]


def update_task_status(task_id: str, status_name: str, push_queue: bool = False) -> None:
    """
    更新任务状态。

    参数：
    - task_id: 任务ID
    - status_name: 状态名称（字符串）
    """
    _tasks_status[task_id] = status_name
    # 只有导入任务会被注册并同步到 Mongo；查询任务仍然只保留内存态状态，
    # 避免把用户问答 session 误当作文件导入历史。
    if task_id in _persistent_task_metadata:
        safe_update_task_status(task_id, status_name)
    if push_queue:
        task_push_queue(task_id)


def task_push_queue(task_id: str):
    push_to_session(task_id, "progress", {
        "status": get_task_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
    })


#
def clear_task(task_id: str):
    _tasks_running_list.pop(task_id, None)
    _tasks_done_list.pop(task_id, None)
    _tasks_status.pop(task_id, None)
    _tasks_result.pop(task_id, None)
    _persistent_task_metadata.pop(task_id, None)
