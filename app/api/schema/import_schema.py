from pydantic import BaseModel, Field


# 上传文件的响应数据类型
class UploadSchema(BaseModel):
    code:int = 200
    message:str
    task_ids:list[str] = Field(default_factory=list)
    document_ids:list[str] = Field(default_factory=list)
    dataset_id:str = ""
    owner_user_id:str = ""
    index_version:int = 0

# 查询任务状态的数据类型
class TaskStatusSchema(BaseModel):
    code:int = 200
    task_id:str
    task_type:str = ""
    status:str # 当前文件整体状态
    done_list:list[str] = Field(default_factory=list) # 当前task_id已完成的节点列表
    running_list:list[str] = Field(default_factory=list) # 当前task_id正在处理的节点列表
    document_id:str = ""
    dataset_id:str = ""
    failed_node:str = ""
    # error_code 是机器可读错误码。旧数据和普通节点异常默认为空；
    # import_service_restarted 表示该任务随旧服务进程退出而中断。
    error_code:str = ""
    error_message:str = ""
    created_at:str = ""
    updated_at:str = ""


class DocumentStatusSchema(BaseModel):
    code:int = 200
    document_id:str
    dataset_id:str = ""
    latest_task_id:str = ""
    file_name:str = ""
    file_path:str = ""
    local_dir:str = ""
    index_version:int = 0
    status:str = ""
    parse_status:str = ""
    index_status:str = ""
    chunk_count:int = 0
    subject_id:str = ""
    standard_subject_name:str = ""
    md_path:str = ""
    image_prefix:str = ""
    parse_result_zip_path:str = ""
    parse_result_dir:str = ""
    deleted_at:str = ""
    failed_node:str = ""
    # document.error_code 投影最新一次导入的机器失败原因，
    # 用于前端稳定区分服务重启中断和普通节点失败。
    error_code:str = ""
    error_message:str = ""
    created_at:str = ""
    updated_at:str = ""


class DocumentListSchema(BaseModel):
    code:int = 200
    items:list[DocumentStatusSchema] = Field(default_factory=list)


class TaskHistorySchema(BaseModel):
    code:int = 200
    document_id:str = ""
    items:list[TaskStatusSchema] = Field(default_factory=list)


class DeleteDocumentSchema(BaseModel):
    code:int = 200
    message:str
    document_id:str
    status:str = "deleted"
    deleted_at:str = ""


class RebuildDocumentSchema(BaseModel):
    code:int = 200
    message:str
    task_id:str
    document_id:str
    dataset_id:str = ""
    index_version:int = 0
