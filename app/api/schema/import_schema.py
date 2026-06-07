from pydantic import BaseModel


# 上传文件的响应数据类型
class UploadSchema(BaseModel):
    code:int = 200
    message:str
    task_ids:list[str]

# 查询任务状态的数据类型
class TaskStatusSchema(BaseModel):
    code:int = 200
    task_id:str
    status:str # 当前文件整体状态
    done_list:list[str] # 当前task_id已完成的节点列表
    running_list:list[str] # 当前task_id正在处理的节点列表

