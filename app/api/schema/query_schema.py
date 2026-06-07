from pydantic import BaseModel

class QueryRequestParam(BaseModel):
    query:str
    session_id:str
    is_stream:bool=False

class QueryStreamResponse(BaseModel):
    message:str
    session_id:str

class QueryNotStreamResponse(BaseModel):
    message: str
    session_id: str
    answer:str
    done_list:list
    image_urls:list