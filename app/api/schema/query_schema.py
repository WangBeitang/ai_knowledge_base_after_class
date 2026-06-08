from typing import Any

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

class ClearHistoryResponse(BaseModel):
    message:str
    deleted_count: int


class HistoryItem(BaseModel):
    id: str
    session_id: str
    role: str
    text: str
    rewritten_query: str
    item_names: list[str]
    image_urls: list[str]
    ts: Any


class HistoryResponse(BaseModel):
    session_id: str
    items: list[HistoryItem]