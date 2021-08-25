from typing import List

from pydantic import BaseModel


class RecordBaseResponse(BaseModel):
    status_code: int = 200
    message: str = None
    result: str = None


class RecordsBaseResponse(BaseModel):
    status_code: int = 200
    message: str = None
    result: List[str] = None
