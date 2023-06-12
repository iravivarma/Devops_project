from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List

class input_params(BaseModel):
    text: str
    min_length: Optional[int] = 30
    max_length: Optional[int] = 150

    class Config:
        orm_mode = True


class OutputSchema(BaseModel):
    text: str = ""
class OutputResponse(BaseModel):
    data: List[OutputSchema]