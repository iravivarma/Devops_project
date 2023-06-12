from fastapi import FastAPI
from pydantic import BaseModel

class Params(BaseModel):
    name: str
    question: str 
    # url: str