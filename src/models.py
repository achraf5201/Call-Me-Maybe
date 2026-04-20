from pydantic import BaseModel
from typing import Dict


class Type(BaseModel):
    type: str


class Prompt(BaseModel):
    prompt: str


class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Type]
    returns: Type


class FunctionCall(BaseModel):
    function: str
    arguments: Dict
