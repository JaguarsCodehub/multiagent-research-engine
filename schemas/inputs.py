from pydantic import BaseModel, Field
from typing import Literal, List
from uuid import uuid4

class ResearchQuery(BaseModel):
    topic: str
    depth: Literal["shallow", "medium", "deep"] = "medium"
    documents: List[str] = []          # local PDF paths
    max_sources: int = 10
    session_id: str = Field(default_factory=lambda: str(uuid4()))

class WebTask(BaseModel):
    queries: List[str]                  # 3-5 search queries
    num_sources_per_query: int = 3
    session_id: str

class DocTask(BaseModel):
    documents: List[str]               # file paths
    query: str
    session_id: str

class FactTask(BaseModel):
    claims: List[str]                  # extracted claims to verify
    query_context: str
    session_id: str
