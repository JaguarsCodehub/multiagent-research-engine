from pydantic import BaseModel
from typing import Literal, List, Dict, Optional
from datetime import datetime

class Source(BaseModel):
    url: str
    title: str
    snippet: str
    retrieved_at: datetime
    embedding_id: str                  # Pinecone doc ID

class ClaimVerification(BaseModel):
    claim: str
    verdict: Literal["VERIFIED", "DISPUTED", "UNVERIFIABLE", "NUANCED"]
    supporting_sources: List[Source]
    dispute_note: Optional[str] = None

class WebResult(BaseModel):
    sources: List[Source]
    claims: List[str]
    embeddings_stored: int

class DocAnalysis(BaseModel):
    chunk_id: str
    content: str
    source_doc: str

class DocResult(BaseModel):
    analysis: List[DocAnalysis]
    citations: List[str]
    claims: List[str] = []

class FactResult(BaseModel):
    verifications: List[ClaimVerification]

class FinalReport(BaseModel):
    topic: str
    session_id: str
    markdown: str
    total_sources: int
    verified_claims: int
    disputed_claims: int
    cost_usd: float
    agent_costs: Dict[str, float]      # per-agent token cost breakdown
    generated_at: datetime
