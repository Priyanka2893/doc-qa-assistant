from pydantic import BaseModel, Field
from enum import Enum


class RoutingDecision(str, Enum):
    VECTOR = "vector"
    WEB = "web"
    HYBRID = "hybrid"


class QueryComplexity(str, Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    doc_id: str | None = None
    session_id: str | None = None


class Citation(BaseModel):
    tag: str
    source_number: int
    text_excerpt: str
    source: str
    score: float


class AgentStep(BaseModel):
    step_type: str
    description: str
    details: dict = {}


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    agent_steps: list[AgentStep]
    retrieval_iterations: int
    tool_calls_made: int
    evidence_quality: str
    routing_decision: str
    query_complexity: str
    total_tokens_used: int
    request_id: str


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    embedding_model: str
    version: str = "1.0.0"
