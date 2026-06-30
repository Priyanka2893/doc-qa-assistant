from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    # Input
    query: str
    doc_id: str | None
    session_id: str | None
    request_id: str

    # Classification
    query_complexity: str          # "simple" | "complex"

    # Planning
    sub_queries: list[str]
    routing_decision: str          # "vector" | "web" | "hybrid"

    # Execution tracking
    retrieved_chunks: list[dict]
    tool_calls_made: int
    retrieval_iteration: int
    previous_chunk_hashes: list[str]

    # CRAG evaluation (populated in A3)
    evaluation_status: str         # "pending"|"sufficient"|"needs_more_info"|"exhausted"
    filtered_chunks: list[dict]
    evidence_quality: str          # "high"|"medium"|"low"|"none"

    # Output
    final_answer: str
    citations: list[dict]
    agent_steps: list[dict]        # trace: list of {step_type, description, details}

    # LLM messages (auto-appended via reducer)
    messages: Annotated[list[BaseMessage], operator.add]

    # Metrics
    total_tokens_used: int
