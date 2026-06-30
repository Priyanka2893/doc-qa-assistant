from fastapi import APIRouter, HTTPException
from app.models import QueryRequest, QueryResponse, AgentStep, Citation
from app.graph import get_graph
from app.graph_state import AgentState
import uuid, time, structlog

log = structlog.get_logger()
router = APIRouter(prefix="/api/v1", tags=["query"])


def _initial_state(request: QueryRequest, request_id: str) -> AgentState:
    return {
        "query": request.question,
        "doc_id": request.doc_id,
        "session_id": request.session_id,
        "request_id": request_id,
        "query_complexity": "simple",
        "sub_queries": [],
        "routing_decision": "vector",
        "retrieved_chunks": [],
        "tool_calls_made": 0,
        "retrieval_iteration": 0,
        "previous_chunk_hashes": [],
        "evaluation_status": "pending",
        "filtered_chunks": [],
        "evidence_quality": "unknown",
        "final_answer": "",
        "citations": [],
        "agent_steps": [],
        "messages": [],
        "total_tokens_used": 0,
    }


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Run a full agentic RAG query. Returns answer with full reasoning trace."""
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    graph = get_graph()
    try:
        final_state = await graph.ainvoke(_initial_state(request, request_id))
    except Exception as e:
        log.error("graph_failed", error=str(e), request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))

    duration_ms = int((time.perf_counter() - start) * 1000)
    log.info("query_complete", request_id=request_id, duration_ms=duration_ms,
             iterations=final_state.get("retrieval_iteration", 0),
             quality=final_state.get("evidence_quality", "unknown"))

    return QueryResponse(
        answer=final_state.get("final_answer", ""),
        citations=[Citation(**c) for c in final_state.get("citations", [])],
        agent_steps=[AgentStep(**s) for s in final_state.get("agent_steps", [])],
        retrieval_iterations=final_state.get("retrieval_iteration", 0),
        tool_calls_made=final_state.get("tool_calls_made", 0),
        evidence_quality=final_state.get("evidence_quality", "unknown"),
        routing_decision=final_state.get("routing_decision", "vector"),
        query_complexity=final_state.get("query_complexity", "simple"),
        total_tokens_used=final_state.get("total_tokens_used", 0),
        request_id=request_id,
    )
