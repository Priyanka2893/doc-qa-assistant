import hashlib
from app.graph_state import AgentState
from app.tools.vector_search import vector_search
from app.tools.web_search import web_search
from app.config import get_settings
import structlog

log = structlog.get_logger()


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


async def execution_node(state: AgentState) -> AgentState:
    s = get_settings()

    if state.get("tool_calls_made", 0) >= s.max_tool_calls_per_task:
        log.warning("tool_storm_guard", calls=state.get("tool_calls_made", 0))
        return {**state, "evaluation_status": "exhausted"}

    routing = state.get("routing_decision", "vector")
    sub_queries = state.get("sub_queries", [state["query"]])
    doc_id = state.get("doc_id")
    tool_calls = state.get("tool_calls_made", 0)
    prev_hashes = set(state.get("previous_chunk_hashes", []))
    all_chunks: list[dict] = []
    new_steps: list[dict] = []

    for sub_query in sub_queries:
        if routing in ("vector", "hybrid"):
            vs_args: dict = {"query": sub_query, "top_k": s.top_k_retrieval}
            if doc_id:
                vs_args["doc_id"] = doc_id
            chunks = await vector_search.ainvoke(vs_args)
            tool_calls += 1
            all_chunks.extend(chunks)
            new_steps.append({
                "step_type": "tool_call",
                "description": f"vector_search: '{sub_query[:50]}'",
                "details": {"tool": "vector_search", "results": len(chunks)},
            })

        if routing in ("web", "hybrid"):
            web_chunks = await web_search.ainvoke({
                "query": sub_query, "max_results": s.tavily_max_results
            })
            tool_calls += 1
            all_chunks.extend(web_chunks)
            new_steps.append({
                "step_type": "tool_call",
                "description": f"web_search: '{sub_query[:50]}'",
                "details": {"tool": "web_search", "results": len(web_chunks)},
            })

    seen_this_run: set[str] = set()
    unique_chunks: list[dict] = []
    new_hashes: list[str] = []
    for chunk in all_chunks:
        h = _hash(chunk["text"])
        if h not in prev_hashes and h not in seen_this_run:
            unique_chunks.append(chunk)
            seen_this_run.add(h)
            new_hashes.append(h)

    if len(prev_hashes) > 0 and len(new_hashes) == 0:
        log.warning("retrieval_thrash_detected")
        return {
            **state,
            "retrieved_chunks": unique_chunks,
            "tool_calls_made": tool_calls,
            "evaluation_status": "exhausted",
            "agent_steps": state.get("agent_steps", []) + new_steps,
        }

    log.info("execution_done", unique_chunks=len(unique_chunks), tool_calls=tool_calls)
    return {
        **state,
        "retrieved_chunks": unique_chunks,
        "tool_calls_made": tool_calls,
        "retrieval_iteration": state.get("retrieval_iteration", 0) + 1,
        "previous_chunk_hashes": list(prev_hashes | seen_this_run),
        "evaluation_status": "pending",
        "agent_steps": state.get("agent_steps", []) + new_steps,
    }
