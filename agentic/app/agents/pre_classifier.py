import re
from app.graph_state import AgentState

COMPLEX_PATTERNS = [
    r'\bcompare\b', r'\bvs\.?\b', r'\bversus\b', r'\bdifference between\b',
    r'\banalyz[es]\b', r'\banalysis\b', r'\bbreakdown\b',
    r'\bsummariz[es] all\b', r'\ball.*document', r'\bacross.*compan',
    r'\bindustry\b.*\bstandard', r'\bbenchmark', r'\bmarket\b.*\bdata',
    r'\bcurrent\b.*\bpric', r'\btrend', r'\bforecast', r'\bexternal\b',
]


async def pre_classifier_node(state: AgentState) -> AgentState:
    """Classify query as simple or complex using regex (no LLM, no latency)."""
    q = state["query"].lower()
    is_complex = any(re.search(p, q) for p in COMPLEX_PATTERNS)
    complexity = "complex" if is_complex else "simple"
    return {
        **state,
        "query_complexity": complexity,
        "agent_steps": state.get("agent_steps", []) + [{
            "step_type": "classification",
            "description": f"Query classified as '{complexity}'",
            "details": {"pattern_matched": is_complex},
        }]
    }


def complexity_router(state: AgentState) -> str:
    return state.get("query_complexity", "simple")
