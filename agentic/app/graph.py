from langgraph.graph import StateGraph, END
from app.graph_state import AgentState
from app.agents.pre_classifier import pre_classifier_node
from app.agents.planner import planning_node
from app.agents.executor import execution_node
from app.agents.generator import generation_node
import structlog

log = structlog.get_logger()


def post_execute_router(state: AgentState) -> str:
    """After execution: exhausted → generate directly, otherwise → generate (CRAG in A3)."""
    if state.get("evaluation_status") == "exhausted":
        return "generate"
    return "generate"   # In A3 this becomes → "evaluate"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("pre_classify", pre_classifier_node)
    graph.add_node("plan", planning_node)
    graph.add_node("execute", execution_node)
    graph.add_node("generate", generation_node)

    graph.set_entry_point("pre_classify")

    graph.add_edge("pre_classify", "plan")
    graph.add_edge("plan", "execute")

    graph.add_conditional_edges("execute", post_execute_router, {"generate": "generate"})

    graph.add_edge("generate", END)

    compiled = graph.compile()
    log.info("langgraph_graph_compiled")
    return compiled


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
