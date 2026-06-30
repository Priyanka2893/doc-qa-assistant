from langchain_core.messages import HumanMessage, SystemMessage
from app.graph_state import AgentState
from app.services.groq_client import get_planning_llm
from app.tools.registry import get_tool_descriptions
import json, re, structlog

log = structlog.get_logger()

PLANNING_PROMPT = """You are a query planning agent for a document Q&A system.
Analyze the user query and create a retrieval plan.

Available tools:
{tool_descriptions}

Respond in JSON only — no explanation, no markdown:
{{
  "routing": "vector" or "web" or "hybrid",
  "sub_queries": ["query1", "query2"],
  "reasoning": "one sentence"
}}

Rules:
- "vector": question about internal documents only
- "web": question requires current external data only
- "hybrid": needs BOTH internal documents AND external data
- sub_queries: for simple = [original_query]; complex = 2-3 specific targeted questions
"""


async def planning_node(state: AgentState) -> AgentState:
    llm = get_planning_llm()
    messages = [
        SystemMessage(content=PLANNING_PROMPT.format(tool_descriptions=get_tool_descriptions())),
        HumanMessage(content=f"Query: {state['query']}")
    ]
    response = await llm.ainvoke(messages)

    try:
        match = re.search(r'\{.*\}', response.content, re.DOTALL)
        plan = json.loads(match.group()) if match else {}
    except Exception:
        plan = {}

    routing = plan.get("routing", "vector")
    sub_queries = plan.get("sub_queries") or [state["query"]]

    log.info("planning_done", routing=routing, sub_queries=len(sub_queries))
    return {
        **state,
        "routing_decision": routing,
        "sub_queries": sub_queries,
        "messages": state.get("messages", []) + messages + [response],
        "agent_steps": state.get("agent_steps", []) + [{
            "step_type": "planning",
            "description": f"Routing to '{routing}' with {len(sub_queries)} sub-queries",
            "details": {"routing": routing, "sub_queries": sub_queries,
                        "reasoning": plan.get("reasoning", "")},
        }]
    }
