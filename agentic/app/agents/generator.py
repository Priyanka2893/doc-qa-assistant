import re
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph_state import AgentState
from app.services.groq_client import get_llm
import structlog

log = structlog.get_logger()

CITED_PROMPT = """You are a precise document analyst. Answer using ONLY the provided context.
Tag every factual claim with [Source N]. If context is insufficient, say:
"Insufficient information found in available sources."

Context:
{context}
"""

PARTIAL_PROMPT = """You are a document analyst. Available evidence is limited.
Answer as best you can, noting any gaps clearly. Tag claims with [Source N].

Available context:
{context}
"""


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        info = f"[Source {i}] {chunk.get('source', 'unknown')}"
        if chunk.get('page_number'):
            info += f" (page {chunk['page_number']})"
        if chunk.get('company'):
            info += f" | {chunk['company']}"
        parts.append(f"{info}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def _parse_citations(answer: str, chunks: list[dict]) -> list[dict]:
    seen, citations = set(), []
    for m in re.finditer(r'\[Source (\d+)\]', answer):
        n = int(m.group(1))
        if 1 <= n <= len(chunks) and n not in seen:
            seen.add(n)
            c = chunks[n - 1]
            citations.append({
                "tag": f"[Source {n}]", "source_number": n,
                "text_excerpt": c["text"][:200],
                "source": c.get("source", ""),
                "score": c.get("score", 0.0),
            })
    return citations


async def generation_node(state: AgentState) -> AgentState:
    chunks = state.get("filtered_chunks") or state.get("retrieved_chunks", [])
    is_partial = state.get("evaluation_status") == "exhausted"

    if not chunks:
        return {
            **state,
            "final_answer": "Insufficient information found in available sources. Please upload relevant documents.",
            "citations": [],
            "agent_steps": state.get("agent_steps", []) + [{
                "step_type": "generation", "description": "No chunks available",
                "details": {"is_partial": True}
            }]
        }

    context = _format_context(chunks)
    prompt = PARTIAL_PROMPT if is_partial else CITED_PROMPT
    llm = get_llm()
    messages = [
        SystemMessage(content=prompt.format(context=context)),
        HumanMessage(content=f"Question: {state['query']}")
    ]
    response = await llm.ainvoke(messages)
    answer = response.content
    citations = _parse_citations(answer, chunks)

    tokens = 0
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        tokens = response.usage_metadata.get("total_tokens", 0)

    log.info("generation_done", citations=len(citations), partial=is_partial)
    return {
        **state,
        "final_answer": answer,
        "citations": citations,
        "total_tokens_used": state.get("total_tokens_used", 0) + tokens,
        "agent_steps": state.get("agent_steps", []) + [{
            "step_type": "generation",
            "description": f"Generated answer with {len(citations)} citations (partial={is_partial})",
            "details": {"is_partial": is_partial}
        }]
    }
