import pytest
from unittest.mock import AsyncMock, patch
from app.agents.pre_classifier import pre_classifier_node
from app.agents.executor import execution_node
from app.graph_state import AgentState


def _base_state(**overrides) -> AgentState:
    state: AgentState = {
        "query": "What is the return policy?",
        "doc_id": None,
        "session_id": None,
        "request_id": "test-req-id",
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
    state.update(overrides)
    return state


# --- Pre-classifier ---

@pytest.mark.asyncio
async def test_pre_classifier_simple():
    state = _base_state(query="What is the return policy?")
    result = await pre_classifier_node(state)
    assert result["query_complexity"] == "simple"


@pytest.mark.asyncio
async def test_pre_classifier_complex():
    state = _base_state(query="Compare our policy with industry standards")
    result = await pre_classifier_node(state)
    assert result["query_complexity"] == "complex"


# --- Planning ---

@pytest.mark.asyncio
async def test_planning_sets_routing():
    from app.agents.planner import planning_node
    from langchain_core.messages import AIMessage

    mock_response = AIMessage(content='{"routing": "hybrid", "sub_queries": ["q1", "q2"], "reasoning": "test"}')
    with patch("app.agents.planner.get_planning_llm") as mock_llm_factory:
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_llm_factory.return_value = mock_llm

        state = _base_state(query="Compare our policy with external benchmarks")
        result = await planning_node(state)

    assert result["routing_decision"] == "hybrid"
    assert len(result["sub_queries"]) == 2
    assert any(s["step_type"] == "planning" for s in result["agent_steps"])


# --- Executor ---

@pytest.mark.asyncio
async def test_execution_increments_counter():
    fake_chunks = [{"text": "chunk text", "source": "doc.pdf", "score": 0.9,
                    "doc_id": "d1", "page_number": 1, "company": None, "category": None, "chunk_index": 0}]

    with patch("app.agents.executor.vector_search") as mock_vs:
        mock_vs.ainvoke = AsyncMock(return_value=fake_chunks)

        state = _base_state(
            routing_decision="vector",
            sub_queries=["What is the return policy?"],
            tool_calls_made=0,
        )
        result = await execution_node(state)

    assert result["tool_calls_made"] == 1
    assert result["retrieval_iteration"] == 1
    assert len(result["retrieved_chunks"]) == 1


@pytest.mark.asyncio
async def test_tool_storm_guard():
    """When tool_calls_made >= max, executor should return exhausted without calling tools."""
    from app.config import get_settings
    s = get_settings()

    state = _base_state(tool_calls_made=s.max_tool_calls_per_task)
    result = await execution_node(state)

    assert result["evaluation_status"] == "exhausted"
    # No new tool calls should have been made
    assert result["tool_calls_made"] == s.max_tool_calls_per_task


# --- Generator ---

@pytest.mark.asyncio
async def test_generation_includes_source_tags():
    from app.agents.generator import generation_node
    from langchain_core.messages import AIMessage

    mock_answer = AIMessage(content="The return window is 30 days [Source 1]. No restocking fee [Source 2].")
    with patch("app.agents.generator.get_llm") as mock_llm_factory:
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_answer)
        mock_llm_factory.return_value = mock_llm

        chunks = [
            {"text": "Returns accepted within 30 days.", "source": "policy.pdf", "score": 0.95,
             "page_number": 1, "company": "Acme"},
            {"text": "No restocking fee applies.", "source": "policy.pdf", "score": 0.90,
             "page_number": 2, "company": "Acme"},
        ]
        state = _base_state(retrieved_chunks=chunks, evaluation_status="pending")
        result = await generation_node(state)

    assert result["final_answer"] != ""
    assert "[Source" in result["final_answer"]
    assert len(result["citations"]) == 2


# --- Full graph integration ---

@pytest.mark.asyncio
async def test_full_graph_invoke():
    from app.graph import build_graph
    from langchain_core.messages import AIMessage

    fake_chunks = [{"text": "Refunds are processed in 5-7 days.", "source": "faq.pdf",
                    "score": 0.88, "doc_id": "d1", "page_number": 1,
                    "company": None, "category": None, "chunk_index": 0}]
    plan_response = AIMessage(content='{"routing": "vector", "sub_queries": ["refund timeline"], "reasoning": "doc query"}')
    gen_response = AIMessage(content="Refunds take 5-7 days [Source 1].")

    with (
        patch("app.agents.planner.get_planning_llm") as mock_plan_factory,
        patch("app.agents.executor.vector_search") as mock_vs,
        patch("app.agents.generator.get_llm") as mock_gen_factory,
    ):
        mock_plan_llm = AsyncMock()
        mock_plan_llm.ainvoke = AsyncMock(return_value=plan_response)
        mock_plan_factory.return_value = mock_plan_llm

        mock_vs.ainvoke = AsyncMock(return_value=fake_chunks)

        mock_gen_llm = AsyncMock()
        mock_gen_llm.ainvoke = AsyncMock(return_value=gen_response)
        mock_gen_factory.return_value = mock_gen_llm

        graph = build_graph()
        initial = _base_state(query="What is the refund timeline?")
        final = await graph.ainvoke(initial)

    assert final["final_answer"] != ""
    assert len(final["agent_steps"]) > 0
