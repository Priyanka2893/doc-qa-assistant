import pytest
from app.tools.registry import ALL_TOOLS, TOOL_MAP
from app.services.embedder import get_embedder


# --- Registry ---

def test_tool_registry_has_3_tools():
    assert len(ALL_TOOLS) == 3


def test_tool_map_keys():
    assert "vector_search" in TOOL_MAP
    assert "filtered_vector_search" in TOOL_MAP
    assert "web_search" in TOOL_MAP


# --- Embedder ---

def test_embedder_singleton():
    m1 = get_embedder()
    m2 = get_embedder()
    assert m1 is m2


# --- Config ---

def test_config_reads_shared_env():
    from app.config import get_settings
    s = get_settings()
    assert s.groq_api_key, "groq_api_key must not be empty — check .env"


# --- Vector search ---

@pytest.mark.asyncio
async def test_vector_search_returns_list():
    from app.tools.vector_search import vector_search
    result = await vector_search.ainvoke({"query": "return policy"})
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_vector_search_with_doc_id():
    from app.tools.vector_search import vector_search
    result = await vector_search.ainvoke({"query": "policy", "doc_id": "nonexistent-doc-id-xyz"})
    assert isinstance(result, list)
    assert len(result) == 0


# --- Web search ---

@pytest.mark.asyncio
async def test_web_search_returns_results():
    from app.config import get_settings
    s = get_settings()
    if not s.tavily_api_key:
        pytest.skip("TAVILY_API_KEY not set")

    from app.tools.web_search import web_search
    result = await web_search.ainvoke({"query": "Python programming language"})
    assert isinstance(result, list)
    assert len(result) > 0
    assert "source" in result[0]
    assert result[0]["source"].startswith("http")


# --- Groq LLM ---

@pytest.mark.asyncio
async def test_groq_client_responds():
    from app.services.groq_client import get_llm
    llm = get_llm()
    response = await llm.ainvoke("Say 'test ok'")
    assert response.content
    assert len(response.content) > 0
