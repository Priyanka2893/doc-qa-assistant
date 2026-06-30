from langchain_core.tools import tool
from tavily import TavilyClient
from app.config import get_settings
import structlog

log = structlog.get_logger()


@tool
async def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the internet for current, real-time, or external information.

    Use this tool when:
    - Internal documents don't contain the answer
    - The question requires current market data, recent news, or live information
    - The question asks about industry standards, benchmarks, or external practices
    - Vector search returned low-confidence or no results

    Args:
        query: Specific search query (be precise for better results)
        max_results: Number of results (default 5)

    Returns:
        List of web results with title, content excerpt, and URL
    """
    s = get_settings()
    if not s.tavily_api_key:
        log.warning("tavily_api_key_missing")
        return []

    client = TavilyClient(api_key=s.tavily_api_key)
    response = client.search(query=query, max_results=max_results, search_depth="basic")

    results = []
    for r in response.get("results", []):
        results.append({
            "text": r.get("content", ""),
            "source": r.get("url", ""),
            "doc_id": "web",
            "score": r.get("score", 0.5),
            "title": r.get("title", ""),
            "page_number": None,
            "company": None,
            "category": "web",
        })

    log.info("web_search_done", query=query[:50], results=len(results))
    return results
