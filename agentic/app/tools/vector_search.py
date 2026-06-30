from langchain_core.tools import tool
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from app.services.embedder import encode_query
from app.config import get_settings
import structlog

log = structlog.get_logger()


def _get_qdrant():
    s = get_settings()
    return QdrantClient(host=s.qdrant_host, port=s.qdrant_port)


@tool
async def vector_search(query: str, doc_id: str = None, top_k: int = 5) -> list[dict]:
    """Search the internal document knowledge base using semantic similarity.

    Use this tool when:
    - The question is about documents uploaded to the system (policies, reports, manuals)
    - You need factual information from internal company documents
    - The query mentions HR, legal, finance, operations, or any company-specific topic

    Args:
        query: The search query text
        doc_id: Optional — restrict search to a single document by its ID
        top_k: Number of results to return (default 5)

    Returns:
        List of relevant text chunks with source filename, score, and metadata
    """
    s = get_settings()
    embedding = await encode_query(query)
    client = _get_qdrant()

    search_filter = None
    if doc_id:
        search_filter = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        )

    results = client.search(
        collection_name=s.qdrant_collection_name,
        query_vector=embedding,
        limit=top_k,
        query_filter=search_filter,
        with_payload=True,
    )

    chunks = []
    for r in results:
        p = r.payload or {}
        chunks.append({
            "text": p.get("text", ""),
            "source": p.get("filename", "unknown"),
            "doc_id": p.get("doc_id", ""),
            "score": float(r.score),
            "page_number": p.get("page_number"),
            "company": p.get("company"),
            "category": p.get("category"),
            "chunk_index": p.get("chunk_index", 0),
        })

    log.info("vector_search_done", query=query[:50], results=len(chunks))
    return chunks


@tool
async def filtered_vector_search(
    query: str, company: str = None, category: str = None, top_k: int = 5
) -> list[dict]:
    """Search documents filtered by company or document category.

    Use this tool when:
    - The question refers to a specific company ("AcmeCorp's policy")
    - The question is about a specific document type ("finance reports", "HR docs")

    Args:
        query: The search query
        company: Filter by company name (e.g. "AcmeCorp")
        category: Filter by category (e.g. "finance", "hr", "legal")
        top_k: Number of results to return
    """
    s = get_settings()
    embedding = await encode_query(query)
    client = _get_qdrant()

    conditions = []
    if company:
        conditions.append(FieldCondition(key="company", match=MatchValue(value=company)))
    if category:
        conditions.append(FieldCondition(key="category", match=MatchValue(value=category)))

    search_filter = Filter(must=conditions) if conditions else None

    results = client.search(
        collection_name=s.qdrant_collection_name,
        query_vector=embedding,
        limit=top_k,
        query_filter=search_filter,
        with_payload=True,
    )

    return [
        {
            "text": r.payload.get("text", ""),
            "source": r.payload.get("filename", ""),
            "doc_id": r.payload.get("doc_id", ""),
            "score": float(r.score),
            "page_number": r.payload.get("page_number"),
            "company": r.payload.get("company"),
            "category": r.payload.get("category"),
        }
        for r in results
    ]
