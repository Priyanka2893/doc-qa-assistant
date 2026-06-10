import uuid

import httpx
import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = structlog.get_logger(__name__)

_QDRANT_RETRY_EXCEPTIONS = (httpx.ConnectError, httpx.TimeoutException, ConnectionError)


def _log_qdrant_retry(retry_state) -> None:
    logger.warning(
        "vector_store.retry_attempt",
        attempt=retry_state.attempt_number,
        error=str(retry_state.outcome.exception()),
    )


def _qdrant_retry(**kwargs):
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(_QDRANT_RETRY_EXCEPTIONS),
        reraise=True,
        before_sleep=_log_qdrant_retry,
        **kwargs,
    )


async def init_collection(
    client: AsyncQdrantClient,
    collection_name: str,
    dimension: int,
) -> None:
    """Create the Qdrant collection if it doesn't already exist."""
    existing = await client.collection_exists(collection_name)
    if not existing:
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=dimension,
                distance=qdrant_models.Distance.COSINE,
            ),
        )
        logger.info("vector_store.collection_created", name=collection_name, dimension=dimension)
    else:
        logger.info("vector_store.collection_exists", name=collection_name)


@_qdrant_retry()
async def upsert_chunks(
    client: AsyncQdrantClient,
    collection_name: str,
    doc_id: str,
    chunks: list[str],
    embeddings: list[list[float]],
    filename: str,
    page_numbers: list[int | None] | None = None,
    language: str | None = None,
    doc_title: str | None = None,
    author: str | None = None,
) -> list[str]:
    """Upsert chunk embeddings with enriched metadata payload into Qdrant.

    Returns the list of point IDs.
    """
    points: list[qdrant_models.PointStruct] = []
    char_offset = 0
    for i, chunk in enumerate(chunks):
        char_start = char_offset
        char_end = char_offset + len(chunk)
        char_offset = char_end
        points.append(
            qdrant_models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i],
                payload={
                    "text": chunk,
                    "doc_id": doc_id,
                    "filename": filename,
                    "chunk_index": i,
                    "page_number": page_numbers[i] if page_numbers else None,
                    "language": language,
                    "doc_title": doc_title,
                    "author": author,
                    "char_start": char_start,
                    "char_end": char_end,
                    "word_count": len(chunk.split()),
                },
            )
        )
    await client.upsert(collection_name=collection_name, points=points)
    logger.info("vector_store.upserted", doc_id=doc_id, chunk_count=len(points))
    return [str(p.id) for p in points]


@_qdrant_retry()
async def search_chunks(
    client: AsyncQdrantClient,
    collection_name: str,
    query_vector: list[float],
    doc_id: str,
    top_k: int,
) -> list[qdrant_models.ScoredPoint]:
    """Search for the top_k most relevant chunks filtered to a specific document."""
    response = await client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="doc_id",
                    match=qdrant_models.MatchValue(value=doc_id),
                )
            ]
        ),
        limit=top_k,
        with_payload=True,
    )
    return response.points


async def delete_document_chunks(
    client: AsyncQdrantClient,
    collection_name: str,
    doc_id: str,
) -> None:
    """Delete all Qdrant points belonging to a specific document."""
    await client.delete(
        collection_name=collection_name,
        points_selector=qdrant_models.FilterSelector(
            filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="doc_id",
                        match=qdrant_models.MatchValue(value=doc_id),
                    )
                ]
            )
        ),
    )
    logger.info("vector_store.deleted", doc_id=doc_id)


@_qdrant_retry()
async def search_chunks_global(
    client: AsyncQdrantClient,
    collection_name: str,
    query_vector: list[float],
    top_k: int,
) -> list[qdrant_models.ScoredPoint]:
    """Search for the top_k most relevant chunks across ALL documents (no doc_id filter)."""
    response = await client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return response.points


async def scroll_document_chunks(
    client: AsyncQdrantClient,
    collection_name: str,
    doc_id: str,
) -> list[dict]:
    """Return all chunks for a doc_id as a list of payload dicts (including 'id')."""
    results: list[dict] = []
    offset = None
    while True:
        points, next_offset = await client.scroll(
            collection_name=collection_name,
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="doc_id",
                        match=qdrant_models.MatchValue(value=doc_id),
                    )
                ]
            ),
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            results.append({"id": str(point.id), **(point.payload or {})})
        if next_offset is None:
            break
        offset = next_offset
    return results


async def count_collection(client: AsyncQdrantClient, collection_name: str) -> int:
    """Return the total number of points in the collection."""
    result = await client.count(collection_name=collection_name, exact=True)
    return result.count
