import time

import structlog
from fastapi import APIRouter, HTTPException, Request

from app import database
from app.models import HealthResponse, MetricsResponse
from app.services.cache import get_semantic_cache
from app.services.embedder import get_embedder
from app.services.vector_store import count_collection

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/health/live")
async def liveness() -> dict:
    """Kubernetes liveness probe — always 200 if the process is running."""
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness(request: Request) -> dict:
    """Kubernetes readiness probe — 503 until startup completes and deps are reachable."""
    if not getattr(request.app.state, "is_ready", False):
        raise HTTPException(status_code=503, detail="Service not ready yet")

    settings = request.app.state.settings
    qdrant_client = request.app.state.qdrant_client

    try:
        await count_collection(qdrant_client, settings.QDRANT_COLLECTION_NAME)
    except Exception as exc:
        logger.warning("health.ready_qdrant_failed", error=str(exc))
        raise HTTPException(status_code=503, detail="Qdrant unreachable")

    try:
        get_embedder(settings.EMBEDDING_MODEL)
    except Exception as exc:
        logger.warning("health.ready_embedder_failed", error=str(exc))
        raise HTTPException(status_code=503, detail="Embedding model unavailable")

    return {"status": "ready"}


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Detailed health — includes Qdrant connectivity and embedding model status."""
    settings = request.app.state.settings
    qdrant_client = request.app.state.qdrant_client

    qdrant_status = "ok"
    try:
        await count_collection(qdrant_client, settings.QDRANT_COLLECTION_NAME)
    except Exception as exc:
        logger.warning("health.qdrant_failed", error=str(exc))
        qdrant_status = "unreachable"

    embedding_status = "ok"
    try:
        get_embedder(settings.EMBEDDING_MODEL)
        embedding_model_name = settings.EMBEDDING_MODEL
    except Exception as exc:
        logger.warning("health.embedder_failed", error=str(exc))
        embedding_status = "unavailable"
        embedding_model_name = settings.EMBEDDING_MODEL

    overall = "ok" if qdrant_status == "ok" and embedding_status == "ok" else "degraded"

    return HealthResponse(
        status=overall,
        qdrant=qdrant_status,
        embedding_model=embedding_model_name,
        version="1.0.0",
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(request: Request) -> MetricsResponse:
    """Operational statistics: documents, chunks, cache, uptime."""
    settings = request.app.state.settings
    qdrant_client = request.app.state.qdrant_client

    docs = await database.list_documents()
    total_chunks = sum(d.get("chunk_count", 0) for d in docs)

    try:
        qdrant_points = await count_collection(qdrant_client, settings.QDRANT_COLLECTION_NAME)
    except Exception:
        qdrant_points = -1

    cache = get_semantic_cache()
    start_time = getattr(request.app.state, "start_time", time.time())

    return MetricsResponse(
        total_documents=len(docs),
        total_chunks=total_chunks,
        cache_size=cache.cache_size(),
        cache_hit_rate=cache.hit_rate(),
        uptime_seconds=int(time.time() - start_time),
        embedding_model=settings.EMBEDDING_MODEL,
        qdrant_points=qdrant_points,
    )
