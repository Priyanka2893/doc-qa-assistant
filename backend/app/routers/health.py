import structlog
from fastapi import APIRouter, Request

from app.models import HealthResponse
from app.services.embedder import get_embedder
from app.services.vector_store import count_collection

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Return service health including Qdrant connectivity and embedding model status."""
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
        embedder = get_embedder(settings.EMBEDDING_MODEL)
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
