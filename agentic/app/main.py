from fastapi import FastAPI
from contextlib import asynccontextmanager
import structlog

log = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("agentic_rag_starting")
    from app.config import get_settings
    from app.services.embedder import get_embedder
    settings = get_settings()
    get_embedder(settings.embedding_model)
    log.info("agentic_rag_ready", port=settings.agentic_app_port)
    yield
    log.info("agentic_rag_shutdown")

app = FastAPI(
    title="Agentic RAG API",
    description="LangGraph autonomous retrieval system",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/api/v1/health")
async def health():
    from app.config import get_settings
    from qdrant_client import QdrantClient
    s = get_settings()
    try:
        QdrantClient(host=s.qdrant_host, port=s.qdrant_port).get_collections()
        qdrant = "connected"
    except Exception:
        qdrant = "error"
    return {"status": "healthy", "qdrant": qdrant, "version": "1.0.0"}
