from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routers.query import router as query_router
import structlog

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("agentic_rag_starting")
    from app.services.embedder import get_embedder
    from app.graph import get_graph
    from app.config import get_settings
    s = get_settings()
    get_embedder(s.embedding_model)
    get_graph()
    log.info("agentic_rag_ready", port=s.agentic_app_port)
    yield
    log.info("agentic_rag_shutdown")


app = FastAPI(title="Agentic RAG API", version="1.0.0", lifespan=lifespan)
app.include_router(query_router)


@app.get("/api/v1/health")
async def health():
    from qdrant_client import QdrantClient
    from app.config import get_settings
    s = get_settings()
    try:
        QdrantClient(host=s.qdrant_host, port=s.qdrant_port).get_collections()
        qdrant = "connected"
    except Exception:
        qdrant = "error"
    return {"status": "healthy", "qdrant": qdrant, "version": "1.0.0"}
