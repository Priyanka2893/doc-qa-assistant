import time
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import AsyncQdrantClient

from app.config import get_settings
from app.database import init_db
from app.routers import documents, health, qa
from app.services.bm25_store import rebuild_indexes_from_qdrant
from app.services.embedder import get_embedder
from app.services.vector_store import init_collection

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer(),
    ],
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings

    # Ensure data directory exists
    Path(__file__).parent.parent.joinpath("data").mkdir(parents=True, exist_ok=True)

    # Init SQLite
    await init_db()

    # Init Qdrant
    qdrant_client = AsyncQdrantClient(
        host=settings.QDRANT_HOST, port=settings.QDRANT_PORT
    )
    await init_collection(qdrant_client, settings.QDRANT_COLLECTION_NAME, settings.EMBEDDING_DIMENSION)
    app.state.qdrant_client = qdrant_client

    # Pre-load embedding model (blocks until ready)
    get_embedder(settings.EMBEDDING_MODEL)

    # Rebuild BM25 indexes from existing Qdrant data
    await rebuild_indexes_from_qdrant(qdrant_client, settings.QDRANT_COLLECTION_NAME)

    logger.info(
        "app.started",
        env=settings.APP_ENV,
        qdrant=f"{settings.QDRANT_HOST}:{settings.QDRANT_PORT}",
        collection=settings.QDRANT_COLLECTION_NAME,
    )

    yield

    await qdrant_client.close()
    logger.info("app.shutdown")


app = FastAPI(
    title="Doc Q&A Assistant",
    description="RAG-powered document question answering via Groq LLM",
    version="1.0.0",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.ALLOWED_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next) -> Response:
    t_start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = int((time.perf_counter() - t_start) * 1000)
    logger.info(
        "http.request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=elapsed_ms,
    )
    return response


app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(documents.router, prefix="/api/v1", tags=["documents"])
app.include_router(qa.router, prefix="/api/v1", tags=["qa"])
