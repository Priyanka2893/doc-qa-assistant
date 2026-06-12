import time
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from qdrant_client import AsyncQdrantClient
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.database import init_db
from app.limiter import limiter
from app.middleware.logging import RequestLoggingMiddleware
from app.middleware.request_id import RequestIDMiddleware
from app.routers import documents, eval, health, qa
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
    app.state.is_ready = False
    app.state.start_time = time.time()

    Path(__file__).parent.parent.joinpath("data").mkdir(parents=True, exist_ok=True)

    await init_db()

    qdrant_client = AsyncQdrantClient(
        host=settings.QDRANT_HOST, port=settings.QDRANT_PORT
    )
    await init_collection(qdrant_client, settings.QDRANT_COLLECTION_NAME, settings.EMBEDDING_DIMENSION)
    app.state.qdrant_client = qdrant_client

    get_embedder(settings.EMBEDDING_MODEL)

    await rebuild_indexes_from_qdrant(qdrant_client, settings.QDRANT_COLLECTION_NAME)

    app.state.is_ready = True
    logger.info(
        "app.started",
        env=settings.APP_ENV,
        qdrant=f"{settings.QDRANT_HOST}:{settings.QDRANT_PORT}",
        collection=settings.QDRANT_COLLECTION_NAME,
    )

    yield

    app.state.is_ready = False
    await qdrant_client.close()
    logger.info("app.shutdown")


app = FastAPI(
    title="Doc Q&A Assistant",
    description="RAG-powered document question answering via Groq LLM",
    version="1.0.0",
    lifespan=lifespan,
)

settings = get_settings()

app.state.limiter = limiter


async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> Response:
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please wait before retrying.", "retry_after": 60},
    )


app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.ALLOWED_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(documents.router, prefix="/api/v1", tags=["documents"])
app.include_router(qa.router, prefix="/api/v1", tags=["qa"])
app.include_router(eval.router, prefix="/api/v1", tags=["eval"])
