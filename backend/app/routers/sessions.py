from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app import database
from app.services.cache import get_semantic_cache
from app.services.corrections import get_correction_store
from app.services.embedder import get_embedder
from app.services.session_memory import get_session_memory

logger = structlog.get_logger(__name__)
router = APIRouter()


class CreateSessionRequest(BaseModel):
    doc_id: str


class SessionInfoResponse(BaseModel):
    session_id: str
    doc_id: str
    turn_count: int
    last_active: str | None
    created_at: str


class CorrectionRequest(BaseModel):
    doc_id: str
    question: str
    original_answer: str
    corrected_answer: str


@router.post("/sessions", status_code=201)
async def create_session(request: Request, body: CreateSessionRequest) -> dict:
    """Create a new conversation session for a document."""
    doc = await database.get_document(body.doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{body.doc_id}' not found.")

    session_id = get_session_memory().create_session(body.doc_id)
    logger.info("session.created_via_api", session_id=session_id, doc_id=body.doc_id)
    return {"session_id": session_id, "doc_id": body.doc_id}


@router.get("/sessions/{session_id}", response_model=SessionInfoResponse)
async def get_session(session_id: str) -> SessionInfoResponse:
    """Get metadata for an existing conversation session."""
    session = get_session_memory().get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    return SessionInfoResponse(
        session_id=session.session_id,
        doc_id=session.doc_id,
        turn_count=len(session.turns),
        last_active=session.last_active.isoformat(),
        created_at=session.created_at.isoformat(),
    )


@router.post("/corrections", status_code=201)
async def submit_correction(request: Request, body: CorrectionRequest) -> dict:
    """Submit an expert correction for a question-answer pair."""
    settings = request.app.state.settings
    doc = await database.get_document(body.doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{body.doc_id}' not found.")

    embedder = get_embedder(settings.EMBEDDING_MODEL)
    store = get_correction_store()
    correction_id = await store.add_correction(
        doc_id=body.doc_id,
        question=body.question,
        original_answer=body.original_answer,
        corrected_answer=body.corrected_answer,
        embedder=embedder,
    )
    return {"correction_id": correction_id}


@router.get("/cache/stats")
async def cache_stats() -> dict:
    """Return semantic cache statistics."""
    stats = get_semantic_cache().get_stats()
    return {
        "total_entries": stats.total_entries,
        "exact_hits": stats.exact_hits,
        "semantic_hits": stats.semantic_hits,
        "misses": stats.misses,
        "hit_rate": stats.hit_rate,
        "avg_response_time_saved_ms": stats.avg_response_time_saved_ms,
    }
