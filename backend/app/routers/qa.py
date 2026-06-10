import json

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app import database
from app.limiter import limiter
from app.middleware.request_id import request_id_var
from app.models import AskRequest, AskResponse, ChunkSource, GlobalAskRequest, GlobalAskResponse, GlobalChunkSource
from app.services.cache import get_semantic_cache
from app.services.llm import generate_answer, generate_answer_stream
from app.services.retriever import retrieve, retrieve_global

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.post("/qa/ask", response_model=AskResponse)
@limiter.limit("60/minute")
async def ask_question(request: Request, body: AskRequest) -> AskResponse:
    """Answer a natural language question grounded in a previously uploaded document."""
    settings = request.app.state.settings
    qdrant_client = request.app.state.qdrant_client
    cache = get_semantic_cache()

    doc = await database.get_document(body.document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{body.document_id}' not found.")

    cached = await cache.get_cached_answer(body.question, body.document_id)
    if cached is not None:
        return cached.model_copy(update={"cache_hit": True})

    results = await retrieve(
        question=body.question,
        doc_id=body.document_id,
        top_k=body.top_k,
        mode=body.search_mode,
        rerank=body.rerank,
        qdrant_client=qdrant_client,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        embedding_model=settings.EMBEDDING_MODEL,
    )

    llm_result = await generate_answer(
        question=body.question,
        chunks=[r.text for r in results],
        model=settings.GROQ_MODEL,
        api_key=settings.GROQ_API_KEY,
    )

    sources = [
        ChunkSource(
            chunk_index=r.chunk_index,
            text_excerpt=r.text[:300],
            score=r.score,
            page_number=r.page_number,
            vector_score=r.vector_score,
            bm25_score=r.bm25_score,
        )
        for r in results
    ]

    response = AskResponse(
        answer=llm_result["answer"],
        sources=sources,
        model=llm_result["model"],
        tokens_used=llm_result["tokens_used"],
        doc_id=body.document_id,
        cache_hit=False,
    )

    await cache.cache_answer(body.question, body.document_id, response)

    logger.info(
        "qa.answered",
        doc_id=body.document_id,
        question_len=len(body.question),
        sources_count=len(sources),
        mode=body.search_mode,
        rerank=body.rerank,
        request_id=request_id_var.get(""),
    )

    return response


@router.post("/qa/ask-stream")
@limiter.limit("60/minute")
async def ask_question_stream(request: Request, body: AskRequest) -> StreamingResponse:
    """Stream an answer token-by-token via Server-Sent Events."""
    settings = request.app.state.settings
    qdrant_client = request.app.state.qdrant_client
    req_id = request_id_var.get("")

    doc = await database.get_document(body.document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{body.document_id}' not found.")

    results = await retrieve(
        question=body.question,
        doc_id=body.document_id,
        top_k=body.top_k,
        mode=body.search_mode,
        rerank=body.rerank,
        qdrant_client=qdrant_client,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        embedding_model=settings.EMBEDDING_MODEL,
    )

    sources = [
        ChunkSource(
            chunk_index=r.chunk_index,
            text_excerpt=r.text[:300],
            score=r.score,
            page_number=r.page_number,
            vector_score=r.vector_score,
            bm25_score=r.bm25_score,
        )
        for r in results
    ]

    sources_payload = [s.model_dump() for s in sources]
    question = body.question
    doc_id = body.document_id

    async def event_generator():
        yield f"data: {json.dumps({'type': 'start', 'doc_id': doc_id, 'request_id': req_id})}\n\n"
        try:
            async for text_chunk in generate_answer_stream(
                question=question,
                chunks=[r.text for r in results],
                model=settings.GROQ_MODEL,
                api_key=settings.GROQ_API_KEY,
            ):
                yield f"data: {json.dumps({'type': 'chunk', 'text': text_chunk})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"
            return
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources_payload})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'tokens_used': 0})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/qa/ask-global", response_model=GlobalAskResponse)
@limiter.limit("30/minute")
async def ask_question_global(request: Request, body: GlobalAskRequest) -> GlobalAskResponse:
    """Answer a question by searching across ALL uploaded documents."""
    settings = request.app.state.settings
    qdrant_client = request.app.state.qdrant_client

    results = await retrieve_global(
        question=body.question,
        top_k=body.top_k,
        mode=body.search_mode,
        rerank=body.rerank,
        qdrant_client=qdrant_client,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        embedding_model=settings.EMBEDDING_MODEL,
    )

    llm_result = await generate_answer(
        question=body.question,
        chunks=[r.text for r in results],
        model=settings.GROQ_MODEL,
        api_key=settings.GROQ_API_KEY,
    )

    sources = [
        GlobalChunkSource(
            chunk_index=r.chunk_index,
            text_excerpt=r.text[:300],
            score=r.score,
            page_number=r.page_number,
            vector_score=r.vector_score,
            bm25_score=r.bm25_score,
            filename=r.filename,
            doc_id=r.doc_id,
        )
        for r in results
    ]

    logger.info(
        "qa.answered_global",
        question_len=len(body.question),
        sources_count=len(sources),
        mode=body.search_mode,
        rerank=body.rerank,
        request_id=request_id_var.get(""),
    )

    return GlobalAskResponse(
        answer=llm_result["answer"],
        sources=sources,
        model=llm_result["model"],
        tokens_used=llm_result["tokens_used"],
    )
