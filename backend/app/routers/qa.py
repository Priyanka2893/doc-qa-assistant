import structlog
from fastapi import APIRouter, HTTPException, Request

from app import database
from app.models import AskRequest, AskResponse, ChunkSource, GlobalAskRequest, GlobalAskResponse, GlobalChunkSource
from app.services.llm import generate_answer
from app.services.retriever import retrieve, retrieve_global

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.post("/qa/ask", response_model=AskResponse)
async def ask_question(request: Request, body: AskRequest) -> AskResponse:
    """Answer a natural language question grounded in a previously uploaded document."""
    settings = request.app.state.settings
    qdrant_client = request.app.state.qdrant_client

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

    logger.info(
        "qa.answered",
        doc_id=body.document_id,
        question_len=len(body.question),
        sources_count=len(sources),
        mode=body.search_mode,
        rerank=body.rerank,
    )

    return AskResponse(
        answer=llm_result["answer"],
        sources=sources,
        model=llm_result["model"],
        tokens_used=llm_result["tokens_used"],
        doc_id=body.document_id,
    )


@router.post("/qa/ask-global", response_model=GlobalAskResponse)
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
    )

    return GlobalAskResponse(
        answer=llm_result["answer"],
        sources=sources,
        model=llm_result["model"],
        tokens_used=llm_result["tokens_used"],
    )
