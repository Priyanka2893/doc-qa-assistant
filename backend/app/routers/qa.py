import json

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app import database
from app.limiter import limiter
from app.middleware.request_id import request_id_var
from app.models import (
    AskRequest,
    AskResponse,
    CitedSource,
    GlobalAskRequest,
    GlobalAskResponse,
    GlobalChunkSource,
    InsufficientEvidenceResponse,
    ResponseMode,
)
from app.services.cache import get_semantic_cache
from app.services.citation_parser import parse_citations
from app.services.confidence_scorer import ScoredChunk, summarize_evidence_quality
from app.services.corrections import get_correction_store
from app.services.embedder import async_encode_query
from app.services.evaluator import evaluate_response
from app.services.hallucination_guard import (
    GateResult,
    log_hallucination_event,
    pre_generation_gate,
    verify_answer,
)
from app.services.llm import generate_answer, generate_answer_stream
from app.services.prompt_builder import build_messages
from app.services.retriever import retrieve, retrieve_global
from app.services.session_memory import get_session_memory

logger = structlog.get_logger(__name__)
router = APIRouter()


def _avg_confidence(chunks: list[ScoredChunk]) -> float:
    if not chunks:
        return 0.0
    return round(sum(c.confidence.composite_score for c in chunks) / len(chunks), 4)


def _cited_sources_from_result(citation_result, chunks: list[ScoredChunk]) -> list[CitedSource]:
    sources: list[CitedSource] = []
    for pc in citation_result.citations:
        chunk = pc.chunk
        sources.append(CitedSource(
            tag=pc.tag,
            source_number=pc.source_number,
            chunk_index=pc.chunk_index,
            page_number=pc.page_number,
            text_excerpt=pc.text_excerpt,
            filename=chunk.filename if chunk else "",
            confidence_score=pc.confidence_score,
            is_unmapped=False,
        ))
    for tag in citation_result.unmapped_citations:
        n = int(tag[len("[Source "):-1])
        sources.append(CitedSource(
            tag=tag,
            source_number=n,
            chunk_index=None,
            page_number=None,
            text_excerpt="",
            filename="",
            confidence_score=None,
            is_unmapped=True,
        ))
    return sources


@router.post("/qa/ask", response_model=AskResponse)
@limiter.limit("60/minute")
async def ask_question(request: Request, body: AskRequest) -> AskResponse:
    """Answer a natural language question grounded in a previously uploaded document."""
    settings = request.app.state.settings
    qdrant_client = request.app.state.qdrant_client
    cache = get_semantic_cache()
    corrections = get_correction_store()
    session_memory = get_session_memory()
    req_id = request_id_var.get("")

    doc = await database.get_document(body.document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{body.document_id}' not found.")

    # Step 1: embed question upfront (reused for cache lookup and semantic correction matching)
    question_embedding = await async_encode_query(settings.EMBEDDING_MODEL, body.question)

    # Auto-create session if not provided — session_id flows back to client in response
    session_id = body.session_id or session_memory.create_session(body.document_id)

    # Step 2: check expert corrections first
    from app.services.embedder import get_embedder
    embedder = get_embedder(settings.EMBEDDING_MODEL)
    correction = await corrections.find_correction(
        question=body.question,
        doc_id=body.document_id,
        embedder=embedder,
    )
    if correction is not None:
        return AskResponse(
            answer=correction.corrected_answer,
            cited_sources=[],
            model="correction",
            tokens_used=0,
            doc_id=body.document_id,
            cache_hit=False,
            cache_hit_type=None,
            session_id=session_id,
            is_correction=True,
        )

    # Step 3: semantic + exact cache check
    cache_entry, hit_type = await cache.get(body.document_id, body.question, question_embedding)
    if cache_entry is not None:
        return cache_entry.response.model_copy(
            update={
                "cache_hit": True,
                "cache_hit_type": hit_type,
                "session_id": session_id,
            }
        )

    # Step 4: inject session context into question for retrieval + LLM
    enriched_question = body.question
    context_prefix = session_memory.get_context_for_query(session_id)
    if context_prefix:
        enriched_question = context_prefix + "\n\n" + body.question

    output = await retrieve(
        question=enriched_question,
        doc_id=body.document_id,
        top_k=body.top_k,
        mode=body.search_mode,
        rerank=body.rerank,
        qdrant_client=qdrant_client,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        embedding_model=settings.EMBEDDING_MODEL,
        min_confidence=settings.MIN_CONFIDENCE_THRESHOLD,
        confidence_weights=settings.CONFIDENCE_WEIGHTS,
    )
    results = output.chunks
    chunks_filtered_out = output.filtered_out

    # Layer 1: Pre-generation gate
    gate_result = pre_generation_gate(
        results, settings.PRE_GEN_CONFIDENCE_GATE, settings.MIN_RAW_VECTOR_SCORE
    )
    if not gate_result.passed:
        await log_hallucination_event(
            request_id=req_id,
            doc_id=body.document_id,
            question=body.question,
            hallucination_risk=0.0,
            ungrounded_sentences=[],
            action_taken="blocked",
            gate_result=gate_result,
        )
        return JSONResponse(
            status_code=422,
            content=InsufficientEvidenceResponse(
                gate_reason=gate_result.reason,
                avg_confidence=gate_result.avg_confidence,
                chunk_count=gate_result.chunk_count,
            ).model_dump(),
        )

    messages = build_messages(enriched_question, results, mode=body.response_mode)
    llm_result = await generate_answer(
        messages=messages,
        model=settings.GROQ_MODEL,
        api_key=settings.GROQ_API_KEY,
        temperature=body.temperature,
    )

    citation_result = parse_citations(llm_result["answer"], results)
    cited_sources = _cited_sources_from_result(citation_result, results)
    evidence_quality = summarize_evidence_quality(results)

    # Layer 2: Post-generation verifier (2-stage: token fast path → semantic cosine fallback)
    verification = await verify_answer(
        llm_result["answer"],
        results,
        token_fast_path_threshold=settings.POST_GEN_TOKEN_FAST_PATH,
        semantic_threshold=settings.POST_GEN_OVERLAP_THRESHOLD,
        high_risk_threshold=settings.HIGH_RISK_THRESHOLD,
    )

    if verification.is_high_risk and settings.HALLUCINATION_ACTION == "block":
        await log_hallucination_event(
            request_id=req_id,
            doc_id=body.document_id,
            question=body.question,
            hallucination_risk=verification.hallucination_risk,
            ungrounded_sentences=verification.ungrounded_sentences,
            action_taken="blocked",
            gate_result=gate_result,
        )
        return JSONResponse(
            status_code=422,
            content=InsufficientEvidenceResponse(
                gate_reason="high_hallucination_risk",
                avg_confidence=gate_result.avg_confidence,
                chunk_count=gate_result.chunk_count,
            ).model_dump(),
        )

    action = "flagged" if verification.is_high_risk else "passed"
    await log_hallucination_event(
        request_id=req_id,
        doc_id=body.document_id,
        question=body.question,
        hallucination_risk=verification.hallucination_risk,
        ungrounded_sentences=verification.ungrounded_sentences,
        action_taken=action,
        gate_result=gate_result,
    )

    await database.insert_citation_audit(
        request_id=req_id,
        doc_id=body.document_id,
        question=body.question,
        answer_preview=llm_result["answer"],
        citation_count=len(citation_result.citations),
        unmapped_count=len(citation_result.unmapped_citations),
        is_abstention=citation_result.is_abstention,
        citation_coverage=citation_result.citation_coverage,
        evidence_quality=evidence_quality,
    )

    eval_metrics = await evaluate_response(
        question=body.question,
        chunks=results,
        answer=llm_result["answer"],
        verification_result=verification,
        is_abstention=citation_result.is_abstention,
    )
    await database.insert_eval_result(
        request_id=req_id,
        doc_id=body.document_id,
        question=body.question,
        context_relevance=eval_metrics.context_relevance,
        faithfulness=eval_metrics.faithfulness,
        answer_relevance=eval_metrics.answer_relevance,
        overall_score=eval_metrics.overall_score,
        chunk_count_used=eval_metrics.chunk_count_used,
        is_abstention=eval_metrics.is_abstention,
        hallucination_risk=eval_metrics.hallucination_risk,
    )

    response = AskResponse(
        answer=llm_result["answer"],
        cited_sources=cited_sources,
        unmapped_citations=citation_result.unmapped_citations,
        is_abstention=citation_result.is_abstention,
        citation_coverage=citation_result.citation_coverage,
        response_mode=body.response_mode,
        model=llm_result["model"],
        tokens_used=llm_result["tokens_used"],
        doc_id=body.document_id,
        cache_hit=False,
        cache_hit_type=None,
        session_id=session_id,
        is_correction=False,
        evidence_quality=evidence_quality,
        avg_confidence=_avg_confidence(results),
        chunks_filtered_out=chunks_filtered_out,
        hallucination_risk=verification.hallucination_risk,
        is_high_risk=verification.is_high_risk,
        ungrounded_sentences=verification.ungrounded_sentences,
        gate_passed=gate_result.passed,
        eval_metrics=eval_metrics,
    )

    # Step 6: store in cache (keyed by original question, not enriched)
    await cache.set(body.document_id, body.question, question_embedding, response)

    # Step 7: update session turn
    session_memory.add_turn(
        session_id=session_id,
        question=body.question,
        answer=llm_result["answer"],
        doc_id=body.document_id,
        cited_sources=cited_sources,
    )

    logger.info(
        "qa.answered",
        doc_id=body.document_id,
        question_len=len(body.question),
        cited_count=len(citation_result.citations),
        unmapped_count=len(citation_result.unmapped_citations),
        is_abstention=citation_result.is_abstention,
        citation_coverage=citation_result.citation_coverage,
        mode=body.search_mode,
        response_mode=body.response_mode,
        evidence_quality=evidence_quality,
        chunks_filtered_out=chunks_filtered_out,
        hallucination_risk=verification.hallucination_risk,
        is_high_risk=verification.is_high_risk,
        hallucination_action=action,
        session_id=body.session_id,
        request_id=req_id,
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

    session_memory = get_session_memory()
    session_id = body.session_id or session_memory.create_session(body.document_id)
    enriched_question = body.question
    context_prefix = session_memory.get_context_for_query(session_id)
    if context_prefix:
        enriched_question = context_prefix + "\n\n" + body.question

    output = await retrieve(
        question=enriched_question,
        doc_id=body.document_id,
        top_k=body.top_k,
        mode=body.search_mode,
        rerank=body.rerank,
        qdrant_client=qdrant_client,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        embedding_model=settings.EMBEDDING_MODEL,
        min_confidence=settings.MIN_CONFIDENCE_THRESHOLD,
        confidence_weights=settings.CONFIDENCE_WEIGHTS,
    )
    results = output.chunks

    messages = build_messages(enriched_question, results, mode=body.response_mode)
    doc_id = body.document_id

    async def event_generator():
        accumulated = ""
        yield f"data: {json.dumps({'type': 'start', 'doc_id': doc_id, 'request_id': req_id})}\n\n"
        try:
            async for text_chunk in generate_answer_stream(
                messages=messages,
                model=settings.GROQ_MODEL,
                api_key=settings.GROQ_API_KEY,
                temperature=body.temperature,
            ):
                accumulated += text_chunk
                yield f"data: {json.dumps({'type': 'chunk', 'text': text_chunk})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"
            return
        session_memory.add_turn(
            session_id=session_id,
            question=body.question,
            answer=accumulated,
            doc_id=body.document_id,
            cited_sources=[],
        )
        yield f"data: {json.dumps({'type': 'done', 'tokens_used': 0, 'session_id': session_id})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/qa/ask-global", response_model=GlobalAskResponse)
@limiter.limit("30/minute")
async def ask_question_global(request: Request, body: GlobalAskRequest) -> GlobalAskResponse:
    """Answer a question by searching across ALL uploaded documents."""
    settings = request.app.state.settings
    qdrant_client = request.app.state.qdrant_client

    output = await retrieve_global(
        question=body.question,
        top_k=body.top_k,
        mode=body.search_mode,
        rerank=body.rerank,
        qdrant_client=qdrant_client,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        embedding_model=settings.EMBEDDING_MODEL,
        min_confidence=settings.MIN_CONFIDENCE_THRESHOLD,
        confidence_weights=settings.CONFIDENCE_WEIGHTS,
    )
    results = output.chunks

    messages = build_messages(body.question, results, mode=ResponseMode.CITED)
    llm_result = await generate_answer(
        messages=messages,
        model=settings.GROQ_MODEL,
        api_key=settings.GROQ_API_KEY,
    )

    sources = [
        GlobalChunkSource(
            chunk_index=r.chunk_index,
            text_excerpt=r.text[:300],
            score=r.confidence.composite_score,
            page_number=r.page_number,
            vector_score=r.vector_score,
            bm25_score=r.bm25_score,
            filename=r.filename,
            doc_id=r.doc_id,
            confidence_score=r.confidence.composite_score,
            freshness_score=r.confidence.freshness_score,
            authority_score=r.confidence.authority_score,
            agreement_score=r.confidence.agreement_score,
            retrieval_score=r.confidence.retrieval_score,
        )
        for r in results
    ]

    evidence_quality = summarize_evidence_quality(results)

    logger.info(
        "qa.answered_global",
        question_len=len(body.question),
        sources_count=len(sources),
        mode=body.search_mode,
        rerank=body.rerank,
        evidence_quality=evidence_quality,
        chunks_filtered_out=output.filtered_out,
        request_id=request_id_var.get(""),
    )

    return GlobalAskResponse(
        answer=llm_result["answer"],
        sources=sources,
        model=llm_result["model"],
        tokens_used=llm_result["tokens_used"],
    )


@router.get("/hallucination/stats")
async def hallucination_stats() -> dict:
    """Aggregated hallucination guard statistics."""
    return await database.get_hallucination_stats()
