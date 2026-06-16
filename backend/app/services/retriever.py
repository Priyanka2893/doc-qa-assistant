import asyncio
import time
from datetime import datetime
from functools import lru_cache
from typing import NamedTuple

import structlog
from sentence_transformers import CrossEncoder

from app import database
from app.models import RetrievalResult, SearchMode
from app.services.bm25_store import get_bm25_store
from app.services.confidence_scorer import ScoredChunk, score_chunks
from app.services.embedder import async_encode_query
from app.services.vector_store import search_chunks, search_chunks_global
from app.telemetry import track_stage, traced

logger = structlog.get_logger(__name__)

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RetrieveOutput(NamedTuple):
    chunks: list[ScoredChunk]
    filtered_out: int


@lru_cache(maxsize=1)
def get_cross_encoder() -> CrossEncoder:
    start = time.perf_counter()
    model = CrossEncoder(CROSS_ENCODER_MODEL)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info("cross_encoder.loaded", model=CROSS_ENCODER_MODEL, load_time_ms=elapsed_ms)
    return model


def reciprocal_rank_fusion(
    vector_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for rank, (chunk_id, _) in enumerate(vector_results):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    for rank, (chunk_id, _) in enumerate(bm25_results):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


async def _rerank(question: str, candidates: list[RetrievalResult]) -> list[RetrievalResult]:
    if not candidates:
        return candidates
    pairs = [(question, c.text) for c in candidates]
    loop = asyncio.get_running_loop()

    def _predict() -> list[float]:
        return get_cross_encoder().predict(pairs)

    scores = await loop.run_in_executor(None, _predict)
    for candidate, score in zip(candidates, scores):
        candidate.score = float(score)
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def _apply_bm25_to_candidates(
    bm25_results: list[tuple[str, float]],
    candidates: dict[str, RetrievalResult],
    fallback_doc_id: str,
    bm25_store,
) -> None:
    for chunk_id, score in bm25_results:
        if chunk_id in candidates:
            candidates[chunk_id].bm25_score = score
        else:
            meta = bm25_store.get_metadata(chunk_id)
            if meta:
                candidates[chunk_id] = RetrievalResult(
                    chunk_id=chunk_id,
                    text=meta["text"],
                    score=score,
                    vector_score=None,
                    bm25_score=score,
                    doc_id=meta.get("doc_id", fallback_doc_id),
                    filename=meta.get("filename", ""),
                    chunk_index=meta.get("chunk_index", 0),
                    page_number=meta.get("page_number"),
                )


def _select_final(
    candidates: dict[str, RetrievalResult],
    vector_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    mode: SearchMode,
    top_k: int,
) -> tuple[list[RetrievalResult], int]:
    """Returns (candidates_after_merge, after_rrf_count)."""
    if mode == SearchMode.HYBRID:
        merged = reciprocal_rank_fusion(vector_results, bm25_results)
        final: list[RetrievalResult] = []
        for chunk_id, rrf_score in merged[:top_k * 2]:
            if chunk_id in candidates:
                candidates[chunk_id].score = rrf_score
                final.append(candidates[chunk_id])
        return final, len(final)
    elif mode == SearchMode.VECTOR:
        result = list(candidates.values())
        return result, len(result)
    else:  # KEYWORD
        result = [candidates[cid] for cid, _ in bm25_results if cid in candidates]
        return result, len(result)


async def _build_doc_maps(
    candidates: list[RetrievalResult],
) -> tuple[dict[str, datetime], dict[str, str]]:
    """Return (uploaded_at_map, authority_map) for all unique doc_ids in candidates."""
    uploaded_at_map: dict[str, datetime] = {}
    authority_map: dict[str, str] = {}
    for doc_id in {c.doc_id for c in candidates}:
        doc = await database.get_document(doc_id)
        if doc and doc.get("uploaded_at"):
            try:
                uploaded_at_map[doc_id] = datetime.fromisoformat(doc["uploaded_at"])
            except ValueError:
                pass
        authority_map[doc_id] = await database.get_document_trust(doc_id)
    return uploaded_at_map, authority_map


@traced("retrieve")
async def retrieve(
    question: str,
    doc_id: str,
    top_k: int = 5,
    mode: SearchMode = SearchMode.HYBRID,
    rerank: bool = True,
    qdrant_client=None,
    collection_name: str = "",
    embedding_model: str = "",
    min_confidence: float = 0.40,
    confidence_weights: dict[str, float] | None = None,
) -> RetrieveOutput:
    bm25_store = get_bm25_store()
    candidates: dict[str, RetrievalResult] = {}
    vector_results: list[tuple[str, float]] = []
    bm25_results: list[tuple[str, float]] = []
    candidate_k = top_k * 2 if mode == SearchMode.HYBRID else top_k

    if mode in (SearchMode.VECTOR, SearchMode.HYBRID):
        query_vector = await async_encode_query(embedding_model, question)
        with track_stage("vector_search"):
            scored_points = await search_chunks(
                client=qdrant_client,
                collection_name=collection_name,
                query_vector=query_vector,
                doc_id=doc_id,
                top_k=candidate_k,
            )
        vector_results = [(str(p.id), p.score) for p in scored_points]
        for p in scored_points:
            candidates[str(p.id)] = RetrievalResult(
                chunk_id=str(p.id),
                text=p.payload["text"],
                score=p.score,
                vector_score=p.score,
                bm25_score=None,
                doc_id=p.payload.get("doc_id", doc_id),
                filename=p.payload.get("filename", ""),
                chunk_index=p.payload.get("chunk_index", 0),
                page_number=p.payload.get("page_number"),
            )

    if mode in (SearchMode.KEYWORD, SearchMode.HYBRID):
        with track_stage("bm25_search"):
            bm25_results = bm25_store.search(doc_id=doc_id, query=question, top_k=candidate_k)
        _apply_bm25_to_candidates(bm25_results, candidates, doc_id, bm25_store)

    with track_stage("rrf_fusion"):
        final_candidates, after_rrf = _select_final(candidates, vector_results, bm25_results, mode, top_k)

    if rerank and final_candidates:
        with track_stage("rerank"):
            final_candidates = await _rerank(question, final_candidates)

    final_candidates = final_candidates[:top_k]
    pre_filter_count = len(final_candidates)

    uploaded_at_map, authority_map = await _build_doc_maps(final_candidates)
    with track_stage("confidence_score"):
        scored = await score_chunks(
            chunks=final_candidates,
            uploaded_at_map=uploaded_at_map,
            authority_map=authority_map,
            min_confidence=min_confidence,
            embedding_model=embedding_model,
            weights=confidence_weights,
        )

    logger.info(
        "retriever.done",
        mode=mode,
        doc_id=doc_id,
        vector_candidates=len(vector_results),
        bm25_candidates=len(bm25_results),
        after_rrf=after_rrf,
        after_rerank=pre_filter_count,
        after_confidence=len(scored),
        rerank=rerank,
    )
    return RetrieveOutput(chunks=scored, filtered_out=pre_filter_count - len(scored))


@traced("retrieve_global")
async def retrieve_global(
    question: str,
    top_k: int = 10,
    mode: SearchMode = SearchMode.HYBRID,
    rerank: bool = True,
    qdrant_client=None,
    collection_name: str = "",
    embedding_model: str = "",
    min_confidence: float = 0.40,
    confidence_weights: dict[str, float] | None = None,
) -> RetrieveOutput:
    bm25_store = get_bm25_store()
    candidates: dict[str, RetrievalResult] = {}
    vector_results: list[tuple[str, float]] = []
    bm25_results: list[tuple[str, float]] = []
    candidate_k = top_k * 2 if mode == SearchMode.HYBRID else top_k

    if mode in (SearchMode.VECTOR, SearchMode.HYBRID):
        query_vector = await async_encode_query(embedding_model, question)
        with track_stage("vector_search"):
            scored_points = await search_chunks_global(
                client=qdrant_client,
                collection_name=collection_name,
                query_vector=query_vector,
                top_k=candidate_k,
            )
        vector_results = [(str(p.id), p.score) for p in scored_points]
        for p in scored_points:
            candidates[str(p.id)] = RetrievalResult(
                chunk_id=str(p.id),
                text=p.payload["text"],
                score=p.score,
                vector_score=p.score,
                bm25_score=None,
                doc_id=p.payload.get("doc_id", ""),
                filename=p.payload.get("filename", ""),
                chunk_index=p.payload.get("chunk_index", 0),
                page_number=p.payload.get("page_number"),
            )

    if mode in (SearchMode.KEYWORD, SearchMode.HYBRID):
        with track_stage("bm25_search"):
            bm25_results = bm25_store.search_all(query=question, top_k=candidate_k)
        _apply_bm25_to_candidates(bm25_results, candidates, "", bm25_store)

    with track_stage("rrf_fusion"):
        final_candidates, after_rrf = _select_final(candidates, vector_results, bm25_results, mode, top_k)

    if rerank and final_candidates:
        with track_stage("rerank"):
            final_candidates = await _rerank(question, final_candidates)

    final_candidates = final_candidates[:top_k]
    pre_filter_count = len(final_candidates)

    uploaded_at_map, authority_map = await _build_doc_maps(final_candidates)
    with track_stage("confidence_score"):
        scored = await score_chunks(
            chunks=final_candidates,
            uploaded_at_map=uploaded_at_map,
            authority_map=authority_map,
            min_confidence=min_confidence,
            embedding_model=embedding_model,
            weights=confidence_weights,
        )

    logger.info(
        "retriever.done_global",
        mode=mode,
        vector_candidates=len(vector_results),
        bm25_candidates=len(bm25_results),
        after_rrf=after_rrf,
        after_rerank=pre_filter_count,
        after_confidence=len(scored),
        rerank=rerank,
    )
    return RetrieveOutput(chunks=scored, filtered_out=pre_filter_count - len(scored))
