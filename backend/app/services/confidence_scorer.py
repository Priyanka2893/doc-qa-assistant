import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import structlog

from app.models import RetrievalResult
from app.services.embedder import async_encode_texts

logger = structlog.get_logger(__name__)

_FRESHNESS_HALF_LIFE_DAYS: float = 180.0

AUTHORITY_SCORES: dict[str, float] = {
    "verified": 1.0,
    "internal": 0.85,
    "external": 0.65,
    "unknown": 0.50,
}


@dataclass
class ConfidenceBreakdown:
    retrieval_score: float
    freshness_score: float
    authority_score: float
    agreement_score: float
    composite_score: float
    w_retrieval: float = 0.50
    w_freshness: float = 0.20
    w_authority: float = 0.20
    w_agreement: float = 0.10


@dataclass
class ScoredChunk:
    text: str
    doc_id: str
    filename: str
    chunk_index: int
    page_number: int | None
    confidence: ConfidenceBreakdown
    vector_score: float | None
    bm25_score: float | None


def _freshness_score(uploaded_at: datetime) -> float:
    now = datetime.now(timezone.utc)
    if uploaded_at.tzinfo is None:
        uploaded_at = uploaded_at.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (now - uploaded_at).total_seconds() / 86400.0)
    return math.exp(-math.log(2) * age_days / _FRESHNESS_HALF_LIFE_DAYS)


def _authority_score(trust_level: str) -> float:
    return AUTHORITY_SCORES.get(trust_level, AUTHORITY_SCORES["unknown"])


def _normalize_retrieval_scores(chunks: list[RetrievalResult]) -> list[float]:
    scores = [c.score for c in chunks]
    min_s, max_s = min(scores), max(scores)
    if max_s > min_s:
        return [(s - min_s) / (max_s - min_s) for s in scores]
    return [1.0] * len(chunks)


async def _compute_agreement_scores(
    chunks: list[RetrievalResult],
    embedding_model: str,
    threshold: float = 0.7,
) -> list[float]:
    if len(chunks) <= 1:
        return [0.0] * len(chunks)

    texts = [c.text for c in chunks]
    embeddings = await async_encode_texts(embedding_model, texts)
    emb_array = np.array(embeddings, dtype=np.float32)

    norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = emb_array / norms

    sim_matrix = normalized @ normalized.T
    n = len(chunks)

    return [
        sum(1 for j in range(n) if i != j and sim_matrix[i, j] > threshold) / (n - 1)
        for i in range(n)
    ]


async def score_chunks(
    chunks: list[RetrievalResult],
    uploaded_at_map: dict[str, datetime],
    authority_map: dict[str, str],
    min_confidence: float = 0.40,
    embedding_model: str = "",
    weights: dict[str, float] | None = None,
) -> list[ScoredChunk]:
    if not chunks:
        return []

    w = weights or {"retrieval": 0.50, "freshness": 0.20, "authority": 0.20, "agreement": 0.10}
    w_r = w.get("retrieval", 0.50)
    w_f = w.get("freshness", 0.20)
    w_a = w.get("authority", 0.20)
    w_ag = w.get("agreement", 0.10)

    retrieval_scores = _normalize_retrieval_scores(chunks)

    if embedding_model and len(chunks) > 1:
        agreement_scores = await _compute_agreement_scores(chunks, embedding_model)
    else:
        agreement_scores = [0.0] * len(chunks)

    scored: list[ScoredChunk] = []
    for chunk, r_score, ag_score in zip(chunks, retrieval_scores, agreement_scores):
        uploaded_at = uploaded_at_map.get(chunk.doc_id)
        f_score = _freshness_score(uploaded_at) if uploaded_at is not None else 0.5

        a_score = _authority_score(authority_map.get(chunk.doc_id, "unknown"))

        composite = w_r * r_score + w_f * f_score + w_a * a_score + w_ag * ag_score

        scored.append(ScoredChunk(
            text=chunk.text,
            doc_id=chunk.doc_id,
            filename=chunk.filename,
            chunk_index=chunk.chunk_index,
            page_number=chunk.page_number,
            confidence=ConfidenceBreakdown(
                retrieval_score=round(r_score, 4),
                freshness_score=round(f_score, 4),
                authority_score=round(a_score, 4),
                agreement_score=round(ag_score, 4),
                composite_score=round(composite, 4),
                w_retrieval=w_r,
                w_freshness=w_f,
                w_authority=w_a,
                w_agreement=w_ag,
            ),
            vector_score=chunk.vector_score,
            bm25_score=chunk.bm25_score,
        ))

    passed = [sc for sc in scored if sc.confidence.composite_score >= min_confidence]
    passed.sort(key=lambda sc: sc.confidence.composite_score, reverse=True)

    logger.info(
        "confidence_scorer.scored",
        total=len(scored),
        passed=len(passed),
        filtered_out=len(scored) - len(passed),
        min_confidence=min_confidence,
    )
    return passed


def summarize_evidence_quality(scored_chunks: list[ScoredChunk]) -> str:
    if not scored_chunks:
        return "none"
    avg = sum(c.confidence.composite_score for c in scored_chunks) / len(scored_chunks)
    if avg >= 0.80:
        return "high"
    if avg >= 0.60:
        return "medium"
    if avg >= 0.40:
        return "low"
    return "insufficient"
