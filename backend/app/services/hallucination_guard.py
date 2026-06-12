import re
from dataclasses import dataclass

import structlog

from app.services.confidence_scorer import ScoredChunk

logger = structlog.get_logger(__name__)

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "ought",
    "i", "you", "he", "she", "it", "we", "they",
    "and", "but", "or", "so", "yet", "for", "nor", "not",
    "in", "on", "at", "to", "of", "with", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
}


@dataclass
class GateResult:
    passed: bool
    reason: str
    avg_confidence: float
    chunk_count: int


@dataclass
class SentenceVerification:
    sentence: str
    is_grounded: bool
    max_overlap_score: float
    best_chunk_index: int | None


@dataclass
class VerificationResult:
    sentences: list[SentenceVerification]
    hallucination_risk: float
    ungrounded_sentences: list[str]
    is_high_risk: bool
    grounded_count: int
    ungrounded_count: int


def pre_generation_gate(
    scored_chunks: list[ScoredChunk],
    min_confidence: float = 0.50,
    min_raw_vector_score: float = 0.30,
) -> GateResult:
    """Check if evidence is strong enough to send to LLM.

    Two checks:
    1. Raw vector score floor — catches irrelevant queries whose scores get
       inflated by within-batch normalization.
    2. Composite confidence average — catches weak evidence after scoring.
    """
    if not scored_chunks:
        return GateResult(passed=False, reason="no_chunks_retrieved", avg_confidence=0.0, chunk_count=0)

    raw_scores = [c.vector_score for c in scored_chunks if c.vector_score is not None]
    if raw_scores:
        avg_raw = sum(raw_scores) / len(raw_scores)
        if avg_raw < min_raw_vector_score:
            return GateResult(
                passed=False,
                reason=f"avg_raw_vector_score_{avg_raw:.2f}_below_threshold_{min_raw_vector_score}",
                avg_confidence=round(avg_raw, 4),
                chunk_count=len(scored_chunks),
            )

    avg = sum(c.confidence.composite_score for c in scored_chunks) / len(scored_chunks)
    if avg < min_confidence:
        return GateResult(
            passed=False,
            reason=f"avg_confidence_{avg:.2f}_below_threshold_{min_confidence}",
            avg_confidence=round(avg, 4),
            chunk_count=len(scored_chunks),
        )
    return GateResult(
        passed=True,
        reason="evidence_sufficient",
        avg_confidence=round(avg, 4),
        chunk_count=len(scored_chunks),
    )


def _tokenize(text: str) -> set[str]:
    words = re.findall(r'\b[a-z]+\b', text.lower())
    return {w for w in words if w not in _STOPWORDS}


def _containment(sentence_tokens: set[str], chunk_tokens: set[str]) -> float:
    """Fraction of sentence tokens present in chunk — asymmetric, chunk size doesn't penalize."""
    if not sentence_tokens:
        return 0.0
    return len(sentence_tokens & chunk_tokens) / len(sentence_tokens)


def verify_answer(
    answer: str,
    chunks: list[ScoredChunk],
    overlap_threshold: float = 0.35,
    high_risk_threshold: float = 0.40,
) -> VerificationResult:
    """Verify each sentence in the answer is grounded in retrieved chunks via token overlap."""
    if answer.startswith("Insufficient information"):
        return VerificationResult(
            sentences=[],
            hallucination_risk=0.0,
            ungrounded_sentences=[],
            is_high_risk=False,
            grounded_count=0,
            ungrounded_count=0,
        )

    chunk_token_sets = [_tokenize(c.text) for c in chunks]
    raw_sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    sentences = [s for s in raw_sentences if s.strip()]

    verified: list[SentenceVerification] = []
    for sentence in sentences:
        clean = re.sub(r'\[Source \d+\]', '', sentence)
        s_tokens = _tokenize(clean)

        best_overlap = 0.0
        best_idx: int | None = None
        for idx, ct in enumerate(chunk_token_sets):
            score = _containment(s_tokens, ct)
            if score > best_overlap:
                best_overlap = score
                best_idx = idx

        verified.append(SentenceVerification(
            sentence=sentence,
            is_grounded=best_overlap >= overlap_threshold,
            max_overlap_score=round(best_overlap, 4),
            best_chunk_index=best_idx,
        ))

    total = len(verified)
    ungrounded = [v for v in verified if not v.is_grounded]
    risk = len(ungrounded) / total if total > 0 else 0.0

    return VerificationResult(
        sentences=verified,
        hallucination_risk=round(risk, 4),
        ungrounded_sentences=[v.sentence for v in ungrounded],
        is_high_risk=risk >= high_risk_threshold,
        grounded_count=total - len(ungrounded),
        ungrounded_count=len(ungrounded),
    )


async def log_hallucination_event(
    request_id: str,
    doc_id: str,
    question: str,
    hallucination_risk: float,
    ungrounded_sentences: list[str],
    action_taken: str,
    gate_result: GateResult,
) -> None:
    from app import database
    gate_blocked = 0 if gate_result.passed else 1
    await database.insert_hallucination_event(
        request_id=request_id,
        doc_id=doc_id,
        question=question,
        gate_blocked=gate_blocked,
        gate_avg_confidence=gate_result.avg_confidence,
        post_gen_risk=hallucination_risk,
        ungrounded_count=len(ungrounded_sentences),
        action_taken=action_taken,
    )
    logger.info(
        "hallucination_event",
        request_id=request_id,
        doc_id=doc_id,
        gate_blocked=bool(gate_blocked),
        gate_avg_confidence=gate_result.avg_confidence,
        post_gen_risk=hallucination_risk,
        ungrounded_count=len(ungrounded_sentences),
        action_taken=action_taken,
    )
