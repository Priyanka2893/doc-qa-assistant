import asyncio
import re
from dataclasses import dataclass

import numpy as np
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
    grounding_method: str  # "token" | "semantic" | "ungrounded"


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


def _max_cosine(sentence_vec: np.ndarray, chunk_matrix: np.ndarray) -> tuple[float, int]:
    """Return (max_cosine_similarity, best_chunk_index) across all chunks."""
    s_norm = sentence_vec / (np.linalg.norm(sentence_vec) + 1e-9)
    c_norms = chunk_matrix / (np.linalg.norm(chunk_matrix, axis=1, keepdims=True) + 1e-9)
    sims = c_norms @ s_norm  # (n_chunks,)
    idx = int(np.argmax(sims))
    return float(sims[idx]), idx


async def verify_answer(
    answer: str,
    chunks: list[ScoredChunk],
    token_fast_path_threshold: float = 0.60,
    semantic_threshold: float = 0.50,
    high_risk_threshold: float = 0.40,
) -> VerificationResult:
    """Verify each sentence via 2-stage grounding check.

    Stage 1 — token containment fast path: if a sentence shares ≥60% of its
    non-stopword tokens with a chunk it is obviously grounded; skip embedding.

    Stage 2 — semantic cosine fallback: sentences that fail stage 1 are embedded
    and compared against pre-embedded chunks via cosine similarity. This catches
    paraphrases that share zero surface tokens but are semantically equivalent,
    and avoids false positives from generic domain tokens.
    """
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
    cleaned = [re.sub(r'\[Source \d+\]', '', s) for s in sentences]

    # Stage 1: token containment for all sentences
    token_results: list[tuple[float, int | None]] = []
    for clean in cleaned:
        s_tokens = _tokenize(clean)
        best_score, best_idx = 0.0, None
        for i, ct in enumerate(chunk_token_sets):
            score = _containment(s_tokens, ct)
            if score > best_score:
                best_score, best_idx = score, i
        token_results.append((best_score, best_idx))

    # Identify sentences that need semantic fallback
    needs_semantic = [i for i, (score, _) in enumerate(token_results) if score < token_fast_path_threshold]

    # Stage 2: batch-embed sentences + chunks, then cosine similarity
    chunk_matrix: np.ndarray | None = None
    sentence_vecs: dict[int, np.ndarray] = {}

    if needs_semantic:
        from app.config import get_settings
        from app.services.embedder import get_embedder

        settings = get_settings()
        embedder = get_embedder(settings.EMBEDDING_MODEL)

        texts = [cleaned[i] for i in needs_semantic] + [c.text for c in chunks]
        loop = asyncio.get_event_loop()
        vecs = await loop.run_in_executor(None, embedder.encode_texts, texts)
        arr = np.array(vecs, dtype=np.float32)

        n_sent = len(needs_semantic)
        sent_matrix = arr[:n_sent]
        chunk_matrix = arr[n_sent:]

        for j, i in enumerate(needs_semantic):
            sentence_vecs[i] = sent_matrix[j]

    # Build final verdicts
    verified: list[SentenceVerification] = []
    for i, sentence in enumerate(sentences):
        token_score, token_idx = token_results[i]

        if token_score >= token_fast_path_threshold:
            verified.append(SentenceVerification(
                sentence=sentence,
                is_grounded=True,
                max_overlap_score=round(token_score, 4),
                best_chunk_index=token_idx,
                grounding_method="token",
            ))
        else:
            assert chunk_matrix is not None
            sem_score, sem_idx = _max_cosine(sentence_vecs[i], chunk_matrix)
            verified.append(SentenceVerification(
                sentence=sentence,
                is_grounded=sem_score >= semantic_threshold,
                max_overlap_score=round(sem_score, 4),
                best_chunk_index=sem_idx,
                grounding_method="semantic" if sem_score >= semantic_threshold else "ungrounded",
            ))

    total = len(verified)
    ungrounded = [v for v in verified if not v.is_grounded]
    risk = len(ungrounded) / total if total > 0 else 0.0

    logger.debug(
        "hallucination_guard.verify",
        total_sentences=total,
        token_grounded=sum(1 for v in verified if v.grounding_method == "token"),
        semantic_grounded=sum(1 for v in verified if v.grounding_method == "semantic"),
        ungrounded=len(ungrounded),
    )

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
