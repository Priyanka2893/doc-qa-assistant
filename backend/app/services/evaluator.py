import asyncio
import re

import numpy as np
import structlog

from app.models import EvalMetrics
from app.services.confidence_scorer import ScoredChunk
from app.services.hallucination_guard import VerificationResult

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

# Token recall floor — below this triggers semantic cosine fallback (mirrors F9 fast-path threshold)
_TOKEN_FAST_PATH: float = 0.50


def _tokenize(text: str) -> set[str]:
    words = re.findall(r'\b[a-z]+\b', text.lower())
    return {w for w in words if w not in _STOPWORDS}


def _recall(query_tokens: set[str], text_tokens: set[str]) -> float:
    """Fraction of query tokens present in text — asymmetric, text length doesn't penalise."""
    if not query_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_n = a / (np.linalg.norm(a) + 1e-9)
    b_n = b / (np.linalg.norm(b) + 1e-9)
    return float(np.clip(np.dot(a_n, b_n), 0.0, 1.0))


def compute_faithfulness(verification_result: VerificationResult, is_abstention: bool) -> float:
    """1 - hallucination_risk from F9 2-stage hybrid (token fast path + semantic cosine).

    Abstentions are always faithful.
    """
    if is_abstention:
        return 1.0
    return round(1.0 - verification_result.hallucination_risk, 4)


async def evaluate_response(
    question: str,
    chunks: list[ScoredChunk],
    answer: str,
    verification_result: VerificationResult,
    is_abstention: bool = False,
) -> EvalMetrics:
    """2-stage evaluation mirroring F9's hallucination verifier.

    Stage 1 — token fast path: if question-token recall >= _TOKEN_FAST_PATH for a
    chunk (or the answer), the score is used directly — no embedding call.

    Stage 2 — semantic cosine fallback: all texts that fall below the fast-path
    floor are batch-embedded together with the question in a single encoder call.
    Final score = max(token_recall, cosine_similarity) so that paraphrases like
    "refund timeline" vs "5-7 business days" are captured by the semantic stage
    instead of being unfairly penalised.
    """
    q_tokens = _tokenize(question)

    # ── Stage 1: token recall ─────────────────────────────────────────────────
    chunk_token_scores = [_recall(q_tokens, _tokenize(c.text)) for c in chunks]
    ans_token_score = _recall(q_tokens, _tokenize(answer)) if not is_abstention else None

    needs_sem_chunks = [i for i, s in enumerate(chunk_token_scores) if s <= _TOKEN_FAST_PATH]
    needs_sem_ans = (
        not is_abstention
        and ans_token_score is not None
        and ans_token_score <= _TOKEN_FAST_PATH
    )

    final_chunk_scores = list(chunk_token_scores)
    final_ans_score = 0.85 if is_abstention else (ans_token_score or 0.0)

    # ── Stage 2: single batched embed for everything below the floor ──────────
    if needs_sem_chunks or needs_sem_ans:
        from app.config import get_settings
        from app.services.embedder import get_embedder

        settings = get_settings()
        embedder = get_embedder(settings.EMBEDDING_MODEL)

        # Layout: [question, chunk_0, chunk_1, ..., answer?]
        texts: list[str] = [question]
        chunk_batch_idx: dict[int, int] = {}
        for i in needs_sem_chunks:
            chunk_batch_idx[i] = len(texts)
            texts.append(chunks[i].text)
        ans_batch_idx: int | None = None
        if needs_sem_ans:
            ans_batch_idx = len(texts)
            texts.append(answer)

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, embedder.encode_texts, texts)
        vecs = np.array(raw, dtype=np.float32)
        q_vec = vecs[0]

        for chunk_i, batch_i in chunk_batch_idx.items():
            sem = _cosine(q_vec, vecs[batch_i])
            final_chunk_scores[chunk_i] = max(chunk_token_scores[chunk_i], sem)
            logger.debug(
                "eval.context_semantic_fallback",
                chunk_i=chunk_i,
                token=chunk_token_scores[chunk_i],
                semantic=round(sem, 4),
                final=round(final_chunk_scores[chunk_i], 4),
            )

        if ans_batch_idx is not None:
            sem = _cosine(q_vec, vecs[ans_batch_idx])
            final_ans_score = max(ans_token_score, sem)  # type: ignore[arg-type]
            logger.debug(
                "eval.answer_semantic_fallback",
                token=ans_token_score,
                semantic=round(sem, 4),
                final=round(final_ans_score, 4),
            )

    ctx = round(sum(final_chunk_scores) / len(final_chunk_scores), 4) if chunks else 0.0
    ans_rel = round(final_ans_score, 4)
    faith = compute_faithfulness(verification_result, is_abstention)
    overall = round(0.30 * ctx + 0.40 * faith + 0.30 * ans_rel, 4)

    logger.debug(
        "evaluator.metrics",
        context_relevance=ctx,
        faithfulness=faith,
        answer_relevance=ans_rel,
        overall_score=overall,
        chunk_count=len(chunks),
        is_abstention=is_abstention,
        semantic_chunks=len(needs_sem_chunks),
        semantic_ans=needs_sem_ans,
    )

    return EvalMetrics(
        context_relevance=ctx,
        faithfulness=faith,
        answer_relevance=ans_rel,
        overall_score=overall,
        chunk_count_used=len(chunks),
        is_abstention=is_abstention,
        hallucination_risk=verification_result.hallucination_risk,
    )
