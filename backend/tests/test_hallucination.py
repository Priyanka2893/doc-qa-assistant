from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

from app.services.confidence_scorer import ConfidenceBreakdown, ScoredChunk
from app.services.hallucination_guard import (
    GateResult,
    pre_generation_gate,
    verify_answer,
)

pytestmark = pytest.mark.anyio


def _make_chunk(text: str, score: float) -> ScoredChunk:
    return ScoredChunk(
        text=text,
        doc_id="doc1",
        filename="file.pdf",
        chunk_index=0,
        page_number=1,
        confidence=ConfidenceBreakdown(
            retrieval_score=score,
            freshness_score=score,
            authority_score=score,
            agreement_score=score,
            composite_score=score,
        ),
        vector_score=score,
        bm25_score=score,
    )


# ── Layer 1: pre-generation gate ─────────────────────────────────────────────

def test_gate_blocks_low_confidence():
    chunks = [_make_chunk("some text", 0.3)]
    result = pre_generation_gate(chunks, min_confidence=0.50)
    assert result.passed is False
    assert "0.30" in result.reason
    assert result.avg_confidence == pytest.approx(0.3, abs=1e-4)
    assert result.chunk_count == 1


def test_gate_passes_high_confidence():
    chunks = [_make_chunk("some text", 0.8), _make_chunk("other text", 0.9)]
    result = pre_generation_gate(chunks, min_confidence=0.50)
    assert result.passed is True
    assert result.reason == "evidence_sufficient"
    assert result.avg_confidence == pytest.approx(0.85, abs=1e-4)


def test_gate_blocks_empty_chunks():
    result = pre_generation_gate([], min_confidence=0.50)
    assert result.passed is False
    assert result.reason == "no_chunks_retrieved"
    assert result.chunk_count == 0


# ── Layer 2: post-generation verifier ────────────────────────────────────────

def test_post_gen_grounded_sentence():
    chunk_text = "The machine learning model achieved high accuracy on the benchmark dataset."
    chunks = [_make_chunk(chunk_text, 0.9)]
    answer = "The machine learning model achieved high accuracy on the benchmark dataset."
    result = verify_answer(answer, chunks, overlap_threshold=0.35)
    assert result.grounded_count == 1
    assert result.ungrounded_count == 0
    assert result.hallucination_risk == 0.0
    assert result.is_high_risk is False


def test_post_gen_hallucinated_sentence():
    chunk_text = "The capital of France is Paris."
    chunks = [_make_chunk(chunk_text, 0.9)]
    # Sentence shares no meaningful tokens with the chunk
    answer = "Quantum entanglement enables faster-than-light communication between particles."
    result = verify_answer(answer, chunks, overlap_threshold=0.35)
    assert result.ungrounded_count >= 1
    assert result.hallucination_risk > 0.0


def test_abstention_skips_verification():
    chunks = [_make_chunk("some relevant text about topic", 0.9)]
    answer = "Insufficient information available to answer the question."
    result = verify_answer(answer, chunks, overlap_threshold=0.35)
    assert result.hallucination_risk == 0.0
    assert result.is_high_risk is False
    assert result.sentences == []


def test_high_risk_flag_when_most_ungrounded():
    chunk_text = "Water boils at 100 degrees Celsius at sea level."
    chunks = [_make_chunk(chunk_text, 0.9)]
    # Multiple completely unrelated sentences → risk >= 0.40
    answer = (
        "Quantum computing uses qubits for calculation. "
        "Neural networks require backpropagation training. "
        "Blockchain technology enables decentralized finance. "
        "Photosynthesis converts sunlight into glucose."
    )
    result = verify_answer(answer, chunks, overlap_threshold=0.35, high_risk_threshold=0.40)
    assert result.is_high_risk is True
    assert result.hallucination_risk >= 0.40


# ── Stats endpoint ─────────────────────────────────────────────────────────

async def test_stats_endpoint(http_client):
    client, _ = http_client
    response = await client.get("/api/v1/hallucination/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_queries" in data
    assert "gate_blocked_count" in data
    assert "high_risk_count" in data
    assert "avg_hallucination_risk" in data
    assert "recent_events" in data
    assert isinstance(data["recent_events"], list)
