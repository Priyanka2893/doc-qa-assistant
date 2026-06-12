from unittest.mock import MagicMock, patch

import pytest

from app.services.confidence_scorer import ConfidenceBreakdown, ScoredChunk
from app.services.hallucination_guard import (
    pre_generation_gate,
    verify_answer,
)

pytestmark = pytest.mark.anyio

_DIM = 8  # small dimension for test vectors


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


def _unit(hot: int) -> list[float]:
    """Unit vector with 1.0 at `hot`, 0.0 elsewhere (dimension=_DIM)."""
    v = [0.0] * _DIM
    v[hot] = 1.0
    return v


def _mock_embedder(vecs: list[list[float]]) -> MagicMock:
    """EmbedderService mock whose encode_texts returns `vecs` in order."""
    m = MagicMock()
    m.encode_texts.return_value = vecs
    return m


def _mock_settings() -> MagicMock:
    m = MagicMock()
    m.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    return m


# ── Layer 1: pre-generation gate (sync, unchanged) ───────────────────────────

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


# ── Layer 2: post-generation verifier (async 2-stage) ────────────────────────

async def test_verbatim_sentence_grounded_via_token_fast_path():
    """Verbatim sentence shares all tokens with chunk → stage 1 passes, no embedding called."""
    chunk_text = "The machine learning model achieved high accuracy on the benchmark dataset."
    chunks = [_make_chunk(chunk_text, 0.9)]

    result = await verify_answer(chunk_text, chunks, token_fast_path_threshold=0.60, semantic_threshold=0.50)

    assert result.grounded_count == 1
    assert result.ungrounded_count == 0
    assert result.hallucination_risk == 0.0
    assert result.is_high_risk is False
    assert result.sentences[0].grounding_method == "token"


async def test_paraphrase_grounded_via_semantic_fallback():
    """Sentence paraphrases the chunk — zero token overlap, but high cosine similarity.

    The token fast path fails (0 shared non-stopword tokens), so the semantic
    stage runs. We inject parallel unit vectors to simulate high cosine similarity.
    """
    chunk_text = "The firm experienced rapid growth in Q3."
    sentence = "The company expanded quickly during the third quarter."
    chunks = [_make_chunk(chunk_text, 0.9)]

    # encode_texts is called with [sentence, chunk_text] — parallel → cosine = 1.0
    vecs = [_unit(0), _unit(0)]

    with patch("app.config.get_settings", return_value=_mock_settings()), \
         patch("app.services.embedder.get_embedder", return_value=_mock_embedder(vecs)):
        result = await verify_answer(sentence, chunks, token_fast_path_threshold=0.60, semantic_threshold=0.50)

    assert result.grounded_count == 1
    assert result.sentences[0].grounding_method == "semantic"
    assert result.sentences[0].is_grounded is True


async def test_generic_tokens_fail_semantic_stage():
    """Sentence shares domain words ('data', 'model') with chunk but is semantically unrelated.

    Token overlap lands below the fast-path floor, so the semantic stage runs.
    We inject perpendicular unit vectors to simulate near-zero cosine similarity.
    """
    chunk_text = "The neural network model processes data through multiple layers."
    # "data" and "model" overlap → containment = 2/4 = 0.50 < 0.60 → semantic fallback
    sentence = "The data model was approved by management."
    chunks = [_make_chunk(chunk_text, 0.9)]

    # perpendicular vectors → cosine = 0.0 < 0.50 → ungrounded
    vecs = [_unit(0), _unit(1)]

    with patch("app.config.get_settings", return_value=_mock_settings()), \
         patch("app.services.embedder.get_embedder", return_value=_mock_embedder(vecs)):
        result = await verify_answer(sentence, chunks, token_fast_path_threshold=0.60, semantic_threshold=0.50)

    assert result.ungrounded_count == 1
    assert result.sentences[0].grounding_method == "ungrounded"
    assert result.sentences[0].is_grounded is False


async def test_fully_hallucinated_sentence_is_ungrounded():
    """Completely off-topic sentence → fails token path, fails semantic → ungrounded."""
    chunk_text = "The capital of France is Paris."
    sentence = "Quantum entanglement enables faster-than-light communication between particles."
    chunks = [_make_chunk(chunk_text, 0.9)]

    vecs = [_unit(0), _unit(1)]  # perpendicular

    with patch("app.config.get_settings", return_value=_mock_settings()), \
         patch("app.services.embedder.get_embedder", return_value=_mock_embedder(vecs)):
        result = await verify_answer(sentence, chunks)

    assert result.ungrounded_count >= 1
    assert result.hallucination_risk > 0.0


async def test_abstention_skips_verification():
    chunks = [_make_chunk("some relevant text about topic", 0.9)]
    answer = "Insufficient information available to answer the question."

    result = await verify_answer(answer, chunks)

    assert result.hallucination_risk == 0.0
    assert result.is_high_risk is False
    assert result.sentences == []


async def test_high_risk_flag_when_most_sentences_ungrounded():
    """Answer with multiple off-topic sentences → risk >= 0.40 → is_high_risk=True."""
    chunk_text = "Water boils at 100 degrees Celsius at sea level."
    chunks = [_make_chunk(chunk_text, 0.9)]
    answer = (
        "Quantum computing uses qubits for calculation. "
        "Neural networks require backpropagation training. "
        "Blockchain technology enables decentralized finance. "
        "Photosynthesis converts sunlight into glucose."
    )
    # 4 sentences + 1 chunk = 5 vectors; all perpendicular → all cosine = 0.0
    vecs = [_unit(0), _unit(1), _unit(2), _unit(3), _unit(4)]

    with patch("app.config.get_settings", return_value=_mock_settings()), \
         patch("app.services.embedder.get_embedder", return_value=_mock_embedder(vecs)):
        result = await verify_answer(answer, chunks, high_risk_threshold=0.40)

    assert result.is_high_risk is True
    assert result.hallucination_risk >= 0.40


async def test_embedder_called_only_for_semantic_sentences():
    """Embedder must NOT be called when every sentence passes the token fast path."""
    chunk_text = "The transformer architecture introduced multi-head self-attention mechanisms."
    # Identical text → containment = 1.0 ≥ 0.60 for every sentence
    answer = chunk_text
    chunks = [_make_chunk(chunk_text, 0.9)]

    mock_embedder = _mock_embedder([])
    with patch("app.services.embedder.get_embedder", return_value=mock_embedder):
        result = await verify_answer(answer, chunks, token_fast_path_threshold=0.60)

    mock_embedder.encode_texts.assert_not_called()
    assert result.grounded_count == 1


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
