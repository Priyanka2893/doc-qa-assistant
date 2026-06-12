from unittest.mock import MagicMock, patch

import pytest

from app.services.confidence_scorer import ConfidenceBreakdown, ScoredChunk
from app.services.evaluator import compute_faithfulness, evaluate_response
from app.services.hallucination_guard import VerificationResult

pytestmark = pytest.mark.anyio

_DIM = 8


def _make_chunk(text: str, score: float = 0.8) -> ScoredChunk:
    return ScoredChunk(
        text=text,
        doc_id="doc1",
        filename="file.pdf",
        chunk_index=0,
        page_number=1,
        confidence=ConfidenceBreakdown(
            retrieval_score=score, freshness_score=score,
            authority_score=score, agreement_score=score, composite_score=score,
        ),
        vector_score=score,
        bm25_score=score,
    )


def _make_verification(risk: float = 0.0, high_risk: bool = False) -> VerificationResult:
    return VerificationResult(
        sentences=[], hallucination_risk=risk, ungrounded_sentences=[],
        is_high_risk=high_risk, grounded_count=0, ungrounded_count=0,
    )


def _unit(hot: int) -> list[float]:
    v = [0.0] * _DIM
    v[hot] = 1.0
    return v


def _mock_embedder(vecs: list[list[float]]) -> MagicMock:
    m = MagicMock()
    m.encode_texts.return_value = vecs
    return m


def _mock_settings() -> MagicMock:
    m = MagicMock()
    m.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    return m


# ── faithfulness (unchanged, sync) ───────────────────────────────────────────

def test_faithfulness_from_verification():
    vr = _make_verification(risk=0.25)
    assert compute_faithfulness(vr, is_abstention=False) == pytest.approx(0.75, abs=1e-4)


def test_abstention_faithfulness():
    vr = _make_verification(risk=0.80, high_risk=True)
    assert compute_faithfulness(vr, is_abstention=True) == 1.0


def test_zero_risk_faithfulness():
    assert compute_faithfulness(_make_verification(risk=0.0), is_abstention=False) == 1.0


# ── context relevance — token fast path ──────────────────────────────────────

async def test_context_relevance_fast_path_high():
    """Question tokens appear verbatim in chunk → token recall ≥ 0.50 → no embedding."""
    question = "What is the machine learning accuracy?"
    chunks = [_make_chunk("The machine learning model achieved high accuracy on the benchmark.")]
    vr = _make_verification()

    mock_embedder = _mock_embedder([])
    with patch("app.services.embedder.get_embedder", return_value=mock_embedder):
        m = await evaluate_response(question, chunks, "machine learning accuracy", vr)

    mock_embedder.encode_texts.assert_not_called()
    assert m.context_relevance >= 0.50


async def test_context_relevance_fast_path_no_chunks():
    vr = _make_verification()
    m = await evaluate_response("any question", [], "any answer", vr)
    assert m.context_relevance == 0.0


# ── context relevance — semantic fallback ────────────────────────────────────

async def test_context_relevance_semantic_fallback_high_similarity():
    """Query word absent from chunk text → token fast path misses → semantic cosine rescues it.

    "timeline" not in chunk → chunk recall = 0.5 < threshold → semantic stage runs.
    Answer contains both query words → answer fast-paths (recall = 1.0), so
    encode_texts is called with exactly [question, chunk] (2 texts).
    Parallel unit vectors → cosine = 1.0 → context_relevance = 1.0.
    """
    question = "refund timeline"
    chunks = [_make_chunk("Credit card refunds take 5-7 business days to process.")]
    # "refund" and "timeline" both in answer → answer recall = 1.0 → fast path
    answer = "The refund timeline is typically 5-7 business days."
    vr = _make_verification()

    # encode_texts receives [question, chunk] — same direction → cosine 1.0
    vecs = [_unit(0), _unit(0)]

    with patch("app.config.get_settings", return_value=_mock_settings()), \
         patch("app.services.embedder.get_embedder", return_value=_mock_embedder(vecs)):
        m = await evaluate_response(question, chunks, answer, vr)

    assert m.context_relevance == pytest.approx(1.0, abs=1e-4)


async def test_context_relevance_semantic_fallback_low_similarity():
    """Semantically unrelated query + chunk → cosine near 0 even after fallback.

    Answer contains both query words so it fast-paths, leaving encode_texts
    with exactly [question, chunk] (2 texts).
    """
    question = "refund timeline"
    chunks = [_make_chunk("The quarterly earnings exceeded analyst expectations significantly.")]
    answer = "The refund timeline is 5-7 business days."   # fast-paths (recall = 1.0)
    vr = _make_verification()

    # perpendicular → cosine 0.0 → chunk score stays at token recall (0.0)
    vecs = [_unit(0), _unit(1)]

    with patch("app.config.get_settings", return_value=_mock_settings()), \
         patch("app.services.embedder.get_embedder", return_value=_mock_embedder(vecs)):
        m = await evaluate_response(question, chunks, answer, vr)

    assert m.context_relevance == pytest.approx(0.0, abs=1e-4)


# ── answer relevance — semantic fallback ─────────────────────────────────────

async def test_answer_relevance_semantic_fallback_rescues_paraphrase():
    """Answer paraphrases the question without sharing exact tokens.

    chunk recall = 0.5 (has "refund", not "timeline") → also needs semantic (≤ threshold).
    Layout passed to encode_texts: [question, chunk, answer].
    All parallel unit vectors → cosine = 1.0 for both chunk and answer.
    """
    question = "refund timeline"
    chunks = [_make_chunk("Our return policy refund process takes several days.")]
    answer = "Processing typically occurs within one week."  # no "refund" or "timeline"
    vr = _make_verification()

    # encode_texts receives [question, chunk, answer] — all same direction → cosine 1.0
    vecs = [_unit(0), _unit(0), _unit(0)]

    with patch("app.config.get_settings", return_value=_mock_settings()), \
         patch("app.services.embedder.get_embedder", return_value=_mock_embedder(vecs)):
        m = await evaluate_response(question, chunks, answer, vr)

    assert m.answer_relevance == pytest.approx(1.0, abs=1e-4)
    assert m.context_relevance == pytest.approx(1.0, abs=1e-4)


async def test_answer_relevance_abstention_score():
    vr = _make_verification()
    m = await evaluate_response("anything", [], "Insufficient information", vr, is_abstention=True)
    assert m.answer_relevance == pytest.approx(0.85)
    assert m.faithfulness == 1.0
    assert m.is_abstention is True


# ── overall formula ───────────────────────────────────────────────────────────

async def test_overall_formula():
    question = "What is the return policy?"
    chunks = [_make_chunk("Our return policy allows returns within 30 days of purchase.")]
    answer = "You may return items within 30 days."
    vr = _make_verification(risk=0.0)

    m = await evaluate_response(question, chunks, answer, vr)

    expected = round(0.30 * m.context_relevance + 0.40 * m.faithfulness + 0.30 * m.answer_relevance, 4)
    assert m.overall_score == pytest.approx(expected, abs=1e-4)
    assert m.hallucination_risk == 0.0
    assert m.chunk_count_used == 1


async def test_embedder_not_called_when_all_fast_path():
    """When token recall ≥ 0.50 for every chunk and the answer, no embedding should happen."""
    question = "return policy"
    chunks = [_make_chunk("The return policy allows customers to return items within 30 days.")]
    answer = "Returns are allowed within 30 days per the return policy."
    vr = _make_verification()

    mock_embedder = _mock_embedder([])
    with patch("app.services.embedder.get_embedder", return_value=mock_embedder):
        await evaluate_response(question, chunks, answer, vr)

    mock_embedder.encode_texts.assert_not_called()


# ── DB persistence ────────────────────────────────────────────────────────────

async def test_eval_stored_in_db():
    from app import database

    await database.init_db()
    await database.insert_eval_result(
        request_id="req-test-001",
        doc_id="doc-test-001",
        question="Test question for persistence",
        context_relevance=0.55,
        faithfulness=0.90,
        answer_relevance=0.65,
        overall_score=0.726,
        chunk_count_used=3,
        is_abstention=False,
        hallucination_risk=0.10,
    )
    summary = await database.get_eval_summary(hours=24)
    assert summary["query_count"] >= 1
    assert summary["avg_faithfulness"] > 0.0


async def test_doc_eval_summary():
    from app import database

    await database.init_db()
    await database.insert_eval_result(
        request_id="req-doc-summary",
        doc_id="doc-summary-test",
        question="Summary test question",
        context_relevance=0.70,
        faithfulness=1.0,
        answer_relevance=0.80,
        overall_score=0.85,
        chunk_count_used=5,
        is_abstention=False,
        hallucination_risk=0.0,
    )
    summary = await database.get_doc_eval_summary("doc-summary-test")
    assert summary["query_count"] >= 1
    assert summary["avg_faithfulness"] >= 1.0


# ── endpoints ─────────────────────────────────────────────────────────────────

async def test_summary_endpoint(http_client):
    client, _ = http_client
    response = await client.get("/api/v1/eval/summary?hours=24")
    assert response.status_code == 200
    data = response.json()
    for key in ("query_count", "avg_context_relevance", "avg_faithfulness",
                "avg_answer_relevance", "avg_overall_score", "abstention_rate",
                "high_risk_rate", "time_window_hours"):
        assert key in data
    assert data["time_window_hours"] == 24


async def test_doc_summary_endpoint_not_found(http_client):
    client, _ = http_client
    response = await client.get("/api/v1/eval/document/nonexistent-doc-id")
    assert response.status_code == 404
