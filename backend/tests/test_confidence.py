from datetime import datetime, timedelta, timezone

import pytest

from app.models import RetrievalResult
from app.services.confidence_scorer import (
    _freshness_score,
    score_chunks,
    summarize_evidence_quality,
)


def _make_result(
    doc_id: str = "doc1",
    score: float = 1.0,
    text: str = "sample text about machine learning",
    chunk_index: int = 0,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"chunk-{chunk_index}",
        text=text,
        score=score,
        vector_score=score,
        bm25_score=None,
        doc_id=doc_id,
        filename="test.pdf",
        chunk_index=chunk_index,
        page_number=1,
    )


@pytest.mark.anyio
async def test_freshness_decay():
    """Documents uploaded more than a year ago should have freshness_score < 0.5."""
    old_date = datetime.now(timezone.utc) - timedelta(days=730)  # ~2 years ago
    score = _freshness_score(old_date)
    assert score < 0.5, f"Expected freshness < 0.5 for 2-year-old doc, got {score:.4f}"


@pytest.mark.anyio
async def test_authority_scoring():
    """A verified document should receive authority_score = 1.0."""
    chunks = [_make_result(doc_id="doc1", score=5.0)]
    uploaded_at_map = {"doc1": datetime.now(timezone.utc)}
    authority_map = {"doc1": "verified"}

    result = await score_chunks(
        chunks=chunks,
        uploaded_at_map=uploaded_at_map,
        authority_map=authority_map,
        min_confidence=0.0,
    )

    assert len(result) == 1
    assert result[0].confidence.authority_score == 1.0


@pytest.mark.anyio
async def test_low_confidence_filter():
    """Chunks with low retrieval scores should be filtered when composite < min_confidence."""
    # Two chunks: high-scorer normalises to 1.0, low-scorer normalises to 0.0
    # For low chunk: 0.5*0 + 0.2*~1 + 0.2*0.5 + 0.1*0 = 0.30 < 0.40 → filtered
    chunks = [
        _make_result(doc_id="doc1", score=10.0, chunk_index=0),
        _make_result(doc_id="doc1", score=0.0, chunk_index=1),
    ]
    uploaded_at_map = {"doc1": datetime.now(timezone.utc)}
    authority_map = {"doc1": "unknown"}

    result = await score_chunks(
        chunks=chunks,
        uploaded_at_map=uploaded_at_map,
        authority_map=authority_map,
        min_confidence=0.40,
        embedding_model="",  # skip agreement to keep scores deterministic
    )

    chunks_filtered_out = len(chunks) - len(result)
    assert chunks_filtered_out > 0, "Expected at least one chunk to be filtered out"


@pytest.mark.anyio
async def test_evidence_quality_high():
    """A single verified, fresh, top-scoring chunk should yield evidence_quality='high'."""
    # Single chunk: retrieval normalises to 1.0, freshness≈1.0, authority=1.0, agreement=0.0
    # composite = 0.5*1 + 0.2*1 + 0.2*1 + 0.1*0 = 0.9 → "high"
    chunks = [_make_result(doc_id="doc1", score=8.0)]
    uploaded_at_map = {"doc1": datetime.now(timezone.utc)}
    authority_map = {"doc1": "verified"}

    result = await score_chunks(
        chunks=chunks,
        uploaded_at_map=uploaded_at_map,
        authority_map=authority_map,
        min_confidence=0.0,
        embedding_model="",
    )

    assert len(result) == 1
    assert result[0].confidence.composite_score >= 0.80
    assert summarize_evidence_quality(result) == "high"
