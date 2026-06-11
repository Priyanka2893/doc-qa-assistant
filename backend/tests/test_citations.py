"""Tests for constrained generation and citation-backed responses (Feature F8)."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.citation_parser import CitationResult, ParsedCitation, parse_citations
from app.services.confidence_scorer import ConfidenceBreakdown, ScoredChunk
from app.services.retriever import RetrieveOutput


def _make_chunk(text: str, filename: str = "doc.pdf", chunk_index: int = 0, page_number: int | None = 1) -> ScoredChunk:
    return ScoredChunk(
        text=text,
        doc_id="doc-1",
        filename=filename,
        chunk_index=chunk_index,
        page_number=page_number,
        confidence=ConfidenceBreakdown(
            retrieval_score=0.9,
            freshness_score=0.8,
            authority_score=0.85,
            agreement_score=0.7,
            composite_score=0.85,
        ),
        vector_score=0.9,
        bm25_score=None,
    )


def _make_retrieve_output(chunks: list[ScoredChunk]) -> RetrieveOutput:
    return RetrieveOutput(chunks=chunks, filtered_out=0)


# ---------------------------------------------------------------------------
# Unit tests for citation_parser
# ---------------------------------------------------------------------------

class TestCitationParser:
    def test_citation_tags_parsed(self):
        chunk = _make_chunk("Return window is 30 days.")
        answer = "The return window is 30 days. [Source 1]"
        result = parse_citations(answer, [chunk])

        assert len(result.citations) == 1
        assert result.citations[0].tag == "[Source 1]"
        assert result.citations[0].source_number == 1
        assert result.citations[0].chunk is chunk
        assert result.is_abstention is False

    def test_citation_mapping_correct_chunk(self):
        chunk_a = _make_chunk("Policy A content.", chunk_index=0)
        chunk_b = _make_chunk("Policy B content.", chunk_index=1)
        answer = "Policy A says something. [Source 1] Policy B adds detail. [Source 2]"
        result = parse_citations(answer, [chunk_a, chunk_b])

        assert len(result.citations) == 2
        assert result.citations[0].chunk is chunk_a
        assert result.citations[1].chunk is chunk_b
        assert result.citations[1].chunk_index == 1

    def test_unmapped_citation_detection(self):
        chunk = _make_chunk("Only one chunk exists.")
        answer = "Some claim. [Source 99]"
        result = parse_citations(answer, [chunk])

        assert "[Source 99]" in result.unmapped_citations
        assert len(result.citations) == 0

    def test_abstention_detected(self):
        chunk = _make_chunk("Irrelevant content.")
        answer = "Insufficient information in the provided documents."
        result = parse_citations(answer, [chunk])

        assert result.is_abstention is True

    def test_citation_coverage_computed(self):
        chunk = _make_chunk("Fact A. Fact B.")
        answer = "Claim one is true. [Source 1] Claim two is also true."
        result = parse_citations(answer, [chunk])

        assert 0.0 < result.citation_coverage <= 1.0

    def test_no_citations_in_plain_answer(self):
        chunk = _make_chunk("The sky is blue.")
        answer = "The sky is blue."
        result = parse_citations(answer, [chunk])

        assert len(result.citations) == 0
        assert len(result.unmapped_citations) == 0
        assert result.citation_coverage == 0.0

    def test_duplicate_tags_counted_once(self):
        chunk = _make_chunk("Some fact.")
        answer = "First claim [Source 1] and again [Source 1]."
        result = parse_citations(answer, [chunk])

        assert len(result.citations) == 1


# ---------------------------------------------------------------------------
# Integration tests via ASGI client
# ---------------------------------------------------------------------------

class TestAskWithCitations:
    async def test_citation_tags_in_answer(self, http_client):
        """[Source N] tags in the LLM answer surface in cited_sources."""
        client, _ = http_client

        chunk = _make_chunk("Return window is 30 days.")
        mock_output = _make_retrieve_output([chunk])

        doc = {"doc_id": "doc-1", "filename": "policy.pdf", "chunk_count": 5,
               "page_count": 1, "uploaded_at": "2026-01-01"}

        with (
            patch("app.routers.qa.database.get_document", new_callable=AsyncMock, return_value=doc),
            patch("app.routers.qa.database.insert_citation_audit", new_callable=AsyncMock),
            patch("app.routers.qa.retrieve", new_callable=AsyncMock, return_value=mock_output),
            patch("app.routers.qa.generate_answer", new_callable=AsyncMock,
                  return_value={"answer": "The return window is 30 days. [Source 1]",
                                "tokens_used": 50, "model": "llama-3.3-70b-versatile"}),
        ):
            resp = await client.post(
                "/api/v1/qa/ask",
                json={"question": "What is the return window?", "document_id": "doc-1"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert "[Source 1]" in body["answer"]
        assert len(body["cited_sources"]) == 1
        assert body["cited_sources"][0]["tag"] == "[Source 1]"
        assert body["cited_sources"][0]["is_unmapped"] is False
        assert body["is_abstention"] is False

    async def test_citation_mapping(self, http_client):
        """cited_sources contain the correct chunk metadata."""
        client, _ = http_client

        chunk = _make_chunk("Policy text here.", filename="policy.pdf", chunk_index=3, page_number=2)
        mock_output = _make_retrieve_output([chunk])

        doc = {"doc_id": "doc-1", "filename": "policy.pdf", "chunk_count": 5,
               "page_count": 2, "uploaded_at": "2026-01-01"}

        with (
            patch("app.routers.qa.database.get_document", new_callable=AsyncMock, return_value=doc),
            patch("app.routers.qa.database.insert_citation_audit", new_callable=AsyncMock),
            patch("app.routers.qa.retrieve", new_callable=AsyncMock, return_value=mock_output),
            patch("app.routers.qa.generate_answer", new_callable=AsyncMock,
                  return_value={"answer": "Policy says so. [Source 1]",
                                "tokens_used": 30, "model": "llama-3.3-70b-versatile"}),
        ):
            resp = await client.post(
                "/api/v1/qa/ask",
                json={"question": "What does the policy say?", "document_id": "doc-1"},
            )

        assert resp.status_code == 200
        body = resp.json()
        src = body["cited_sources"][0]
        assert src["filename"] == "policy.pdf"
        assert src["chunk_index"] == 3
        assert src["page_number"] == 2
        assert src["confidence_score"] == pytest.approx(0.85, abs=0.01)

    async def test_unmapped_citation_detection(self, http_client):
        """[Source N] tags beyond chunk count land in unmapped_citations."""
        client, _ = http_client

        chunk = _make_chunk("Only one chunk.")
        mock_output = _make_retrieve_output([chunk])

        doc = {"doc_id": "doc-1", "filename": "policy.pdf", "chunk_count": 1,
               "page_count": 1, "uploaded_at": "2026-01-01"}

        with (
            patch("app.routers.qa.database.get_document", new_callable=AsyncMock, return_value=doc),
            patch("app.routers.qa.database.insert_citation_audit", new_callable=AsyncMock),
            patch("app.routers.qa.retrieve", new_callable=AsyncMock, return_value=mock_output),
            patch("app.routers.qa.generate_answer", new_callable=AsyncMock,
                  return_value={"answer": "Some claim. [Source 99]",
                                "tokens_used": 20, "model": "llama-3.3-70b-versatile"}),
        ):
            resp = await client.post(
                "/api/v1/qa/ask",
                json={"question": "What does this say?", "document_id": "doc-1"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert "[Source 99]" in body["unmapped_citations"]
        unmapped_srcs = [s for s in body["cited_sources"] if s["is_unmapped"]]
        assert any(s["tag"] == "[Source 99]" for s in unmapped_srcs)

    async def test_abstention_detected_in_response(self, http_client):
        """is_abstention=True when LLM returns the abstention phrase."""
        client, _ = http_client

        chunk = _make_chunk("Unrelated content.")
        mock_output = _make_retrieve_output([chunk])

        doc = {"doc_id": "doc-1", "filename": "policy.pdf", "chunk_count": 1,
               "page_count": 1, "uploaded_at": "2026-01-01"}

        with (
            patch("app.routers.qa.database.get_document", new_callable=AsyncMock, return_value=doc),
            patch("app.routers.qa.database.insert_citation_audit", new_callable=AsyncMock),
            patch("app.routers.qa.retrieve", new_callable=AsyncMock, return_value=mock_output),
            patch("app.routers.qa.generate_answer", new_callable=AsyncMock,
                  return_value={"answer": "Insufficient information in the provided documents.",
                                "tokens_used": 10, "model": "llama-3.3-70b-versatile"}),
        ):
            resp = await client.post(
                "/api/v1/qa/ask",
                json={"question": "What is the capital of Mars?", "document_id": "doc-1"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["is_abstention"] is True

    async def test_plain_mode_accepted(self, http_client):
        """response_mode=plain is accepted and reflected in the response."""
        client, _ = http_client

        chunk = _make_chunk("The sky is blue.")
        mock_output = _make_retrieve_output([chunk])

        doc = {"doc_id": "doc-1", "filename": "facts.pdf", "chunk_count": 1,
               "page_count": 1, "uploaded_at": "2026-01-01"}

        with (
            patch("app.routers.qa.database.get_document", new_callable=AsyncMock, return_value=doc),
            patch("app.routers.qa.database.insert_citation_audit", new_callable=AsyncMock),
            patch("app.routers.qa.retrieve", new_callable=AsyncMock, return_value=mock_output),
            patch("app.routers.qa.generate_answer", new_callable=AsyncMock,
                  return_value={"answer": "The sky is blue.",
                                "tokens_used": 10, "model": "llama-3.3-70b-versatile"}),
        ):
            resp = await client.post(
                "/api/v1/qa/ask",
                json={"question": "What color is the sky?", "document_id": "doc-1",
                      "response_mode": "plain"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["response_mode"] == "plain"

    async def test_temperature_param_accepted(self, http_client):
        """temperature field is accepted and does not cause a 422."""
        client, _ = http_client

        chunk = _make_chunk("Some text.")
        mock_output = _make_retrieve_output([chunk])

        doc = {"doc_id": "doc-1", "filename": "doc.pdf", "chunk_count": 1,
               "page_count": 1, "uploaded_at": "2026-01-01"}

        with (
            patch("app.routers.qa.database.get_document", new_callable=AsyncMock, return_value=doc),
            patch("app.routers.qa.database.insert_citation_audit", new_callable=AsyncMock),
            patch("app.routers.qa.retrieve", new_callable=AsyncMock, return_value=mock_output),
            patch("app.routers.qa.generate_answer", new_callable=AsyncMock,
                  return_value={"answer": "Some answer.",
                                "tokens_used": 15, "model": "llama-3.3-70b-versatile"}),
        ):
            resp = await client.post(
                "/api/v1/qa/ask",
                json={"question": "Tell me something?", "document_id": "doc-1",
                      "temperature": 0.7},
            )

        assert resp.status_code == 200

    async def test_temperature_out_of_range_rejected(self, http_client):
        """temperature > 1.0 returns 422."""
        client, _ = http_client
        resp = await client.post(
            "/api/v1/qa/ask",
            json={"question": "Tell me something?", "document_id": "doc-1",
                  "temperature": 2.0},
        )
        assert resp.status_code == 422
