"""Integration tests for all API routers using httpx against the live ASGI app."""
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Health router
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    async def test_returns_200_when_healthy(self, http_client):
        client, mock_qdrant = http_client
        count_result = MagicMock()
        count_result.count = 42
        mock_qdrant.count = AsyncMock(return_value=count_result)

        with patch("app.routers.health.get_embedder", return_value=MagicMock()):
            resp = await client.get("/api/v1/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("ok", "degraded")
        assert "qdrant" in body
        assert "version" in body

    async def test_degraded_when_qdrant_unreachable(self, http_client):
        client, mock_qdrant = http_client
        mock_qdrant.count = AsyncMock(side_effect=Exception("connection refused"))

        with patch("app.routers.health.get_embedder", return_value=MagicMock()):
            resp = await client.get("/api/v1/health")

        assert resp.status_code == 200
        assert resp.json()["status"] == "degraded"
        assert resp.json()["qdrant"] == "unreachable"


# ---------------------------------------------------------------------------
# Documents router
# ---------------------------------------------------------------------------

class TestListDocuments:
    async def test_returns_empty_list(self, http_client):
        client, _ = http_client
        with patch("app.routers.documents.database.list_documents", new_callable=AsyncMock, return_value=[]):
            resp = await client.get("/api/v1/documents")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_returns_document_list(self, http_client):
        client, _ = http_client
        docs = [
            {
                "doc_id": "abc-123",
                "filename": "policy.pdf",
                "chunk_count": 10,
                "page_count": 2,
                "uploaded_at": "2026-01-01 10:00:00",
            }
        ]
        with patch("app.routers.documents.database.list_documents", new_callable=AsyncMock, return_value=docs):
            resp = await client.get("/api/v1/documents")
        assert resp.status_code == 200
        assert len(resp.json()) == 1
        assert resp.json()[0]["doc_id"] == "abc-123"


class TestUploadDocument:
    async def test_upload_txt_succeeds(self, http_client):
        from app.services.parser import DocumentMetadata as ParserDocMeta, ParseResult

        client, mock_qdrant = http_client
        mock_qdrant.upsert = AsyncMock()

        content = b"This is a test document. " * 30
        fake_result = ParseResult(
            text="chunk1 chunk2",
            chunks=["chunk1", "chunk2"],
            page_count=1,
            metadata=ParserDocMeta(language="en", word_count=4, file_format="txt"),
        )

        with (
            patch("app.routers.documents.database.get_document_by_hash", new_callable=AsyncMock, return_value=None),
            patch("app.routers.documents.database.insert_document", new_callable=AsyncMock),
            patch("app.routers.documents.parse_and_chunk", return_value=fake_result),
            patch("app.routers.documents.get_embedder", return_value=MagicMock()),
            patch("app.routers.documents.async_encode_texts", new_callable=AsyncMock,
                  return_value=[[0.1] * 384, [0.2] * 384]),
            patch("app.routers.documents.upsert_chunks", new_callable=AsyncMock),
            patch("app.routers.documents.database.update_document_ingested", new_callable=AsyncMock),
        ):
            resp = await client.post(
                "/api/v1/documents/upload",
                files={"file": ("test.txt", io.BytesIO(content), "text/plain")},
            )

        assert resp.status_code == 201
        body = resp.json()
        assert body["filename"] == "test.txt"
        assert body["chunk_count"] == 2
        assert body["status"] == "success"
        assert "doc_id" in body
        assert "ingestion_report" in body
        assert "document_metadata" in body

    async def test_upload_unsupported_type_returns_400(self, http_client):
        client, _ = http_client
        resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("report.xlsx", io.BytesIO(b"data"), "application/octet-stream")},
        )
        assert resp.status_code == 400

    async def test_upload_oversized_file_returns_413(self, http_client):
        client, _ = http_client
        big = b"x" * (51 * 1024 * 1024)  # 51 MB > 50 MB limit
        resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("big.txt", io.BytesIO(big), "text/plain")},
        )
        assert resp.status_code == 413


class TestDeleteDocument:
    async def test_delete_existing_doc(self, http_client):
        client, _ = http_client
        doc = {"doc_id": "doc-1", "filename": "f.pdf", "chunk_count": 5, "page_count": 1, "uploaded_at": "2026-01-01"}
        with (
            patch("app.routers.documents.database.get_document", new_callable=AsyncMock, return_value=doc),
            patch("app.routers.documents.delete_document_chunks", new_callable=AsyncMock),
            patch("app.routers.documents.database.delete_document", new_callable=AsyncMock),
        ):
            resp = await client.delete("/api/v1/documents/doc-1")
        assert resp.status_code == 200
        assert resp.json() == {"status": "deleted", "doc_id": "doc-1"}

    async def test_delete_nonexistent_returns_404(self, http_client):
        client, _ = http_client
        with patch("app.routers.documents.database.get_document", new_callable=AsyncMock, return_value=None):
            resp = await client.delete("/api/v1/documents/ghost-id")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# QA router
# ---------------------------------------------------------------------------

class TestAskEndpoint:
    async def test_ask_returns_answer(self, http_client):
        from app.services.confidence_scorer import ConfidenceBreakdown, ScoredChunk
        from app.services.retriever import RetrieveOutput

        client, _ = http_client

        doc = {"doc_id": "doc-1", "filename": "policy.pdf", "chunk_count": 10,
               "page_count": 2, "uploaded_at": "2026-01-01"}

        mock_chunk = ScoredChunk(
            text="Return window is 30 days.",
            doc_id="doc-1",
            filename="policy.pdf",
            chunk_index=0,
            page_number=1,
            confidence=ConfidenceBreakdown(
                retrieval_score=0.9, freshness_score=0.8,
                authority_score=0.85, agreement_score=0.7, composite_score=0.85,
            ),
            vector_score=0.9,
            bm25_score=None,
        )
        mock_output = RetrieveOutput(chunks=[mock_chunk], filtered_out=0)

        with (
            patch("app.routers.qa.database.get_document", new_callable=AsyncMock, return_value=doc),
            patch("app.routers.qa.database.insert_citation_audit", new_callable=AsyncMock),
            patch("app.routers.qa.retrieve", new_callable=AsyncMock, return_value=mock_output),
            patch("app.routers.qa.generate_answer", new_callable=AsyncMock,
                  return_value={"answer": "30 days. [Source 1]", "tokens_used": 100,
                                "model": "llama-3.3-70b-versatile"}),
        ):
            resp = await client.post(
                "/api/v1/qa/ask",
                json={"question": "What is return window?", "document_id": "doc-1", "top_k": 5},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert "30 days." in body["answer"]
        assert body["doc_id"] == "doc-1"
        assert body["tokens_used"] == 100
        assert len(body["cited_sources"]) == 1
        assert body["cited_sources"][0]["tag"] == "[Source 1]"
        assert body["is_abstention"] is False

    async def test_ask_nonexistent_doc_returns_404(self, http_client):
        client, _ = http_client
        with patch("app.routers.qa.database.get_document", new_callable=AsyncMock, return_value=None):
            resp = await client.post(
                "/api/v1/qa/ask",
                json={"question": "What is this?", "document_id": "ghost-id", "top_k": 5},
            )
        assert resp.status_code == 404

    async def test_ask_question_too_short_returns_422(self, http_client):
        client, _ = http_client
        resp = await client.post(
            "/api/v1/qa/ask",
            json={"question": "Hi", "document_id": "doc-1"},
        )
        assert resp.status_code == 422
