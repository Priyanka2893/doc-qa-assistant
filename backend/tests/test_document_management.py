"""Tests for Phase 3 document management features."""
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestDocumentManagement:
    async def test_duplicate_detection(self, http_client):
        """Uploading a file whose hash already exists should return 409."""
        client, _ = http_client
        existing_doc = {
            "doc_id": "existing-123",
            "filename": "test.txt",
            "chunk_count": 5,
            "page_count": 1,
            "file_size_bytes": 100,
            "uploaded_at": "2026-01-01 10:00:00",
            "status": "ready",
            "content_hash": "abc123hash",
        }
        content = b"This is duplicate content. " * 30

        with patch(
            "app.routers.documents.database.get_document_by_hash",
            new_callable=AsyncMock,
            return_value=existing_doc,
        ):
            resp = await client.post(
                "/api/v1/documents/upload",
                files={"file": ("test.txt", io.BytesIO(content), "text/plain")},
            )

        assert resp.status_code == 409
        body = resp.json()
        assert body["detail"] == "Document already exists"
        assert body["existing_doc_id"] == "existing-123"
        assert body["filename"] == "test.txt"

    async def test_status_tracking(self, http_client):
        """Uploaded document should appear with status='ready' in the documents list."""
        from unittest.mock import MagicMock

        from app.services.parser import DocumentMetadata as ParserDocMeta, ParseResult

        client, mock_qdrant = http_client
        mock_qdrant.upsert = AsyncMock()
        content = b"Status tracking test content. " * 30
        fake_result = ParseResult(
            text="chunk1 chunk2",
            chunks=["chunk1", "chunk2"],
            page_count=2,
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
            upload_resp = await client.post(
                "/api/v1/documents/upload",
                files={"file": ("status_test.txt", io.BytesIO(content), "text/plain")},
            )

        assert upload_resp.status_code == 201
        doc_id = upload_resp.json()["doc_id"]

        ready_doc = {
            "doc_id": doc_id,
            "filename": "status_test.txt",
            "chunk_count": 2,
            "page_count": 2,
            "file_size_bytes": len(content),
            "uploaded_at": "2026-01-01 10:00:00",
            "status": "ready",
            "content_hash": "somehash",
        }
        with patch(
            "app.routers.documents.database.list_documents",
            new_callable=AsyncMock,
            return_value=[ready_doc],
        ):
            list_resp = await client.get("/api/v1/documents")

        assert list_resp.status_code == 200
        docs = list_resp.json()
        assert len(docs) == 1
        assert docs[0]["status"] == "ready"
        assert docs[0]["doc_id"] == doc_id

    async def test_global_ask(self, http_client):
        """ask-global should search across all docs and return sources with filename/doc_id."""
        from app.services.confidence_scorer import ConfidenceBreakdown, ScoredChunk
        from app.services.retriever import RetrieveOutput

        client, _ = http_client

        def _chunk(text, doc_id, filename, chunk_index, page_number):
            return ScoredChunk(
                text=text,
                doc_id=doc_id,
                filename=filename,
                chunk_index=chunk_index,
                page_number=page_number,
                confidence=ConfidenceBreakdown(
                    retrieval_score=0.9, freshness_score=0.8,
                    authority_score=0.85, agreement_score=0.7, composite_score=0.85,
                ),
                vector_score=0.9,
                bm25_score=None,
            )

        mock_output = RetrieveOutput(
            chunks=[
                _chunk("The policy covers remote work arrangements.", "doc-1", "policy.pdf", 0, 1),
                _chunk("Employee benefits include health insurance.", "doc-2", "benefits.txt", 2, 3),
            ],
            filtered_out=0,
        )

        with (
            patch("app.routers.qa.retrieve_global", new_callable=AsyncMock, return_value=mock_output),
            patch("app.routers.qa.generate_answer", new_callable=AsyncMock,
                  return_value={
                      "answer": "Remote work is covered; benefits include health insurance.",
                      "tokens_used": 200,
                      "model": "llama-3.3-70b-versatile",
                  }),
        ):
            resp = await client.post(
                "/api/v1/qa/ask-global",
                json={"question": "What are the policies?", "top_k": 10},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert "Remote work" in body["answer"]
        assert len(body["sources"]) == 2
        assert body["sources"][0]["filename"] == "policy.pdf"
        assert body["sources"][0]["doc_id"] == "doc-1"
        assert body["sources"][1]["filename"] == "benefits.txt"
        assert body["sources"][1]["doc_id"] == "doc-2"
        assert body["tokens_used"] == 200

    async def test_get_single_document(self, http_client):
        """GET /documents/{doc_id} should return full DocumentInfo."""
        client, _ = http_client
        doc = {
            "doc_id": "doc-abc",
            "filename": "manual.pdf",
            "chunk_count": 12,
            "page_count": 4,
            "file_size_bytes": 48000,
            "uploaded_at": "2026-01-15 09:30:00",
            "status": "ready",
            "content_hash": "deadbeef1234",
        }
        with patch(
            "app.routers.documents.database.get_document",
            new_callable=AsyncMock,
            return_value=doc,
        ):
            resp = await client.get("/api/v1/documents/doc-abc")

        assert resp.status_code == 200
        body = resp.json()
        assert body["doc_id"] == "doc-abc"
        assert body["filename"] == "manual.pdf"
        assert body["status"] == "ready"
        assert body["file_size_bytes"] == 48000
        assert body["content_hash"] == "deadbeef1234"

    async def test_get_single_document_not_found(self, http_client):
        """GET /documents/{doc_id} returns 404 for unknown doc."""
        client, _ = http_client
        with patch(
            "app.routers.documents.database.get_document",
            new_callable=AsyncMock,
            return_value=None,
        ):
            resp = await client.get("/api/v1/documents/ghost-id")

        assert resp.status_code == 404

    async def test_global_ask_short_question_returns_422(self, http_client):
        """ask-global respects min_length validation."""
        client, _ = http_client
        resp = await client.post("/api/v1/qa/ask-global", json={"question": "Hi"})
        assert resp.status_code == 422
