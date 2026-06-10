import pytest
from pydantic import ValidationError

from app.models import AskRequest, AskResponse, ChunkSource, DocumentInfo, HealthResponse, UploadResponse


class TestAskRequest:
    def test_valid(self):
        req = AskRequest(question="What is the return policy?", document_id="abc-123")
        assert req.top_k == 5

    def test_question_too_short(self):
        with pytest.raises(ValidationError):
            AskRequest(question="Hi", document_id="abc-123")

    def test_question_too_long(self):
        with pytest.raises(ValidationError):
            AskRequest(question="x" * 1001, document_id="abc-123")

    def test_top_k_bounds(self):
        with pytest.raises(ValidationError):
            AskRequest(question="Valid question?", document_id="abc", top_k=0)
        with pytest.raises(ValidationError):
            AskRequest(question="Valid question?", document_id="abc", top_k=21)

    def test_top_k_boundaries_accepted(self):
        assert AskRequest(question="Valid question?", document_id="abc", top_k=1).top_k == 1
        assert AskRequest(question="Valid question?", document_id="abc", top_k=20).top_k == 20


class TestUploadResponse:
    def test_default_status(self):
        r = UploadResponse(
            doc_id="id", filename="f.pdf", chunk_count=5, page_count=2, ingestion_time_ms=100
        )
        assert r.status == "success"


class TestChunkSource:
    def test_page_number_optional(self):
        src = ChunkSource(chunk_index=0, text_excerpt="text", score=0.9, page_number=None)
        assert src.page_number is None

        src2 = ChunkSource(chunk_index=0, text_excerpt="text", score=0.9, page_number=3)
        assert src2.page_number == 3


class TestHealthResponse:
    def test_fields_present(self):
        h = HealthResponse(status="ok", qdrant="ok", embedding_model="all-MiniLM-L6-v2", version="1.0.0")
        assert h.status == "ok"


class TestDocumentInfo:
    def test_construction(self):
        d = DocumentInfo(
            doc_id="x", filename="doc.pdf", chunk_count=10, page_count=3, uploaded_at="2026-01-01"
        )
        assert d.chunk_count == 10
