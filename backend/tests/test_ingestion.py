"""Integration-style tests for the enriched document ingestion pipeline (Phase 6)."""
import io
from unittest.mock import AsyncMock, MagicMock, patch

import fitz
import pytest


def _make_pdf_bytes(text: str, pages: int = 1) -> bytes:
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), text)
    return doc.tobytes()


def _make_docx_bytes(paragraphs: list[str]) -> bytes:
    import docx

    doc = docx.Document()
    for para in paragraphs:
        doc.add_paragraph(para)
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.read()


def _make_html_bytes(body: str, with_nav: bool = True) -> bytes:
    nav = "<nav><ul><li>Home</li><li>About</li></ul></nav>" if with_nav else ""
    footer = "<footer>Copyright 2026</footer>" if with_nav else ""
    html = f"""<!DOCTYPE html>
<html><head><title>Test Page</title></head>
<body>
{nav}
<main>{body}</main>
{footer}
</body></html>"""
    return html.encode("utf-8")


_COMMON_PATCHES = [
    ("app.routers.documents.database.get_document_by_hash", None),
    ("app.routers.documents.database.insert_document", None),
    ("app.routers.documents.database.update_document_ingested", None),
    ("app.routers.documents.upsert_chunks", None),
]


def _build_patches(extra_pairs=None):
    pairs = list(_COMMON_PATCHES)
    if extra_pairs:
        pairs.extend(extra_pairs)
    return pairs


class TestDocxUpload:
    async def test_docx_upload(self, http_client):
        """A valid DOCX file should upload successfully with chunk_count > 0."""
        client, mock_qdrant = http_client
        mock_qdrant.upsert = AsyncMock()

        content = _make_docx_bytes([
            "The quick brown fox jumps over the lazy dog. " * 5,
            "Artificial intelligence is transforming modern industries. " * 5,
            "Python is a popular programming language for data science. " * 5,
        ])

        with (
            patch("app.routers.documents.database.get_document_by_hash", new_callable=AsyncMock, return_value=None),
            patch("app.routers.documents.database.insert_document", new_callable=AsyncMock),
            patch("app.routers.documents.get_embedder", return_value=MagicMock()),
            patch("app.routers.documents.async_encode_texts", new_callable=AsyncMock,
                  return_value=[[0.1] * 384]),
            patch("app.routers.documents.upsert_chunks", new_callable=AsyncMock, return_value=["id1"]),
            patch("app.routers.documents.database.update_document_ingested", new_callable=AsyncMock),
        ):
            resp = await client.post(
                "/api/v1/documents/upload",
                files={"file": ("test.docx", io.BytesIO(content),
                                "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            )

        assert resp.status_code == 201
        body = resp.json()
        assert body["chunk_count"] > 0
        assert body["document_metadata"]["file_format"] == "docx"


class TestHtmlUpload:
    async def test_html_upload_strips_nav_content(self, http_client):
        """HTML nav/footer should be stripped; only main body text in chunks."""
        client, mock_qdrant = http_client
        mock_qdrant.upsert = AsyncMock()

        main_text = "This is the main article content about machine learning. " * 10
        content = _make_html_bytes(main_text, with_nav=True)

        captured_chunks: list[str] = []

        async def capture_upsert(**kwargs):
            captured_chunks.extend(kwargs.get("chunks", []))
            return ["id1"]

        with (
            patch("app.routers.documents.database.get_document_by_hash", new_callable=AsyncMock, return_value=None),
            patch("app.routers.documents.database.insert_document", new_callable=AsyncMock),
            patch("app.routers.documents.get_embedder", return_value=MagicMock()),
            patch("app.routers.documents.async_encode_texts", new_callable=AsyncMock,
                  return_value=[[0.1] * 384]),
            patch("app.routers.documents.upsert_chunks", new_callable=AsyncMock,
                  side_effect=capture_upsert),
            patch("app.routers.documents.database.update_document_ingested", new_callable=AsyncMock),
        ):
            resp = await client.post(
                "/api/v1/documents/upload",
                files={"file": ("page.html", io.BytesIO(content), "text/html")},
            )

        assert resp.status_code == 201
        combined = " ".join(captured_chunks).lower()
        assert "machine learning" in combined
        assert "home" not in combined or "about" not in combined  # nav items stripped
        assert "copyright" not in combined  # footer stripped


class TestExactDedup:
    async def test_exact_dedup_removes_repeated_chunks(self, http_client):
        """A TXT file with 3 identical long paragraphs should report exact_dedup_removed >= 2."""
        client, mock_qdrant = http_client
        mock_qdrant.upsert = AsyncMock()

        # Each paragraph is ~600 chars so each forms its own chunk (chunk_size=500 by default,
        # but with RecursiveCharacterTextSplitter the paragraph will split into 1-2 chunks).
        # Use 3 identical blocks to guarantee duplicates.
        paragraph = ("The annual performance review process evaluates employee contributions. " * 8).strip()
        content = (paragraph + "\n\n") * 3
        assert len(content) > 600

        with (
            patch("app.routers.documents.database.get_document_by_hash", new_callable=AsyncMock, return_value=None),
            patch("app.routers.documents.database.insert_document", new_callable=AsyncMock),
            patch("app.routers.documents.get_embedder", return_value=MagicMock()),
            patch("app.routers.documents.async_encode_texts", new_callable=AsyncMock,
                  return_value=[[0.1] * 384]),
            patch("app.routers.documents.upsert_chunks", new_callable=AsyncMock, return_value=["id1"]),
            patch("app.routers.documents.database.update_document_ingested", new_callable=AsyncMock),
        ):
            resp = await client.post(
                "/api/v1/documents/upload",
                files={"file": ("dedup_test.txt", io.BytesIO(content.encode()), "text/plain")},
            )

        assert resp.status_code == 201
        body = resp.json()
        report = body["ingestion_report"]
        assert report["exact_dedup_removed"] >= 2
        assert report["final_chunks"] < report["original_chunks"]


class TestMetadataExtraction:
    async def test_pdf_metadata_in_response(self, http_client):
        """Uploading a PDF should return language and word_count in document_metadata."""
        client, mock_qdrant = http_client
        mock_qdrant.upsert = AsyncMock()

        pdf_text = "The quick brown fox jumps over the lazy dog. " * 20
        content = _make_pdf_bytes(pdf_text)

        with (
            patch("app.routers.documents.database.get_document_by_hash", new_callable=AsyncMock, return_value=None),
            patch("app.routers.documents.database.insert_document", new_callable=AsyncMock),
            patch("app.routers.documents.get_embedder", return_value=MagicMock()),
            patch("app.routers.documents.async_encode_texts", new_callable=AsyncMock,
                  return_value=[[0.1] * 384]),
            patch("app.routers.documents.upsert_chunks", new_callable=AsyncMock, return_value=["id1"]),
            patch("app.routers.documents.database.update_document_ingested", new_callable=AsyncMock),
        ):
            resp = await client.post(
                "/api/v1/documents/upload",
                files={"file": ("doc.pdf", io.BytesIO(content), "application/pdf")},
            )

        assert resp.status_code == 201
        meta = resp.json()["document_metadata"]
        assert meta["language"] != ""
        assert meta["word_count"] > 0
        assert meta["file_format"] == "pdf"


class TestUnsupportedFormat:
    async def test_xlsx_returns_400(self, http_client):
        """Uploading an unsupported file type should return 400."""
        client, _ = http_client
        resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("spreadsheet.xlsx", io.BytesIO(b"fake xlsx data"), "application/octet-stream")},
        )
        assert resp.status_code == 400
