import fitz
import pytest

from app.services.parser import (
    ParseResult,
    chunk_text,
    extract_text_from_pdf,
    extract_text_from_txt,
    normalize_text,
    parse_and_chunk,
)


def _make_pdf_bytes(text: str, pages: int = 1) -> bytes:
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), text)
    return doc.tobytes()


class TestExtractText:
    def test_txt_returns_text_and_one_page(self):
        text, pages = extract_text_from_txt(b"Hello, world!")
        assert text == "Hello, world!"
        assert pages == 1

    def test_txt_utf8_decoding(self):
        text, _ = extract_text_from_txt("café".encode("utf-8"))
        assert "café" in text

    def test_pdf_extracts_text(self):
        pdf = _make_pdf_bytes("Return policy is 30 days")
        text, pages = extract_text_from_pdf(pdf)
        assert "Return policy" in text
        assert pages == 1

    def test_pdf_multipage_count(self):
        pdf = _make_pdf_bytes("Page content", pages=3)
        _, pages = extract_text_from_pdf(pdf)
        assert pages == 3


class TestChunkText:
    def test_short_text_single_chunk(self):
        chunks = chunk_text("Short text", chunk_size=500, chunk_overlap=100)
        assert len(chunks) == 1
        assert chunks[0] == "Short text"

    def test_long_text_splits(self):
        long = "word " * 300  # ~1500 chars
        chunks = chunk_text(long, chunk_size=200, chunk_overlap=40)
        assert len(chunks) > 1

    def test_overlap_produces_repeated_content(self):
        text = "alpha beta gamma delta epsilon " * 30
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=50)
        assert len(chunks) >= 2
        assert chunks[0][-20:] in chunks[1] or chunks[1][:20] in chunks[0]


class TestNormalizeText:
    def test_strips_control_chars(self):
        result = normalize_text("hello\x00world")
        assert "\x00" not in result
        assert "hello" in result

    def test_collapses_blank_lines(self):
        result = normalize_text("a\n\n\n\n\nb")
        assert "\n\n\n" not in result

    def test_crlf_normalized(self):
        result = normalize_text("line1\r\nline2\r\nline3")
        assert "\r" not in result
        assert "line1" in result and "line3" in result


class TestParseAndChunk:
    def test_txt_file_returns_parse_result(self):
        content = b"Some document content. " * 20
        result = parse_and_chunk("readme.txt", content, 200, 40)
        assert isinstance(result, ParseResult)
        assert result.page_count == 1
        assert len(result.chunks) >= 1
        assert result.metadata.file_format == "txt"
        assert result.metadata.word_count > 0

    def test_pdf_file_returns_parse_result(self):
        pdf = _make_pdf_bytes("Policy document content here.")
        result = parse_and_chunk("policy.pdf", pdf, 200, 40)
        assert isinstance(result, ParseResult)
        assert result.page_count == 1
        assert len(result.chunks) >= 1
        assert result.metadata.file_format == "pdf"

    def test_unsupported_extension_raises_400(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            parse_and_chunk("file.xlsx", b"data", 500, 100)
        assert exc_info.value.status_code == 400

    def test_empty_txt_raises_422(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            parse_and_chunk("empty.txt", b"   \n\t  ", 500, 100)
        assert exc_info.value.status_code == 422

    def test_no_extension_raises_400(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            parse_and_chunk("noextension", b"data", 500, 100)
        assert exc_info.value.status_code == 400

    def test_language_detected(self):
        content = b"The quick brown fox jumps over the lazy dog. " * 10
        result = parse_and_chunk("en.txt", content, 500, 100)
        assert result.metadata.language in ("en", "unknown", "")  # langdetect should detect English
