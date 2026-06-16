import io
import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # pymupdf
import magic
import structlog
from fastapi import HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.telemetry import track_stage

logger = structlog.get_logger(__name__)

# ─── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class DocumentMetadata:
    author: str | None = None
    created_at: str | None = None  # ISO 8601
    title: str | None = None
    language: str = "en"
    word_count: int = 0
    file_format: str = ""


@dataclass
class RawParseResult:
    text: str
    page_count: int
    metadata: DocumentMetadata


@dataclass
class ParseResult:
    text: str
    chunks: list[str]
    page_count: int
    metadata: DocumentMetadata


# ─── MIME allowlist (per extension) ───────────────────────────────────────────

_EXT_MIME_MAP: dict[str, set[str]] = {
    ".pdf": {"application/pdf"},
    ".txt": {"text/plain", "text/x-python", "text/x-script.python", "inode/x-empty"},
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip",
        "application/octet-stream",
    },
    ".html": {"text/html", "text/plain"},
    ".htm": {"text/html", "text/plain"},
    ".png": {"image/png"},
    ".jpg": {"image/jpeg"},
    ".jpeg": {"image/jpeg"},
    ".tiff": {"image/tiff", "image/x-tiff"},
}


def validate_file_type(content: bytes, filename: str) -> None:
    """Raise HTTPException(400) if magic bytes don't match the expected MIME for this extension."""
    detected = magic.from_buffer(content, mime=True)
    ext = Path(filename).suffix.lower()
    allowed = _EXT_MIME_MAP.get(ext, set())
    if detected not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"File content doesn't match allowed types for '{ext}'. Detected: {detected}",
        )


# ─── Format parsers ───────────────────────────────────────────────────────────


def parse_pdf(content: bytes) -> RawParseResult:
    doc = fitz.open(stream=content, filetype="pdf")
    pages_text: list[str] = []
    for page in doc:
        text = page.get_text()
        if not text.strip():
            try:
                import pytesseract
                from PIL import Image

                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img)
            except Exception:
                pass
        pages_text.append(text)

    raw_meta = doc.metadata or {}
    author = raw_meta.get("author") or None
    title = raw_meta.get("title") or None
    created_at = _parse_pdf_date(raw_meta.get("creationDate") or "")
    doc.close()

    return RawParseResult(
        text="\n\n".join(pages_text),
        page_count=len(pages_text),
        metadata=DocumentMetadata(author=author, title=title, created_at=created_at, file_format="pdf"),
    )


def _parse_pdf_date(raw: str) -> str | None:
    """Convert PDF date format 'D:YYYYMMDD...' to ISO 8601 date string."""
    if not raw:
        return None
    clean = re.sub(r"^[Dd]:", "", raw.strip())
    if len(clean) >= 8:
        try:
            return f"{clean[0:4]}-{clean[4:6]}-{clean[6:8]}"
        except Exception:
            pass
    return None


def parse_docx(content: bytes) -> RawParseResult:
    import docx

    doc = docx.Document(io.BytesIO(content))
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    props = doc.core_properties
    author = props.author or None
    title = props.title or None
    created_at = props.created.isoformat() if props.created else None
    page_count = max(1, len(doc.sections))
    return RawParseResult(
        text=text,
        page_count=page_count,
        metadata=DocumentMetadata(author=author, title=title, created_at=created_at, file_format="docx"),
    )


def parse_html(content: bytes) -> RawParseResult:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(content, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    for cls_pattern in ("cookie", "banner", "menu"):
        for el in soup.find_all(class_=lambda c, p=cls_pattern: c and p in " ".join(c).lower()):
            el.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else None
    text = soup.get_text(separator="\n", strip=True)
    text = "\n".join(line for line in text.splitlines() if line.strip())
    return RawParseResult(
        text=text,
        page_count=1,
        metadata=DocumentMetadata(title=title, file_format="html"),
    )


def parse_image(content: bytes) -> RawParseResult:
    import pytesseract
    from PIL import Image

    image = Image.open(io.BytesIO(content))
    text = pytesseract.image_to_string(image, config="--psm 3")
    if len(text.strip()) < 20:
        raise HTTPException(
            status_code=422,
            detail="Could not extract text from image. Ensure the image contains readable text.",
        )
    return RawParseResult(
        text=text,
        page_count=1,
        metadata=DocumentMetadata(file_format="image"),
    )


def parse_txt(content: bytes) -> RawParseResult:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1")
    return RawParseResult(
        text=text,
        page_count=1,
        metadata=DocumentMetadata(file_format="txt"),
    )


# ─── Format router ────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS: dict[str, object] = {
    ".pdf": parse_pdf,
    ".txt": parse_txt,
    ".docx": parse_docx,
    ".html": parse_html,
    ".htm": parse_html,
    ".png": parse_image,
    ".jpg": parse_image,
    ".jpeg": parse_image,
    ".tiff": parse_image,
}


# ─── Text normalization ───────────────────────────────────────────────────────


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [re.sub(r" {2,}", " ", line) for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text


# ─── Chunking ─────────────────────────────────────────────────────────────────


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)


# ─── Public entry point ───────────────────────────────────────────────────────


def parse_and_chunk(
    filename: str,
    content: bytes,
    chunk_size: int,
    chunk_overlap: int,
) -> ParseResult:
    """Route file to the correct extractor by extension, normalize, detect language, and chunk.

    Raises HTTPException(400) for unsupported extensions or magic-byte mismatches.
    Raises HTTPException(422) for empty/unparseable files.
    """
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(SUPPORTED_EXTENSIONS.keys())}",
        )

    validate_file_type(content, filename)

    parser_fn = SUPPORTED_EXTENSIONS[ext]
    with track_stage("parse"):
        raw_result: RawParseResult = parser_fn(content)  # type: ignore[operator]
        cleaned_text = normalize_text(raw_result.text)

    if not cleaned_text.strip():
        raise HTTPException(status_code=422, detail="File is empty or could not be parsed")

    try:
        from langdetect import detect

        raw_result.metadata.language = detect(cleaned_text[:500])
    except Exception:
        raw_result.metadata.language = "en"

    raw_result.metadata.word_count = len(cleaned_text.split())

    with track_stage("chunk"):
        chunks = chunk_text(cleaned_text, chunk_size, chunk_overlap)
    logger.info(
        "parser.chunked",
        filename=filename,
        page_count=raw_result.page_count,
        chunk_count=len(chunks),
        language=raw_result.metadata.language,
    )

    return ParseResult(
        text=cleaned_text,
        chunks=chunks,
        page_count=raw_result.page_count,
        metadata=raw_result.metadata,
    )


# ─── Legacy helpers kept for backward compatibility with existing tests ────────


def extract_text_from_pdf(content: bytes) -> tuple[str, int]:
    result = parse_pdf(content)
    return result.text, result.page_count


def extract_text_from_txt(content: bytes) -> tuple[str, int]:
    result = parse_txt(content)
    return result.text, result.page_count
