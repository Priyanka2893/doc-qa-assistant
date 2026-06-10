import fitz  # pymupdf
import structlog
from fastapi import HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = structlog.get_logger(__name__)


def extract_text_from_pdf(content: bytes) -> tuple[str, int]:
    """Extract full text and page count from a PDF byte payload."""
    doc = fitz.open(stream=content, filetype="pdf")
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages), len(pages)


def extract_text_from_txt(content: bytes) -> tuple[str, int]:
    """Decode a plain-text byte payload and return (text, 1)."""
    return content.decode("utf-8", errors="replace"), 1


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)


def parse_and_chunk(
    filename: str,
    content: bytes,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[str], int]:
    """Route file to the correct extractor by extension, then chunk.

    Returns (chunks, page_count).
    Raises HTTPException(400) for unsupported types, HTTPException(422) for empty files.
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "pdf":
        text, page_count = extract_text_from_pdf(content)
    elif ext == "txt":
        text, page_count = extract_text_from_txt(content)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. Allowed: .pdf, .txt",
        )

    if not text.strip():
        raise HTTPException(status_code=422, detail="File is empty or contains no extractable text.")

    chunks = chunk_text(text, chunk_size, chunk_overlap)
    logger.info(
        "parser.chunked",
        filename=filename,
        page_count=page_count,
        chunk_count=len(chunks),
    )
    return chunks, page_count
