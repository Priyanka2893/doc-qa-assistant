import hashlib
import time
import uuid

import structlog
from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app import database
from app.models import DocumentInfo, UploadResponse
from app.services.bm25_store import get_bm25_store
from app.services.embedder import async_encode_texts
from app.services.parser import parse_and_chunk
from app.services.vector_store import delete_document_chunks, upsert_chunks

logger = structlog.get_logger(__name__)
router = APIRouter()

_ALLOWED_EXTENSIONS = {".pdf", ".txt"}


@router.post("/documents/upload", response_model=UploadResponse, status_code=201)
async def upload_document(request: Request, file: UploadFile) -> UploadResponse:
    """Ingest a PDF or TXT document: parse, chunk, embed, and store in Qdrant + SQLite."""
    settings = request.app.state.settings
    qdrant_client = request.app.state.qdrant_client

    filename = file.filename or "unknown"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(_ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {settings.MAX_FILE_SIZE_MB} MB limit.",
        )

    content_hash = hashlib.sha256(content).hexdigest()
    existing = await database.get_document_by_hash(content_hash)
    if existing:
        return JSONResponse(
            status_code=409,
            content={
                "detail": "Document already exists",
                "existing_doc_id": existing["doc_id"],
                "filename": existing["filename"],
            },
        )

    doc_id = str(uuid.uuid4())
    await database.insert_document(
        doc_id=doc_id,
        filename=filename,
        file_size_bytes=len(content),
        content_hash=content_hash,
        status="processing",
    )

    t_start = time.perf_counter()
    try:
        chunks, page_count = parse_and_chunk(
            filename, content, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP
        )
        embeddings = await async_encode_texts(settings.EMBEDDING_MODEL, chunks)
        chunk_ids = await upsert_chunks(
            client=qdrant_client,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            doc_id=doc_id,
            chunks=chunks,
            embeddings=embeddings,
            filename=filename,
        )
        get_bm25_store().build_index(doc_id=doc_id, chunks=chunks, chunk_ids=chunk_ids, filename=filename)
        await database.update_document_ingested(
            doc_id=doc_id,
            chunk_count=len(chunks),
            page_count=page_count,
        )
    except Exception as exc:
        await database.update_document_status(doc_id, "error")
        logger.error("documents.upload_failed", doc_id=doc_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Document ingestion failed.")

    ingestion_ms = int((time.perf_counter() - t_start) * 1000)
    logger.info(
        "documents.uploaded",
        doc_id=doc_id,
        filename=filename,
        chunks=len(chunks),
        ingestion_ms=ingestion_ms,
    )

    return UploadResponse(
        doc_id=doc_id,
        filename=filename,
        chunk_count=len(chunks),
        page_count=page_count,
        ingestion_time_ms=ingestion_ms,
    )


@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents() -> list[DocumentInfo]:
    """Return metadata for all ingested documents."""
    rows = await database.list_documents()
    return [DocumentInfo(**row) for row in rows]


@router.get("/documents/{doc_id}", response_model=DocumentInfo)
async def get_document(doc_id: str) -> DocumentInfo:
    """Return full metadata for a single document. 404 if not found."""
    doc = await database.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return DocumentInfo(**doc)


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, request: Request) -> dict:
    """Remove a document and all its chunks from Qdrant and SQLite."""
    settings = request.app.state.settings
    qdrant_client = request.app.state.qdrant_client

    doc = await database.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    await delete_document_chunks(qdrant_client, settings.QDRANT_COLLECTION_NAME, doc_id)
    get_bm25_store().remove_document(doc_id)
    await database.delete_document(doc_id)
    logger.info("documents.deleted", doc_id=doc_id)
    return {"status": "deleted", "doc_id": doc_id}
