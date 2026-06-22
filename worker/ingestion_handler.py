import asyncio
import os

import structlog
from qdrant_client import AsyncQdrantClient

from app.config import get_settings
from app.database import get_document_by_hash, insert_document_kafka
from app.services.deduplicator import deduplicate_exact, deduplicate_semantic
from app.services.embedder import async_encode_texts, get_embedder
from app.services.parser import parse_and_chunk
from app.services.vector_store import init_collection, upsert_chunks
from worker.s3_client import download_document
from worker.schemas import KafkaDocumentMessage
from worker.status_publisher import publish_status

log = structlog.get_logger()


def _get_qdrant_client() -> AsyncQdrantClient:
    settings = get_settings()
    return AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)


async def handle_document_message(msg: KafkaDocumentMessage) -> dict:
    """
    Process a single Kafka message through the full RAG ingestion pipeline.
    Returns: {"status": "completed|skipped|failed", "doc_id": str, "chunk_count": int}
    """
    settings = get_settings()
    bound_log = log.bind(doc_id=msg.doc_id, company=msg.company, category=msg.category)

    # ── Step 1: Duplicate check ──────────────────────────────────────────────
    existing = await get_document_by_hash(msg.content_hash)
    if existing:
        bound_log.info(
            "document_skipped_duplicate",
            existing_doc_id=existing["doc_id"],
            content_hash=msg.content_hash,
        )
        publish_status(msg.doc_id, "duplicate", {"existing_doc_id": existing["doc_id"]})
        return {"status": "skipped", "doc_id": msg.doc_id, "chunk_count": 0}

    publish_status(msg.doc_id, "processing")

    # ── Step 2: Download from MinIO ──────────────────────────────────────────
    bound_log.info("downloading_from_s3", s3_key=msg.s3_key)
    loop = asyncio.get_event_loop()
    content = await loop.run_in_executor(None, download_document, msg.s3_key)

    # ── Step 3: Parse and chunk ──────────────────────────────────────────────
    bound_log.info("parsing_document", filename=msg.filename)
    parse_result = await loop.run_in_executor(
        None,
        parse_and_chunk,
        msg.filename,
        content,
        settings.CHUNK_SIZE,
        settings.CHUNK_OVERLAP,
    )

    # ── Step 4: Deduplicate chunks ───────────────────────────────────────────
    unique_chunks, exact_removed = deduplicate_exact(parse_result.chunks)
    embedder = get_embedder(settings.EMBEDDING_MODEL)
    final_chunks, semantic_removed = await deduplicate_semantic(unique_chunks, embedder, 0.95)
    bound_log.info(
        "chunks_prepared",
        original=len(parse_result.chunks),
        after_exact_dedup=len(unique_chunks),
        after_semantic_dedup=len(final_chunks),
    )

    if not final_chunks:
        bound_log.warning("no_chunks_after_dedup")
        publish_status(msg.doc_id, "failed", {"reason": "no_chunks_after_dedup"})
        return {"status": "failed", "doc_id": msg.doc_id, "chunk_count": 0}

    # ── Step 5: Embed chunks ─────────────────────────────────────────────────
    embeddings = await async_encode_texts(settings.EMBEDDING_MODEL, final_chunks)

    # ── Step 6: Upsert to Qdrant with enriched metadata ─────────────────────
    client = _get_qdrant_client()
    try:
        await init_collection(client, settings.QDRANT_COLLECTION_NAME, settings.EMBEDDING_DIMENSION)
        await upsert_chunks(
            client=client,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            doc_id=msg.doc_id,
            chunks=final_chunks,
            embeddings=embeddings,
            filename=msg.filename,
            language=parse_result.metadata.language,
            doc_title=parse_result.metadata.title,
            author=parse_result.metadata.author,
            extra_payload={
                "company": msg.company,
                "category": msg.category,
                "ingestion_source": "kafka",
                "airflow_dag_run_id": msg.airflow_dag_run_id,
            },
        )
    finally:
        await client.close()

    # ── Step 7: Register in SQLite ───────────────────────────────────────────
    await insert_document_kafka(
        doc_id=msg.doc_id,
        filename=msg.filename,
        file_size_bytes=msg.file_size_bytes,
        page_count=parse_result.page_count,
        chunk_count=len(final_chunks),
        content_hash=msg.content_hash,
        author=parse_result.metadata.author,
        doc_title=parse_result.metadata.title,
        language=parse_result.metadata.language or "en",
        word_count=parse_result.metadata.word_count,
        file_format=msg.file_extension,
        exact_dedup_removed=exact_removed,
        semantic_dedup_removed=semantic_removed,
        company=msg.company,
        category=msg.category,
    )

    publish_status(
        msg.doc_id,
        "completed",
        {"chunk_count": len(final_chunks), "company": msg.company},
    )
    bound_log.info(
        "document_ingested_success",
        chunk_count=len(final_chunks),
        page_count=parse_result.page_count,
        ingestion_source="kafka",
    )
    return {"status": "completed", "doc_id": msg.doc_id, "chunk_count": len(final_chunks)}
