import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import aiosqlite
import structlog

logger = structlog.get_logger(__name__)

_DB_PATH = Path(__file__).parent.parent / "data" / "documents.db"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_size_bytes INTEGER,
    page_count INTEGER,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    uploaded_at TEXT NOT NULL DEFAULT (datetime('now')),
    status TEXT NOT NULL DEFAULT 'processing',
    content_hash TEXT,
    author TEXT,
    doc_title TEXT,
    language TEXT DEFAULT 'en',
    word_count INTEGER DEFAULT 0,
    file_format TEXT,
    exact_dedup_removed INTEGER DEFAULT 0,
    semantic_dedup_removed INTEGER DEFAULT 0
);
"""

_MIGRATIONS = [
    "ALTER TABLE documents ADD COLUMN status TEXT NOT NULL DEFAULT 'ready'",
    "ALTER TABLE documents ADD COLUMN content_hash TEXT",
    "ALTER TABLE documents ADD COLUMN author TEXT",
    "ALTER TABLE documents ADD COLUMN doc_title TEXT",
    "ALTER TABLE documents ADD COLUMN language TEXT DEFAULT 'en'",
    "ALTER TABLE documents ADD COLUMN word_count INTEGER DEFAULT 0",
    "ALTER TABLE documents ADD COLUMN file_format TEXT",
    "ALTER TABLE documents ADD COLUMN exact_dedup_removed INTEGER DEFAULT 0",
    "ALTER TABLE documents ADD COLUMN semantic_dedup_removed INTEGER DEFAULT 0",
]


async def init_db() -> None:
    """Create the data directory and documents table if they don't exist."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(_DB_PATH) as db:
        await db.execute(_CREATE_TABLE_SQL)
        for col_sql in _MIGRATIONS:
            try:
                await db.execute(col_sql)
            except Exception:
                pass  # column already exists
        await db.commit()
    logger.info("database.initialized", path=str(_DB_PATH))


@asynccontextmanager
async def get_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    """Async context manager yielding a connected aiosqlite connection."""
    async with aiosqlite.connect(_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        yield db


async def insert_document(
    doc_id: str,
    filename: str,
    file_size_bytes: int,
    page_count: int = 0,
    chunk_count: int = 0,
    content_hash: str | None = None,
    status: str = "processing",
) -> None:
    """Insert a new document record (call before ingestion with status='processing')."""
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO documents
                (doc_id, filename, file_size_bytes, page_count, chunk_count, content_hash, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (doc_id, filename, file_size_bytes, page_count, chunk_count, content_hash, status),
        )
        await db.commit()


async def update_document_ingested(
    doc_id: str,
    chunk_count: int,
    page_count: int,
    author: str | None = None,
    doc_title: str | None = None,
    language: str = "en",
    word_count: int = 0,
    file_format: str = "",
    exact_dedup_removed: int = 0,
    semantic_dedup_removed: int = 0,
) -> None:
    """Mark a document as ready after successful ingestion, storing enriched metadata."""
    async with get_db() as db:
        await db.execute(
            """
            UPDATE documents
            SET status = 'ready', chunk_count = ?, page_count = ?,
                author = ?, doc_title = ?, language = ?, word_count = ?,
                file_format = ?, exact_dedup_removed = ?, semantic_dedup_removed = ?
            WHERE doc_id = ?
            """,
            (
                chunk_count, page_count,
                author, doc_title, language, word_count,
                file_format, exact_dedup_removed, semantic_dedup_removed,
                doc_id,
            ),
        )
        await db.commit()


async def update_document_status(doc_id: str, status: str) -> None:
    """Update just the status of a document (e.g., to 'error')."""
    async with get_db() as db:
        await db.execute(
            "UPDATE documents SET status = ? WHERE doc_id = ?",
            (status, doc_id),
        )
        await db.commit()


_SELECT_COLS = """
    doc_id, filename, chunk_count, page_count, file_size_bytes, uploaded_at, status,
    content_hash, author, doc_title, language, word_count, file_format,
    exact_dedup_removed, semantic_dedup_removed
"""


async def get_document_by_hash(content_hash: str) -> dict | None:
    """Return a document record matching the given SHA256 hash, or None."""
    async with get_db() as db:
        cursor = await db.execute(
            f"SELECT {_SELECT_COLS} FROM documents WHERE content_hash = ?",
            (content_hash,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None


async def list_documents() -> list[dict]:
    """Return all document records ordered by upload time descending."""
    async with get_db() as db:
        cursor = await db.execute(
            f"SELECT {_SELECT_COLS} FROM documents ORDER BY uploaded_at DESC"
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def get_document(doc_id: str) -> dict | None:
    """Return a single document record or None if not found."""
    async with get_db() as db:
        cursor = await db.execute(
            f"SELECT {_SELECT_COLS} FROM documents WHERE doc_id = ?",
            (doc_id,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None


async def delete_document(doc_id: str) -> None:
    """Remove a document record from the database."""
    async with get_db() as db:
        await db.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        await db.commit()
