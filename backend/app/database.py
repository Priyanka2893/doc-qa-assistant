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
    semantic_dedup_removed INTEGER DEFAULT 0,
    document_trust TEXT DEFAULT 'unknown'
);
"""

_CREATE_HALLUCINATION_EVENTS_SQL = """
CREATE TABLE IF NOT EXISTS hallucination_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    question TEXT NOT NULL,
    gate_blocked INTEGER DEFAULT 0,
    gate_avg_confidence REAL,
    post_gen_risk REAL,
    ungrounded_count INTEGER DEFAULT 0,
    action_taken TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
"""

_CREATE_EVAL_RESULTS_SQL = """
CREATE TABLE IF NOT EXISTS eval_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    question_preview TEXT,
    context_relevance REAL,
    faithfulness REAL,
    answer_relevance REAL,
    overall_score REAL,
    chunk_count_used INTEGER,
    is_abstention INTEGER,
    hallucination_risk REAL,
    created_at TEXT DEFAULT (datetime('now'))
);
"""

_CREATE_CITATION_AUDIT_SQL = """
CREATE TABLE IF NOT EXISTS citation_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer_preview TEXT,
    citation_count INTEGER DEFAULT 0,
    unmapped_count INTEGER DEFAULT 0,
    is_abstention INTEGER DEFAULT 0,
    citation_coverage REAL DEFAULT 0,
    evidence_quality TEXT,
    created_at TEXT DEFAULT (datetime('now'))
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
    "ALTER TABLE documents ADD COLUMN document_trust TEXT DEFAULT 'unknown'",
]


async def init_db() -> None:
    """Create the data directory and documents table if they don't exist."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(_DB_PATH) as db:
        await db.execute(_CREATE_TABLE_SQL)
        await db.execute(_CREATE_CITATION_AUDIT_SQL)
        await db.execute(_CREATE_HALLUCINATION_EVENTS_SQL)
        await db.execute(_CREATE_EVAL_RESULTS_SQL)
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
    exact_dedup_removed, semantic_dedup_removed, document_trust
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


async def set_document_trust(doc_id: str, trust_level: str) -> None:
    async with get_db() as db:
        await db.execute(
            "UPDATE documents SET document_trust = ? WHERE doc_id = ?",
            (trust_level, doc_id),
        )
        await db.commit()


async def get_document_trust(doc_id: str) -> str:
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT document_trust FROM documents WHERE doc_id = ?",
            (doc_id,),
        )
        row = await cursor.fetchone()
        return row["document_trust"] if row else "unknown"


async def insert_hallucination_event(
    request_id: str,
    doc_id: str,
    question: str,
    gate_blocked: int,
    gate_avg_confidence: float,
    post_gen_risk: float,
    ungrounded_count: int,
    action_taken: str,
) -> None:
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO hallucination_events
                (request_id, doc_id, question, gate_blocked,
                 gate_avg_confidence, post_gen_risk, ungrounded_count, action_taken)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (request_id, doc_id, question, gate_blocked,
             gate_avg_confidence, post_gen_risk, ungrounded_count, action_taken),
        )
        await db.commit()


async def get_hallucination_stats() -> dict:
    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT
                COUNT(*) AS total_queries,
                SUM(gate_blocked) AS gate_blocked_count,
                SUM(CASE WHEN action_taken = 'flagged' THEN 1 ELSE 0 END) AS high_risk_count,
                AVG(post_gen_risk) AS avg_hallucination_risk
            FROM hallucination_events
            """
        )
        row = await cursor.fetchone()
        stats = dict(row) if row else {}

        cursor2 = await db.execute(
            """
            SELECT id, request_id, doc_id, question, gate_blocked, gate_avg_confidence,
                   post_gen_risk, ungrounded_count, action_taken, created_at
            FROM hallucination_events
            ORDER BY created_at DESC
            LIMIT 10
            """
        )
        recent = [dict(r) for r in await cursor2.fetchall()]

    return {
        "total_queries": stats.get("total_queries") or 0,
        "gate_blocked_count": stats.get("gate_blocked_count") or 0,
        "high_risk_count": stats.get("high_risk_count") or 0,
        "avg_hallucination_risk": round(stats.get("avg_hallucination_risk") or 0.0, 4),
        "recent_events": recent,
    }


async def insert_eval_result(
    request_id: str,
    doc_id: str,
    question: str,
    context_relevance: float,
    faithfulness: float,
    answer_relevance: float,
    overall_score: float,
    chunk_count_used: int,
    is_abstention: bool,
    hallucination_risk: float,
) -> None:
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO eval_results
                (request_id, doc_id, question_preview, context_relevance, faithfulness,
                 answer_relevance, overall_score, chunk_count_used, is_abstention, hallucination_risk)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id, doc_id, question[:200],
                context_relevance, faithfulness, answer_relevance, overall_score,
                chunk_count_used, int(is_abstention), hallucination_risk,
            ),
        )
        await db.commit()


async def get_eval_summary(hours: int = 24) -> dict:
    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT
                COUNT(*) AS query_count,
                AVG(context_relevance) AS avg_context_relevance,
                AVG(faithfulness) AS avg_faithfulness,
                AVG(answer_relevance) AS avg_answer_relevance,
                AVG(overall_score) AS avg_overall_score,
                AVG(CAST(is_abstention AS REAL)) AS abstention_rate,
                AVG(CASE WHEN hallucination_risk >= 0.40 THEN 1.0 ELSE 0.0 END) AS high_risk_rate
            FROM eval_results
            WHERE created_at >= datetime('now', ?)
            """,
            (f"-{hours} hours",),
        )
        row = await cursor.fetchone()
    row = dict(row) if row else {}
    return {
        "query_count": row.get("query_count") or 0,
        "avg_context_relevance": round(row.get("avg_context_relevance") or 0.0, 4),
        "avg_faithfulness": round(row.get("avg_faithfulness") or 0.0, 4),
        "avg_answer_relevance": round(row.get("avg_answer_relevance") or 0.0, 4),
        "avg_overall_score": round(row.get("avg_overall_score") or 0.0, 4),
        "abstention_rate": round(row.get("abstention_rate") or 0.0, 4),
        "high_risk_rate": round(row.get("high_risk_rate") or 0.0, 4),
        "time_window_hours": hours,
    }


async def get_doc_eval_summary(doc_id: str) -> dict:
    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT
                COUNT(*) AS query_count,
                AVG(context_relevance) AS avg_context_relevance,
                AVG(faithfulness) AS avg_faithfulness,
                AVG(answer_relevance) AS avg_answer_relevance,
                AVG(overall_score) AS avg_overall_score,
                AVG(CAST(is_abstention AS REAL)) AS abstention_rate,
                AVG(CASE WHEN hallucination_risk >= 0.40 THEN 1.0 ELSE 0.0 END) AS high_risk_rate
            FROM eval_results
            WHERE doc_id = ?
            """,
            (doc_id,),
        )
        row = await cursor.fetchone()
    row = dict(row) if row else {}
    return {
        "query_count": row.get("query_count") or 0,
        "avg_context_relevance": round(row.get("avg_context_relevance") or 0.0, 4),
        "avg_faithfulness": round(row.get("avg_faithfulness") or 0.0, 4),
        "avg_answer_relevance": round(row.get("avg_answer_relevance") or 0.0, 4),
        "avg_overall_score": round(row.get("avg_overall_score") or 0.0, 4),
        "abstention_rate": round(row.get("abstention_rate") or 0.0, 4),
        "high_risk_rate": round(row.get("high_risk_rate") or 0.0, 4),
        "time_window_hours": 0,
    }


async def insert_citation_audit(
    request_id: str,
    doc_id: str,
    question: str,
    answer_preview: str,
    citation_count: int,
    unmapped_count: int,
    is_abstention: bool,
    citation_coverage: float,
    evidence_quality: str,
) -> None:
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO citation_audit
                (request_id, doc_id, question, answer_preview,
                 citation_count, unmapped_count, is_abstention,
                 citation_coverage, evidence_quality)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id, doc_id, question, answer_preview[:500],
                citation_count, unmapped_count, int(is_abstention),
                citation_coverage, evidence_quality,
            ),
        )
        await db.commit()
