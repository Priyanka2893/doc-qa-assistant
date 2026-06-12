# CLAUDE.md — AI Document Q&A Assistant
## Project Briefing for Claude Code

### Project Overview
A production-grade RAG (Retrieval-Augmented Generation) Document Q&A system. 
Users upload PDFs/text files and ask natural language questions. The backend retrieves 
relevant chunks and generates answers via Groq LLM.

### Current Status
- Phase: Bootstrap complete
- Backend: Not yet scaffolded (start with phase-01-backend-core.md)
- Frontend: Not yet built
- Vector DB: Qdrant running on localhost:6333

### Tech Stack
- Backend: Python 3.12, FastAPI, uv (package manager)
- Vector DB: Qdrant (Docker, localhost:6333)
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (local, CPU)
- LLM: Groq API (llama-3.3-70b-versatile)
- Document Parsing: pymupdf (fitz)
- Chunking: langchain-text-splitters RecursiveCharacterTextSplitter
- Frontend: Next.js 14, Tailwind CSS, shadcn/ui (Phase 2)
- Config: pydantic-settings, .env file
- Logging: structlog

### Project Structure
doc-qa-assistant/
├── CLAUDE.md ← this file
├── docker-compose.yml ← Qdrant
├── .env ← secrets (git-ignored)
├── backend/
│ ├── pyproject.toml ← uv project
│ ├── app/
│ │ ├── main.py ← FastAPI app entry
│ │ ├── config.py ← pydantic settings
│ │ ├── models.py ← pydantic request/response models
│ │ ├── routers/
│ │ │ ├── documents.py ← /upload endpoint
│ │ │ └── qa.py ← /ask endpoint
│ │ └── services/
│ │ ├── parser.py ← PDF/text extraction + chunking
│ │ ├── embedder.py ← sentence-transformer embeddings
│ │ ├── vector_store.py ← Qdrant operations
│ │ └── llm.py ← Groq API calls
│ └── tests/
└── frontend/ ← Phase 2

### Key Rules (Always Follow)
1. Always use `uv run` to execute Python commands in backend/
2. Read settings ONLY from pydantic-settings (no hardcoded values)
3. All endpoints must be async
4. Use structlog for logging (never print())
5. Handle all errors with proper HTTP status codes
6. Never commit .env — it's git-ignored
7. Chunk size: 500 tokens, overlap: 100 tokens
8. Qdrant collection name: from settings (QDRANT_COLLECTION_NAME)
9. Top-K retrieval: 5 chunks

### Running the Project
```bash
# Start Qdrant {#start-qdrant  data-source-line="280"}
docker compose up -d

# Start backend (from backend/ directory) {#start-backend-from-backend-directory  data-source-line="283"}
uv run uvicorn app.main:app --reload --port 8000

# API docs {#api-docs  data-source-line="286"}
open http://localhost:8000/docs
``` {data-source-line="288"}

### API Endpoints (Planned)
- POST /api/v1/documents/upload — Upload & ingest a document
- GET  /api/v1/documents — List all documents
- POST /api/v1/qa/ask — Ask a question
- GET  /api/v1/health — Health check

### Phase 1 — COMPLETE ✅
Backend core is fully implemented and tested.

**Completed endpoints:**
- POST /api/v1/documents/upload — file ingestion pipeline
- GET /api/v1/documents — list all documents
- DELETE /api/v1/documents/{doc_id} — delete document
- POST /api/v1/qa/ask — question answering
- GET /api/v1/health — health check

**Key files:**
- backend/app/main.py — FastAPI app entry point
- backend/app/config.py — settings (pydantic-settings)
- backend/app/services/embedder.py — SentenceTransformer singleton
- backend/app/services/vector_store.py — Qdrant operations
- backend/app/services/llm.py — Groq API integration

**Backend runs on:** http://localhost:8000
**API docs:** http://localhost:8000/docs
**Next phase:** Build React frontend (phase-02-frontend.md)

### Phase 2 — COMPLETE ✅
Next.js 14 frontend is implemented.

**Frontend stack:**
- Next.js 14 App Router, TypeScript
- Tailwind CSS + shadcn/ui components
- Dark theme (#0a0a0a background, #3b82f6 accent)

**Key components:**
- src/components/FileUpload.tsx — drag-drop upload with progress
- src/components/DocumentSidebar.tsx — document management
- src/components/ChatWindow.tsx — chat interface
- src/components/MessageBubble.tsx — messages with source citations
- src/lib/api.ts — typed API client

**Frontend runs on:** http://localhost:3000
**API proxy:** /api/backend/* → http://localhost:8000/api/v1/*
**Next phase:** Multi-document management (phase-03-document-management.md)

### Phase 3 — COMPLETE ✅
Multi-document management implemented.

**New capabilities:**
- Duplicate detection via SHA256 content hash
- Document status tracking (processing/ready/error)
- GET /api/v1/documents/{doc_id} — single document details
- POST /api/v1/qa/ask-global — cross-document search
- Frontend: global search mode, status badges, richer metadata display

**Data model addition:** content_hash column in SQLite documents table

### Phase 4 — COMPLETE ✅
Hybrid search and reranking implemented.

**New retrieval pipeline:**
- BM25 keyword index (in-memory, per-document, rebuilt on startup)
- Reciprocal Rank Fusion (RRF) to merge vector + BM25 results
- Cross-encoder reranking (cross-encoder/ms-marco-MiniLM-L-6-v2)
- Default mode: HYBRID with reranking enabled

**New files:**
- backend/app/services/bm25_store.py — BM25 index management
- backend/app/services/retriever.py — retrieval orchestrator

**AskRequest now accepts:** search_mode (vector|hybrid|keyword), rerank (bool)

### Phase 5 — COMPLETE ✅
Production hardening implemented.

**New capabilities:**
- Rate limiting via slowapi: upload 10/min, ask 60/min, ask-global 30/min (429 on exceed)
- Request ID middleware: UUID4 per request, X-Request-ID response header, contextvars for structlog
- Request logging middleware: structured logs with method, path, status, duration_ms, request_id
- Semantic answer cache: TTLCache(500 entries, 1h TTL), invalidated on document delete; AskResponse.cache_hit field
- Retry with tenacity: Groq API (3 attempts, exp backoff) + Qdrant connection errors
- File magic byte validation: python-magic MIME check before text extraction
- SSE streaming endpoint: POST /api/v1/qa/ask-stream — tokens arrive token-by-token
- Metrics endpoint: GET /api/v1/metrics — docs, chunks, cache, uptime, Qdrant points
- Liveness/readiness probes: /api/v1/health/live (always 200), /api/v1/health/ready (503 until ready)
- Frontend streaming: per-document queries use fetch+ReadableStream; tokens appear as they arrive

**New files:**
- backend/app/limiter.py — slowapi Limiter singleton
- backend/app/middleware/request_id.py — RequestIDMiddleware + request_id_var ContextVar
- backend/app/middleware/logging.py — RequestLoggingMiddleware
- backend/app/services/cache.py — SemanticCache with TTLCache



### Feature F6 — COMPLETE ✅
Advanced ingestion pipeline implemented.

**Supported formats:** .pdf, .txt, .docx, .html, .htm, .png, .jpg, .jpeg, .tiff
**New capabilities:**
- Multi-format parsing (DOCX via python-docx, HTML via BeautifulSoup, images/scanned PDFs via Tesseract OCR)
- Text normalization: whitespace, control chars, unicode
- Exact chunk deduplication (SHA256)
- Semantic chunk deduplication (cosine similarity > 0.95 threshold)
- Rich metadata: author, title, language, word_count, file_format

**New files:**
- backend/app/services/deduplicator.py — exact + semantic dedup
**Updated files:**
- backend/app/services/parser.py — multi-format router + normalizer
- backend/app/database.py — new metadata columns
- backend/app/models.py — IngestionReport, enriched DocumentInfo

**UploadResponse now includes:** ingestion_report, document_metadata

### Feature F7 — COMPLETE ✅
Source confidence scoring implemented.

**Scoring formula:** composite = 0.5*retrieval + 0.2*freshness + 0.2*authority + 0.1*agreement
**Trust levels:** verified(1.0), internal(0.85), external(0.65), unknown(0.50)
**Min threshold:** 0.40 (configurable via MIN_CONFIDENCE_THRESHOLD)

**New files:** backend/app/services/confidence_scorer.py
**New endpoint:** PATCH /api/v1/documents/{doc_id}/trust
**AskResponse now includes:** evidence_quality, avg_confidence, chunks_filtered_out
**ChunkSource now includes:** confidence_score, freshness_score, authority_score, agreement_score

### Feature F8 — COMPLETE ✅
Constrained generation and citation-backed responses implemented.

**Response modes:** cited (default), plain, strict_abstain
**Citation flow:** LLM outputs [Source N] tags → citation_parser.py maps to actual chunks
**Audit trail:** citation_audit table in SQLite logs every Q&A
**Temperature:** default 0.1 (deterministic), configurable per request

**New files:**
- backend/app/services/prompt_builder.py — all prompt templates live here
- backend/app/services/citation_parser.py — [Source N] tag parsing and mapping

**AskRequest new fields:** response_mode, temperature
**AskResponse new fields:** cited_sources, unmapped_citations, is_abstention, citation_coverage

### Feature F9 — COMPLETE ✅
Two-layer hallucination guard implemented.

**Layer 1 (pre-gen gate):** avg_confidence < PRE_GEN_CONFIDENCE_GATE (0.50) → block LLM call
**Layer 2 (post-gen verifier):** Jaccard token overlap per sentence → hallucination_risk score
**Action on high risk:** configurable "flag" or "block" (HALLUCINATION_ACTION setting)

**New file:** backend/app/services/hallucination_guard.py
**New table:** hallucination_events in SQLite
**New endpoint:** GET /api/v1/hallucination/stats
**AskResponse new fields:** hallucination_risk, is_high_risk, ungrounded_sentences, gate_passed
