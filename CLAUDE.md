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
Two-layer hallucination guard implemented, with post-gen verifier upgraded to a 2-stage hybrid.

**Layer 1 (pre-gen gate):** avg_confidence < PRE_GEN_CONFIDENCE_GATE (0.50) → block LLM call
**Layer 2 (post-gen verifier):** 2-stage per-sentence grounding check:
- Stage 1 — token fast path: token containment ≥ POST_GEN_TOKEN_FAST_PATH (0.60) → grounded immediately, no embedding
- Stage 2 — semantic cosine fallback: sentences failing stage 1 are batch-embedded via the existing SentenceTransformer singleton; cosine similarity ≥ POST_GEN_OVERLAP_THRESHOLD (0.50) → grounded
- Note: semantic cosine cannot detect negation ("Patient has no diabetes" vs "Patient has diabetes" scores ~0.90). For production, replace Stage 2 with an NLI cross-encoder or LLM-as-judge run async off the response path.

**Action on high risk:** configurable "flag" or "block" (HALLUCINATION_ACTION setting)

**New file:** backend/app/services/hallucination_guard.py
**New table:** hallucination_events in SQLite
**New endpoint:** GET /api/v1/hallucination/stats
**AskResponse new fields:** hallucination_risk, is_high_risk, ungrounded_sentences, gate_passed
**SentenceVerification new field:** grounding_method ("token" | "semantic" | "ungrounded") — indicates which stage made the grounding decision
**Config:** POST_GEN_TOKEN_FAST_PATH (0.60) — token containment floor for stage 1; POST_GEN_OVERLAP_THRESHOLD (0.50) — semantic cosine threshold for stage 2

### Feature F10 — COMPLETE ✅
Continuous evaluation pipeline implemented.

**Scoring formula:** overall = 0.30*context_relevance + 0.40*faithfulness + 0.30*answer_relevance

**Metric computation — 2-stage hybrid (mirrors F9):**
- context_relevance: per-chunk question-token recall; if recall < 0.50 → semantic cosine fallback via SentenceTransformer; final = max(token, semantic); return avg across chunks
- faithfulness: 1.0 - hallucination_risk (directly from F9 VerificationResult — already semantically grounded)
- answer_relevance: question-token recall against answer; if recall < 0.50 → semantic cosine fallback; final = max(token, semantic); abstentions score 0.85
- Token fast path threshold: 0.50 (same spirit as F9's POST_GEN_TOKEN_FAST_PATH)
- All low-scoring texts (chunks + answer) are batch-embedded in a single encoder call

**Why 2-stage matters:** pure token recall scores "refund timeline" vs "5-7 business days" as 0.50 (only "refund" matches). After semantic fallback the cosine similarity rescues the paraphrase and scores it ~0.85+.

**New files:**
- backend/app/services/evaluator.py — compute_faithfulness (sync), evaluate_response (async, 2-stage hybrid)
- backend/app/routers/eval.py — eval endpoints
- backend/app/eval_runner.py — standalone CLI benchmark runner

**New endpoints:**
- GET /api/v1/eval/summary?hours=24 — aggregated EvalSummary for last N hours
- GET /api/v1/eval/document/{doc_id} — per-document EvalSummary
- POST /api/v1/eval/benchmark — run keyword-recall benchmark suite against a document

**New table:** eval_results in SQLite
**AskResponse new field:** eval_metrics (EvalMetrics | None) — computed after every successful query
**CLI usage:** uv run python -m app.eval_runner --doc_id=... --questions_file=... [--base_url=...]
**CI exit code:** 0 if avg_overall >= 0.70, else 1

### Feature F11 — COMPLETE ✅
Full observability stack implemented.

**Tracing:** OpenTelemetry spans on every pipeline stage
**Metrics:** Prometheus (GET /metrics) — stage latencies, quality scores, cache, errors
**Dashboard:** Grafana at http://localhost:3001 (admin/admin) — pre-provisioned RAG dashboard
**Log-trace correlation:** trace_id injected into every structlog message

**New files:** backend/app/telemetry.py, monitoring/prometheus.yml, monitoring/grafana/provisioning/...
**Docker services added:** rag_prometheus (9090), rag_grafana (3001)

cat >> ~/projects/doc-qa-assistant/CLAUDE.md << 'EOF'

### Feature F12 — COMPLETE ✅
Semantic cache and session memory implemented.

**Cache:** Exact (SHA256) + Semantic (cosine > 0.97), 500-entry TTL cache, invalidated on doc delete
**Session memory:** Multi-turn conversations with 30-min expiry, context injected into retrieval
**Corrections:** Expert corrections override cache and retrieval for similar questions

**New files:**
- backend/app/services/session_memory.py
- backend/app/services/corrections.py

**Updated:** backend/app/services/cache.py (full semantic cache upgrade)
**New endpoints:** POST /sessions, GET /sessions/{id}, POST /corrections, GET /cache/stats

**AskRequest new field:** session_id
**AskResponse new fields:** cache_hit, cache_hit_type, session_id, is_correction

### 🎉 ALL FEATURES COMPLETE — Production-Grade RAG System
