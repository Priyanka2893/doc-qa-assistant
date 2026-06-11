# Feature F7 — Source Confidence Scoring

## What It Does

Every chunk returned by the `/ask` (and `/ask-global`) endpoint now carries a **composite confidence score** (0–1) instead of a raw retrieval score. Chunks that score below a configurable threshold are silently filtered out before they are sent to the LLM. The API response also surfaces three summary fields: `evidence_quality`, `avg_confidence`, and `chunks_filtered_out`.

A separate `PATCH /api/v1/documents/{doc_id}/trust` endpoint lets callers assign a trust label (`verified`, `internal`, `external`, `unknown`) to any stored document, which then feeds into the scoring formula.

---

## Scoring Formula

```
composite = 0.50 × retrieval
          + 0.20 × freshness
          + 0.20 × authority
          + 0.10 × agreement
```

| Component   | Range | What it measures |
|-------------|-------|------------------|
| `retrieval` | 0–1   | Min-max normalised retrieval score from the reranker / RRF |
| `freshness` | 0–1   | Exponential decay: half-life = 180 days from upload date |
| `authority` | 0–1   | Mapped from document trust label (see table below) |
| `agreement` | 0–1   | Fraction of _other_ chunks whose cosine similarity ≥ 0.7 |

**Authority mapping** (`AUTHORITY_SCORES` in `confidence_scorer.py:15`):

| Trust label  | Score |
|-------------|-------|
| `verified`  | 1.00  |
| `internal`  | 0.85  |
| `external`  | 0.65  |
| `unknown`   | 0.50  |

**Evidence quality thresholds** (`summarize_evidence_quality` in `confidence_scorer.py:160`):

| avg composite | Label          |
|---------------|----------------|
| ≥ 0.80        | `high`         |
| ≥ 0.60        | `medium`       |
| ≥ 0.40        | `low`          |
| < 0.40        | `insufficient` |
| no chunks     | `none`         |

---

## End-to-End Request Flow

```
POST /api/v1/qa/ask
       │
       ▼
 routers/qa.py — ask_question()
       │
       ├─ 1. database.get_document()        check doc exists
       ├─ 2. cache.get_cached_answer()      return early if hit
       │
       ▼
 services/retriever.py — retrieve()
       │
       ├─ 3. async_encode_query()           embed the question
       ├─ 4. search_chunks() [Qdrant]       vector results
       ├─ 5. bm25_store.search()            keyword results
       ├─ 6. reciprocal_rank_fusion()       merge vector+BM25
       ├─ 7. _rerank()  [cross-encoder]     re-score pairs
       │
       ├─ 8. _build_doc_maps()
       │      ├─ database.get_document()    → uploaded_at
       │      └─ database.get_document_trust() → authority label
       │
       └─ 9. score_chunks()                ← THE SCORER
              │
              ├─ _normalize_retrieval_scores()
              ├─ _freshness_score()  (per chunk)
              ├─ _authority_score()  (per chunk)
              ├─ _compute_agreement_scores() [optional re-encode]
              ├─ composite = weighted sum
              └─ filter chunks < min_confidence
              returns list[ScoredChunk]
       │
       ▼
 routers/qa.py (continued)
       │
       ├─ 10. generate_answer()             LLM call with filtered chunks
       ├─ 11. _chunk_source_from_scored()   map ScoredChunk → ChunkSource
       ├─ 12. summarize_evidence_quality()  label from avg composite
       └─ 13. return AskResponse            with evidence_quality, avg_confidence, …
```

---

## File-by-File Code Walkthrough

### 1. `backend/app/config.py` — Settings

```python
# config.py:34
MIN_CONFIDENCE_THRESHOLD: float = 0.40
CONFIDENCE_WEIGHTS: dict[str, float] = {
    "retrieval": 0.50,
    "freshness": 0.20,
    "authority": 0.20,
    "agreement": 0.10,
}
```

Both values are read from `.env` (pydantic-settings). The router passes them directly into `retrieve()` so no logic is hardcoded. Override them in `.env` without touching code.

---

### 2. `backend/app/database.py` — Persistence

**Schema addition** (`database.py:30`):

```python
document_trust TEXT DEFAULT 'unknown'
```

The column is added via `_MIGRATIONS` so existing databases get it automatically on the next startup.

**Two new async helpers** (`database.py:180`):

```python
async def set_document_trust(doc_id: str, trust_level: str) -> None:
    # UPDATE documents SET document_trust = ? WHERE doc_id = ?

async def get_document_trust(doc_id: str) -> str:
    # SELECT document_trust FROM documents WHERE doc_id = ?
    # returns "unknown" if row missing
```

`get_document_trust` is called once per unique `doc_id` inside `_build_doc_maps()` in the retriever. The result feeds directly into `_authority_score()`.

---

### 3. `backend/app/models.py` — Pydantic Contracts

**`TrustUpdateRequest`** (`models.py:43`) — body of the PATCH endpoint:

```python
class TrustUpdateRequest(BaseModel):
    trust_level: str = Field(pattern="^(verified|internal|external|unknown)$")
```

Regex validation rejects any value outside the four allowed labels at the FastAPI layer before any DB call.

**`ChunkSource`** (`models.py:24`) — now carries the full confidence breakdown:

```python
class ChunkSource(BaseModel):
    ...
    confidence_score: float = 0.0   # composite
    freshness_score:  float = 0.0
    authority_score:  float = 0.0
    agreement_score:  float = 0.0
    retrieval_score:  float = 0.0
```

**`AskResponse`** (`models.py:82`) — three new summary fields:

```python
class AskResponse(BaseModel):
    ...
    evidence_quality:    str   = "none"
    avg_confidence:      float = 0.0
    chunks_filtered_out: int   = 0
```

**`DocumentInfo`** (`models.py:108`) — exposes trust to the document list endpoint:

```python
document_trust: str | None = "unknown"
```

---

### 4. `backend/app/services/confidence_scorer.py` — Core Scorer

This is the only new file introduced by F7.

#### Data classes

```python
@dataclass
class ConfidenceBreakdown:        # all four sub-scores + weights
    retrieval_score: float
    freshness_score: float
    authority_score: float
    agreement_score: float
    composite_score: float
    w_retrieval: float = 0.50
    ...

@dataclass
class ScoredChunk:                # wraps a chunk with its ConfidenceBreakdown
    text: str
    doc_id: str
    filename: str
    chunk_index: int
    page_number: int | None
    confidence: ConfidenceBreakdown
    vector_score: float | None
    bm25_score:   float | None
```

`ScoredChunk` replaces `RetrievalResult` everywhere after scoring. The retriever returns `list[ScoredChunk]` and the router consumes it directly.

#### `_freshness_score(uploaded_at)` — `confidence_scorer.py:48`

```python
age_days = (now - uploaded_at).total_seconds() / 86400.0
return math.exp(-math.log(2) * age_days / 180.0)
```

Exponential decay with a **180-day half-life**: a document uploaded today scores 1.0; at 180 days it scores 0.5; at 2 years it scores ≈ 0.078.

#### `_normalize_retrieval_scores(chunks)` — `confidence_scorer.py:60`

Min-max normalisation over all chunks in the batch so the raw scale (vector cosine, BM25 BM25 score, or cross-encoder logit) doesn't matter — only relative ranking does.

#### `_compute_agreement_scores(chunks, embedding_model)` — `confidence_scorer.py:68`

Re-encodes all chunk texts with the same SentenceTransformer, builds a pairwise cosine similarity matrix, and for each chunk counts the fraction of _other_ chunks with similarity > 0.7. A chunk that agrees with many peers is more trustworthy. Skipped if `embedding_model=""` or only one chunk.

#### `score_chunks(...)` — `confidence_scorer.py:93`

The main entry point called from the retriever:

1. Normalise retrieval scores across the batch.
2. Optionally compute agreement scores (async re-encode).
3. For each chunk: compute freshness, look up authority, compute composite.
4. Filter out any chunk with `composite < min_confidence`.
5. Sort survivors descending by composite.
6. Log total / passed / filtered.

```python
composite = w_r * r_score + w_f * f_score + w_a * a_score + w_ag * ag_score
passed = [sc for sc in scored if sc.confidence.composite_score >= min_confidence]
```

#### `summarize_evidence_quality(scored_chunks)` — `confidence_scorer.py:160`

```python
avg = mean(composite scores)
return "high" if avg >= 0.80 else "medium" if avg >= 0.60 else "low" if avg >= 0.40 else "insufficient"
```

Called in the router after scoring to populate `AskResponse.evidence_quality`.

---

### 5. `backend/app/services/retriever.py` — Integration Point

`_build_doc_maps()` (`retriever.py:114`) — runs **after** reranking, before scoring:

```python
async def _build_doc_maps(candidates):
    for doc_id in {c.doc_id for c in candidates}:
        doc = await database.get_document(doc_id)
        uploaded_at_map[doc_id] = datetime.fromisoformat(doc["uploaded_at"])
        authority_map[doc_id]   = await database.get_document_trust(doc_id)
    return uploaded_at_map, authority_map
```

One DB call per unique document, not per chunk.

`retrieve()` (`retriever.py:131`) and `retrieve_global()` (`retriever.py:208`) both end with:

```python
uploaded_at_map, authority_map = await _build_doc_maps(final_candidates)
scored = await score_chunks(
    chunks=final_candidates,
    uploaded_at_map=uploaded_at_map,
    authority_map=authority_map,
    min_confidence=min_confidence,
    embedding_model=embedding_model,
    weights=confidence_weights,
)
return RetrieveOutput(chunks=scored, filtered_out=pre_filter_count - len(scored))
```

`RetrieveOutput` is a `NamedTuple(chunks: list[ScoredChunk], filtered_out: int)`.

---

### 6. `backend/app/routers/documents.py` — Trust Endpoint

```python
# documents.py:168
@router.patch("/documents/{doc_id}/trust")
async def set_trust_level(doc_id: str, body: TrustUpdateRequest) -> dict:
    doc = await database.get_document(doc_id)
    if not doc:
        raise HTTPException(404, ...)
    await database.set_document_trust(doc_id, body.trust_level)
    return {"doc_id": doc_id, "trust_level": body.trust_level}
```

No rate limit because it is a low-frequency admin operation. Returns the new state for easy confirmation.

---

### 7. `backend/app/routers/qa.py` — Response Assembly

Two helpers defined at the top:

```python
# qa.py:20
def _chunk_source_from_scored(r: ScoredChunk) -> ChunkSource:
    return ChunkSource(
        ...
        confidence_score=r.confidence.composite_score,
        freshness_score= r.confidence.freshness_score,
        authority_score= r.confidence.authority_score,
        agreement_score= r.confidence.agreement_score,
        retrieval_score= r.confidence.retrieval_score,
    )

def _avg_confidence(chunks: list[ScoredChunk]) -> float:
    return round(sum(c.confidence.composite_score for c in chunks) / len(chunks), 4)
```

Inside `ask_question()` (`qa.py:42`):

```python
output = await retrieve(..., min_confidence=settings.MIN_CONFIDENCE_THRESHOLD, ...)
results            = output.chunks          # list[ScoredChunk], already filtered
chunks_filtered_out = output.filtered_out

sources          = [_chunk_source_from_scored(r) for r in results]
evidence_quality = summarize_evidence_quality(results)

return AskResponse(
    ...
    evidence_quality=evidence_quality,
    avg_confidence=_avg_confidence(results),
    chunks_filtered_out=chunks_filtered_out,
)
```

`ask_question_global()` follows the same pattern using `retrieve_global()`.

---

### 8. `backend/tests/test_confidence.py` — Tests

| Test | What it verifies |
|------|-----------------|
| `test_freshness_decay` | A 2-year-old document gets freshness < 0.5 |
| `test_authority_scoring` | `verified` trust → `authority_score == 1.0` |
| `test_low_confidence_filter` | Low-retrieval chunk (normalised to 0) is filtered at threshold 0.40 |
| `test_evidence_quality_high` | Fresh + verified + top-score → composite ≥ 0.80 → label `high` |

All tests are `async` (marked `@pytest.mark.anyio`) because `score_chunks` is an `async` function.

---

## API Changes Summary

### New Endpoint

```
PATCH /api/v1/documents/{doc_id}/trust
Body:  { "trust_level": "verified" | "internal" | "external" | "unknown" }
Response: { "doc_id": "...", "trust_level": "verified" }
```

### Modified Response: `AskResponse`

```json
{
  "answer": "...",
  "sources": [
    {
      "chunk_index": 2,
      "text_excerpt": "...",
      "score": 0.831,
      "confidence_score": 0.831,
      "freshness_score": 0.974,
      "authority_score": 1.0,
      "agreement_score": 0.25,
      "retrieval_score": 0.912,
      "vector_score": 0.85,
      "bm25_score": 12.3,
      "page_number": 4
    }
  ],
  "evidence_quality": "high",
  "avg_confidence": 0.831,
  "chunks_filtered_out": 1,
  "cache_hit": false,
  "model": "llama-3.3-70b-versatile",
  "tokens_used": 512,
  "doc_id": "abc-123"
}
```

### Modified Response: `DocumentInfo` (GET /documents)

```json
{
  "doc_id": "...",
  "document_trust": "verified",
  ...
}
```

---

## Configuration Reference

Add to `.env` to override defaults:

```env
MIN_CONFIDENCE_THRESHOLD=0.40   # drop chunks below this
# Weights must sum to 1.0
CONFIDENCE_WEIGHTS='{"retrieval":0.50,"freshness":0.20,"authority":0.20,"agreement":0.10}'
```
