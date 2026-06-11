# Ingestion Pipeline — Complete Code Walkthrough

## What is Ingestion?

When a user uploads a file, the system must:
1. Validate it is safe and supported
2. Extract text from it (regardless of format)
3. Clean and normalize the text
4. Split it into small searchable chunks
5. Remove duplicate chunks
6. Convert chunks to embedding vectors
7. Store everything in the database

This end-to-end process is called **ingestion**.

---

## Entry Point — `POST /api/v1/documents/upload`

**File:** `backend/app/routers/documents.py`

```python
@router.post("/documents/upload", response_model=UploadResponse, status_code=201)
@limiter.limit("10/minute")
async def upload_document(request: Request, file: UploadFile) -> UploadResponse:
```

This is the HTTP endpoint the user hits. It orchestrates every step below.

---

## Full Flow Diagram

```
User: POST /upload (file bytes)
         │
         ▼
[1] Extension check
    Is the extension in SUPPORTED_EXTENSIONS?
    No  → HTTP 400
         │
         ▼
[2] File size check
    Exceeds MAX_FILE_SIZE_MB?
    Yes → HTTP 413
         │
         ▼
[3] Duplicate document check (SHA256 of full file)
    Already in SQLite?
    Yes → HTTP 409 (already exists)
         │
         ▼
[4] Insert record in SQLite with status = "processing"
         │
         ▼
[5] parse_and_chunk()          ← parser.py
    ├── validate_file_type()   magic bytes check
    ├── format router          picks parser by extension
    ├── parser function        extracts raw text + metadata
    ├── normalize_text()       cleans whitespace/control chars
    ├── detect language        langdetect on first 500 chars
    ├── count words
    └── chunk_text()           RecursiveCharacterTextSplitter
         │
         ▼
[6] deduplicate_exact()        ← deduplicator.py
    SHA256 each chunk, drop exact copies
         │
         ▼
[7] deduplicate_semantic()     ← deduplicator.py
    Embed all chunks, drop near-duplicates (cosine ≥ 0.95)
         │
         ▼
[8] async_encode_texts()       ← embedder.py
    Convert final chunks to float vectors
         │
         ▼
[9] upsert_chunks()            ← vector_store.py
    Store vectors + metadata in Qdrant
         │
         ▼
[10] build_index()             ← bm25_store.py
     Build BM25 keyword index for hybrid search
         │
         ▼
[11] update_document_ingested() ← database.py
     Update SQLite record: status="ready", metadata, dedup stats
         │
         ▼
[12] Return UploadResponse
     chunk_count, ingestion_report, document_metadata
```

---

## Step 1 & 2 — Extension + Size Guard

**File:** `backend/app/routers/documents.py:33-45`

```python
ext = Path(filename).suffix.lower()
if ext not in SUPPORTED_EXTENSIONS:
    raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'.")

content = await file.read()
max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
if len(content) > max_bytes:
    raise HTTPException(status_code=413, detail=f"File exceeds {settings.MAX_FILE_SIZE_MB} MB limit.")
```

Fast-fail before any heavy processing.

---

## Step 3 — Whole-File Duplicate Detection

**File:** `backend/app/routers/documents.py:47-57`

```python
content_hash = hashlib.sha256(content).hexdigest()
existing = await database.get_document_by_hash(content_hash)
if existing:
    return JSONResponse(status_code=409, content={
        "detail": "Document already exists",
        "existing_doc_id": existing["doc_id"],
    })
```

SHA256 the entire file bytes. If the same file was uploaded before → return 409 immediately without re-processing.

---

## Step 4 — Insert with status = "processing"

**File:** `backend/app/routers/documents.py:59-66`

```python
doc_id = str(uuid.uuid4())
await database.insert_document(
    doc_id=doc_id,
    filename=filename,
    file_size_bytes=len(content),
    content_hash=content_hash,
    status="processing",
)
```

Record is created first so the document is visible (as "processing") even if ingestion takes time.

---

## Step 5 — parse_and_chunk()

**File:** `backend/app/services/parser.py`

This is the main parsing orchestrator.

```python
def parse_and_chunk(filename, content, chunk_size, chunk_overlap) -> ParseResult:
    ext = Path(filename).suffix.lower()
    validate_file_type(content, filename)   # magic bytes check
    parser_fn = SUPPORTED_EXTENSIONS[ext]   # pick the right parser
    raw_result = parser_fn(content)         # extract text + metadata
    cleaned_text = normalize_text(raw_result.text)
    raw_result.metadata.language = detect(cleaned_text[:500])
    raw_result.metadata.word_count = len(cleaned_text.split())
    chunks = chunk_text(cleaned_text, chunk_size, chunk_overlap)
    return ParseResult(text=cleaned_text, chunks=chunks, ...)
```

### 5a — Magic Byte Validation

```python
_EXT_MIME_MAP = {
    ".pdf":  {"application/pdf"},
    ".docx": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/zip"},
    ".png":  {"image/png"},
    ".jpg":  {"image/jpeg"},
    # ...
}

def validate_file_type(content: bytes, filename: str) -> None:
    detected = magic.from_buffer(content, mime=True)  # reads actual file bytes
    allowed = _EXT_MIME_MAP.get(ext, set())
    if detected not in allowed:
        raise HTTPException(400, detail=f"Content doesn't match extension. Detected: {detected}")
```

Why: a user could rename `malware.exe` to `report.pdf`. Magic bytes read the real file signature, not the name.

### 5b — Format Router

```python
SUPPORTED_EXTENSIONS = {
    ".pdf":  parse_pdf,
    ".txt":  parse_txt,
    ".docx": parse_docx,
    ".html": parse_html,
    ".htm":  parse_html,
    ".png":  parse_image,
    ".jpg":  parse_image,
    ".jpeg": parse_image,
    ".tiff": parse_image,
}
```

Each extension maps to a dedicated parser function. All return `RawParseResult(text, page_count, metadata)`.

### 5c — Per-Format Parsers

**PDF** (`parse_pdf`):
```python
def parse_pdf(content: bytes) -> RawParseResult:
    doc = fitz.open(stream=content, filetype="pdf")
    for page in doc:
        text = page.get_text()
        if not text.strip():                  # scanned / image-only page
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", ...)
            text = pytesseract.image_to_string(img)   # OCR fallback
        pages_text.append(text)
    # also extracts: author, title, creationDate from PDF metadata
```

If a PDF page has no digital text (it's a photo of a document), OCR reads it automatically.

**DOCX** (`parse_docx`):
```python
def parse_docx(content: bytes) -> RawParseResult:
    doc = docx.Document(io.BytesIO(content))
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    # reads core_properties: author, title, created date
```

**HTML** (`parse_html`):
```python
def parse_html(content: bytes) -> RawParseResult:
    soup = BeautifulSoup(content, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()                        # strip UI chrome
    text = soup.get_text(separator="\n", strip=True)
```

Removes navigation, footers, scripts — keeps only readable content.

**Images / PNG / JPG / TIFF** (`parse_image`):
```python
def parse_image(content: bytes) -> RawParseResult:
    image = Image.open(io.BytesIO(content))
    text = pytesseract.image_to_string(image, config="--psm 3")
    if len(text.strip()) < 20:
        raise HTTPException(422, "Could not extract text from image.")
```

Pure OCR — reads the image and returns the text found in it.

### 5d — Text Normalization

```python
def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")  # Windows/Mac line endings
    text = text.replace("\t", " ")                          # tabs → spaces
    text = re.sub(r"\n{3,}", "\n\n", text)                 # max 2 consecutive blank lines
    lines = [re.sub(r" {2,}", " ", line) for line in text.split("\n")]  # collapse spaces
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)      # strip control chars
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text
```

Raw extracted text from PDFs / images is messy. This irons it out before chunking.

### 5e — Chunking

```python
def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,      # 500 chars (from settings)
        chunk_overlap=chunk_overlap # 100 chars overlap
    )
    return splitter.split_text(text)
```

`RecursiveCharacterTextSplitter` tries to split on `\n\n`, then `\n`, then `.`, then spaces — to keep semantically coherent chunks rather than cutting mid-sentence.

---

## Step 6 — Exact Deduplication

**File:** `backend/app/services/deduplicator.py:27-38`

```python
def deduplicate_exact(chunks: list[str]) -> tuple[list[str], int]:
    seen: set[str] = set()
    unique: list[str] = []
    for chunk in chunks:
        h = hashlib.sha256(chunk.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(chunk)
    removed = len(chunks) - len(unique)
    return unique, removed
```

**Why:** Documents often repeat boilerplate (terms, headers, disclaimers). Storing duplicates wastes Qdrant space and biases retrieval toward repeated content.

**How:** SHA256 fingerprint each chunk. If we've seen that fingerprint → skip.

Example:
```
chunk_1: "All rights reserved. © Acme Corp."   → hash_A → keep
chunk_7: "All rights reserved. © Acme Corp."   → hash_A → DROP (exact duplicate)
```

---

## Step 7 — Semantic Deduplication

**File:** `backend/app/services/deduplicator.py:42-73`

```python
async def deduplicate_semantic(chunks, embedder, similarity_threshold=0.95):
    if len(chunks) <= 3:
        return chunks, 0            # not worth the compute

    embeddings = await loop.run_in_executor(None, embedder.encode_texts, chunks)

    kept_indices = [0]
    kept_vecs = [embeddings[0]]

    for i in range(1, len(chunks)):
        emb = embeddings[i]
        max_sim = max(_cosine(emb, k) for k in kept_vecs)
        if max_sim >= similarity_threshold:
            continue          # near-duplicate, skip
        kept_indices.append(i)
        kept_vecs.append(emb)

    return [chunks[i] for i in kept_indices], removed
```

**Why:** Exact dedup misses paraphrases. "The company was founded in 1990" and "The company was established in 1990" are different strings but convey identical meaning — redundant to store both.

**How:** Convert every chunk to a vector (numbers that encode meaning). Compare each new chunk to all already-kept chunks using **cosine similarity**. If similarity ≥ 0.95 (95% similar) → treat as near-duplicate → drop.

```python
def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)    # 1.0 = identical, 0.0 = unrelated
```

**Threshold 0.95:** Very strict — only drops chunks that are essentially saying the same thing.

---

## Step 8 — Embedding

**File:** `backend/app/services/embedder.py`

```python
embeddings = await async_encode_texts(settings.EMBEDDING_MODEL, chunks)
```

Converts each final (deduplicated) chunk into a list of floats — a vector in 384-dimensional space. Uses `sentence-transformers/all-MiniLM-L6-v2` locally on CPU.

---

## Step 9 — Upsert to Qdrant

**File:** `backend/app/services/vector_store.py`

```python
chunk_ids = await upsert_chunks(
    client=qdrant_client,
    collection_name=settings.QDRANT_COLLECTION_NAME,
    doc_id=doc_id,
    chunks=chunks,
    embeddings=embeddings,
    filename=filename,
    language=parse_result.metadata.language,
    doc_title=parse_result.metadata.title,
    author=parse_result.metadata.author,
)
```

Each chunk is stored as a **Qdrant point**: a vector + payload (text, doc_id, filename, language, author, title). This enables semantic search at query time.

---

## Step 10 — Build BM25 Index

**File:** `backend/app/services/bm25_store.py`

```python
get_bm25_store().build_index(doc_id=doc_id, chunks=chunks, chunk_ids=chunk_ids, filename=filename)
```

Builds an in-memory keyword (BM25) index for this document. Used by the hybrid retriever at search time (vector search + keyword search merged via Reciprocal Rank Fusion).

---

## Step 11 — Update SQLite Record

**File:** `backend/app/database.py`

```python
await database.update_document_ingested(
    doc_id=doc_id,
    chunk_count=len(chunks),
    page_count=parse_result.page_count,
    author=parse_result.metadata.author,
    doc_title=parse_result.metadata.title,
    language=parse_result.metadata.language,
    word_count=parse_result.metadata.word_count,
    file_format=parse_result.metadata.file_format,
    exact_dedup_removed=exact_removed,
    semantic_dedup_removed=semantic_removed,
)
```

Status changes from `"processing"` → `"ready"`. All metadata and dedup stats are persisted.

---

## Step 12 — Response

```json
{
  "doc_id": "d3f1a2b4-...",
  "filename": "report.pdf",
  "chunk_count": 42,
  "page_count": 8,
  "ingestion_time_ms": 1243,
  "ingestion_report": {
    "original_chunks": 50,
    "exact_dedup_removed": 3,
    "semantic_dedup_removed": 5,
    "final_chunks": 42,
    "dedup_rate": 0.16
  },
  "document_metadata": {
    "author": "Jane Smith",
    "doc_title": "Annual Report 2024",
    "language": "en",
    "word_count": 8423,
    "file_format": "pdf"
  }
}
```

`dedup_rate: 0.16` means 16% of chunks were removed as duplicates before storage.

---

## Error Handling

| Condition | HTTP Code | Where |
|-----------|-----------|-------|
| Unsupported extension | 400 | router + parser |
| Magic bytes mismatch | 400 | `validate_file_type()` |
| File too large | 413 | router |
| Same file already ingested | 409 | router (SHA256 check) |
| Empty / unparseable file | 422 | `parse_and_chunk()` |
| Image with no readable text | 422 | `parse_image()` |
| Any unexpected error | 500 | router catch-all |

On any failure after DB insert, the document status is set to `"error"`:
```python
except Exception as exc:
    await database.update_document_status(doc_id, "error")
    raise HTTPException(status_code=500, detail="Document ingestion failed.")
```

---

## File Map

| File | Responsibility |
|------|---------------|
| `routers/documents.py` | HTTP endpoint, orchestration, duplicate check |
| `services/parser.py` | Format routing, per-format parsers, normalization, chunking |
| `services/deduplicator.py` | Exact dedup (SHA256) + semantic dedup (cosine similarity) |
| `services/embedder.py` | SentenceTransformer — text → vector |
| `services/vector_store.py` | Qdrant upsert / delete |
| `services/bm25_store.py` | In-memory BM25 keyword index |
| `database.py` | SQLite document record CRUD |
