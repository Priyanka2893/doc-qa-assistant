# Backend — Every Line Explained (Beginner Friendly)

This file explains every single line of every `.py` file in `backend/app/`.
Written as if you have never seen these libraries before.

---

## THE BIG PICTURE FIRST

Before diving into code, understand what this app does:

1. **You upload a PDF** → the app reads it, cuts it into small pieces (chunks),
   converts each piece into a list of numbers (called a vector/embedding),
   and stores everything in two databases.

2. **You ask a question** → the app converts your question into numbers too,
   finds the 5 most similar chunks in the database, sends them + your question
   to an AI (Groq), and returns the AI's answer.

That's it. Everything in the code exists to do these two things.

---

## HOW FILES RELATE TO EACH OTHER

```
main.py  ← starts the app, connects everything
  │
  ├── config.py        ← reads your .env file (API keys, settings)
  ├── database.py      ← SQLite database (stores doc names/sizes/counts)
  │
  ├── routers/
  │   ├── documents.py ← handles /upload and /documents URLs
  │   ├── qa.py        ← handles /ask URL
  │   └── health.py    ← handles /health URL
  │
  └── services/
      ├── parser.py       ← reads PDF/TXT files, splits into chunks
      ├── embedder.py     ← converts text to number-vectors
      ├── vector_store.py ← talks to Qdrant (vector database)
      └── llm.py          ← talks to Groq AI
```

---

---

# FILE 1: `config.py`

This file reads settings from your `.env` file and makes them available everywhere.

```python
from functools import lru_cache
```
`functools` is a standard Python library (comes built-in).
`lru_cache` is a tool inside it. LRU = "Least Recently Used".
It memorizes the result of a function so the function body doesn't run again on future calls.
We use it so settings are read from disk only once, not on every request.

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
```
`pydantic_settings` is an installed library (added in pyproject.toml).
`BaseSettings` = a special base class that can read values from environment variables and `.env` files automatically.
`SettingsConfigDict` = a helper to tell BaseSettings where to find the `.env` file.

---

```python
class Settings(BaseSettings):
```
We create a class called `Settings` that INHERITS from `BaseSettings`.
Inheriting means our class gets all the powers of `BaseSettings` for free.
The power here: Python will automatically read values from `.env` and put them into this class.

```python
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
```
`model_config` is a special class variable that pydantic reads.
`env_file=".env"` → "look for a file called `.env`" (in the same folder you run the server from).
`env_file_encoding="utf-8"` → read the file as UTF-8 text (standard text encoding).

---

```python
    GROQ_API_KEY: str
```
This means: "there must be a line `GROQ_API_KEY=abc123` in `.env`".
The type hint `: str` means "this must be a string".
There is no default value → if it's missing from `.env`, the app will crash on startup (intentional — you can't run without an API key).

```python
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
```
Same idea, but this one has a default value `"llama-3.3-70b-versatile"`.
If you don't put `GROQ_MODEL=...` in `.env`, this default is used.

```python
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
```
The name of the local AI model that converts text to vectors.
Default is "all-MiniLM-L6-v2" — a small, fast, free model that runs on your CPU.

```python
    EMBEDDING_DIMENSION: int = 384
```
This model produces vectors of exactly 384 numbers.
Different models produce different sizes. This must match the model you chose.

```python
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
```
Where Qdrant (the vector database) is running.
`localhost` means "on this same computer".
`6333` is Qdrant's default port.

```python
    QDRANT_COLLECTION_NAME: str = "documents"
```
In Qdrant, data is organized into "collections" (similar to "tables" in SQL).
This is the name of our collection. Can be anything — we chose "documents".

```python
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
```
When splitting a document into chunks:
- Each chunk is at most 500 characters long.
- Neighboring chunks share 100 characters.
  The overlap prevents losing meaning when a sentence is cut at a boundary.

```python
    TOP_K_RESULTS: int = 5
```
When searching for relevant chunks, return the top 5 matches.

```python
    MAX_FILE_SIZE_MB: int = 50
```
Reject uploaded files bigger than 50 MB.

```python
    APP_ENV: str = "development"
```
A label. You'd set this to "production" when deploying.
Useful for showing extra debug info only in development.

```python
    LOG_LEVEL: str = "INFO"
```
Controls how much logging output you see.
INFO = normal messages. DEBUG = everything including tiny details.

```python
    ALLOWED_ORIGINS: str = "http://localhost:3000"
```
The frontend URL that is allowed to call this backend.
CORS (browser security) blocks other websites from calling your API — this is the exception list.

---

```python
@lru_cache()
def get_settings() -> Settings:
    return Settings()
```
`@lru_cache()` is a "decorator" — it wraps the function with extra behavior.
Here it means: "run this function once, remember the result, and return the cached result every time after".

`def get_settings()` → a simple function named `get_settings`.
`-> Settings` → type hint: "this function returns a Settings object".
`return Settings()` → create a new Settings object (which reads the `.env` file) and return it.

Because of `@lru_cache()`:
- First call: reads `.env`, creates Settings, caches it.
- Every later call: returns the already-created Settings without reading `.env` again.

---

---

# FILE 2: `models.py`

This file defines the "shape" of data sent to and from the API.
Think of them as forms — they define which fields exist and what type each field is.
FastAPI uses these to automatically validate incoming data and format outgoing responses.

```python
from pydantic import BaseModel, Field
```
`pydantic` is a library for data validation.
`BaseModel` = base class for all our data shapes.
`Field` = adds extra rules to a field (like min/max length, min/max value).

---

```python
class ChunkSource(BaseModel):
```
A class representing one "source" shown under the AI answer.
Each source is one chunk of text that helped answer the question.

```python
    chunk_index: int
```
Which number chunk this is (0 = first chunk, 1 = second chunk, etc.).

```python
    text_excerpt: str
```
The first 300 characters of the chunk text — shown to the user as proof/citation.

```python
    score: float
```
How similar this chunk was to the question. Float between 0 and 1.
1.0 = perfect match, 0.0 = completely unrelated.

```python
    page_number: int | None
```
Which PDF page this chunk came from. Can be `None` (for .txt files that have no pages).
`int | None` means "either an integer or nothing".

---

```python
class UploadResponse(BaseModel):
```
What the server sends back after you upload a document.

```python
    doc_id: str
```
A unique ID we generated for this document (like "a3f2-8e19-...").
You need this ID to later ask questions about the document.

```python
    filename: str
```
The original filename you uploaded (e.g., "annual_report.pdf").

```python
    chunk_count: int
```
How many pieces the document was split into.

```python
    page_count: int
```
How many pages the PDF had (always 1 for .txt files).

```python
    status: str = "success"
```
A string field with a default value of "success". Since we only return this when successful, it's always "success".

```python
    ingestion_time_ms: int
```
How many milliseconds the whole upload process took (parsing + embedding + storing).

---

```python
class AskRequest(BaseModel):
```
The data you must send in the body of a POST request to `/ask`.

```python
    question: str = Field(min_length=3, max_length=1000)
```
Your question. `Field(min_length=3, max_length=1000)` means:
- Must be at least 3 characters (prevents empty questions).
- Must be at most 1000 characters.
FastAPI automatically rejects requests that violate these rules with a 422 error.

```python
    document_id: str
```
The `doc_id` you got when you uploaded the document.
This tells us WHICH document to search in.

```python
    top_k: int = Field(default=5, ge=1, le=20)
```
How many chunks to retrieve. Default is 5.
`ge=1` means "greater than or equal to 1" (can't ask for 0 chunks).
`le=20` means "less than or equal to 20" (can't ask for 1000 chunks).

---

```python
class AskResponse(BaseModel):
```
What the server sends back after you ask a question.

```python
    answer: str
```
The AI's answer text.

```python
    sources: list[ChunkSource]
```
A list of ChunkSource objects — the chunks the AI used to answer.
`list[ChunkSource]` = "a list where every item is a ChunkSource".

```python
    model: str
```
Which AI model was used (e.g., "llama-3.3-70b-versatile").

```python
    tokens_used: int
```
How many tokens the AI processed. Groq bills by tokens.

```python
    doc_id: str
```
Which document was searched (echoed back to the client).

---

```python
class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    page_count: int
    uploaded_at: str
```
Used when listing all uploaded documents.
`uploaded_at` is a string like "2025-06-08 10:32:11" — SQLite stores dates as text.

---

```python
class HealthResponse(BaseModel):
    status: str        # "ok" or "degraded"
    qdrant: str        # "ok" or "unreachable"
    embedding_model: str
    version: str       # app version like "1.0.0"
```
What the health check endpoint returns.

---

---

# FILE 3: `database.py`

This file handles SQLite — a simple database stored as a single file on disk.
SQLite stores document metadata: names, sizes, chunk counts, upload times.
The actual text/vectors go into Qdrant (a different database).

```python
import os
```
Standard Python library. Gives access to operating system features.
(Actually not heavily used here — likely a leftover import.)

```python
from contextlib import asynccontextmanager
```
`contextlib` is a standard Python library.
`asynccontextmanager` is a decorator that lets you write an `async with` helper using `yield`.
We use it for `get_db()` so the database connection is always closed after use.

```python
from pathlib import Path
```
`Path` is a modern way to work with file paths in Python.
Much nicer than string concatenation like `"folder" + "/" + "file.db"`.

```python
from typing import AsyncGenerator
```
`AsyncGenerator` is a type hint. It means "a function that yields values asynchronously".
Used in the type signature of `get_db()`.

```python
import aiosqlite
```
An installed library. The regular `sqlite3` library blocks (freezes) while waiting for disk.
`aiosqlite` is the async version — while waiting for SQLite, other requests can run.

```python
import structlog
```
An installed library for structured logging.
Instead of `print("done")`, we write `logger.info("done")` which outputs nicely formatted log lines.

```python
logger = structlog.get_logger(__name__)
```
Creates a logger tied to this file's name.
`__name__` is a Python built-in that equals the current module name (e.g., `"app.database"`).
Having the module name in logs helps you know WHICH file produced each log line.

---

```python
_DB_PATH = Path(__file__).parent.parent / "data" / "documents.db"
```
Let's break this down step by step:
- `__file__` = full path to this file, e.g. `/projects/backend/app/database.py`
- `Path(__file__)` = wrap it as a Path object
- `.parent` = go up one level → `/projects/backend/app`
- `.parent` again = go up one more level → `/projects/backend`
- `/ "data"` = append "data" → `/projects/backend/data`
- `/ "documents.db"` = append filename → `/projects/backend/data/documents.db`

The `_` prefix on `_DB_PATH` is a Python convention meaning "this is private, don't import it".
This path is computed ONCE when the module loads and reused everywhere.

---

```python
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_size_bytes INTEGER,
    page_count INTEGER,
    chunk_count INTEGER NOT NULL,
    uploaded_at TEXT NOT NULL DEFAULT (datetime('now')),
    status TEXT NOT NULL DEFAULT 'ready'
);
"""
```
This is just a SQL string stored as a Python variable.
`"""..."""` = triple quotes = multi-line string.

The SQL creates a table called `documents` with these columns:
- `doc_id TEXT PRIMARY KEY` → text column, must be unique (like a primary key in any DB)
- `filename TEXT NOT NULL` → cannot be empty
- `file_size_bytes INTEGER` → can be NULL (optional)
- `page_count INTEGER` → can be NULL
- `chunk_count INTEGER NOT NULL` → required
- `uploaded_at TEXT NOT NULL DEFAULT (datetime('now'))` → auto-fills with current time if not provided
- `status TEXT NOT NULL DEFAULT 'ready'` → defaults to "ready"
- `IF NOT EXISTS` → don't crash if table already exists — safe to run on every startup

---

```python
async def init_db() -> None:
```
`async def` = this is an async function (must be called with `await`).
`-> None` = returns nothing.

```python
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
```
`_DB_PATH.parent` = the `data/` folder.
`.mkdir(parents=True, exist_ok=True)`:
- `parents=True` = also create parent folders if they don't exist
- `exist_ok=True` = don't error if the folder already exists

```python
    async with aiosqlite.connect(_DB_PATH) as db:
```
`async with` = open something and automatically close it when done (like Python's `with open(...)`).
`aiosqlite.connect(_DB_PATH)` = open (or create) the SQLite file.
`as db` = name the connection `db` so we can use it inside the block.

```python
        await db.execute(_CREATE_TABLE_SQL)
```
Run the SQL command to create the table.
`await` = wait for the async operation to finish before continuing.

```python
        await db.commit()
```
SQLite requires you to call `commit()` to save changes to disk.
Without this, changes exist only in memory and are lost when the connection closes.

```python
    logger.info("database.initialized", path=str(_DB_PATH))
```
Log a message. `"database.initialized"` is the event name. `path=str(_DB_PATH)` is extra context shown alongside.
`str(_DB_PATH)` = convert the Path object to a string for display.

---

```python
@asynccontextmanager
async def get_db() -> AsyncGenerator[aiosqlite.Connection, None]:
```
`@asynccontextmanager` = makes this usable as `async with get_db() as db:`.
`AsyncGenerator[aiosqlite.Connection, None]` = type hint saying "this yields a Connection object".

```python
    async with aiosqlite.connect(_DB_PATH) as db:
```
Open the database connection.

```python
        db.row_factory = aiosqlite.Row
```
By default, SQLite returns rows as plain tuples: `(value1, value2, ...)`.
Setting `row_factory = aiosqlite.Row` makes rows behave like dictionaries.
After this, you can do `row["doc_id"]` instead of `row[0]`.

```python
        yield db
```
`yield` = "here is the database connection — let the caller use it".
After the caller is done (the `async with` block ends), Python comes back here and runs any code after `yield` (there is none — the `async with aiosqlite.connect` closes it automatically).

---

```python
async def insert_document(
    doc_id: str,
    filename: str,
    file_size_bytes: int,
    page_count: int,
    chunk_count: int,
) -> None:
```
A function to save a new document record. Takes 5 arguments, returns nothing.

```python
    async with get_db() as db:
```
Use our helper to get a database connection.

```python
        await db.execute(
            """
            INSERT INTO documents (doc_id, filename, file_size_bytes, page_count, chunk_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            (doc_id, filename, file_size_bytes, page_count, chunk_count),
        )
```
Insert a new row. 
The `?` are placeholders — SQLite fills them in with the values in the tuple `(doc_id, filename, ...)`.
Why placeholders? Safety! If you wrote `f"VALUES ('{doc_id}')"`, a malicious doc_id could break the query (SQL injection). Placeholders prevent this.

```python
        await db.commit()
```
Save the insertion to disk.

---

```python
async def list_documents() -> list[dict]:
```
Returns a list of dictionaries — one dict per document row.

```python
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT doc_id, filename, chunk_count, page_count, uploaded_at FROM documents ORDER BY uploaded_at DESC"
        )
```
`cursor` = an object that holds the query results (like a pointer to the results).
`ORDER BY uploaded_at DESC` = newest documents first.
`SELECT` lists specific columns instead of `SELECT *` to only fetch what we need.

```python
        rows = await cursor.fetchall()
```
`fetchall()` = retrieve ALL rows at once into a list.

```python
        return [dict(row) for row in rows]
```
`dict(row)` = convert each Row object (which behaves like a dict) to a plain Python dict.
`[dict(row) for row in rows]` = list comprehension = do this for every row.

---

```python
async def get_document(doc_id: str) -> dict | None:
```
Returns one document dict, or `None` if not found.
`dict | None` = "either a dict or None" (Python 3.10+ union type syntax).

```python
        cursor = await db.execute(
            "SELECT ... FROM documents WHERE doc_id = ?",
            (doc_id,),
        )
```
`WHERE doc_id = ?` = only get the row where doc_id matches.
`(doc_id,)` = a tuple with one element. The trailing comma makes it a tuple (not just parentheses around a value).

```python
        row = await cursor.fetchone()
```
`fetchone()` = get just one row (or None if no rows match).

```python
        return dict(row) if row else None
```
"If row exists, convert and return it. Otherwise return None."

---

```python
async def delete_document(doc_id: str) -> None:
    async with get_db() as db:
        await db.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        await db.commit()
```
Delete the row with this doc_id. Then commit to save the deletion.

---

---

# FILE 4: `services/parser.py`

This file reads PDF and TXT files, extracts their text, and splits the text into chunks.

```python
import fitz  # pymupdf
```
`fitz` is the import name for the `pymupdf` library.
PyMuPDF can read PDFs, extract text, images, and more.
The comment `# pymupdf` clarifies that `fitz` = pymupdf (the names don't match!).

```python
import structlog
```
For logging (same as before).

```python
from fastapi import HTTPException
```
`HTTPException` = a special exception that FastAPI catches and turns into an HTTP error response.
Example: `raise HTTPException(status_code=400, detail="bad request")` → API returns `{"detail": "bad request"}` with status 400.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```
`langchain_text_splitters` is an installed library (from LangChain project).
`RecursiveCharacterTextSplitter` = a smart text splitter that tries to split at:
1. Paragraphs (double newline `\n\n`) first
2. Then sentences (single newline `\n`)
3. Then spaces (` `)
4. Then individual characters (last resort)
This keeps chunks as semantically meaningful as possible.

```python
logger = structlog.get_logger(__name__)
```
Create a logger for this file. Same pattern as database.py.

---

```python
def extract_text_from_pdf(content: bytes) -> tuple[str, int]:
```
`def` (not `async def`) = regular synchronous function, no `await` needed.
`content: bytes` = raw binary file data (not a file path, not a string — bytes).
`-> tuple[str, int]` = returns a tuple of (text string, page count).

```python
    doc = fitz.open(stream=content, filetype="pdf")
```
Open the PDF from bytes in memory.
`stream=content` = "read from this bytes object, not from a file path".
`filetype="pdf"` = hint to fitz about what format it is.
`doc` is now a fitz Document object representing the PDF.

```python
    pages = [page.get_text() for page in doc]
```
Loop over every page in the PDF, extract text from each page.
`page.get_text()` = returns a string of all text on that page.
`[... for page in doc]` = list comprehension = build a list.
Result: `pages` is a list of strings, one per page.

```python
    doc.close()
```
Release the memory used by the PDF. Always close what you open.

```python
    return "\n".join(pages), len(pages)
```
`"\n".join(pages)` = join all page texts into one big string, with newline between pages.
`len(pages)` = number of pages.
Returns both values as a tuple: `(full_text, page_count)`.

---

```python
def extract_text_from_txt(content: bytes) -> tuple[str, int]:
```
Same signature as the PDF extractor.

```python
    return content.decode("utf-8", errors="replace"), 1
```
`content.decode("utf-8")` = convert bytes to a Python string using UTF-8 encoding.
`errors="replace"` = if a byte sequence can't be decoded, replace with `?` instead of crashing.
Returns `(text, 1)` — text files always have 1 "page".

---

```python
def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
```
Takes the full document text and returns a list of chunk strings.

```python
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
```
Create a splitter object with our settings.
`length_function=len` = use Python's built-in `len()` function to measure chunk size in characters.
(Alternative would be measuring in tokens, but character count is simpler and fast.)

```python
    return splitter.split_text(text)
```
Do the actual splitting. Returns a `list[str]`.
Example: A 5000-char document with chunk_size=500 and overlap=100 →
roughly 11 chunks (each 500 chars, but neighboring chunks share 100 chars).

---

```python
def parse_and_chunk(
    filename: str,
    content: bytes,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[str], int]:
```
The main function that routers call. Combines extraction + chunking.
Returns `(list_of_chunks, page_count)`.

```python
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
```
This extracts the file extension. Let's decode it:
- `if "." in filename` = check if there's any dot in the filename
- `filename.rsplit(".", 1)` = split the filename on `.` from the RIGHT, max 1 split
  - `"report.final.pdf".rsplit(".", 1)` → `["report.final", "pdf"]`
  - Regular `.split(".", 1)` → `["report", "final.pdf"]` ← wrong!
- `[-1]` = take the last element → `"pdf"`
- `.lower()` = make lowercase so `".PDF"` and `".pdf"` both work
- If no `.` in filename → use `""` as extension (will trigger the error below)

```python
    if ext == "pdf":
        text, page_count = extract_text_from_pdf(content)
    elif ext == "txt":
        text, page_count = extract_text_from_txt(content)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. Allowed: .pdf, .txt",
        )
```
Route to the right extractor based on file extension.
If neither PDF nor TXT → raise an HTTPException which FastAPI turns into a 400 error response.
`f"..."` = f-string = Python's way to embed variables in strings: `f"type '.{ext}'"` → `"type '.docx'"`.

```python
    if not text.strip():
        raise HTTPException(status_code=422, detail="File is empty or contains no extractable text.")
```
`text.strip()` = remove whitespace from both ends of the string.
`not text.strip()` = True if the stripped text is empty (empty string is falsy in Python).
This catches PDFs that are images (scanned) — fitz can't read image-based PDFs, so text comes back empty.
422 = "Unprocessable Entity" — the request format was correct but content is invalid.

```python
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    logger.info(
        "parser.chunked",
        filename=filename,
        page_count=page_count,
        chunk_count=len(chunks),
    )
    return chunks, page_count
```
Split the text into chunks, log the result, return.

---

---

# FILE 5: `services/embedder.py`

This file converts text into vectors (lists of numbers).
The same sentence always produces the same vector.
Similar sentences produce similar vectors — this is the magic behind semantic search.

```python
import asyncio
```
Standard Python library for async programming.
We use it here to run the CPU-heavy embedding model in a background thread.

```python
import time
```
Standard Python library. Used to measure how long the model takes to load.

```python
from functools import lru_cache
```
Same caching decorator as in config.py. Used to load the model only once.

```python
import structlog
from sentence_transformers import SentenceTransformer
```
`sentence_transformers` is an installed library.
`SentenceTransformer` = a class that loads a pre-trained AI model and can encode text to vectors.

---

```python
class EmbedderService:
    """Singleton wrapper around a SentenceTransformer model."""
```
A class that wraps the AI model and provides clean methods to use it.
"Singleton" = only one instance ever exists (achieved via `lru_cache`).

```python
    def __init__(self, model_name: str) -> None:
```
`__init__` = constructor. Called automatically when you do `EmbedderService("model-name")`.
`model_name: str` = the name of the model to load (e.g., "all-MiniLM-L6-v2").

```python
        start = time.perf_counter()
```
`time.perf_counter()` = returns the current time in seconds, with very high precision.
Used to calculate elapsed time. We save the start time.

```python
        self._model = SentenceTransformer(model_name)
```
`self._model` = store the model as an instance variable (accessible anywhere in the class via `self._model`).
`SentenceTransformer(model_name)` = download and load the model (takes a few seconds on first run).
The `_` prefix on `_model` means "private — don't access this from outside the class".

```python
        elapsed_ms = int((time.perf_counter() - start) * 1000)
```
`time.perf_counter() - start` = seconds elapsed since `start`.
`* 1000` = convert to milliseconds.
`int(...)` = round down to a whole number (drop the decimal).

```python
        logger.info("embedder.loaded", model=model_name, load_time_ms=elapsed_ms)
```
Log that the model loaded, including which model and how long it took.

---

```python
    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts into embedding vectors."""
```
Takes a list of strings. Returns a list of vectors (each vector is a list of floats).
`list[str]` = a list where every element is a string.
`list[list[float]]` = a list of lists of floating-point numbers.

```python
        vectors = self._model.encode(
            texts,
            batch_size=32,
```
`batch_size=32` = process 32 texts at a time.
Why not all at once? If you have 1000 chunks, encoding all at once could use too much RAM.
Batching processes 32 at a time, uses less memory.

```python
            convert_to_tensor=False,
```
"Don't return PyTorch Tensors — return numpy arrays instead."
Numpy arrays are easier to work with and we convert them to Python lists next.

```python
            show_progress_bar=False,
        )
```
"Don't print a loading bar to the console." (We use structured logs instead.)

```python
        return [v.tolist() for v in vectors]
```
`v.tolist()` = convert each numpy array to a plain Python list of floats.
Qdrant expects plain Python lists, not numpy arrays.
Result: `[[0.1, 0.3, ...], [0.2, 0.8, ...], ...]` — one list of 384 numbers per text.

---

```python
    def encode_query(self, query: str) -> list[float]:
        """Encode a single query string into an embedding vector."""
        vector = self._model.encode(
            query,
            convert_to_tensor=False,
            show_progress_bar=False,
        )
        return vector.tolist()
```
Same as `encode_texts` but for a single string.
Returns just one vector (not a list of vectors).
Used when encoding a question (query) during the ask flow.

---

```python
    @property
    def model_name(self) -> str:
        return str(self._model.model_card_data.base_model or "unknown")
```
`@property` = makes this method accessible like an attribute: `embedder.model_name` (no parentheses).
Returns the model's name as a string.
`... or "unknown"` = if `base_model` is None/empty, fall back to "unknown".

---

```python
@lru_cache(maxsize=1)
def get_embedder(model_name: str) -> EmbedderService:
    """Return (and cache) the singleton EmbedderService instance."""
    return EmbedderService(model_name)
```
`@lru_cache(maxsize=1)` = cache only 1 result (we only use one model).
First call with `"all-MiniLM-L6-v2"` → loads the model, returns EmbedderService, caches it.
Every subsequent call with the same argument → returns the cached EmbedderService immediately.
This ensures the model is loaded only ONCE for the lifetime of the server process.

---

```python
async def async_encode_texts(model_name: str, texts: list[str]) -> list[list[float]]:
    """Run encode_texts in a thread executor to keep the event loop unblocked."""
```
This is the function routers actually call (the async wrapper).

```python
    loop = asyncio.get_event_loop()
```
`asyncio.get_event_loop()` = get the currently running async event loop.
FastAPI runs in an async event loop that handles all requests. We need a reference to it.

```python
    embedder = get_embedder(model_name)
```
Get the cached EmbedderService (loads the model if first time).

```python
    return await loop.run_in_executor(None, embedder.encode_texts, texts)
```
`run_in_executor(None, func, *args)` = run `func(*args)` in a background thread.
`None` = use the default thread pool executor.

WHY: `encode_texts` is CPU-heavy — it takes seconds. If we called it directly in an async function,
it would FREEZE the entire server (no other requests could be handled while it runs).
`run_in_executor` moves it to a background thread, so the event loop stays free to handle other requests.
`await` = wait for the background thread to finish, then continue.

---

```python
async def async_encode_query(model_name: str, query: str) -> list[float]:
    """Run encode_query in a thread executor to keep the event loop unblocked."""
    loop = asyncio.get_event_loop()
    embedder = get_embedder(model_name)
    return await loop.run_in_executor(None, embedder.encode_query, query)
```
Identical idea, but wraps `encode_query` (single string → single vector).
Used in the ask flow to encode the user's question.

---

---

# FILE 6: `services/vector_store.py`

This file talks to Qdrant — the database that stores vectors.
Qdrant is NOT SQL. You can't query it with SELECT. Instead you give it a vector, and it finds similar vectors.

```python
import uuid
```
Standard Python library. `uuid.uuid4()` generates a random unique identifier.
Example: `"a3f29e10-7b2d-4c1a-8e72-1234567890ab"`.

```python
import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models
```
`AsyncQdrantClient` = async version of the Qdrant client (doesn't block while waiting for Qdrant).
`qdrant_models` = data classes for Qdrant structures (VectorParams, Filter, etc.).
We alias it as `qdrant_models` to avoid writing the full path every time.

---

```python
async def init_collection(
    client: AsyncQdrantClient,
    collection_name: str,
    dimension: int,
) -> None:
```
Creates the Qdrant collection if it doesn't exist. Called once on startup.

```python
    existing = await client.collection_exists(collection_name)
```
Ask Qdrant: "does a collection with this name already exist?"
Returns `True` or `False`.

```python
    if not existing:
        await client.create_collection(
            collection_name=collection_name,
```
If the collection doesn't exist yet, create it.

```python
            vectors_config=qdrant_models.VectorParams(
                size=dimension,
```
`size=dimension` = each vector has this many numbers (384 for our model).
This is set in stone when the collection is created — all vectors must be this size.

```python
                distance=qdrant_models.Distance.COSINE,
```
The similarity metric. COSINE measures the angle between two vectors.
Used in recommendation systems, search engines, document similarity.
`COSINE` score: 1.0 = identical direction (same meaning), 0.0 = 90° apart (unrelated), -1.0 = opposite.

---

```python
async def upsert_chunks(
    client: AsyncQdrantClient,
    collection_name: str,
    doc_id: str,
    chunks: list[str],
    embeddings: list[list[float]],
    filename: str,
    page_numbers: list[int | None] | None = None,
) -> None:
```
Save chunks and their vectors to Qdrant.
`page_numbers: list[int | None] | None = None` = optional. Can be:
- `None` = page numbers not provided (e.g., for .txt files)
- A list like `[1, 1, 2, 2, 3]` = which page each chunk came from

```python
    points = [
        qdrant_models.PointStruct(
```
Build a list of `PointStruct` objects. Each point = one chunk.
`PointStruct` is Qdrant's data class for a single vector entry.

```python
            id=str(uuid.uuid4()),
```
Each point needs a unique ID. We generate a random UUID for each chunk.
`str(...)` = convert UUID object to string (Qdrant accepts string IDs).

```python
            vector=embeddings[i],
```
The 384-float vector for this chunk.

```python
            payload={
                "text": chunks[i],
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": i,
                "page_number": page_numbers[i] if page_numbers else None,
            },
```
`payload` = metadata stored alongside the vector.
When we search and get results back, we read these values.
- `"text"` = the actual chunk text (what we send to the AI)
- `"doc_id"` = used to filter searches to one document
- `"chunk_index"` = position in the document (0, 1, 2...)
- `"page_number"` = for source citations in the UI
`page_numbers[i] if page_numbers else None` = "if page_numbers list was provided, get index i; otherwise None".

```python
        )
        for i in range(len(chunks))
    ]
```
`for i in range(len(chunks))` = loop from 0 to number_of_chunks - 1.
This is a list comprehension — builds the entire list of PointStructs in one expression.

```python
    await client.upsert(collection_name=collection_name, points=points)
```
`upsert` = "insert or update". If a point with the same ID exists, update it. Otherwise insert.
Sends ALL points to Qdrant in one network call (efficient).

---

```python
async def search_chunks(
    client: AsyncQdrantClient,
    collection_name: str,
    query_vector: list[float],
    doc_id: str,
    top_k: int,
) -> list[qdrant_models.ScoredPoint]:
```
Find the most similar chunks to the query vector.
Returns a list of `ScoredPoint` objects (vector + score + payload).

```python
    response = await client.query_points(
        collection_name=collection_name,
        query=query_vector,
```
`query=query_vector` = the 384-float vector of the user's question.
Qdrant computes cosine similarity between this and every stored vector.

```python
        query_filter=qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="doc_id",
                    match=qdrant_models.MatchValue(value=doc_id),
                )
            ]
        ),
```
This is a filter — only search chunks belonging to this specific document.
`must` = all conditions in this list must be true.
`FieldCondition` = checks a payload field.
`key="doc_id"` = check the `doc_id` field in the payload.
`MatchValue(value=doc_id)` = it must exactly equal our doc_id.
Without this filter, the search would return results from ALL documents.

```python
        limit=top_k,
```
Return at most `top_k` results (e.g., 5).

```python
        with_payload=True,
```
Include the payload (metadata) in the results.
If `False`, you'd get vectors back but no text/doc_id/page_number — useless for us.

```python
    )
    return response.points
```
`response.points` = the list of `ScoredPoint` objects.
Each has `.score` (similarity), `.payload` (dict with text/doc_id/etc), `.vector` (the numbers).

---

```python
async def delete_document_chunks(
    client: AsyncQdrantClient,
    collection_name: str,
    doc_id: str,
) -> None:
    await client.delete(
        collection_name=collection_name,
        points_selector=qdrant_models.FilterSelector(
            filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="doc_id",
                        match=qdrant_models.MatchValue(value=doc_id),
                    )
                ]
            )
        ),
    )
```
Delete ALL points where `payload.doc_id == doc_id`.
`FilterSelector` = "select points matching this filter for deletion".
Same filter structure as search — it matches on the `doc_id` payload field.

---

```python
async def count_collection(client: AsyncQdrantClient, collection_name: str) -> int:
    result = await client.count(collection_name=collection_name, exact=True)
    return result.count
```
`exact=True` = count precisely (vs. approximate count which is faster but less accurate).
`result.count` = the integer count.
Used only by the health check to verify Qdrant is responding.

---

---

# FILE 7: `services/llm.py`

This file sends the retrieved chunks and the user's question to the Groq AI and gets an answer back.

```python
import groq
```
The official Groq Python SDK (installed library).

```python
import structlog
from fastapi import HTTPException
```
For logging and error handling.

---

```python
_SYSTEM_PROMPT = (
    "You are an expert document analyst. Answer questions based ONLY on the provided document "
    "excerpts. If the answer is not in the context, say 'I couldn't find information about that "
    "in the document.' Never make up information."
)
```
A string stored in a module-level variable (created once, reused forever).
`_SYSTEM_PROMPT` starts with `_` = module-private (don't import from other files).

This is the "system message" — instructions to the AI about how to behave.
It tells the AI:
1. Only use the provided text to answer
2. If you don't know, say so
3. Never invent facts (important for a document Q&A — hallucination would be misleading)

The parentheses around the string are just for line-wrapping — Python automatically joins adjacent strings.

---

```python
async def generate_answer(
    question: str,
    chunks: list[str],
    model: str,
    api_key: str,
) -> dict:
```
The main function. Takes a question + list of chunk texts + model name + API key.
Returns a dict with keys: `answer`, `tokens_used`, `model`.

```python
    context = "\n\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(chunks))
```
Let's decode this carefully:

`enumerate(chunks)` = iterate with index and value → `(0, "chunk text"), (1, "chunk text"), ...`

`for i, chunk in enumerate(chunks)` = for each (index, text) pair...

`f"[{i + 1}] {chunk}"` = format as `"[1] first chunk text"`, `"[2] second chunk text"`, etc.
`i + 1` because we want labels starting at 1, not 0.

`"\n\n".join(...)` = join all formatted chunks with a blank line between them.

Result looks like:
```
[1] The company was founded in 1990 by John Smith...

[2] Revenue grew by 15% in the last fiscal year...

[3] The main product is a cloud-based software...
```

```python
    user_message = f"Document excerpts:\n{context}\n\nQuestion: {question}"
```
Build the full user message by combining:
- The label "Document excerpts:"
- The context block (all 5 chunks)
- A blank line
- "Question:" + the actual question

```python
    client = groq.AsyncGroq(api_key=api_key)
```
Create a Groq API client using the API key from settings.
`AsyncGroq` = the async version (doesn't block the server while waiting for Groq's response).

```python
    try:
        response = await client.chat.completions.create(
```
`try:` = attempt this block, if it fails jump to `except`.
`await` = wait for Groq to respond.
`chat.completions.create` = the method to generate a chat completion (standard LLM API pattern).

```python
            model=model,
```
Which Groq model to use. Comes from settings: `"llama-3.3-70b-versatile"`.

```python
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
```
Chat models expect a list of messages, each with a `role` and `content`:
- `"system"` = background instructions for the AI (not part of the conversation)
- `"user"` = what the user said (our combined chunks + question)
- (there could also be `"assistant"` for previous AI replies in multi-turn chats)

```python
            temperature=0.1,
```
Controls randomness/creativity:
- `0.0` = completely deterministic (same input → same output every time)
- `1.0` = very creative/random
- `0.1` = almost deterministic with tiny variation. Good for factual Q&A where consistency matters.

```python
            max_tokens=1024,
```
Maximum length of the AI's response in tokens.
1 token ≈ 0.75 words. So 1024 tokens ≈ ~750 words.
This prevents extremely long responses and controls API costs.

```python
    except groq.RateLimitError as exc:
        logger.warning("llm.rate_limit", error=str(exc))
        raise HTTPException(status_code=429, detail="LLM rate limit exceeded. Please retry shortly.")
```
`RateLimitError` = you've sent too many requests to Groq in a short time.
429 = "Too Many Requests" — standard HTTP code, tells the client to wait and retry.

```python
    except groq.APIError as exc:
        logger.error("llm.api_error", error=str(exc))
        raise HTTPException(status_code=502, detail="LLM API error. Please try again later.")
```
`APIError` = Groq had an internal server error.
502 = "Bad Gateway" — our server got a bad response from an upstream service (Groq).
`logger.error` (vs `.warning`) = more severe — shows up more prominently in logs.

```python
    answer = response.choices[0].message.content or ""
```
`response.choices` = a list of possible responses. We always use the first one (`[0]`).
`.message.content` = the actual text response from the AI.
`or ""` = if content is somehow `None`, use empty string instead (avoids errors later).

```python
    tokens_used = response.usage.total_tokens if response.usage else 0
```
`response.usage` = token usage statistics. Could theoretically be `None`.
`if response.usage else 0` = use the count if available, otherwise 0.
`total_tokens` = input tokens + output tokens combined.

```python
    logger.info("llm.generated", model=model, tokens_used=tokens_used)
    return {"answer": answer, "tokens_used": tokens_used, "model": model}
```
Log success, then return a plain dict (the router will use the values to build an `AskResponse`).

---

---

# FILE 8: `main.py`

This is the app entry point. It creates the FastAPI app, connects all the pieces, and defines startup/shutdown logic.

```python
import time
```
For timing request durations in the middleware.

```python
from contextlib import asynccontextmanager
```
To define the lifespan function (startup + shutdown logic).

```python
from pathlib import Path
```
For creating the `data/` directory on startup.

```python
import structlog
from fastapi import FastAPI, Request, Response
```
`FastAPI` = the main application class.
`Request` = represents an incoming HTTP request (gives access to headers, body, app state, etc.).
`Response` = represents an outgoing HTTP response.

```python
from fastapi.middleware.cors import CORSMiddleware
```
CORS middleware: allows the browser to call our API from a different origin (our frontend).

```python
from qdrant_client import AsyncQdrantClient
```
To create the Qdrant connection during startup.

```python
from app.config import get_settings
from app.database import init_db
from app.routers import documents, health, qa
from app.services.embedder import get_embedder
from app.services.vector_store import init_collection
```
Import our own code from other files in the project.
`from app.routers import documents, health, qa` = import all three router modules.

---

```python
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
```
Configure the logging system. `20` = logging.INFO level (from Python's standard `logging` module).
This means DEBUG messages (level 10) are hidden; INFO and above are shown.

```python
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
```
Add an ISO timestamp (like `"2025-06-08T10:32:11.123Z"`) to every log entry.

```python
        structlog.processors.add_log_level,
```
Add the log level name (`"info"`, `"error"`, etc.) to every log entry.

```python
        structlog.processors.StackInfoRenderer(),
```
If an exception is logged, include the full stack trace.

```python
        structlog.dev.ConsoleRenderer(),
```
Format logs for human reading in the terminal (colors, aligned columns).
(In production you'd use `JSONRenderer()` for log aggregation tools.)

---

```python
logger = structlog.get_logger(__name__)
```
Create a logger for main.py.

---

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
```
`@asynccontextmanager` = makes this an async context manager.
FastAPI calls this function and uses `yield` as the dividing line:
- Code before `yield` = runs BEFORE the server starts accepting requests (startup).
- Code after `yield` = runs AFTER the server stops (shutdown).

`app: FastAPI` = the FastAPI application object is passed in so we can store things on `app.state`.

```python
    settings = get_settings()
    app.state.settings = settings
```
Load settings from `.env`.
`app.state` = a generic storage namespace on the FastAPI app.
Storing settings here makes them available to ALL requests via `request.app.state.settings`.
Without this, every router would have to call `get_settings()` directly (still works due to caching, but this is cleaner).

```python
    Path(__file__).parent.parent.joinpath("data").mkdir(parents=True, exist_ok=True)
```
Create the `backend/data/` folder if it doesn't exist.
`Path(__file__)` = `/path/to/backend/app/main.py`.
`.parent.parent` = `/path/to/backend/`.
`.joinpath("data")` = `/path/to/backend/data`.
`.mkdir(parents=True, exist_ok=True)` = create it (and any missing parent folders), don't error if it exists.

```python
    await init_db()
```
Create the SQLite table if it doesn't exist.

```python
    qdrant_client = AsyncQdrantClient(
        host=settings.QDRANT_HOST, port=settings.QDRANT_PORT
    )
```
Create the Qdrant connection. This doesn't actually connect yet — it just sets up the client object.

```python
    await init_collection(qdrant_client, settings.QDRANT_COLLECTION_NAME, settings.EMBEDDING_DIMENSION)
```
Connect to Qdrant and create the "documents" collection if it doesn't exist.

```python
    app.state.qdrant_client = qdrant_client
```
Store the Qdrant client on `app.state` so routers can access it via `request.app.state.qdrant_client`.

```python
    get_embedder(settings.EMBEDDING_MODEL)
```
Pre-load the embedding model NOW (at startup) rather than on the first request.
Loading takes ~5-10 seconds. Without this, the first upload would be very slow.
We don't store the return value because `get_embedder` caches it internally via `lru_cache`.

```python
    logger.info(
        "app.started",
        env=settings.APP_ENV,
        qdrant=f"{settings.QDRANT_HOST}:{settings.QDRANT_PORT}",
        collection=settings.QDRANT_COLLECTION_NAME,
    )
```
Log that startup completed successfully. Very useful for confirming everything initialized.

```python
    yield
```
The dividing line. Everything above runs at startup. Everything below runs at shutdown.
After `yield`, the app is running and accepting requests.

```python
    await qdrant_client.close()
    logger.info("app.shutdown")
```
When the server is stopped (Ctrl+C), close the Qdrant connection cleanly.

---

```python
app = FastAPI(
    title="Doc Q&A Assistant",
    description="RAG-powered document question answering via Groq LLM",
    version="1.0.0",
    lifespan=lifespan,
)
```
Create the FastAPI application.
`title` and `description` appear in the auto-generated docs at `/docs`.
`lifespan=lifespan` = "use our lifespan function for startup/shutdown".

---

```python
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.ALLOWED_ORIGINS.split(",")],
```
`settings.ALLOWED_ORIGINS` = `"http://localhost:3000"` (from `.env`).
`.split(",")` = split by comma → `["http://localhost:3000"]` (or multiple if you added more).
`[o.strip() for o in ...]` = strip whitespace from each origin (safety against `" http://..."` with a space).
Result: a list of allowed origins.

```python
    allow_credentials=True,
```
Allow cookies and Authorization headers to be sent cross-origin (needed for auth).

```python
    allow_methods=["*"],
```
Allow all HTTP methods (GET, POST, DELETE, OPTIONS, etc.).

```python
    allow_headers=["*"],
```
Allow all HTTP headers.

---

```python
@app.middleware("http")
async def log_requests(request: Request, call_next) -> Response:
```
`@app.middleware("http")` = register this function as middleware that wraps every HTTP request.
`call_next` = a function that runs the actual endpoint handler and returns the response.
Every request → enters `log_requests` → `call_next` runs the real handler → log results → return.

```python
    t_start = time.perf_counter()
```
Record start time.

```python
    response = await call_next(request)
```
Run the actual endpoint handler (e.g., `upload_document` or `ask_question`).
We `await` it because FastAPI endpoints are async.

```python
    elapsed_ms = int((time.perf_counter() - t_start) * 1000)
    logger.info(
        "http.request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=elapsed_ms,
    )
    return response
```
Calculate elapsed time, log the request details, then return the response to the client.
This produces log lines like: `method=POST path=/api/v1/documents/upload status=201 duration_ms=1823`.

---

```python
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(documents.router, prefix="/api/v1", tags=["documents"])
app.include_router(qa.router, prefix="/api/v1", tags=["qa"])
```
Register each router's endpoints with the main app.
`prefix="/api/v1"` = prepend `/api/v1` to all routes in that router.
So `@router.get("/health")` in health.py becomes `GET /api/v1/health` in the app.
`tags=[...]` = groups endpoints in the Swagger docs at `/docs`.

---

---

# FILE 9: `routers/documents.py`

Handles three endpoints: upload, list, and delete documents.

```python
import time
import uuid
```
`time` for measuring ingestion duration. `uuid` for generating unique document IDs.

```python
import structlog
from fastapi import APIRouter, HTTPException, Request, UploadFile
```
`APIRouter` = a mini-app for grouping related routes. Gets merged into the main app.
`UploadFile` = FastAPI's special type for file uploads. Handles multipart form data automatically.

```python
from app import database
from app.models import DocumentInfo, UploadResponse
from app.services.embedder import async_encode_texts
from app.services.parser import parse_and_chunk
from app.services.vector_store import delete_document_chunks, upsert_chunks
```
Import from our own code. Notice we import just the specific functions/classes we need.

```python
logger = structlog.get_logger(__name__)
router = APIRouter()
```
Create the router. Routes defined below use this router.

```python
_ALLOWED_EXTENSIONS = {".pdf", ".txt"}
```
A Python `set` (curly braces with no key-value pairs = set, not dict).
Sets have O(1) lookup — checking `ext in _ALLOWED_EXTENSIONS` is instant regardless of size.

---

## POST /api/v1/documents/upload

```python
@router.post("/documents/upload", response_model=UploadResponse, status_code=201)
async def upload_document(request: Request, file: UploadFile) -> UploadResponse:
```
`@router.post` = register this function as the handler for POST requests to `/documents/upload`.
`response_model=UploadResponse` = FastAPI will convert the returned object to JSON matching UploadResponse.
`status_code=201` = successful response will have HTTP 201 (Created), not the default 200 (OK).
`request: Request` = gives us access to `request.app.state` (settings, qdrant_client).
`file: UploadFile` = FastAPI automatically extracts the uploaded file from the multipart request.

```python
    settings = request.app.state.settings
    qdrant_client = request.app.state.qdrant_client
```
Retrieve the objects we stored on `app.state` during startup.

```python
    filename = file.filename or "unknown"
```
`file.filename` = the original filename from the upload (could be `None` if client didn't send it).
`or "unknown"` = if it's None/empty, use "unknown" as the name.

```python
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
```
Extract extension. Same logic as parser.py.
The `"." +` at the start adds the dot back so we get `".pdf"` not `"pdf"`.
This way it matches `_ALLOWED_EXTENSIONS = {".pdf", ".txt"}`.

```python
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(_ALLOWED_EXTENSIONS)}",
        )
```
`ext not in _ALLOWED_EXTENSIONS` = if the extension is not in our allowed set.
`{', '.join(_ALLOWED_EXTENSIONS)}` = join the set into a readable string like `".pdf, .txt"`.
400 = "Bad Request" — the client sent something we don't accept.

```python
    content = await file.read()
```
`await file.read()` = read ALL bytes of the uploaded file into memory.
`await` because reading from the network/disk is async.
`content` is now a `bytes` object.

```python
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
```
Convert MB to bytes: `50 * 1024 * 1024 = 52,428,800 bytes = 50 MB`.
`1024 * 1024 = 1,048,576 = 1 MB` (1 kilobyte = 1024 bytes, 1 megabyte = 1024 kilobytes).

```python
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {settings.MAX_FILE_SIZE_MB} MB limit.",
        )
```
`len(content)` = size of the file in bytes.
413 = "Content Too Large" (formerly "Payload Too Large").

```python
    t_start = time.perf_counter()
```
Start timing the ingestion process (everything below is what we time).

```python
    chunks, page_count = parse_and_chunk(
        filename, content, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP
    )
```
Call the parser service. It returns `(list_of_chunks, page_count)`.
We "unpack" the tuple directly into `chunks` and `page_count`.

```python
    embeddings = await async_encode_texts(settings.EMBEDDING_MODEL, chunks)
```
Convert all chunks to vectors. This runs in a background thread (won't freeze the server).
`embeddings` is now a `list[list[float]]` — one 384-float list per chunk.

```python
    doc_id = str(uuid.uuid4())
```
Generate a random unique ID for this document.
Example: `"f81d4fae-7dec-11d0-a765-00a0c91e6bf6"`.
Every call generates a new unique ID.

```python
    await upsert_chunks(
        client=qdrant_client,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        doc_id=doc_id,
        chunks=chunks,
        embeddings=embeddings,
        filename=filename,
    )
```
Save all chunks + vectors to Qdrant. Each chunk becomes one point in the collection.

```python
    await database.insert_document(
        doc_id=doc_id,
        filename=filename,
        file_size_bytes=len(content),
        page_count=page_count,
        chunk_count=len(chunks),
    )
```
Save document metadata to SQLite.
`len(content)` = file size in bytes.
`len(chunks)` = how many chunks were created.

```python
    ingestion_ms = int((time.perf_counter() - t_start) * 1000)
```
How many milliseconds the whole pipeline took (parse + embed + store).

```python
    logger.info(
        "documents.uploaded",
        doc_id=doc_id,
        filename=filename,
        chunks=len(chunks),
        ingestion_ms=ingestion_ms,
    )
```
Log success with key metrics.

```python
    return UploadResponse(
        doc_id=doc_id,
        filename=filename,
        chunk_count=len(chunks),
        page_count=page_count,
        ingestion_time_ms=ingestion_ms,
    )
```
Create and return the response object. FastAPI serializes it to JSON automatically.

---

## GET /api/v1/documents

```python
@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents() -> list[DocumentInfo]:
    rows = await database.list_documents()
    return [DocumentInfo(**row) for row in rows]
```
`rows` = list of dicts from SQLite.
`DocumentInfo(**row)` = `**row` unpacks the dict as keyword arguments.
Example: `DocumentInfo(**{"doc_id": "abc", "filename": "report.pdf", ...})` = `DocumentInfo(doc_id="abc", filename="report.pdf", ...)`.
Returns the list. FastAPI validates it matches `list[DocumentInfo]` and converts to JSON.

---

## DELETE /api/v1/documents/{doc_id}

```python
@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, request: Request) -> dict:
```
`{doc_id}` in the path = URL parameter. FastAPI extracts it automatically.
Example: `DELETE /api/v1/documents/abc-123` → `doc_id = "abc-123"`.

```python
    settings = request.app.state.settings
    qdrant_client = request.app.state.qdrant_client

    doc = await database.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
```
First check the document exists. 404 = "Not Found".

```python
    await delete_document_chunks(qdrant_client, settings.QDRANT_COLLECTION_NAME, doc_id)
    await database.delete_document(doc_id)
    logger.info("documents.deleted", doc_id=doc_id)
    return {"status": "deleted", "doc_id": doc_id}
```
Delete from Qdrant first (vectors), then from SQLite (metadata).
Return a plain dict — FastAPI converts it to JSON automatically.

---

---

# FILE 10: `routers/qa.py`

Handles the ask question endpoint.

```python
import structlog
from fastapi import APIRouter, HTTPException, Request

from app import database
from app.models import AskRequest, AskResponse, ChunkSource
from app.services.embedder import async_encode_query
from app.services.llm import generate_answer
from app.services.vector_store import search_chunks

logger = structlog.get_logger(__name__)
router = APIRouter()
```
Standard setup — same as documents.py.

---

## POST /api/v1/qa/ask

```python
@router.post("/qa/ask", response_model=AskResponse)
async def ask_question(request: Request, body: AskRequest) -> AskResponse:
```
`body: AskRequest` = FastAPI reads the JSON request body and validates it as `AskRequest`.
If the JSON is missing `question` or `document_id`, FastAPI returns a 422 error automatically.

```python
    settings = request.app.state.settings
    qdrant_client = request.app.state.qdrant_client
```
Retrieve app-level resources.

```python
    doc = await database.get_document(body.document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{body.document_id}' not found.")
```
Verify the document exists before doing any expensive operations.
If it doesn't exist, fail fast with a clear error.

```python
    query_vector = await async_encode_query(settings.EMBEDDING_MODEL, body.question)
```
Convert the user's question to a 384-float vector.
Same model that was used to embed the document chunks — must be identical for similarity to work.

```python
    scored_points = await search_chunks(
        client=qdrant_client,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        query_vector=query_vector,
        doc_id=body.document_id,
        top_k=body.top_k,
    )
```
Search Qdrant for the top `top_k` chunks most similar to the question vector, filtered to this document.
`scored_points` = list of Qdrant `ScoredPoint` objects.

```python
    chunk_texts = [point.payload["text"] for point in scored_points]
```
Extract just the text from each result.
`point.payload` = the dict we stored when uploading (contains `"text"`, `"doc_id"`, `"page_number"`, etc.).
`["text"]` = get just the chunk text.
`chunk_texts` = `["chunk 1 text", "chunk 2 text", ...]` — a plain list of strings.

```python
    llm_result = await generate_answer(
        question=body.question,
        chunks=chunk_texts,
        model=settings.GROQ_MODEL,
        api_key=settings.GROQ_API_KEY,
    )
```
Send the question + top chunks to Groq AI.
`llm_result` = a dict with keys `"answer"`, `"tokens_used"`, `"model"`.

```python
    sources = [
        ChunkSource(
            chunk_index=point.payload.get("chunk_index", i),
```
`point.payload.get("chunk_index", i)` = try to get "chunk_index" from payload.
If it's not there (e.g., old data), fall back to `i` (the loop index).
`.get(key, default)` is a safer dict access than `["key"]` (doesn't crash if key missing).

```python
            text_excerpt=point.payload["text"][:300],
```
`[:300]` = slice → get first 300 characters only.
We don't show the full chunk to the user — just a preview as the citation.

```python
            score=point.score,
```
The cosine similarity score from Qdrant (float between 0 and 1).

```python
            page_number=point.payload.get("page_number"),
```
Which page. `.get("page_number")` = returns `None` if not present (safer than `["page_number"]`).

```python
        )
        for i, point in enumerate(scored_points)
    ]
```
Build a list of ChunkSource objects — one per retrieved chunk.

```python
    logger.info(
        "qa.answered",
        doc_id=body.document_id,
        question_len=len(body.question),
        sources_count=len(sources),
    )
```
Log the event. We log `question_len` not the question itself (privacy — questions might be sensitive).

```python
    return AskResponse(
        answer=llm_result["answer"],
        sources=sources,
        model=llm_result["model"],
        tokens_used=llm_result["tokens_used"],
        doc_id=body.document_id,
    )
```
Build and return the final response.

---

---

# FILE 11: `routers/health.py`

A health check endpoint. Used by monitoring systems and load balancers to verify the service is alive.

```python
import structlog
from fastapi import APIRouter, Request

from app.models import HealthResponse
from app.services.embedder import get_embedder
from app.services.vector_store import count_collection

logger = structlog.get_logger(__name__)
router = APIRouter()
```

---

```python
@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
```
A GET endpoint at `/health` (→ `/api/v1/health` with prefix).

```python
    settings = request.app.state.settings
    qdrant_client = request.app.state.qdrant_client

    qdrant_status = "ok"
    try:
        await count_collection(qdrant_client, settings.QDRANT_COLLECTION_NAME)
    except Exception as exc:
        logger.warning("health.qdrant_failed", error=str(exc))
        qdrant_status = "unreachable"
```
`try: ... except Exception:` = catch ANY exception.
If counting works → Qdrant is alive → `"ok"`.
If it throws any error (connection refused, timeout, etc.) → `"unreachable"`.
We use `except Exception` (broad) here deliberately — we want to catch ALL possible errors, not just specific ones.

```python
    embedding_status = "ok"
    try:
        embedder = get_embedder(settings.EMBEDDING_MODEL)
        embedding_model_name = settings.EMBEDDING_MODEL
    except Exception as exc:
        logger.warning("health.embedder_failed", error=str(exc))
        embedding_status = "unavailable"
        embedding_model_name = settings.EMBEDDING_MODEL
```
Check the embedding model. `get_embedder` is cached — it just returns the existing object (very fast).
If somehow it fails (e.g., model files corrupted), mark as unavailable.

```python
    overall = "ok" if qdrant_status == "ok" and embedding_status == "ok" else "degraded"
```
`"ok" if A and B else "degraded"` = ternary expression.
Both must be "ok" for the overall status to be "ok".
If either is down, overall is "degraded".

```python
    return HealthResponse(
        status=overall,
        qdrant=qdrant_status,
        embedding_model=embedding_model_name,
        version="1.0.0",
    )
```
Return the health status. This endpoint NEVER raises an exception — it always returns a response (even if services are down). The response body tells you what's wrong.

---

---

# COMPLETE FLOW: Every Step When You Upload a File

```
1. Browser sends: POST /api/v1/documents/upload (multipart file)
   │
2. FastAPI receives it, runs log_requests middleware (starts timer)
   │
3. FastAPI routes to upload_document() in routers/documents.py
   │
4. Validation:
   ├── Check extension (.pdf or .txt only) — 400 if wrong
   └── Check file size (≤ 50 MB) — 413 if too big
   │
5. parse_and_chunk() in services/parser.py:
   ├── fitz.open() — read the PDF bytes
   ├── page.get_text() for each page — extract text strings
   └── RecursiveCharacterTextSplitter — split into ~500 char chunks
   Result: ["chunk 1 text", "chunk 2 text", ..., "chunk N text"]
   │
6. async_encode_texts() in services/embedder.py:
   ├── run_in_executor — move to background thread
   └── SentenceTransformer.encode() — convert each chunk to 384 floats
   Result: [[0.1, 0.3, ...], [0.2, 0.8, ...], ...]  (N vectors)
   │
7. doc_id = uuid4() — generate random unique ID
   │
8. upsert_chunks() in services/vector_store.py:
   └── qdrant_client.upsert() — save N points to Qdrant
       Each point: { id: uuid, vector: [384 floats], payload: {text, doc_id, filename, ...} }
   │
9. database.insert_document() in database.py:
   └── SQLite INSERT — save {doc_id, filename, file_size_bytes, page_count, chunk_count}
   │
10. Log "documents.uploaded" with timing
    │
11. Return UploadResponse { doc_id, filename, chunk_count, page_count, ingestion_time_ms }
    │
12. log_requests middleware logs the HTTP request details
    │
13. Browser receives JSON response with doc_id
```

---

# COMPLETE FLOW: Every Step When You Ask a Question

```
1. Browser sends: POST /api/v1/qa/ask
   Body: { "question": "What is revenue?", "document_id": "abc-123", "top_k": 5 }
   │
2. FastAPI middleware starts timing
   │
3. FastAPI validates the body against AskRequest:
   ├── question: 3-1000 chars ✓
   ├── document_id: string ✓
   └── top_k: 1-20 ✓ (default 5 if not sent)
   │
4. ask_question() in routers/qa.py:
   │
5. database.get_document("abc-123"):
   └── SQLite SELECT — verify document exists → 404 if not
   │
6. async_encode_query() in services/embedder.py:
   └── SentenceTransformer.encode("What is revenue?") → [384 floats]
   │
7. search_chunks() in services/vector_store.py:
   └── qdrant_client.query_points(
           query=[384 floats],
           filter: doc_id == "abc-123",
           limit: 5
       )
   Result: 5 ScoredPoint objects (each with .score and .payload.text)
   │
8. Extract chunk_texts = ["revenue grew...", "fiscal year shows...", ...]
   │
9. generate_answer() in services/llm.py:
   ├── Build context: "[1] revenue grew...\n\n[2] fiscal year shows..."
   ├── Build user_message: "Document excerpts:\n...\n\nQuestion: What is revenue?"
   ├── groq.AsyncGroq.chat.completions.create(
   │       system: "Answer ONLY from document excerpts..."
   │       user: context + question
   │       temperature: 0.1
   │   )
   └── Extract answer text + tokens_used from response
   │
10. Build list of ChunkSource objects (chunk_index, text_excerpt[:300], score, page_number)
    │
11. Return AskResponse { answer, sources, model, tokens_used, doc_id }
    │
12. Browser receives the answer + sources as JSON
```

---

# KEY PYTHON CONCEPTS USED (Quick Reference)

| Concept | What It Means | Example in This Code |
|---|---|---|
| `async def` | Function that can pause without blocking the server | `async def upload_document(...)` |
| `await` | Pause here until the async operation finishes | `await file.read()` |
| `lru_cache` | Remember function result so it runs only once | `get_settings()`, `get_embedder()` |
| `__init__` | Constructor — runs when class is created | `EmbedderService.__init__` |
| `self` | Refers to the current object instance | `self._model = SentenceTransformer(...)` |
| `yield` | In context managers, splits startup from cleanup | `lifespan()`, `get_db()` |
| `f"..."` | F-string — embed variables in strings | `f"File '.{ext}' not supported"` |
| `[:300]` | Slice — get first 300 characters | `text[:300]` |
| `dict[key]` | Get value from dict (crashes if missing) | `point.payload["text"]` |
| `dict.get(key)` | Get value from dict (returns None if missing) | `point.payload.get("page_number")` |
| `**dict` | Unpack dict as keyword arguments | `DocumentInfo(**row)` |
| `A if condition else B` | Ternary expression | `"ok" if x == "ok" else "degraded"` |
| `[x for x in list]` | List comprehension — build list in one line | `[dict(row) for row in rows]` |
| `raise HTTPException` | Return an HTTP error response | `raise HTTPException(status_code=404, ...)` |
| `try/except` | Catch errors without crashing | Error handling in health.py |
| `@decorator` | Wrap a function with extra behavior | `@router.post(...)`, `@lru_cache()` |
