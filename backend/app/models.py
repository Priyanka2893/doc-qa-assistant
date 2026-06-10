from enum import Enum

from pydantic import BaseModel, Field


class SearchMode(str, Enum):
    VECTOR = "vector"
    HYBRID = "hybrid"
    KEYWORD = "keyword"


class ChunkSource(BaseModel):
    chunk_index: int
    text_excerpt: str
    score: float
    page_number: int | None
    vector_score: float | None = None
    bm25_score: float | None = None


class GlobalChunkSource(ChunkSource):
    filename: str
    doc_id: str


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    page_count: int
    status: str = "success"
    ingestion_time_ms: int


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    document_id: str
    top_k: int = Field(default=5, ge=1, le=20)
    search_mode: SearchMode = SearchMode.HYBRID
    rerank: bool = True


class AskResponse(BaseModel):
    answer: str
    sources: list[ChunkSource]
    model: str
    tokens_used: int
    doc_id: str


class GlobalAskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=50)
    search_mode: SearchMode = SearchMode.HYBRID
    rerank: bool = True


class GlobalAskResponse(BaseModel):
    answer: str
    sources: list[GlobalChunkSource]
    model: str
    tokens_used: int


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    page_count: int
    uploaded_at: str
    status: str = "ready"
    file_size_bytes: int | None = None
    content_hash: str | None = None


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    embedding_model: str
    version: str
