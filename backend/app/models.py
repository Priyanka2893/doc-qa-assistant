from enum import Enum

from pydantic import BaseModel, Field


class SearchMode(str, Enum):
    VECTOR = "vector"
    HYBRID = "hybrid"
    KEYWORD = "keyword"


class RetrievalResult(BaseModel):
    chunk_id: str
    text: str
    score: float
    vector_score: float | None
    bm25_score: float | None
    doc_id: str
    filename: str
    chunk_index: int
    page_number: int | None


class ChunkSource(BaseModel):
    chunk_index: int
    text_excerpt: str
    score: float
    page_number: int | None
    vector_score: float | None = None
    bm25_score: float | None = None
    confidence_score: float = 0.0
    freshness_score: float = 0.0
    authority_score: float = 0.0
    agreement_score: float = 0.0
    retrieval_score: float = 0.0


class GlobalChunkSource(ChunkSource):
    filename: str
    doc_id: str


class TrustUpdateRequest(BaseModel):
    trust_level: str = Field(pattern="^(verified|internal|external|unknown)$")


class DocumentMetadata(BaseModel):
    author: str | None = None
    doc_title: str | None = None
    language: str = "en"
    word_count: int = 0
    file_format: str = ""


class IngestionReport(BaseModel):
    original_chunks: int
    exact_dedup_removed: int
    semantic_dedup_removed: int
    final_chunks: int
    dedup_rate: float


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    page_count: int
    status: str = "success"
    ingestion_time_ms: int
    ingestion_report: IngestionReport
    document_metadata: DocumentMetadata


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
    cache_hit: bool = False
    evidence_quality: str = "none"
    avg_confidence: float = 0.0
    chunks_filtered_out: int = 0


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
    author: str | None = None
    doc_title: str | None = None
    language: str | None = "en"
    word_count: int | None = 0
    file_format: str | None = None
    exact_dedup_removed: int | None = 0
    semantic_dedup_removed: int | None = 0
    document_trust: str | None = "unknown"
    supported_formats: list[str] = Field(
        default=[".pdf", ".txt", ".docx", ".html", ".htm", ".png", ".jpg", ".jpeg", ".tiff"]
    )


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    embedding_model: str
    version: str


class MetricsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    cache_size: int
    cache_hit_rate: float
    uptime_seconds: int
    embedding_model: str
    qdrant_points: int
