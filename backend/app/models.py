from enum import Enum

from pydantic import BaseModel, Field


class SearchMode(str, Enum):
    VECTOR = "vector"
    HYBRID = "hybrid"
    KEYWORD = "keyword"


class ResponseMode(str, Enum):
    CITED = "cited"
    PLAIN = "plain"
    STRICT_ABSTAIN = "strict_abstain"


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


class CitedSource(BaseModel):
    tag: str
    source_number: int
    chunk_index: int | None
    page_number: int | None
    text_excerpt: str
    filename: str
    confidence_score: float | None
    is_unmapped: bool = False


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


class InsufficientEvidenceResponse(BaseModel):
    answer: str = "Insufficient evidence found in the provided documents."
    is_gate_blocked: bool = True
    gate_reason: str
    avg_confidence: float
    chunk_count: int
    suggestion: str = "Try uploading more relevant documents or rephrasing your question."


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    document_id: str
    top_k: int = Field(default=5, ge=1, le=20)
    search_mode: SearchMode = SearchMode.HYBRID
    rerank: bool = True
    response_mode: ResponseMode = ResponseMode.CITED
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    session_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    cited_sources: list[CitedSource]
    unmapped_citations: list[str] = []
    is_abstention: bool = False
    citation_coverage: float = 0.0
    response_mode: str = ResponseMode.CITED
    model: str
    tokens_used: int
    doc_id: str
    cache_hit: bool = False
    cache_hit_type: str | None = None   # "exact" | "semantic" | None
    session_id: str | None = None
    is_correction: bool = False
    evidence_quality: str = "none"
    avg_confidence: float = 0.0
    chunks_filtered_out: int = 0
    hallucination_risk: float = 0.0
    is_high_risk: bool = False
    ungrounded_sentences: list[str] = []
    gate_passed: bool = True
    eval_metrics: "EvalMetrics | None" = None


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


class EvalMetrics(BaseModel):
    context_relevance: float
    faithfulness: float
    answer_relevance: float
    overall_score: float
    chunk_count_used: int
    is_abstention: bool
    hallucination_risk: float


class EvalSummary(BaseModel):
    query_count: int
    avg_context_relevance: float
    avg_faithfulness: float
    avg_answer_relevance: float
    avg_overall_score: float
    abstention_rate: float
    high_risk_rate: float
    time_window_hours: int
