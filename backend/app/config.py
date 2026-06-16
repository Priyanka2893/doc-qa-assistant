from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # LLM
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # Embeddings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # Vector DB
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "documents"

    # RAG
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    TOP_K_RESULTS: int = 5

    # App
    MAX_FILE_SIZE_MB: int = 50
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    ALLOWED_ORIGINS: str = "http://localhost:3000"

    # Confidence scoring
    MIN_CONFIDENCE_THRESHOLD: float = 0.40
    CONFIDENCE_WEIGHTS: dict[str, float] = {
        "retrieval": 0.50,
        "freshness": 0.20,
        "authority": 0.20,
        "agreement": 0.10,
    }

    # Semantic cache
    CACHE_SEMANTIC_THRESHOLD: float = 0.82  # lower (e.g. 0.75) to catch looser paraphrases

    # Hallucination guard
    PRE_GEN_CONFIDENCE_GATE: float = 0.50
    POST_GEN_TOKEN_FAST_PATH: float = 0.60   # token containment floor → grounded without embedding
    POST_GEN_OVERLAP_THRESHOLD: float = 0.50  # semantic cosine threshold for fallback stage
    HIGH_RISK_THRESHOLD: float = 0.40
    HALLUCINATION_ACTION: str = "flag"  # "flag" | "block"
    MIN_RAW_VECTOR_SCORE: float = 0.30  # absolute cosine similarity floor before normalization


@lru_cache()
def get_settings() -> Settings:
    return Settings()
