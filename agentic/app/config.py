from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pathlib import Path

# Path to shared .env — one level up from agentic/
ENV_FILE = Path(__file__).parent.parent.parent / ".env"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",          # ignore keys from .env that aren't in this model
    )

    # LLM (from existing .env)
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = 0.1

    # Embeddings (must match backend — same Qdrant collection)
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Qdrant (shared Docker service)
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "documents"

    # Tools
    tavily_api_key: str = ""
    tavily_max_results: int = 5

    # LangSmith
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "agentic-rag-production"

    # Agentic limits
    max_retrieval_iterations: int = 3
    max_tool_calls_per_task: int = 12
    new_evidence_similarity_threshold: float = 0.85
    top_k_retrieval: int = 5

    # App
    agentic_app_port: int = 8010
    app_env: str = "development"
    log_level: str = "INFO"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
