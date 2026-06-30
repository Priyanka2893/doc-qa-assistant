from langchain_groq import ChatGroq
from functools import lru_cache
from app.config import get_settings
import structlog

log = structlog.get_logger()


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    s = get_settings()
    return ChatGroq(
        api_key=s.groq_api_key,
        model=s.groq_model,
        temperature=s.groq_temperature,
        max_tokens=2048,
    )


def get_planning_llm() -> ChatGroq:
    """Lighter config for planning: deterministic, short output."""
    s = get_settings()
    return ChatGroq(
        api_key=s.groq_api_key,
        model=s.groq_model,
        temperature=0.0,
        max_tokens=512,
    )
