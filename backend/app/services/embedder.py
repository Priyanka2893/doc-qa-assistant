import asyncio
import time
from functools import lru_cache

import structlog
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger(__name__)


class EmbedderService:
    """Singleton wrapper around a SentenceTransformer model."""

    def __init__(self, model_name: str) -> None:
        start = time.perf_counter()
        self._model = SentenceTransformer(model_name)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info("embedder.loaded", model=model_name, load_time_ms=elapsed_ms)

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts into embedding vectors."""
        vectors = self._model.encode(
            texts,
            batch_size=32,
            convert_to_tensor=False,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def encode_query(self, query: str) -> list[float]:
        """Encode a single query string into an embedding vector."""
        vector = self._model.encode(
            query,
            convert_to_tensor=False,
            show_progress_bar=False,
        )
        return vector.tolist()

    @property
    def model_name(self) -> str:
        return str(self._model.model_card_data.base_model or "unknown")


@lru_cache(maxsize=1)
def get_embedder(model_name: str) -> EmbedderService:
    """Return (and cache) the singleton EmbedderService instance."""
    return EmbedderService(model_name)


async def async_encode_texts(model_name: str, texts: list[str]) -> list[list[float]]:
    """Run encode_texts in a thread executor to keep the event loop unblocked."""
    loop = asyncio.get_event_loop()
    embedder = get_embedder(model_name)
    return await loop.run_in_executor(None, embedder.encode_texts, texts)


async def async_encode_query(model_name: str, query: str) -> list[float]:
    """Run encode_query in a thread executor to keep the event loop unblocked."""
    loop = asyncio.get_event_loop()
    embedder = get_embedder(model_name)
    return await loop.run_in_executor(None, embedder.encode_query, query)
