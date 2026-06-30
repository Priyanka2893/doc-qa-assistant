from sentence_transformers import SentenceTransformer
from functools import lru_cache
import asyncio
import structlog

log = structlog.get_logger()


@lru_cache(maxsize=1)
def _load_model(model_name: str) -> SentenceTransformer:
    log.info("loading_embedding_model", model=model_name)
    return SentenceTransformer(model_name)


def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    return _load_model(model_name)


async def encode_query(query: str, model_name: str = "all-MiniLM-L6-v2") -> list[float]:
    loop = asyncio.get_event_loop()
    model = get_embedder(model_name)
    return await loop.run_in_executor(None, lambda: model.encode(query).tolist())


async def encode_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> list[list[float]]:
    loop = asyncio.get_event_loop()
    model = get_embedder(model_name)
    return await loop.run_in_executor(None, lambda: model.encode(texts).tolist())
