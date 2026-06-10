import asyncio
import hashlib
import math
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DedupResult:
    chunks: list[str]
    exact_removed: int
    semantic_removed: int
    original_count: int

    @property
    def final_count(self) -> int:
        return len(self.chunks)

    @property
    def dedup_rate(self) -> float:
        return 1 - self.final_count / max(self.original_count, 1)


def deduplicate_exact(chunks: list[str]) -> tuple[list[str], int]:
    """Remove exact duplicate chunks by SHA256 hash. Returns (unique_chunks, removed_count)."""
    seen: set[str] = set()
    unique: list[str] = []
    for chunk in chunks:
        h = hashlib.sha256(chunk.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(chunk)
    removed = len(chunks) - len(unique)
    if removed:
        logger.info("deduplicator.exact_removed", count=removed)
    return unique, removed


async def deduplicate_semantic(
    chunks: list[str],
    embedder,
    similarity_threshold: float = 0.95,
) -> tuple[list[str], int]:
    """Remove near-duplicate chunks via cosine similarity. Returns (unique_chunks, removed_count).

    Greedy: keeps the first occurrence; drops any later chunk whose max cosine
    similarity to the kept set is >= similarity_threshold.
    Skipped when len(chunks) <= 3 (not worth the overhead).
    """
    if len(chunks) <= 3:
        return chunks, 0

    loop = asyncio.get_event_loop()
    embeddings: list[list[float]] = await loop.run_in_executor(None, embedder.encode_texts, chunks)

    kept_indices = [0]
    kept_vecs = [embeddings[0]]

    for i in range(1, len(chunks)):
        emb = embeddings[i]
        max_sim = max(_cosine(emb, k) for k in kept_vecs)
        if max_sim >= similarity_threshold:
            continue
        kept_indices.append(i)
        kept_vecs.append(emb)

    unique_chunks = [chunks[i] for i in kept_indices]
    removed = len(chunks) - len(unique_chunks)
    if removed:
        logger.info("deduplicator.semantic_removed", count=removed)
    return unique_chunks, removed


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
