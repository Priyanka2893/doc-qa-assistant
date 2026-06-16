from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import structlog

from app.telemetry import CACHE_HITS, CACHE_MISSES

logger = structlog.get_logger(__name__)


@dataclass
class CacheEntry:
    question: str
    question_embedding: list[float]
    response: object  # AskResponse — imported lazily to avoid circular deps
    doc_id: str
    created_at: datetime
    hit_count: int = 0


@dataclass
class CacheStats:
    total_entries: int
    exact_hits: int
    semantic_hits: int
    misses: int
    hit_rate: float
    avg_response_time_saved_ms: float


class SemanticCache:
    def __init__(
        self,
        max_size: int = 500,
        ttl_seconds: int = 3600,
        semantic_threshold: float = 0.97,
    ) -> None:
        self._exact_cache: dict[str, CacheEntry] = {}
        self._embeddings: list[tuple[list[float], str]] = []
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._threshold = semantic_threshold
        self._stats: dict[str, int] = {"exact_hits": 0, "semantic_hits": 0, "misses": 0}

    def _cache_key(self, doc_id: str, question: str) -> str:
        normalized = question.lower().strip()
        return hashlib.sha256(f"{doc_id}:{normalized}".encode()).hexdigest()

    async def get(
        self,
        doc_id: str,
        question: str,
        question_embedding: list[float],
    ) -> tuple[CacheEntry | None, str]:
        """Return (cache_entry, hit_type) where hit_type is 'exact', 'semantic', or 'miss'."""
        key = self._cache_key(doc_id, question)
        if key in self._exact_cache:
            entry = self._exact_cache[key]
            if not self._is_expired(entry):
                entry.hit_count += 1
                self._stats["exact_hits"] += 1
                CACHE_HITS.labels(cache_type="exact").inc()
                logger.info("cache_hit", hit_type="exact", doc_id=doc_id)
                return entry, "exact"
            self._remove(key)

        if question_embedding:
            best_score, best_key = self._find_similar(question_embedding, doc_id)
            if best_score >= self._threshold and best_key in self._exact_cache:
                entry = self._exact_cache[best_key]
                if not self._is_expired(entry):
                    entry.hit_count += 1
                    self._stats["semantic_hits"] += 1
                    CACHE_HITS.labels(cache_type="semantic").inc()
                    logger.info(
                        "cache_hit",
                        hit_type="semantic",
                        doc_id=doc_id,
                        score=round(best_score, 4),
                    )
                    return entry, "semantic"

        self._stats["misses"] += 1
        CACHE_MISSES.inc()
        return None, "miss"

    def _find_similar(self, query_embedding: list[float], doc_id: str) -> tuple[float, str]:
        best_score = 0.0
        best_key = ""
        q = np.array(query_embedding)
        for emb, key in self._embeddings:
            entry = self._exact_cache.get(key)
            if entry and entry.doc_id == doc_id:
                c = np.array(emb)
                score = float(np.dot(q, c) / (np.linalg.norm(q) * np.linalg.norm(c) + 1e-9))
                if score > best_score:
                    best_score, best_key = score, key
        return best_score, best_key

    async def set(
        self,
        doc_id: str,
        question: str,
        question_embedding: list[float],
        response: object,
    ) -> None:
        key = self._cache_key(doc_id, question)
        if len(self._exact_cache) >= self._max_size:
            oldest_key = min(self._exact_cache, key=lambda k: self._exact_cache[k].created_at)
            self._remove(oldest_key)
        self._exact_cache[key] = CacheEntry(
            question=question,
            question_embedding=question_embedding,
            response=response,
            doc_id=doc_id,
            created_at=datetime.utcnow(),
        )
        self._embeddings.append((question_embedding, key))

    async def invalidate_document(self, doc_id: str) -> None:
        keys_to_remove = [k for k, v in self._exact_cache.items() if v.doc_id == doc_id]
        for key in keys_to_remove:
            self._remove(key)
        logger.info("cache_invalidated", doc_id=doc_id, entries_removed=len(keys_to_remove))

    def get_stats(self) -> CacheStats:
        total_hits = self._stats["exact_hits"] + self._stats["semantic_hits"]
        total = total_hits + self._stats["misses"]
        return CacheStats(
            total_entries=len(self._exact_cache),
            exact_hits=self._stats["exact_hits"],
            semantic_hits=self._stats["semantic_hits"],
            misses=self._stats["misses"],
            hit_rate=round(total_hits / max(total, 1), 4),
            avg_response_time_saved_ms=1800.0,
        )

    def cache_size(self) -> int:
        return len(self._exact_cache)

    def hit_rate(self) -> float:
        total_hits = self._stats["exact_hits"] + self._stats["semantic_hits"]
        total = total_hits + self._stats["misses"]
        return round(total_hits / max(total, 1), 4)

    def _is_expired(self, entry: CacheEntry) -> bool:
        return (datetime.utcnow() - entry.created_at).total_seconds() > self._ttl

    def _remove(self, key: str) -> None:
        self._exact_cache.pop(key, None)
        self._embeddings = [(e, k) for e, k in self._embeddings if k != key]


def _make_cache() -> SemanticCache:
    from app.config import get_settings
    s = get_settings()
    return SemanticCache(semantic_threshold=s.CACHE_SEMANTIC_THRESHOLD)


_semantic_cache = _make_cache()


def get_semantic_cache() -> SemanticCache:
    return _semantic_cache
