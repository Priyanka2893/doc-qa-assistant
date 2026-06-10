from __future__ import annotations

import structlog
from cachetools import TTLCache

logger = structlog.get_logger(__name__)

_cache: TTLCache = TTLCache(maxsize=500, ttl=3600)
_hits: int = 0
_misses: int = 0


class SemanticCache:
    async def get_cached_answer(self, question: str, doc_id: str):
        """Return cached AskResponse or None. Imports AskResponse lazily to avoid circular deps."""
        global _hits, _misses
        key = f"{doc_id}:{question.lower().strip()}"
        result = _cache.get(key)
        if result is not None:
            _hits += 1
            logger.info(
                "cache_hit",
                doc_id=doc_id,
                question_preview=question[:50],
            )
            return result
        _misses += 1
        return None

    async def cache_answer(self, question: str, doc_id: str, response) -> None:
        key = f"{doc_id}:{question.lower().strip()}"
        _cache[key] = response

    async def invalidate_document(self, doc_id: str) -> None:
        keys_to_delete = [k for k in list(_cache.keys()) if isinstance(k, str) and k.startswith(f"{doc_id}:")]
        for k in keys_to_delete:
            _cache.pop(k, None)
        if keys_to_delete:
            logger.info("cache.invalidated", doc_id=doc_id, count=len(keys_to_delete))

    def cache_size(self) -> int:
        return len(_cache)

    def hit_rate(self) -> float:
        total = _hits + _misses
        if total == 0:
            return 0.0
        return round(_hits / total, 4)


_semantic_cache = SemanticCache()


def get_semantic_cache() -> SemanticCache:
    return _semantic_cache
