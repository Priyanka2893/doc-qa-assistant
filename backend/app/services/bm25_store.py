from collections import defaultdict
from functools import lru_cache

import structlog
from rank_bm25 import BM25Okapi

logger = structlog.get_logger(__name__)


class BM25Store:
    def __init__(self) -> None:
        self._indexes: dict[str, BM25Okapi] = {}
        self._corpus: dict[str, list[str]] = {}
        self._chunk_ids: dict[str, list[str]] = {}
        self._metadata: dict[str, dict] = {}  # chunk_id -> payload dict

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def build_index(
        self,
        doc_id: str,
        chunks: list[str],
        chunk_ids: list[str],
        filename: str = "",
        chunk_metas: list[dict] | None = None,
    ) -> None:
        tokenized = [self._tokenize(chunk) for chunk in chunks]
        self._corpus[doc_id] = chunks
        self._chunk_ids[doc_id] = chunk_ids
        self._indexes[doc_id] = BM25Okapi(tokenized)
        for i, chunk_id in enumerate(chunk_ids):
            if chunk_metas:
                self._metadata[chunk_id] = chunk_metas[i]
            else:
                self._metadata[chunk_id] = {
                    "text": chunks[i],
                    "doc_id": doc_id,
                    "filename": filename,
                    "chunk_index": i,
                    "page_number": None,
                }
        logger.info("bm25_store.indexed", doc_id=doc_id, chunk_count=len(chunks))

    def search(self, doc_id: str, query: str, top_k: int) -> list[tuple[str, float]]:
        if doc_id not in self._indexes:
            logger.warning("bm25_store.doc_not_found", doc_id=doc_id)
            return []
        tokenized_query = self._tokenize(query)
        scores = self._indexes[doc_id].get_scores(tokenized_query)
        chunk_ids = self._chunk_ids[doc_id]
        max_score = float(max(scores)) if len(scores) > 0 else 1.0
        if max_score == 0.0:
            max_score = 1.0
        normalized = [(chunk_ids[i], float(scores[i]) / max_score) for i in range(len(scores))]
        normalized.sort(key=lambda x: x[1], reverse=True)
        return normalized[:top_k]

    def search_all(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Search across all document indexes; normalize scores within each doc before merging."""
        tokenized_query = self._tokenize(query)
        all_results: list[tuple[str, float]] = []
        for doc_id, index in self._indexes.items():
            scores = index.get_scores(tokenized_query)
            chunk_ids = self._chunk_ids[doc_id]
            max_score = float(max(scores)) if len(scores) > 0 else 1.0
            if max_score == 0.0:
                max_score = 1.0
            for i, score in enumerate(scores):
                all_results.append((chunk_ids[i], float(score) / max_score))
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]

    def get_metadata(self, chunk_id: str) -> dict | None:
        return self._metadata.get(chunk_id)

    def remove_document(self, doc_id: str) -> None:
        for chunk_id in self._chunk_ids.get(doc_id, []):
            self._metadata.pop(chunk_id, None)
        self._indexes.pop(doc_id, None)
        self._corpus.pop(doc_id, None)
        self._chunk_ids.pop(doc_id, None)
        logger.info("bm25_store.removed", doc_id=doc_id)

    def has_document(self, doc_id: str) -> bool:
        return doc_id in self._indexes


@lru_cache(maxsize=1)
def get_bm25_store() -> BM25Store:
    return BM25Store()


async def rebuild_indexes_from_qdrant(client, collection_name: str) -> None:
    """Scroll all Qdrant points and rebuild in-memory BM25 indexes grouped by doc_id."""
    store = get_bm25_store()
    all_chunks: dict[str, list[dict]] = defaultdict(list)
    offset = None

    while True:
        points, next_offset = await client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            if point.payload and "doc_id" in point.payload:
                all_chunks[point.payload["doc_id"]].append({
                    "id": str(point.id),
                    "text": point.payload.get("text", ""),
                    "doc_id": point.payload["doc_id"],
                    "filename": point.payload.get("filename", ""),
                    "chunk_index": point.payload.get("chunk_index", 0),
                    "page_number": point.payload.get("page_number"),
                })
        if next_offset is None:
            break
        offset = next_offset

    for doc_id, chunks in all_chunks.items():
        chunks.sort(key=lambda c: c["chunk_index"])
        chunk_metas = [
            {
                "text": c["text"],
                "doc_id": c["doc_id"],
                "filename": c["filename"],
                "chunk_index": c["chunk_index"],
                "page_number": c["page_number"],
            }
            for c in chunks
        ]
        store.build_index(
            doc_id=doc_id,
            chunks=[c["text"] for c in chunks],
            chunk_ids=[c["id"] for c in chunks],
            chunk_metas=chunk_metas,
        )

    logger.info("bm25_store.rebuilt", doc_count=len(all_chunks))
