"""Tests for semantic cache, session memory, and corrections."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.services.cache import CacheEntry, SemanticCache
from app.services.corrections import Correction, CorrectionStore
from app.services.session_memory import SessionMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embedding(seed: int, size: int = 8) -> list[float]:
    """Return a unit vector seeded deterministically."""
    rng = np.random.default_rng(seed)
    v = rng.random(size).astype(float)
    return (v / np.linalg.norm(v)).tolist()


def _near_embedding(base: list[float], noise: float = 0.01) -> list[float]:
    """Return a vector very close to base (cosine > 0.99)."""
    v = np.array(base) + noise * np.random.default_rng(99).random(len(base))
    return (v / np.linalg.norm(v)).tolist()


def _fake_ask_response(answer: str = "42"):
    """Minimal stand-in for AskResponse (cache stores any object)."""
    resp = MagicMock()
    resp.answer = answer
    resp.model_copy = lambda update=None: resp
    return resp


# ---------------------------------------------------------------------------
# SemanticCache — unit tests
# ---------------------------------------------------------------------------

class TestExactCacheHit:
    @pytest.fixture
    def cache(self):
        return SemanticCache(max_size=50, ttl_seconds=3600, semantic_threshold=0.97)

    async def test_exact_cache_hit(self, cache):
        emb = _make_embedding(1)
        resp = _fake_ask_response("the answer")
        await cache.set("doc-1", "What is the refund policy?", emb, resp)

        entry, hit_type = await cache.get("doc-1", "What is the refund policy?", emb)
        assert hit_type == "exact"
        assert entry is not None
        assert entry.response is resp

    async def test_miss_on_different_question(self, cache):
        emb = _make_embedding(1)
        resp = _fake_ask_response()
        await cache.set("doc-1", "What is the refund policy?", emb, resp)

        entry, hit_type = await cache.get("doc-1", "What is the return window?", _make_embedding(2))
        assert hit_type == "miss"
        assert entry is None

    async def test_miss_on_different_doc(self, cache):
        emb = _make_embedding(1)
        resp = _fake_ask_response()
        await cache.set("doc-1", "What is the refund policy?", emb, resp)

        entry, hit_type = await cache.get("doc-2", "What is the refund policy?", emb)
        assert hit_type == "miss"

    async def test_stats_track_exact_hit(self, cache):
        emb = _make_embedding(1)
        resp = _fake_ask_response()
        await cache.set("doc-1", "hello?", emb, resp)
        await cache.get("doc-1", "hello?", emb)

        stats = cache.get_stats()
        assert stats.exact_hits == 1
        assert stats.misses == 0


class TestSemanticCacheHit:
    @pytest.fixture
    def cache(self):
        return SemanticCache(max_size=50, ttl_seconds=3600, semantic_threshold=0.97)

    async def test_semantic_cache_hit(self, cache):
        base_emb = _make_embedding(42, size=32)
        near_emb = _near_embedding(base_emb, noise=0.005)  # cosine will be > 0.99

        resp = _fake_ask_response("semantically similar answer")
        await cache.set("doc-1", "return policy?", base_emb, resp)

        entry, hit_type = await cache.get("doc-1", "what's the return policy?", near_emb)
        assert hit_type == "semantic"
        assert entry is not None

    async def test_no_semantic_hit_below_threshold(self, cache):
        base_emb = _make_embedding(1, size=32)
        distant_emb = _make_embedding(99, size=32)  # unrelated vector

        resp = _fake_ask_response()
        await cache.set("doc-1", "return policy?", base_emb, resp)

        entry, hit_type = await cache.get("doc-1", "something unrelated", distant_emb)
        assert hit_type == "miss"

    async def test_semantic_hits_not_counted_for_different_doc(self, cache):
        base_emb = _make_embedding(42, size=32)
        near_emb = _near_embedding(base_emb, noise=0.005)

        resp = _fake_ask_response()
        await cache.set("doc-A", "question?", base_emb, resp)

        entry, hit_type = await cache.get("doc-B", "question?", near_emb)
        assert hit_type == "miss"


class TestCacheInvalidation:
    async def test_cache_invalidation(self):
        cache = SemanticCache()
        emb = _make_embedding(1)
        resp = _fake_ask_response()
        await cache.set("doc-del", "test question?", emb, resp)

        entry, _ = await cache.get("doc-del", "test question?", emb)
        assert entry is not None  # cached before deletion

        await cache.invalidate_document("doc-del")

        entry, hit_type = await cache.get("doc-del", "test question?", emb)
        assert hit_type == "miss"
        assert entry is None

    async def test_invalidation_only_removes_target_doc(self):
        cache = SemanticCache()
        emb = _make_embedding(1)
        await cache.set("doc-1", "q1?", emb, _fake_ask_response())
        await cache.set("doc-2", "q2?", emb, _fake_ask_response())

        await cache.invalidate_document("doc-1")

        entry1, ht1 = await cache.get("doc-1", "q1?", emb)
        entry2, ht2 = await cache.get("doc-2", "q2?", emb)
        assert ht1 == "miss"
        assert ht2 == "exact"

    async def test_embeddings_list_shrinks_on_remove(self):
        cache = SemanticCache()
        emb = _make_embedding(1)
        await cache.set("doc-x", "question?", emb, _fake_ask_response())
        assert len(cache._embeddings) == 1

        await cache.invalidate_document("doc-x")
        assert len(cache._embeddings) == 0

    async def test_eviction_when_at_max_size(self):
        cache = SemanticCache(max_size=3)
        for i in range(3):
            await cache.set(f"doc-{i}", "q?", _make_embedding(i), _fake_ask_response(str(i)))
        assert cache.cache_size() == 3

        # Adding a 4th should evict the oldest
        await cache.set("doc-3", "q?", _make_embedding(3), _fake_ask_response("3"))
        assert cache.cache_size() == 3


# ---------------------------------------------------------------------------
# SessionMemory — unit tests
# ---------------------------------------------------------------------------

class TestSessionCreation:
    def test_create_session_returns_session_id(self):
        sm = SessionMemory()
        sid = sm.create_session("doc-abc")
        assert sid is not None
        assert len(sid) == 36  # UUID4 format

    def test_get_session_returns_session(self):
        sm = SessionMemory()
        sid = sm.create_session("doc-abc")
        session = sm.get_session(sid)
        assert session is not None
        assert session.doc_id == "doc-abc"
        assert session.session_id == sid

    def test_get_session_missing_returns_none(self):
        sm = SessionMemory()
        assert sm.get_session("nonexistent-id") is None


class TestMultiTurnSession:
    def test_multi_turn_session(self):
        sm = SessionMemory()
        sid = sm.create_session("doc-1")

        sm.add_turn(sid, "What is the policy?", "The policy is X.", "doc-1", [])
        sm.add_turn(sid, "How do I apply?", "You apply via Y.", "doc-1", [])

        session = sm.get_session(sid)
        assert len(session.turns) == 2
        assert session.turns[0].question == "What is the policy?"
        assert session.turns[1].question == "How do I apply?"

    def test_get_context_for_query_contains_turns(self):
        sm = SessionMemory()
        sid = sm.create_session("doc-1")
        sm.add_turn(sid, "What is the policy?", "The policy is X.", "doc-1", [])

        ctx = sm.get_context_for_query(sid)
        assert "What is the policy?" in ctx
        assert "The policy is X" in ctx

    def test_get_context_empty_for_new_session(self):
        sm = SessionMemory()
        sid = sm.create_session("doc-1")
        assert sm.get_context_for_query(sid) == ""

    def test_cleanup_expired_removes_sessions(self):
        sm = SessionMemory()
        sid = sm.create_session("doc-1")
        # Manually expire the session
        sm._sessions[sid].last_active = datetime(2000, 1, 1)

        removed = sm.cleanup_expired()
        assert removed == 1
        assert sm.get_session(sid) is None


# ---------------------------------------------------------------------------
# CorrectionStore — unit tests
# ---------------------------------------------------------------------------

class TestCorrectionOverride:
    @pytest.fixture
    def mock_embedder(self):
        embedder = MagicMock()
        # Same embedding for all calls → cosine = 1.0 → will always match
        embedder.encode_query = MagicMock(return_value=_make_embedding(7, size=16))
        return embedder

    async def test_add_and_find_correction(self, mock_embedder):
        store = CorrectionStore()
        cid = await store.add_correction(
            doc_id="doc-1",
            question="What is the refund time?",
            original_answer="7 days",
            corrected_answer="5 business days",
            embedder=mock_embedder,
        )
        assert cid is not None

        result = await store.find_correction(
            question="What is the refund time?",
            doc_id="doc-1",
            embedder=mock_embedder,
            threshold=0.92,
        )
        assert result is not None
        assert result.corrected_answer == "5 business days"
        assert result.use_count == 1

    async def test_no_correction_for_different_doc(self, mock_embedder):
        store = CorrectionStore()
        await store.add_correction(
            doc_id="doc-1",
            question="What is the refund time?",
            original_answer="7 days",
            corrected_answer="5 business days",
            embedder=mock_embedder,
        )
        result = await store.find_correction(
            question="What is the refund time?",
            doc_id="doc-2",
            embedder=mock_embedder,
        )
        assert result is None

    async def test_no_match_below_threshold(self):
        store = CorrectionStore()

        low_emb = _make_embedding(1, size=16)
        high_emb = _make_embedding(99, size=16)  # unrelated vector

        embedder_add = MagicMock()
        embedder_add.encode_query = MagicMock(return_value=low_emb)

        embedder_find = MagicMock()
        embedder_find.encode_query = MagicMock(return_value=high_emb)

        await store.add_correction(
            doc_id="doc-1",
            question="original question",
            original_answer="wrong",
            corrected_answer="right",
            embedder=embedder_add,
        )
        result = await store.find_correction(
            question="totally different question",
            doc_id="doc-1",
            embedder=embedder_find,
            threshold=0.92,
        )
        assert result is None


# ---------------------------------------------------------------------------
# HTTP integration tests — sessions and corrections endpoints
# ---------------------------------------------------------------------------

@pytest.mark.anyio
class TestSessionEndpoints:
    async def test_session_creation_returns_session_id(self, http_client):
        client, _ = http_client
        with patch(
            "app.routers.sessions.database.get_document",
            new_callable=AsyncMock,
            return_value={"doc_id": "doc-1", "filename": "test.pdf"},
        ):
            resp = await client.post("/api/v1/sessions", json={"doc_id": "doc-1"})

        assert resp.status_code == 201
        body = resp.json()
        assert "session_id" in body
        assert body["doc_id"] == "doc-1"

    async def test_session_creation_404_for_missing_doc(self, http_client):
        client, _ = http_client
        with patch(
            "app.routers.sessions.database.get_document",
            new_callable=AsyncMock,
            return_value=None,
        ):
            resp = await client.post("/api/v1/sessions", json={"doc_id": "missing"})

        assert resp.status_code == 404

    async def test_get_session_info(self, http_client):
        from app.services.session_memory import get_session_memory

        client, _ = http_client
        sid = get_session_memory().create_session("doc-xyz")

        resp = await client.get(f"/api/v1/sessions/{sid}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == sid
        assert body["turn_count"] == 0

    async def test_get_session_404_for_missing(self, http_client):
        client, _ = http_client
        resp = await client.get("/api/v1/sessions/nonexistent-session-id")
        assert resp.status_code == 404


@pytest.mark.anyio
class TestCacheStatsEndpoint:
    async def test_cache_stats_returns_expected_keys(self, http_client):
        client, _ = http_client
        resp = await client.get("/api/v1/cache/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert "total_entries" in body
        assert "exact_hits" in body
        assert "semantic_hits" in body
        assert "hit_rate" in body


@pytest.mark.anyio
class TestCorrectionEndpoint:
    async def test_submit_correction_returns_correction_id(self, http_client):
        client, _ = http_client

        fake_embedder = MagicMock()
        fake_embedder.encode_query = MagicMock(return_value=[0.1] * 384)

        with (
            patch(
                "app.routers.sessions.database.get_document",
                new_callable=AsyncMock,
                return_value={"doc_id": "doc-1"},
            ),
            patch("app.routers.sessions.get_embedder", return_value=fake_embedder),
        ):
            resp = await client.post(
                "/api/v1/corrections",
                json={
                    "doc_id": "doc-1",
                    "question": "What is the return policy?",
                    "original_answer": "30 days",
                    "corrected_answer": "60 days for premium members",
                },
            )

        assert resp.status_code == 201
        body = resp.json()
        assert "correction_id" in body
