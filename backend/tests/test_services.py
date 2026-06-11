"""Unit tests for vector_store and llm services using mocked external clients."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from qdrant_client.http import models as qdrant_models


# ---------------------------------------------------------------------------
# vector_store tests
# ---------------------------------------------------------------------------

class TestInitCollection:
    async def test_creates_collection_when_absent(self):
        from app.services.vector_store import init_collection

        client = AsyncMock()
        client.collection_exists = AsyncMock(return_value=False)
        client.create_collection = AsyncMock()

        await init_collection(client, "docs", 384)

        client.create_collection.assert_awaited_once()
        call_kwargs = client.create_collection.call_args.kwargs
        assert call_kwargs["collection_name"] == "docs"

    async def test_skips_creation_when_exists(self):
        from app.services.vector_store import init_collection

        client = AsyncMock()
        client.collection_exists = AsyncMock(return_value=True)
        client.create_collection = AsyncMock()

        await init_collection(client, "docs", 384)

        client.create_collection.assert_not_awaited()


class TestUpsertChunks:
    async def test_upserts_correct_number_of_points(self):
        from app.services.vector_store import upsert_chunks

        client = AsyncMock()
        chunks = ["chunk one", "chunk two", "chunk three"]
        embeddings = [[0.1] * 384, [0.2] * 384, [0.3] * 384]

        await upsert_chunks(client, "docs", "doc-1", chunks, embeddings, "test.pdf")

        client.upsert.assert_awaited_once()
        points = client.upsert.call_args.kwargs["points"]
        assert len(points) == 3
        assert all(p.payload["doc_id"] == "doc-1" for p in points)
        assert all(p.payload["filename"] == "test.pdf" for p in points)
        for i, point in enumerate(points):
            assert point.payload["chunk_index"] == i


class TestSearchChunks:
    async def test_returns_scored_points(self):
        from app.services.vector_store import search_chunks

        mock_point = MagicMock()
        mock_point.payload = {"text": "relevant chunk", "doc_id": "doc-1", "chunk_index": 0}
        mock_point.score = 0.87

        client = AsyncMock()
        mock_response = MagicMock()
        mock_response.points = [mock_point]
        client.query_points = AsyncMock(return_value=mock_response)

        results = await search_chunks(client, "docs", [0.1] * 384, "doc-1", top_k=5)

        assert len(results) == 1
        assert results[0].score == 0.87
        client.query_points.assert_awaited_once()
        call_kwargs = client.query_points.call_args.kwargs
        assert call_kwargs["limit"] == 5


class TestDeleteDocumentChunks:
    async def test_calls_delete_with_filter(self):
        from app.services.vector_store import delete_document_chunks

        client = AsyncMock()
        await delete_document_chunks(client, "docs", "doc-42")

        client.delete.assert_awaited_once()
        call_kwargs = client.delete.call_args.kwargs
        assert call_kwargs["collection_name"] == "docs"


class TestCountCollection:
    async def test_returns_count(self):
        from app.services.vector_store import count_collection

        client = AsyncMock()
        count_result = MagicMock()
        count_result.count = 99
        client.count = AsyncMock(return_value=count_result)

        result = await count_collection(client, "docs")
        assert result == 99


# ---------------------------------------------------------------------------
# llm tests
# ---------------------------------------------------------------------------

class TestGenerateAnswer:
    async def test_returns_answer_dict(self):
        from app.services.llm import generate_answer

        mock_choice = MagicMock()
        mock_choice.message.content = "The return window is 30 days."
        mock_usage = MagicMock()
        mock_usage.total_tokens = 150
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [
            {"role": "system", "content": "Answer from context."},
            {"role": "user", "content": "Question: What is return window?"},
        ]
        with patch("app.services.llm.groq.AsyncGroq", return_value=mock_client):
            result = await generate_answer(
                messages=messages,
                model="llama-3.3-70b-versatile",
                api_key="test-key",
            )

        assert result["answer"] == "The return window is 30 days."
        assert result["tokens_used"] == 150
        assert result["model"] == "llama-3.3-70b-versatile"

    async def test_rate_limit_raises_429(self):
        import groq as groq_lib

        from app.services.llm import generate_answer

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=groq_lib.RateLimitError("rate limited", response=MagicMock(), body={})
        )

        _msgs = [{"role": "user", "content": "Q?"}]
        with patch("app.services.llm.groq.AsyncGroq", return_value=mock_client):
            with pytest.raises(HTTPException) as exc_info:
                await generate_answer(_msgs, "model", "key")
        assert exc_info.value.status_code == 429

    async def test_api_error_raises_502(self):
        import groq as groq_lib

        from app.services.llm import generate_answer

        mock_client = AsyncMock()
        # APIConnectionError is a subclass of APIError; all args are keyword-only
        mock_client.chat.completions.create = AsyncMock(
            side_effect=groq_lib.APIConnectionError(request=MagicMock())
        )

        _msgs = [{"role": "user", "content": "Q?"}]
        with patch("app.services.llm.groq.AsyncGroq", return_value=mock_client):
            with pytest.raises(HTTPException) as exc_info:
                await generate_answer(_msgs, "model", "key")
        assert exc_info.value.status_code == 502
