"""
Tests for the Kafka consumer and ingestion handler.

Each test mocks at the boundary — Kafka, S3, and backend services —
so no external infrastructure is needed.
"""

import asyncio
import json
import signal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from worker.consumer import process_message_with_retry, publish_to_dlq
from worker.schemas import KafkaDocumentMessage

_SAMPLE_MSG = KafkaDocumentMessage(
    doc_id="doc-001",
    company="Acme",
    category="Finance",
    filename="report__Acme__Finance__20240101.pdf",
    s3_bucket="rag-docs",
    s3_key="upload/report__Acme__Finance__20240101.pdf",
    content_hash="sha256:abc123",
    file_size_bytes=1024,
    file_extension=".pdf",
    uploaded_at="2024-01-01T00:00:00",
    airflow_dag_run_id="run-001",
)


def _msg_bytes(msg: KafkaDocumentMessage = _SAMPLE_MSG) -> bytes:
    return msg.model_dump_json().encode()


# ── test_duplicate_skip ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_duplicate_skip():
    existing_doc = {"doc_id": "doc-existing", "filename": "same.pdf"}

    with (
        patch(
            "worker.ingestion_handler.get_document_by_hash",
            new=AsyncMock(return_value=existing_doc),
        ),
        patch("worker.ingestion_handler.publish_status") as mock_pub,
    ):
        from worker.ingestion_handler import handle_document_message

        result = await handle_document_message(_SAMPLE_MSG)

    assert result["status"] == "skipped"
    assert result["doc_id"] == "doc-001"
    assert result["chunk_count"] == 0
    mock_pub.assert_called_once_with(
        "doc-001", "duplicate", {"existing_doc_id": "doc-existing"}
    )


# ── test_successful_ingestion ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_successful_ingestion():
    from app.services.parser import DocumentMetadata, ParseResult

    fake_parse = ParseResult(
        text="hello world",
        chunks=["chunk one", "chunk two"],
        page_count=1,
        metadata=DocumentMetadata(
            author="Alice", title="Report", language="en", word_count=2
        ),
    )

    with (
        patch(
            "worker.ingestion_handler.get_document_by_hash",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "worker.ingestion_handler.download_document",
            return_value=b"%PDF-fake",
        ),
        patch(
            "worker.ingestion_handler.parse_and_chunk",
            return_value=fake_parse,
        ),
        patch(
            "worker.ingestion_handler.deduplicate_exact",
            return_value=(["chunk one", "chunk two"], 0),
        ),
        patch(
            "worker.ingestion_handler.deduplicate_semantic",
            new=AsyncMock(return_value=(["chunk one", "chunk two"], 0)),
        ),
        patch(
            "worker.ingestion_handler.async_encode_texts",
            new=AsyncMock(return_value=[[0.1] * 384, [0.2] * 384]),
        ),
        patch(
            "worker.ingestion_handler._get_qdrant_client",
        ) as mock_client_factory,
        patch(
            "worker.ingestion_handler.init_collection",
            new=AsyncMock(),
        ),
        patch(
            "worker.ingestion_handler.upsert_chunks",
            new=AsyncMock(return_value=["id1", "id2"]),
        ),
        patch(
            "worker.ingestion_handler.insert_document_kafka",
            new=AsyncMock(),
        ) as mock_db,
        patch("worker.ingestion_handler.publish_status") as mock_pub,
    ):
        mock_qdrant = AsyncMock()
        mock_qdrant.close = AsyncMock()
        mock_client_factory.return_value = mock_qdrant

        from worker.ingestion_handler import handle_document_message

        result = await handle_document_message(_SAMPLE_MSG)

    assert result["status"] == "completed"
    assert result["chunk_count"] == 2
    mock_db.assert_awaited_once()
    assert mock_pub.call_args_list[0][0] == ("doc-001", "processing")
    assert mock_pub.call_args_list[1][0][1] == "completed"


# ── test_dlq_on_parse_error ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dlq_on_parse_error():
    malformed = b"not-valid-json!!!"
    dlq_producer = MagicMock()
    dlq_producer.produce = MagicMock()
    dlq_producer.flush = MagicMock()

    committed = await process_message_with_retry(malformed, dlq_producer)

    assert committed is True
    dlq_producer.produce.assert_called_once()
    call_kwargs = dlq_producer.produce.call_args
    dlq_value = json.loads(call_kwargs[1]["value"].decode())
    assert "parse_error" in dlq_value["error_reason"]


# ── test_retry_on_transient_error ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retry_on_transient_error():
    call_count = 0

    async def flaky_handler(msg):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("transient failure")
        return {"status": "completed", "doc_id": msg.doc_id, "chunk_count": 5}

    dlq_producer = MagicMock()

    with (
        patch("worker.consumer.handle_document_message", side_effect=flaky_handler),
        patch("worker.consumer.asyncio.sleep", new=AsyncMock()),
    ):
        committed = await process_message_with_retry(_msg_bytes(), dlq_producer)

    assert committed is True
    assert call_count == 3
    dlq_producer.produce.assert_not_called()


# ── test_graceful_shutdown ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_graceful_shutdown():
    """Sending SIGTERM while idle should stop the poll loop and close the consumer."""

    mock_consumer = MagicMock()
    # Return None on every poll (no messages)
    msg_sequence = iter([None, None])

    def fake_poll(timeout):
        try:
            return next(msg_sequence)
        except StopIteration:
            # Send SIGTERM after exhausting the sequence
            signal.raise_signal(signal.SIGTERM)
            return None

    mock_consumer.poll.side_effect = fake_poll
    mock_consumer.subscribe = MagicMock()
    mock_consumer.close = MagicMock()

    with (
        patch("worker.consumer.Consumer", return_value=mock_consumer),
        patch("worker.consumer.get_dlq_producer", return_value=MagicMock()),
    ):
        from worker.consumer import run_consumer

        await run_consumer()

    mock_consumer.close.assert_called_once()
