"""
RAG Kafka Consumer

Polls rag.documents.ingest topic and processes each document
through the existing RAG ingestion pipeline.

Usage: cd backend && PYTHONPATH=/path/to/doc-qa-assistant uv run python -m worker.consumer
"""

import asyncio
import datetime
import json
import os
import signal

import structlog
from confluent_kafka import Consumer, KafkaError, Producer as KafkaProducer

from worker.ingestion_handler import handle_document_message
from worker.schemas import KafkaDocumentMessage

log = structlog.get_logger()

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC_INGEST", "rag.documents.ingest")
KAFKA_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "rag-ingestion-group")
KAFKA_DLQ_TOPIC = os.getenv("KAFKA_TOPIC_DLQ", "rag.documents.deadletter")
MAX_RETRIES = int(os.getenv("KAFKA_MAX_RETRIES", "3"))
POLL_TIMEOUT = float(os.getenv("KAFKA_POLL_TIMEOUT_SECONDS", "1.0"))


def get_dlq_producer() -> KafkaProducer:
    return KafkaProducer({"bootstrap.servers": KAFKA_BOOTSTRAP, "acks": "1"})


def publish_to_dlq(
    dlq_producer: KafkaProducer,
    original_msg_value: bytes,
    error_reason: str,
    doc_id: str,
) -> None:
    try:
        original: object = json.loads(original_msg_value.decode())
    except Exception:
        original = original_msg_value.decode(errors="replace")
    dlq_msg = {
        "original_message": original,
        "error_reason": error_reason,
        "failed_at": datetime.datetime.utcnow().isoformat(),
        "consumer_group": KAFKA_GROUP,
    }
    dlq_producer.produce(
        KAFKA_DLQ_TOPIC,
        key=doc_id.encode(),
        value=json.dumps(dlq_msg).encode(),
    )
    dlq_producer.flush(timeout=10)


async def process_message_with_retry(
    msg_value: bytes, dlq_producer: KafkaProducer
) -> bool:
    """Process a single Kafka message. Returns True when offset should be committed."""
    doc_message = None
    try:
        doc_message = KafkaDocumentMessage.from_kafka_bytes(msg_value)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = await handle_document_message(doc_message)
                log.info(
                    "message_processed",
                    doc_id=doc_message.doc_id,
                    status=result["status"],
                    attempt=attempt,
                )
                return True

            except Exception as e:
                log.warning(
                    "processing_attempt_failed",
                    doc_id=doc_message.doc_id,
                    attempt=attempt,
                    max_retries=MAX_RETRIES,
                    error=str(e),
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(2**attempt)
                else:
                    log.error("message_sent_to_dlq", doc_id=doc_message.doc_id, error=str(e))
                    publish_to_dlq(dlq_producer, msg_value, str(e), doc_message.doc_id)
                    return True

    except Exception as parse_err:
        log.error("message_parse_failed", error=str(parse_err))
        publish_to_dlq(dlq_producer, msg_value, f"parse_error: {parse_err}", "unknown")
        return True

    return True


async def run_consumer() -> None:
    log.info(
        "consumer_starting",
        topic=KAFKA_TOPIC,
        group=KAFKA_GROUP,
        bootstrap=KAFKA_BOOTSTRAP,
    )

    consumer = Consumer(
        {
            "bootstrap.servers": KAFKA_BOOTSTRAP,
            "group.id": KAFKA_GROUP,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
            "max.poll.interval.ms": 300000,
            "session.timeout.ms": 30000,
            "heartbeat.interval.ms": 3000,
        }
    )

    dlq_producer = get_dlq_producer()
    running = True

    def shutdown_handler(sig, frame):
        nonlocal running
        log.info("shutdown_signal_received", signal=str(sig))
        running = False

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    consumer.subscribe([KAFKA_TOPIC])
    log.info("consumer_subscribed", topic=KAFKA_TOPIC)

    try:
        while running:
            msg = consumer.poll(timeout=POLL_TIMEOUT)

            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    log.debug(
                        "partition_eof",
                        topic=msg.topic(),
                        partition=msg.partition(),
                        offset=msg.offset(),
                    )
                else:
                    log.error("kafka_consumer_error", error=msg.error())
                continue

            log.info(
                "message_received",
                topic=msg.topic(),
                partition=msg.partition(),
                offset=msg.offset(),
                key=msg.key().decode() if msg.key() else None,
            )

            should_commit = await process_message_with_retry(msg.value(), dlq_producer)

            if should_commit:
                consumer.commit(message=msg, asynchronous=False)
                log.debug(
                    "offset_committed",
                    partition=msg.partition(),
                    offset=msg.offset(),
                )

    finally:
        consumer.close()
        log.info("consumer_stopped")


if __name__ == "__main__":
    asyncio.run(run_consumer())
