import datetime
import json
import os

import structlog
from confluent_kafka import Producer

log = structlog.get_logger()

_producer: Producer | None = None


def get_status_producer() -> Producer:
    global _producer
    if _producer is None:
        _producer = Producer(
            {
                "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092"),
                "client.id": "rag-consumer-status",
                "acks": "1",
            }
        )
    return _producer


def publish_status(doc_id: str, status: str, details: dict | None = None) -> None:
    """Publish processing status. status: 'processing' | 'completed' | 'failed' | 'duplicate'"""
    topic = os.getenv("KAFKA_TOPIC_STATUS", "rag.documents.status")
    msg = {
        "doc_id": doc_id,
        "status": status,
        "details": details or {},
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }
    p = get_status_producer()
    p.produce(topic, key=doc_id.encode(), value=json.dumps(msg).encode())
    p.poll(0)
