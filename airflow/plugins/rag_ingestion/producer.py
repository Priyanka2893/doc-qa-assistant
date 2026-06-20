from confluent_kafka import Producer, KafkaException
import json
import structlog

log = structlog.get_logger()


def get_producer(bootstrap_servers: str) -> Producer:
    """Create idempotent Kafka producer."""
    return Producer(
        {
            "bootstrap.servers": bootstrap_servers,
            "enable.idempotence": True,
            "acks": "all",
            "retries": 5,
            "max.in.flight.requests.per.connection": 5,
            "client.id": "airflow-rag-producer",
            "linger.ms": 5,
            "batch.size": 65536,
        }
    )


def publish_message(
    producer: Producer,
    topic: str,
    message: dict,
    partition_key: str,
) -> None:
    """Publish a single message. Blocks until delivery confirmed or raises.
    partition_key: typically company name — routes all docs from same company to same partition.
    """

    def delivery_callback(err, msg):
        if err:
            log.error("kafka_delivery_failed", error=str(err), topic=topic)
            raise KafkaException(err)
        log.info(
            "kafka_message_delivered",
            topic=msg.topic(),
            partition=msg.partition(),
            offset=msg.offset(),
            doc_id=message.get("doc_id"),
        )

    producer.produce(
        topic=topic,
        key=partition_key.encode("utf-8"),
        value=json.dumps(message).encode("utf-8"),
        callback=delivery_callback,
    )
    producer.flush(timeout=30)
