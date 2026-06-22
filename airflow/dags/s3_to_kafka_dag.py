from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from datetime import timedelta
import os

S3_ENDPOINT = os.getenv("MINIO_ENDPOINT_INTERNAL", "http://minio:9000")
S3_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "rag-docs")
KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS_INTERNAL", "redpanda:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC_INGEST", "rag.documents.ingest")

default_args = {
    "owner": "rag-team",
    "retries": 3,
    "retry_delay": timedelta(seconds=60),
    "retry_exponential_backoff": True,
    "email_on_failure": False,
}


@dag(
    dag_id="s3_to_kafka_dag",
    description="Detect new files in MinIO upload/ and publish to Kafka",
    schedule_interval="*/5 * * * *",
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["rag", "ingestion", "kafka"],
)
def s3_to_kafka_dag():

    @task
    def list_new_files(**context) -> list[dict]:
        """List all files in MinIO upload/ prefix. Returns list of S3 object dicts."""
        from rag_ingestion.s3_utils import get_s3_client, list_upload_files
        import structlog

        s3 = get_s3_client(S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY)
        files = list_upload_files(s3, S3_BUCKET, prefix="upload/")
        structlog.get_logger().info(
            "files_discovered", count=len(files), dag_run_id=context["run_id"]
        )
        # boto3 datetime objects are not JSON-serialisable; convert before XCom push
        serialisable = []
        for obj in files:
            serialisable.append(
                {
                    "Key": obj["Key"],
                    "Size": obj["Size"],
                    "ETag": obj["ETag"],
                    "LastModified": obj["LastModified"].isoformat(),
                }
            )
        return serialisable

    @task
    def process_file(s3_obj: dict, **context) -> dict:
        """For one file: build message, publish to Kafka, move to done/ or reject/.
        Returns result dict with doc_id and status."""
        from rag_ingestion.schemas import KafkaDocumentMessage
        from rag_ingestion.producer import get_producer, publish_message
        from rag_ingestion.s3_utils import get_s3_client, move_s3_object
        from datetime import datetime
        import structlog

        log = structlog.get_logger()
        s3 = get_s3_client(S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY)
        s3_key = s3_obj["Key"]

        # Re-inflate LastModified as datetime (was serialised to ISO string via XCom)
        s3_obj_normalised = dict(s3_obj)
        if isinstance(s3_obj_normalised["LastModified"], str):
            s3_obj_normalised["LastModified"] = datetime.fromisoformat(
                s3_obj_normalised["LastModified"]
            )

        try:
            filename = s3_key.split("/")[-1]
            done_key = f"done/{filename}"

            # Move first so the consumer always downloads from done/
            move_s3_object(s3, S3_BUCKET, s3_key, done_key)

            msg = KafkaDocumentMessage.from_s3_object(s3_obj_normalised, context["run_id"])
            msg.s3_bucket = S3_BUCKET
            msg.s3_key = done_key  # consumer downloads from done/

            producer = get_producer(KAFKA_SERVERS)
            publish_message(producer, KAFKA_TOPIC, msg.model_dump(), partition_key=msg.company)

            log.info("file_processed_success", doc_id=msg.doc_id, s3_key=done_key)
            return {"doc_id": msg.doc_id, "status": "published", "s3_key": done_key}

        except Exception as e:
            log.error("file_processing_failed", s3_key=s3_key, error=str(e))
            try:
                filename = s3_key.split("/")[-1]
                move_s3_object(s3, S3_BUCKET, s3_key, f"reject/{filename}")
            except Exception as move_err:
                log.error("failed_to_move_to_reject", s3_key=s3_key, error=str(move_err))
            raise

    @task
    def log_summary(results: list[dict], **context) -> None:
        """Log a summary of this DAG run."""
        import structlog

        published = [r for r in results if r.get("status") == "published"]
        structlog.get_logger().info(
            "dag_run_summary",
            dag_run_id=context["run_id"],
            total_files=len(results),
            published=len(published),
            failed=len(results) - len(published),
        )

    files = list_new_files()
    results = process_file.expand(s3_obj=files)
    log_summary(results)


dag_instance = s3_to_kafka_dag()
