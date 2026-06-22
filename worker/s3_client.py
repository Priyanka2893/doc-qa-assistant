import os

import boto3
import structlog
from botocore.client import Config

log = structlog.get_logger()


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", "minioadmin123"),
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def download_document(s3_key: str, bucket: str | None = None) -> bytes:
    """Download a document from MinIO. Returns raw bytes.

    If the key is not found in upload/ (DAG may have moved it already),
    falls back to the same filename under done/.
    """
    bucket = bucket or os.getenv("S3_BUCKET_NAME", "rag-docs")
    s3 = get_s3_client()
    try:
        response = s3.get_object(Bucket=bucket, Key=s3_key)
    except s3.exceptions.NoSuchKey:
        filename = s3_key.split("/")[-1]
        fallback_key = f"done/{filename}"
        log.warning("s3_key_not_found_trying_done", original_key=s3_key, fallback_key=fallback_key)
        response = s3.get_object(Bucket=bucket, Key=fallback_key)
        s3_key = fallback_key
    content = response["Body"].read()
    log.info("document_downloaded", s3_key=s3_key, size_bytes=len(content))
    return content
