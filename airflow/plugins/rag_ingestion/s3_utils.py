import boto3
from botocore.client import Config
import structlog

log = structlog.get_logger()


def get_s3_client(endpoint_url: str, access_key: str, secret_key: str):
    """Create and return a boto3 S3 client for MinIO."""
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def list_upload_files(s3_client, bucket: str, prefix: str = "upload/") -> list[dict]:
    """List all files in the upload/ prefix. Returns list of S3 object dicts.
    Excludes .keep placeholder files."""
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    objects = []
    for page in pages:
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".keep"):
                continue
            objects.append(obj)

    return objects


def download_file_bytes(s3_client, bucket: str, key: str) -> bytes:
    """Download file content from S3/MinIO. Returns bytes."""
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


def move_s3_object(s3_client, bucket: str, src_key: str, dest_key: str) -> None:
    """Copy object to dest_key then delete from src_key (atomic move simulation)."""
    s3_client.copy_object(
        Bucket=bucket,
        CopySource={"Bucket": bucket, "Key": src_key},
        Key=dest_key,
    )
    s3_client.delete_object(Bucket=bucket, Key=src_key)
    log.info("s3_object_moved", bucket=bucket, src_key=src_key, dest_key=dest_key)
