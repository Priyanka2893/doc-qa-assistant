from pydantic import BaseModel, Field
from pathlib import Path
from datetime import datetime
import uuid


class KafkaDocumentMessage(BaseModel):
    schema_version: str = "1.0"
    event_type: str = "document.ingest.requested"
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    company: str
    category: str
    filename: str
    s3_bucket: str
    s3_key: str
    content_hash: str
    file_size_bytes: int
    file_extension: str
    uploaded_at: str
    airflow_dag_run_id: str
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def from_s3_object(cls, s3_obj: dict, dag_run_id: str) -> "KafkaDocumentMessage":
        """Build a message from an S3/MinIO object metadata dict.
        s3_obj has keys: Key, Size, ETag, LastModified
        """
        s3_key = s3_obj["Key"]
        filename = s3_key.split("/")[-1]

        # Parse filename convention: name__Company__Category__timestamp.ext
        stem = Path(filename).stem
        parts = stem.split("__")
        if len(parts) >= 4:
            company = parts[1]
            category = parts[2]
        else:
            company = "unknown"
            category = "general"

        file_extension = Path(filename).suffix.lower()
        content_hash = f"md5:{s3_obj['ETag'].strip(chr(34))}"

        last_modified = s3_obj["LastModified"]
        uploaded_at = (
            last_modified.isoformat()
            if isinstance(last_modified, datetime)
            else str(last_modified)
        )

        return cls(
            company=company,
            category=category,
            filename=filename,
            s3_bucket="",  # caller sets bucket separately; filled in DAG
            s3_key=s3_key,
            content_hash=content_hash,
            file_size_bytes=s3_obj["Size"],
            file_extension=file_extension,
            uploaded_at=uploaded_at,
            airflow_dag_run_id=dag_run_id,
        )
