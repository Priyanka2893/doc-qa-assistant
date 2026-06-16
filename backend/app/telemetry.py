"""Centralised Prometheus metrics + OpenTelemetry tracer for the RAG backend."""
from __future__ import annotations

import time
from contextlib import contextmanager
from functools import wraps
from typing import Generator

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from prometheus_client import Counter, Gauge, Histogram

# ─── Prometheus metrics ───────────────────────────────────────────────────────

REQUEST_DURATION = Histogram(
    "rag_request_duration_seconds",
    "Request duration by endpoint",
    ["method", "endpoint", "status_code"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

PIPELINE_STAGE_DURATION = Histogram(
    "rag_pipeline_stage_seconds",
    "Duration of each pipeline stage",
    ["stage"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

DOCUMENTS_TOTAL = Gauge("rag_documents_total", "Total documents in system")
CHUNKS_TOTAL = Gauge("rag_chunks_total", "Total chunks in vector DB")

FAITHFULNESS_GAUGE = Gauge("rag_faithfulness_score", "Current average faithfulness score")
CONTEXT_RELEVANCE_GAUGE = Gauge("rag_context_relevance_score", "Current avg context relevance")
ANSWER_RELEVANCE_GAUGE = Gauge("rag_answer_relevance_score", "Current avg answer relevance")
HALLUCINATION_RATE_GAUGE = Gauge("rag_hallucination_rate", "Current hallucination rate")

CACHE_HITS = Counter("rag_cache_hits_total", "Cache hit count", ["cache_type"])
CACHE_MISSES = Counter("rag_cache_misses_total", "Cache miss count")

ERRORS_TOTAL = Counter("rag_errors_total", "Error count", ["error_type"])

INGESTION_TOTAL = Counter("rag_ingestion_total", "Documents ingested", ["file_format", "status"])
CHUNKS_DEDUPED = Counter("rag_chunks_deduped_total", "Chunks removed by dedup", ["dedup_type"])


# ─── Pipeline stage timing ────────────────────────────────────────────────────

@contextmanager
def track_stage(stage_name: str) -> Generator[None, None, None]:
    """Record the wall-clock duration of a named pipeline stage to Prometheus."""
    start = time.perf_counter()
    try:
        yield
    finally:
        PIPELINE_STAGE_DURATION.labels(stage=stage_name).observe(time.perf_counter() - start)


# ─── OpenTelemetry tracer ─────────────────────────────────────────────────────

def _setup_tracing(service_name: str = "rag-backend") -> trace.Tracer:
    resource = Resource.create({"service.name": service_name, "service.version": "1.0.0"})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)


TRACER: trace.Tracer = _setup_tracing()


def get_trace_id() -> str:
    """Return the current OTel trace ID as a 32-char hex string, or 'no-trace'."""
    span = trace.get_current_span()
    ctx = span.get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else "no-trace"


# ─── Span decorator for async functions ───────────────────────────────────────

def traced(span_name: str | None = None):
    """Wrap an async function in an OTel span. Sets OK/ERROR status automatically."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            name = span_name or func.__name__
            with TRACER.start_as_current_span(name) as span:
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(trace.StatusCode.OK)
                    return result
                except Exception as exc:
                    span.set_status(trace.StatusCode.ERROR, str(exc))
                    span.record_exception(exc)
                    raise
        return wrapper
    return decorator
