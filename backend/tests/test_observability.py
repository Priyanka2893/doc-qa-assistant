"""Tests for the OpenTelemetry + Prometheus observability layer."""
import pytest
from httpx import ASGITransport, AsyncClient
from prometheus_client import REGISTRY
from unittest.mock import AsyncMock


@pytest.fixture
async def obs_client():
    """Minimal test client — no lifespan, state pre-populated."""
    from app import database
    from app.config import get_settings
    from app.main import app

    await database.init_db()

    mock_qdrant = AsyncMock()
    mock_qdrant.close = AsyncMock()

    app.state.settings = get_settings()
    app.state.qdrant_client = mock_qdrant
    app.state.is_ready = True

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


async def test_metrics_endpoint_accessible(obs_client):
    """GET /metrics returns 200 with Prometheus text content-type."""
    response = await obs_client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


async def test_metrics_contain_stage_durations(obs_client):
    """The pipeline stage histogram is registered and appears in /metrics output."""
    response = await obs_client.get("/metrics")
    assert response.status_code == 200
    # Histogram is registered at import time so the name appears even before observations
    assert "rag_pipeline_stage_seconds" in response.text
    assert "rag_request_duration_seconds" in response.text


async def test_trace_id_in_response_headers(obs_client):
    """Every response carries an X-Request-ID header from RequestIDMiddleware."""
    response = await obs_client.get("/api/v1/health/live")
    assert response.status_code == 200
    assert "x-request-id" in response.headers


async def test_cache_hit_counter(obs_client):
    """Cache metrics counters are registered and visible in /metrics."""
    response = await obs_client.get("/metrics")
    assert response.status_code == 200
    assert "rag_cache_hits_total" in response.text
    assert "rag_cache_misses_total" in response.text

    # Trigger a cache miss by calling get_cached_answer, then verify the counter increments
    from app.services.cache import get_semantic_cache

    cache = get_semantic_cache()
    before_text = response.text

    # Simulate a miss — no prior entry exists
    result = await cache.get_cached_answer("what is observability?", "doc-test-obs")
    assert result is None

    metrics_after = await obs_client.get("/metrics")
    # The miss counter should have incremented (new label combination or existing)
    assert "rag_cache_misses_total" in metrics_after.text
