from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def http_client():
    """FastAPI test client with app.state pre-populated.

    httpx ASGITransport does not send ASGI lifespan events, so app.state is
    never populated by the lifespan handler. We set it here directly.

    Yields (AsyncClient, mock_qdrant) so tests can program mock_qdrant responses.
    """
    from app import database
    from app.config import get_settings
    from app.main import app

    await database.init_db()

    mock_qdrant = AsyncMock()
    mock_qdrant.close = AsyncMock()

    app.state.settings = get_settings()
    app.state.qdrant_client = mock_qdrant

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client, mock_qdrant
