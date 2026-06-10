import time

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.middleware.request_id import request_id_var

logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        t_start = time.perf_counter()
        response = await call_next(request)
        duration_ms = int((time.perf_counter() - t_start) * 1000)

        req_id = request_id_var.get("")
        status = response.status_code

        log_kwargs = dict(
            event="request",
            method=request.method,
            path=request.url.path,
            status=status,
            duration_ms=duration_ms,
            request_id=req_id,
        )

        if status >= 500:
            logger.error(**log_kwargs)
        elif status >= 400:
            logger.warning(**log_kwargs)
        else:
            logger.info(**log_kwargs)

        return response
