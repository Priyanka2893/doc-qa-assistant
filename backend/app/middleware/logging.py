import time

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.middleware.request_id import request_id_var
from app.telemetry import REQUEST_DURATION

logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        t_start = time.perf_counter()
        response = await call_next(request)
        duration_s = time.perf_counter() - t_start
        duration_ms = int(duration_s * 1000)

        req_id = request_id_var.get("")
        status = response.status_code

        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=str(status),
        ).observe(duration_s)

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
