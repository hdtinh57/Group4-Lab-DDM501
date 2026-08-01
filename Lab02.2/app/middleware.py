import time
from starlette.middleware.base import BaseHTTPMiddleware
from app.metrics import REQUEST_COUNT, REQUEST_LATENCY


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            endpoint = request.url.path
            REQUEST_COUNT.labels(request.method, endpoint, status).inc()
            REQUEST_LATENCY.labels(request.method, endpoint).observe(time.perf_counter() - started)
