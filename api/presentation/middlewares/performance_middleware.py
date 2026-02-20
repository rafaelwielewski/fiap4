import time
import json
from datetime import datetime
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from api.utils.logger import logger


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Middleware to capture performance metrics for all API calls."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        try:
            response = await call_next(request)

            duration = time.time() - start_time
            duration_ms = round(duration * 1000, 2)

            log_data = {
                'timestamp': datetime.now().isoformat(),
                'method': request.method,
                'path': request.url.path,
                'status_code': response.status_code,
                'duration_ms': duration_ms
            }

            logger.log_request_json(log_data)

            response.headers['X-Response-Time'] = f'{duration_ms}ms'
            response.headers['X-Request-ID'] = str(time.time())

            return response

        except Exception as e:
            duration = time.time() - start_time
            duration_ms = round(duration * 1000, 2)

            error_data = {
                'timestamp': datetime.now().isoformat(),
                'method': request.method,
                'path': request.url.path,
                'error_type': type(e).__name__,
                'error_message': str(e)
            }

            logger.log_error_json(error_data)
            raise
