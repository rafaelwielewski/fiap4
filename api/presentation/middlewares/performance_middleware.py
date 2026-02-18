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
            # Process the request
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time
            duration_ms = round(duration * 1000, 2)

            # Create log data with timestamp
            log_data = {
                'timestamp': datetime.now().isoformat(),
                'method': request.method,
                'path': request.url.path,
                'status_code': response.status_code,
                'duration_ms': duration_ms
            }

            # Log the request/response
            logger.log_request_json(log_data)

            # Add performance headers
            response.headers['X-Response-Time'] = f'{duration_ms}ms'
            response.headers['X-Request-ID'] = str(time.time())

            return response

        except Exception as e:
            # Calculate duration even for errors
            duration = time.time() - start_time
            duration_ms = round(duration * 1000, 2)

            # Create error log data
            error_data = {
                'timestamp': datetime.now().isoformat(),
                'method': request.method,
                'path': request.url.path,
                'error_type': type(e).__name__,
                'error_message': str(e)
            }

            # Log the error
            logger.log_error_json(error_data)

            # Re-raise the exception
            raise
