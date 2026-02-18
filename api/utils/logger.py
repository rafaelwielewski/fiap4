import datetime
import os
import json
import threading
from typing import Dict, List, Any, Optional
from collections import deque


class Logger:
    """Unified logger for serverless environments with file and memory storage."""

    def __init__(self, max_logs: int = 1000):
        self.max_logs = max_logs
        self.logs = deque(maxlen=max_logs)
        self.lock = threading.Lock()

        # Check environment
        self.is_local = self._is_local_environment()

        # Initialize storage backends
        self.file_logging_enabled = False

        # Setup storage based on environment
        self._setup_storage()

    def _is_local_environment(self) -> bool:
        """Check if we're running in local environment."""
        local_indicators = [
            os.getenv('ENV') == 'LOCAL',
        ]
        return any(local_indicators)

    def _setup_storage(self):
        """Setup storage backends based on environment."""
        if self.is_local:
            self._setup_file_logging()

    def _setup_file_logging(self):
        """Setup file logging for local development."""
        try:
            log_dir = 'logs'
            log_file = os.path.join(log_dir, 'api.log')

            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            with open(log_file, 'a') as f:
                f.write('')

            self.log_file = log_file
            self.file_logging_enabled = True
            print('✅ File logging enabled for local development')
        except Exception as e:
            print(f'❌ File logging setup failed: {e}')
            self.file_logging_enabled = False

    def _store_in_file(self, log_data: Dict[str, Any]) -> bool:
        """Store log entry in file."""
        if not self.file_logging_enabled:
            return False

        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_data) + '\n')
            return True
        except Exception as e:
            print(f'Failed to store log in file: {e}')
            return False

    def __log(self, level: str, message: str, data: Optional[Dict] = None):
        """Internal method to log messages."""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = {
            'timestamp': timestamp,
            'level': level,
            'message': message,
            'data': data
        }

        # Print to console (always available)
        print(f'[{timestamp}] [{level}] {message}')
        if data:
            print(f'Data: {json.dumps(data, indent=2)}')

        # Store in memory for all environments
        with self.lock:
            self.logs.append(log_entry)

        # Store in file for local development
        if self.is_local:
            self._store_in_file(log_entry)

    def debug(self, message: str, data: Optional[Dict] = None):
        """Log a debug message."""
        self.__log('DEBUG', message, data)

    def info(self, message: str, data: Optional[Dict] = None):
        """Log an info message."""
        self.__log('INFO', message, data)

    def warning(self, message: str, data: Optional[Dict] = None):
        """Log a warning message."""
        self.__log('WARNING', message, data)

    def error(self, message: str, data: Optional[Dict] = None):
        """Log an error message."""
        self.__log('ERROR', message, data)

    def critical(self, message: str, data: Optional[Dict] = None):
        """Log a critical message."""
        self.__log('CRITICAL', message, data)

    def log_request_json(self, log_data: Dict[str, Any]):
        """Log an API request in JSON format."""
        self.info(f'API_CALL: {json.dumps(log_data)}', log_data)

    def log_error_json(self, error_data: Dict[str, Any]):
        """Log an API error in JSON format."""
        self.error(f'API_ERROR: {json.dumps(error_data)}', error_data)

    def get_logs(self, level: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
        """Get stored logs, optionally filtered by level."""
        with self.lock:
            logs = list(self.logs)

        # Sort by timestamp (newest first)
        logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        # Filter by level
        if level:
            logs = [log for log in logs if log.get('level') == level]

        # Apply limit
        if limit:
            logs = logs[:limit]

        return logs

    def clear_logs(self) -> bool:
        """Clear all stored logs."""
        try:
            with self.lock:
                self.logs.clear()

            if self.file_logging_enabled:
                try:
                    with open(self.log_file, 'w') as f:
                        f.write('')
                except Exception as e:
                    print(f'Failed to clear file logs: {e}')

            return True
        except Exception as e:
            print(f'Failed to clear logs: {e}')
            return False


# Global logger instance
logger = Logger()
