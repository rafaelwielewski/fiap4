"""
Serviço de analytics que processa logs do logger global
e calcula métricas agregadas para o dashboard de monitoramento.
"""

import json
import statistics
from collections import Counter
from typing import Any, Dict, List, Optional

from api.utils.logger import logger


class AnalyticsService:
    """Processa logs em memória e calcula métricas de performance da API."""

    def _get_request_logs(self) -> List[Dict[str, Any]]:
        """Extrai logs de requisições (API_CALL) do logger."""
        all_logs = logger.get_logs()
        request_logs = []
        for log in all_logs:
            if log.get('data') and 'duration_ms' in log.get('data', {}):
                request_logs.append(log['data'])
        return request_logs

    def _get_error_logs(self) -> List[Dict[str, Any]]:
        """Extrai logs de erros do logger."""
        all_logs = logger.get_logs(level='ERROR')
        error_logs = []
        for log in all_logs:
            if log.get('data') and 'error_type' in log.get('data', {}):
                error_logs.append(log['data'])
        return error_logs

    def get_metrics(self) -> Dict[str, Any]:
        """Calcula métricas gerais da API."""
        request_logs = self._get_request_logs()
        error_logs = self._get_error_logs()

        total_requests = len(request_logs)
        total_errors = len(error_logs)

        # Tempo médio de resposta
        durations = [r['duration_ms'] for r in request_logs if 'duration_ms' in r]
        avg_response_time = statistics.mean(durations) if durations else 0

        # Taxa de erros
        error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0

        # Requests por endpoint
        endpoint_counter = Counter(r.get('path', 'unknown') for r in request_logs)
        requests_by_endpoint = dict(endpoint_counter.most_common())

        # Requests por método HTTP
        method_counter = Counter(r.get('method', 'unknown') for r in request_logs)
        requests_by_method = dict(method_counter.most_common())

        # Requests por status code
        status_counter = Counter(r.get('status_code', 0) for r in request_logs)
        requests_by_status = {str(k): v for k, v in status_counter.most_common()}

        # Top endpoints
        top_endpoints = [
            {'endpoint': endpoint, 'count': count}
            for endpoint, count in endpoint_counter.most_common(10)
        ]

        # Atividade recente (últimas 20)
        recent_activity = sorted(
            request_logs, key=lambda x: x.get('timestamp', ''), reverse=True
        )[:20]

        return {
            'total_requests': total_requests,
            'total_errors': total_errors,
            'average_response_time': round(avg_response_time, 2),
            'error_rate': round(error_rate, 2),
            'requests_by_endpoint': requests_by_endpoint,
            'requests_by_method': requests_by_method,
            'requests_by_status': requests_by_status,
            'top_endpoints': top_endpoints,
            'recent_activity': recent_activity,
        }

    def get_performance(self) -> Dict[str, Any]:
        """Calcula métricas detalhadas de performance."""
        request_logs = self._get_request_logs()
        error_logs = self._get_error_logs()

        total_requests = len(request_logs)
        total_errors = len(error_logs)

        # Distribuição de response time
        durations = sorted([r['duration_ms'] for r in request_logs if 'duration_ms' in r])
        rt_distribution = {}
        if durations:
            rt_distribution = {
                'min': round(min(durations), 2),
                'max': round(max(durations), 2),
                'mean': round(statistics.mean(durations), 2),
                'median': round(statistics.median(durations), 2),
                'p95': round(durations[int(len(durations) * 0.95)] if len(durations) > 1 else durations[0], 2),
                'p99': round(durations[int(len(durations) * 0.99)] if len(durations) > 1 else durations[0], 2),
            }

        # Performance por endpoint
        endpoint_perf = {}
        for r in request_logs:
            path = r.get('path', 'unknown')
            dur = r.get('duration_ms', 0)
            if path not in endpoint_perf:
                endpoint_perf[path] = {'durations': [], 'count': 0}
            endpoint_perf[path]['durations'].append(dur)
            endpoint_perf[path]['count'] += 1

        endpoint_performance = {}
        for path, data in endpoint_perf.items():
            durs = data['durations']
            endpoint_performance[path] = {
                'request_count': data['count'],
                'avg_response_time': round(statistics.mean(durs), 2),
                'min_response_time': round(min(durs), 2),
                'max_response_time': round(max(durs), 2),
            }

        # Error breakdown
        error_counter = Counter(e.get('error_type', 'unknown') for e in error_logs)
        error_breakdown = dict(error_counter.most_common())

        # System health
        success_count = total_requests - total_errors
        success_rate = (success_count / total_requests * 100) if total_requests > 0 else 100

        system_health = {
            'total_requests': total_requests,
            'error_count': total_errors,
            'success_rate': round(success_rate, 2),
            'uptime_percentage': round(success_rate, 1),
        }

        return {
            'system_health': system_health,
            'response_time_distribution': rt_distribution,
            'endpoint_performance': endpoint_performance,
            'error_breakdown': error_breakdown,
        }
