from fastapi import APIRouter

from api.domain.usecases.analytics_service import AnalyticsService
from api.presentation.routes.router import DefaultRouter

router = APIRouter(route_class=DefaultRouter)

analytics_service = AnalyticsService()


@router.get('/metrics', summary='Retorna métricas gerais da API')
def get_metrics():
    """
    Retorna métricas agregadas: total de requests, tempo médio de resposta,
    taxa de erros, requests por endpoint/método, atividade recente.
    """
    return analytics_service.get_metrics()


@router.get('/performance', summary='Retorna métricas de performance detalhadas')
def get_performance():
    """
    Retorna distribuição de response time (min, max, median, p95, p99),
    performance por endpoint e breakdown de erros.
    """
    return analytics_service.get_performance()
