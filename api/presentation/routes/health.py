from fastapi import APIRouter, Depends
from api.domain.repositories.stock_repository import StockRepository
from api.domain.usecases.health.get_health_status import GetHealthStatusUseCase
from api.presentation.routes.router import DefaultRouter
from api.presentation.factories.repository_factory import build_stock_repository

router = APIRouter(route_class=DefaultRouter)


@router.get('/', summary='Verifica o status da API e leitura dos dados')
def health_check(repository: StockRepository = Depends(build_stock_repository)):
    """Verifica o status de saúde da aplicação."""
    use_case = GetHealthStatusUseCase(repository)
    return use_case.execute()
