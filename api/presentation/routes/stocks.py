from typing import Optional

from fastapi import APIRouter, Depends, Query

from api.domain.models.stock import StockHistory
from api.domain.repositories.stock_repository import StockRepository
from api.domain.usecases.stocks.get_stock_history import GetStockHistoryUseCase
from api.domain.usecases.stocks.get_stock_data import GetStockDataUseCase
from api.presentation.routes.router import DefaultRouter
from api.presentation.factories.repository_factory import build_stock_repository

router = APIRouter(route_class=DefaultRouter)


@router.get('/history',
            summary='Retorna histórico de dados da ação',
            response_model=StockHistory)
def get_history(
    limit: Optional[int] = Query(None, ge=1, le=5000, description='Número máximo de registros'),
    repository: StockRepository = Depends(build_stock_repository)
):
    """Retorna o histórico completo ou limitado de dados da ação Apple (AAPL)."""
    use_case = GetStockHistoryUseCase(repository)
    return use_case.execute(limit=limit)


@router.get('/latest',
            summary='Retorna os dados mais recentes da ação')
def get_latest(
    n: int = Query(30, ge=1, le=500, description='Quantidade de registros recentes'),
    repository: StockRepository = Depends(build_stock_repository)
):
    """Retorna os N registros mais recentes de dados da ação Apple (AAPL)."""
    use_case = GetStockDataUseCase(repository)
    return use_case.execute(n=n)
