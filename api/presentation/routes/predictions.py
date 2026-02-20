from fastapi import APIRouter, Depends, HTTPException, Query

from api.domain.models.prediction import PredictionResponse, ModelInfo
from api.domain.repositories.stock_repository import StockRepository
from api.domain.usecases.predictions.predict_stock_price import PredictStockPriceUseCase
from api.domain.usecases.predictions.get_model_info import GetModelInfoUseCase
from api.presentation.routes.router import DefaultRouter
from api.presentation.factories.repository_factory import build_stock_repository
from api.utils.logger import logger

router = APIRouter(route_class=DefaultRouter)


@router.get('/predict',
            summary='Prediz o preço de fechamento daqui 5 dias úteis',
            response_model=PredictionResponse)
def predict_price(
    days: int = Query(default=90, ge=60, description='Número de pregões históricos a considerar (mínimo 60)'),
    repository: StockRepository = Depends(build_stock_repository),
):
    """
    Retorna a previsão de preço de fechamento da AAPL para 5 dias úteis à frente.
    O parâmetro `days` define quantos pregões históricos serão usados (padrão: 90).
    """
    try:
        use_case = PredictStockPriceUseCase(repository)
        result = use_case.execute(days=days)
        logger.info(f'Predição realizada com {days} dias históricos')
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f'Erro na predição: {str(e)}')
        raise HTTPException(status_code=500, detail=f'Erro ao realizar predição: {str(e)}') from e


@router.get('/model-info',
            summary='Retorna informações do modelo LSTM',
            response_model=ModelInfo)
def get_model_info():
    """Retorna informações detalhadas sobre o modelo LSTM, incluindo métricas de performance."""
    try:
        use_case = GetModelInfoUseCase()
        return use_case.execute()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f'Erro ao obter info do modelo: {str(e)}')
        raise HTTPException(status_code=500, detail=f'Erro ao obter informações do modelo: {str(e)}') from e
