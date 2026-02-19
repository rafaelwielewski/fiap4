from fastapi import APIRouter, Depends, HTTPException

from api.domain.models.prediction import PredictionRequest, CustomPredictionRequest, PredictionResponse, ModelInfo
from api.domain.repositories.stock_repository import StockRepository
from api.domain.usecases.predictions.predict_stock_price import PredictStockPriceUseCase
from api.domain.usecases.predictions.get_model_info import GetModelInfoUseCase
from api.presentation.routes.router import DefaultRouter
from api.presentation.factories.repository_factory import build_stock_repository
from api.utils.logger import logger

router = APIRouter(route_class=DefaultRouter)


@router.post('/predict',
             summary='Prediz preços futuros da ação',
             response_model=PredictionResponse)
def predict_price(
    request: PredictionRequest,
    repository: StockRepository = Depends(build_stock_repository)
):
    """
    Realiza predição de preços futuros usando o modelo LSTM.

    Envie o número de dias para prever (1-30) e receba as predições.
    Utiliza os dados históricos pré-carregados da ação PETR4.SA.
    """
    try:
        use_case = PredictStockPriceUseCase(repository)
        result = use_case.execute(days_ahead=request.days_ahead)
        logger.info(f'Predição realizada: {request.days_ahead} dias')
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f'Erro na predição: {str(e)}')
        raise HTTPException(status_code=500, detail=f'Erro ao realizar predição: {str(e)}') from e


@router.post('/predict-custom',
             summary='Prediz preços futuros com dados fornecidos pelo usuário',
             response_model=PredictionResponse)
def predict_custom(
    request: CustomPredictionRequest,
    repository: StockRepository = Depends(build_stock_repository)
):
    """
    Realiza predição de preços futuros usando dados históricos fornecidos pelo usuário.

    O usuário deve fornecer ao menos 60 registros de preços históricos (date + close)
    e o número de dias para prever (1-30). O modelo LSTM processa os dados e retorna
    as predições futuras.
    """
    try:
        use_case = PredictStockPriceUseCase(repository)
        historical_data = [{'date': p.date, 'close': p.close} for p in request.historical_data]
        result = use_case.execute_custom(
            historical_prices=historical_data,
            days_ahead=request.days_ahead
        )
        logger.info(f'Predição customizada: {len(request.historical_data)} registros, {request.days_ahead} dias')
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f'Erro na predição customizada: {str(e)}')
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
