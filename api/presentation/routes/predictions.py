from fastapi import APIRouter, Depends, HTTPException

from api.domain.models.prediction import CustomPredictionRequest, PredictionResponse, ModelInfo
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
def predict_price(repository: StockRepository = Depends(build_stock_repository)):
    """
    Retorna a previsão de preço de fechamento da AAPL para 5 dias úteis à frente,
    utilizando o modelo LSTM treinado com janela de 60 dias.
    """
    try:
        use_case = PredictStockPriceUseCase(repository)
        result = use_case.execute()
        logger.info('Predição realizada: 5 dias úteis à frente')
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f'Erro na predição: {str(e)}')
        raise HTTPException(status_code=500, detail=f'Erro ao realizar predição: {str(e)}') from e


@router.post('/predict-custom',
             summary='Prediz o preço daqui 5 dias com dados fornecidos pelo usuário',
             response_model=PredictionResponse)
def predict_custom(
    request: CustomPredictionRequest,
    repository: StockRepository = Depends(build_stock_repository)
):
    """
    Recebe dados históricos de fechamento (mínimo 90 registros) e retorna
    a previsão de preço para 5 dias úteis à frente.
    """
    try:
        use_case = PredictStockPriceUseCase(repository)
        historical_data = [{'date': p.date, 'close': p.close} for p in request.historical_data]
        result = use_case.execute_custom(historical_prices=historical_data)
        logger.info(f'Predição customizada: {len(request.historical_data)} registros')
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
