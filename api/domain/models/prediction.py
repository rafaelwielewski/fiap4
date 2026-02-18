from typing import List, Dict, Any, Optional
from datetime import datetime

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Parâmetros para predição de preço de ação."""
    days_ahead: int = Field(default=7, ge=1, le=30, description='Número de dias para prever (1-30)')


class PredictedPrice(BaseModel):
    """Um preço previsto para uma data futura."""
    date: str
    predicted_close: float


class PredictionResponse(BaseModel):
    """Resultado da predição de preços."""
    symbol: str
    predictions: List[PredictedPrice]
    model_version: str
    generated_at: str
    metrics: Dict[str, float]


class ModelMetrics(BaseModel):
    """Métricas de avaliação do modelo."""
    mae: float = Field(description='Mean Absolute Error')
    rmse: float = Field(description='Root Mean Square Error')
    mape: float = Field(description='Mean Absolute Percentage Error')


class ModelInfo(BaseModel):
    """Informações sobre o modelo LSTM."""
    model_name: str
    description: str
    version: str
    symbol: str
    training_period: Dict[str, str]
    sequence_length: int
    features_used: List[str]
    metrics: ModelMetrics
    last_trained: str
    is_active: bool
