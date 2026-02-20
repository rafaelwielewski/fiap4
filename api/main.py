from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from api.presentation.middlewares.error_handlers import (
    request_validation_error_handler,
    validation_error_handler,
)
from api.presentation.middlewares.performance_middleware import PerformanceMiddleware
from api.presentation.routes import health, stocks, predictions, analytics


app = FastAPI(
    title='Stock Price Prediction API',
    version='1.0.0',
    description='API para predição de preços de ações usando modelo LSTM (Deep Learning). '
                'Projeto FIAP - Tech Challenge Fase 4.'
)

app.add_middleware(PerformanceMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(health.router, prefix='/api/v1/health', tags=['Health'])
app.include_router(stocks.router, prefix='/api/v1/stocks', tags=['Stocks'])
app.include_router(predictions.router, prefix='/api/v1/predictions', tags=['Predictions'])
app.include_router(analytics.router, prefix='/api/v1/analytics', tags=['Analytics'])

app.add_exception_handler(ValidationError, validation_error_handler)
app.add_exception_handler(RequestValidationError, request_validation_error_handler)


@app.get('/')
def root():
    return {
        'message': 'Stock Price Prediction API - FIAP Tech Challenge Fase 4',
        'docs': '/docs',
        'endpoints': {
            'health': '/api/v1/health',
            'stocks_history': '/api/v1/stocks/history',
            'stocks_latest': '/api/v1/stocks/latest',
            'predict': '/api/v1/predictions/predict',
            'model_info': '/api/v1/predictions/model-info'
        }
    }
