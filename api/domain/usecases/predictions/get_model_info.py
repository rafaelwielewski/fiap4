import json
import os

from api.domain.models.prediction import ModelInfo, ModelMetrics
from api.utils.logger import logger


class GetModelInfoUseCase:
    """Use case for getting LSTM model information."""

    def execute(self) -> ModelInfo:
        """Execute the use case to get model information."""
        try:
            artifacts_dir = 'artifacts'
            metadata_path = os.path.join(artifacts_dir, 'metadata.json')
            metrics_path = os.path.join(artifacts_dir, 'metrics.json')

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            metrics_data = {}
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics_data = json.load(f)

            model_metrics = metrics_data.get('model', {})
            symbol = metadata.get('symbol', 'AAPL')

            return ModelInfo(
                model_name='LSTM Stock Predictor v2',
                description=f'Modelo LSTM multi-feature para predição de preços de {symbol} com 16 indicadores técnicos',
                version='2.0.0',
                symbol=symbol,
                training_period={
                    'start': metadata.get('start_date', '2018-01-01'),
                    'end': metadata.get('end_date', 'N/A')
                },
                sequence_length=metadata.get('lookback', 60),
                features_used=metadata.get('features', []),
                metrics=ModelMetrics(
                    mae=model_metrics.get('mae_price', 0.0),
                    rmse=model_metrics.get('rmse_price', 0.0),
                    mape=model_metrics.get('mape_price_pct', 0.0),
                    directional_accuracy=model_metrics.get('directional_accuracy_pct', 0.0),
                ),
                last_trained=metadata.get('trained_at', 'N/A'),
                is_active=True,
                horizon_days=metadata.get('horizon_days', 5),
                baselines=metrics_data.get('baselines', {}),
            )
        except FileNotFoundError:
            logger.error('Arquivos do modelo não encontrados em artifacts/')
            raise ValueError('Modelo não encontrado. Execute o script de treinamento primeiro.')
        except Exception as e:
            logger.error(f'Erro ao carregar info do modelo: {str(e)}')
            raise
