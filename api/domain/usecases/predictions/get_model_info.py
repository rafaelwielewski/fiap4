import json
import os

from api.domain.models.prediction import ModelInfo, ModelMetrics
from api.utils.logger import logger


class GetModelInfoUseCase:
    """Use case for getting LSTM model information."""

    def execute(self) -> ModelInfo:
        """Execute the use case to get model information."""
        try:
            weights_path = os.path.join('data', 'model_weights.json')
            scaler_path = os.path.join('data', 'scaler_params.json')

            with open(weights_path, 'r') as f:
                model_data = json.load(f)

            with open(scaler_path, 'r') as f:
                scaler_data = json.load(f)

            metrics_data = model_data.get('metrics', {})

            return ModelInfo(
                model_name='LSTM Stock Predictor',
                description='Modelo LSTM para predição de preços de fechamento de ações da Petrobras (PETR4.SA)',
                version=model_data.get('version', '1.0.0'),
                symbol='PETR4.SA',
                training_period={
                    'start': model_data.get('training_start', '2018-01-01'),
                    'end': model_data.get('training_end', '2024-07-20')
                },
                sequence_length=model_data.get('sequence_length', 60),
                features_used=['close'],
                metrics=ModelMetrics(
                    mae=metrics_data.get('mae', 0.0),
                    rmse=metrics_data.get('rmse', 0.0),
                    mape=metrics_data.get('mape', 0.0)
                ),
                last_trained=model_data.get('trained_at', 'N/A'),
                is_active=True
            )
        except FileNotFoundError:
            logger.error('Arquivos do modelo não encontrados')
            raise ValueError('Modelo não encontrado. Execute o script de treinamento primeiro.')
        except Exception as e:
            logger.error(f'Erro ao carregar info do modelo: {str(e)}')
            raise
