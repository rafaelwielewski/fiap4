import json
import os
from datetime import datetime, timedelta
from typing import List

import numpy as np

from api.domain.models.prediction import PredictionResponse, PredictedPrice, ModelInfo, ModelMetrics
from api.domain.models.stock import StockData
from api.domain.repositories.stock_repository import StockRepository
from api.utils.logger import logger


class PredictStockPriceUseCase:
    """Use case for predicting stock prices using pre-trained LSTM model."""

    def __init__(self, stock_repository: StockRepository):
        self.repository = stock_repository
        self.model_weights = None
        self.scaler_params = None
        self.sequence_length = 60
        self._load_model()

    def _load_model(self):
        """Load pre-trained model weights and scaler parameters."""
        try:
            weights_path = os.path.join('data', 'model_weights.json')
            scaler_path = os.path.join('data', 'scaler_params.json')

            with open(weights_path, 'r') as f:
                self.model_weights = json.load(f)

            with open(scaler_path, 'r') as f:
                self.scaler_params = json.load(f)

            logger.info('Modelo LSTM carregado com sucesso')
        except Exception as e:
            logger.error(f'Erro ao carregar modelo: {str(e)}')
            self.model_weights = None
            self.scaler_params = None

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize data using saved scaler parameters."""
        min_val = self.scaler_params['min']
        max_val = self.scaler_params['max']
        if max_val - min_val == 0:
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)

    def _denormalize(self, data: np.ndarray) -> np.ndarray:
        """Denormalize data back to original scale."""
        min_val = self.scaler_params['min']
        max_val = self.scaler_params['max']
        return data * (max_val - min_val) + min_val

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _tanh(self, x: np.ndarray) -> np.ndarray:
        """Tanh activation function."""
        return np.tanh(x)

    def _lstm_layer_forward(self, x_sequence: np.ndarray, layer_key: str, return_sequences: bool = False) -> np.ndarray:
        """Forward pass through a single LSTM layer using saved weights."""
        weights = self.model_weights[layer_key]

        W = np.array(weights['kernel'])        # (input_size, 4 * units)
        U = np.array(weights['recurrent'])     # (units, 4 * units)
        b = np.array(weights['bias'])           # (4 * units,)

        units = U.shape[0]
        h = np.zeros(units)
        c = np.zeros(units)

        outputs = []

        for t in range(x_sequence.shape[0]):
            x_t = x_sequence[t]

            z = np.dot(x_t, W) + np.dot(h, U) + b

            i = self._sigmoid(z[:units])
            f = self._sigmoid(z[units:2 * units])
            c_candidate = self._tanh(z[2 * units:3 * units])
            o = self._sigmoid(z[3 * units:])

            c = f * c + i * c_candidate
            h = o * self._tanh(c)

            if return_sequences:
                outputs.append(h.copy())

        if return_sequences:
            return np.array(outputs)
        return h

    def _dense_forward(self, x: np.ndarray, layer_key: str, activation: str = None) -> np.ndarray:
        """Forward pass through a dense layer."""
        weights = self.model_weights[layer_key]
        W = np.array(weights['kernel'])
        b = np.array(weights['bias'])
        output = np.dot(x, W) + b

        if activation == 'relu':
            output = np.maximum(0, output)

        return output

    def _predict_single(self, sequence: np.ndarray) -> float:
        """Predict a single value from a sequence.

        Model architecture: LSTM1(return_sequences=True) -> LSTM2 -> Dense(25) -> Dense(1)
        Dropout layers are skipped during inference.
        """
        # First LSTM layer (return_sequences=True)
        lstm1_output = self._lstm_layer_forward(sequence, 'lstm1', return_sequences=True)

        # Second LSTM layer (return_sequences=False)
        lstm2_output = self._lstm_layer_forward(lstm1_output, 'lstm', return_sequences=False)

        # Dense(25)
        dense1_output = self._dense_forward(lstm2_output, 'dense1')

        # Dense(1)
        prediction = self._dense_forward(dense1_output, 'dense')

        return float(prediction[0])

    def execute(self, days_ahead: int = 7) -> PredictionResponse:
        """Execute the prediction use case."""
        if self.model_weights is None or self.scaler_params is None:
            raise ValueError('Modelo não carregado. Execute o script de treinamento primeiro.')

        latest_data = self.repository.get_latest_data(self.sequence_length)

        if len(latest_data) < self.sequence_length:
            raise ValueError(
                f'Dados insuficientes. Necessário: {self.sequence_length}, disponível: {len(latest_data)}'
            )

        close_prices = np.array([d.close for d in latest_data])
        normalized = self._normalize(close_prices).reshape(-1, 1)

        predictions = []
        current_sequence = normalized.copy()

        current_date = datetime.strptime(latest_data[-1].date, '%Y-%m-%d')

        for i in range(days_ahead):
            pred_normalized = self._predict_single(current_sequence)
            pred_price = self._denormalize(np.array([pred_normalized]))[0]

            # Move to next business day
            current_date += timedelta(days=1)
            while current_date.weekday() >= 5:
                current_date += timedelta(days=1)

            predictions.append(PredictedPrice(
                date=current_date.strftime('%Y-%m-%d'),
                predicted_close=round(float(pred_price), 2)
            ))

            # Update sequence for next prediction
            new_entry = np.array([[pred_normalized]])
            current_sequence = np.vstack([current_sequence[1:], new_entry])

        return PredictionResponse(
            symbol='PETR4.SA',
            predictions=predictions,
            model_version=self.model_weights.get('version', '1.0.0'),
            generated_at=datetime.now().isoformat(),
            metrics=self.model_weights.get('metrics', {})
        )

    def execute_custom(self, historical_prices: List[dict], days_ahead: int = 7) -> PredictionResponse:
        """
        Execute prediction using user-provided historical data.

        Args:
            historical_prices: Lista de dicts com 'date' e 'close'
            days_ahead: Número de dias para prever
        """
        if self.model_weights is None or self.scaler_params is None:
            raise ValueError('Modelo não carregado. Execute o script de treinamento primeiro.')

        if len(historical_prices) < self.sequence_length:
            raise ValueError(
                f'Dados insuficientes. Forneça ao menos {self.sequence_length} registros. '
                f'Recebido: {len(historical_prices)}'
            )

        # Use the last sequence_length prices
        recent_prices = historical_prices[-self.sequence_length:]
        close_prices = np.array([p['close'] for p in recent_prices])
        normalized = self._normalize(close_prices).reshape(-1, 1)

        predictions = []
        current_sequence = normalized.copy()

        # Start predictions from the last date provided
        last_date = datetime.strptime(recent_prices[-1]['date'], '%Y-%m-%d')
        current_date = last_date

        for i in range(days_ahead):
            pred_normalized = self._predict_single(current_sequence)
            pred_price = self._denormalize(np.array([pred_normalized]))[0]

            # Move to next business day
            current_date += timedelta(days=1)
            while current_date.weekday() >= 5:
                current_date += timedelta(days=1)

            predictions.append(PredictedPrice(
                date=current_date.strftime('%Y-%m-%d'),
                predicted_close=round(float(pred_price), 2)
            ))

            # Update sequence for next prediction
            new_entry = np.array([[pred_normalized]])
            current_sequence = np.vstack([current_sequence[1:], new_entry])

        return PredictionResponse(
            symbol='Custom',
            predictions=predictions,
            model_version=self.model_weights.get('version', '1.0.0'),
            generated_at=datetime.now().isoformat(),
            metrics=self.model_weights.get('metrics', {})
        )
