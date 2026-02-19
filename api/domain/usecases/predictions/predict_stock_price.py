import json
import os
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
import joblib

from api.domain.models.prediction import PredictionResponse, PredictedPrice
from api.domain.models.stock import StockData
from api.domain.repositories.stock_repository import StockRepository
from api.utils.logger import logger


class PredictStockPriceUseCase:
    """Use case for predicting stock prices using the new multi-feature LSTM model."""

    def __init__(self, stock_repository: StockRepository):
        self.repository = stock_repository
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.metadata = None
        self.metrics = None
        self.lookback = 60
        self.horizon = 5
        self._load_model()

    def _load_model(self):
        """Load keras model, scalers, and metadata from artifacts/."""
        try:
            artifacts_dir = os.path.join('artifacts')

            # Load keras model
            # Load ONNX model
            import onnxruntime as ort
            model_path = os.path.join(artifacts_dir, 'final_model.onnx')
            self.model_sess = ort.InferenceSession(model_path)
            self.input_name = self.model_sess.get_inputs()[0].name

            # Load scalers
            self.scaler_X = joblib.load(os.path.join(artifacts_dir, 'scaler_X.joblib'))
            self.scaler_y = joblib.load(os.path.join(artifacts_dir, 'scaler_y.joblib'))

            # Load metadata
            with open(os.path.join(artifacts_dir, 'metadata.json'), 'r') as f:
                self.metadata = json.load(f)

            self.lookback = self.metadata.get('lookback', 60)
            self.horizon = self.metadata.get('horizon_days', 5)

            # Load metrics
            metrics_path = os.path.join(artifacts_dir, 'metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    self.metrics = json.load(f)

            logger.info('Modelo LSTM (ONNX) carregado com sucesso')
        except Exception as e:
            logger.error(f'Erro ao carregar modelo: {str(e)}')
            self.model_sess = None

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = series.diff()
        up = delta.clip(lower=0)
        down = (-delta.clip(upper=0))
        roll_up = up.rolling(period).mean()
        roll_down = down.rolling(period).mean()
        rs = roll_up / (roll_down + 1e-9)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _build_features_from_df(df: pd.DataFrame) -> pd.DataFrame:
        """Build the same 16 features used during training."""
        feats = pd.DataFrame({
            'close': df['close'].astype(float),
            'high': df['high'].astype(float),
            'low': df['low'].astype(float),
            'open': df['open'].astype(float),
            'volume': df['volume'].astype(float),
        })

        feats['ret_1'] = feats['close'].pct_change()
        feats['log_ret_1'] = np.log(feats['close']).diff()

        feats['sma_7'] = feats['close'].rolling(7).mean()
        feats['sma_21'] = feats['close'].rolling(21).mean()
        feats['ema_12'] = feats['close'].ewm(span=12, adjust=False).mean()
        feats['ema_26'] = feats['close'].ewm(span=26, adjust=False).mean()

        feats['macd'] = feats['ema_12'] - feats['ema_26']
        feats['macd_signal'] = feats['macd'].ewm(span=9, adjust=False).mean()

        feats['rsi_14'] = PredictStockPriceUseCase._rsi(feats['close'], 14)

        feats['vol_7'] = feats['ret_1'].rolling(7).std()
        feats['vol_21'] = feats['ret_1'].rolling(21).std()

        feats = feats.dropna()
        return feats

    def _predict_future(self, raw_df: pd.DataFrame, last_date: str, days_ahead: int) -> List[PredictedPrice]:
        """
        Generate predictions for the next N days.

        Receives raw OHLCV data and rebuilds all features from scratch at each
        step, so derived indicators (SMA, EMA, RSI…) evolve naturally with the
        predicted prices instead of remaining frozen.
        """
        feature_cols = [
            'close', 'high', 'low', 'open', 'volume',
            'ret_1', 'log_ret_1',
            'sma_7', 'sma_21', 'ema_12', 'ema_26',
            'macd', 'macd_signal',
            'rsi_14',
            'vol_7', 'vol_21'
        ]

        # Validate: after dropna inside _build_features_from_df we must still
        # have at least lookback rows available.
        initial_feats = self._build_features_from_df(raw_df)
        if len(initial_feats) < self.lookback:
            raise ValueError(
                f'Dados insuficientes após feature engineering. '
                f'Necessário: {self.lookback}, disponível: {len(initial_feats)}'
            )

        raw_df = raw_df[['close', 'high', 'low', 'open', 'volume']].reset_index(drop=True).copy()

        predictions = []
        current_date = datetime.strptime(last_date, '%Y-%m-%d')

        steps = (days_ahead + self.horizon - 1) // self.horizon  # ceiling division
        for _ in range(steps):
            # Rebuild all features from the current raw OHLCV window
            feats = self._build_features_from_df(raw_df)

            X_scaled = self.scaler_X.transform(feats[feature_cols].values)
            input_seq = X_scaled[-self.lookback:].reshape(1, self.lookback, len(feature_cols)).astype(np.float32)

            pred_delta_scaled = self.model_sess.run(None, {self.input_name: input_seq})[0]
            pred_delta = self.scaler_y.inverse_transform(pred_delta_scaled)[0][0]

            current_close = float(feats['close'].iloc[-1])
            pred_price = current_close + float(pred_delta)

            # Advance by horizon business days
            target_date = current_date
            for _ in range(self.horizon):
                target_date += timedelta(days=1)
                while target_date.weekday() >= 5:
                    target_date += timedelta(days=1)

            predictions.append(PredictedPrice(
                date=target_date.strftime('%Y-%m-%d'),
                predicted_close=round(pred_price, 2)
            ))

            if len(predictions) >= days_ahead:
                break

            # Append a synthetic OHLCV row using the predicted close.
            # Spreads and volume are estimated from recent rolling averages
            # so that derived indicators (SMA, EMA, RSI…) evolve naturally.
            avg_spread_high = float((raw_df['high'] - raw_df['close']).tail(20).mean())
            avg_spread_low = float((raw_df['close'] - raw_df['low']).tail(20).mean())
            avg_open_diff = float((raw_df['open'] - raw_df['close']).tail(20).mean())
            avg_volume = float(raw_df['volume'].tail(20).mean())

            new_row = pd.DataFrame([{
                'close': pred_price,
                'high': pred_price + avg_spread_high,
                'low': pred_price - avg_spread_low,
                'open': pred_price + avg_open_diff,
                'volume': avg_volume,
            }])
            raw_df = pd.concat([raw_df, new_row], ignore_index=True)

            current_date = target_date

        return predictions[:days_ahead]

    def execute(self, days_ahead: int = 7) -> PredictionResponse:
        """Execute prediction using pre-loaded stock data."""
        if self.model_sess is None:
            raise ValueError('Modelo não carregado. Execute o script de treinamento primeiro.')

        # Get enough data for features (need ~30 extra rows for indicators like SMA21)
        needed = self.lookback + 30
        latest_data = self.repository.get_latest_data(needed)

        if len(latest_data) < needed:
            raise ValueError(
                f'Dados insuficientes. Necessário: {needed}, disponível: {len(latest_data)}'
            )

        # Convert to DataFrame
        df = pd.DataFrame([{
            'date': d.date, 'open': d.open, 'high': d.high,
            'low': d.low, 'close': d.close, 'volume': d.volume
        } for d in latest_data])

        last_date = latest_data[-1].date

        predictions = self._predict_future(df, last_date, days_ahead)

        # Build metrics response
        metrics_dict = {}
        if self.metrics and 'model' in self.metrics:
            m = self.metrics['model']
            metrics_dict = {
                'mae': m.get('mae_price', 0),
                'rmse': m.get('rmse_price', 0),
                'mape': m.get('mape_price_pct', 0),
                'directional_accuracy': m.get('directional_accuracy_pct', 0),
            }

        symbol = self.metadata.get('symbol', 'AAPL') if self.metadata else 'AAPL'

        return PredictionResponse(
            symbol=symbol,
            predictions=predictions,
            model_version='2.0.0',
            generated_at=datetime.now().isoformat(),
            metrics=metrics_dict
        )

    def execute_custom(self, historical_prices: List[dict], days_ahead: int = 7) -> PredictionResponse:
        """Execute prediction using user-provided historical data."""
        if self.model_sess is None:
            raise ValueError('Modelo não carregado. Execute o script de treinamento primeiro.')

        needed = self.lookback + 30
        if len(historical_prices) < needed:
            raise ValueError(
                f'Dados insuficientes. Forneça ao menos {needed} registros para garantir a precisão dos indicadores. '
                f'Recebido: {len(historical_prices)}'
            )

        # Convert to DataFrame
        df = pd.DataFrame(historical_prices)
        # Rename 'close' column if other OHLCV columns are missing
        if 'open' not in df.columns:
            df['open'] = df['close']
        if 'high' not in df.columns:
            df['high'] = df['close']
        if 'low' not in df.columns:
            df['low'] = df['close']
        if 'volume' not in df.columns:
            df['volume'] = 0

        last_date = historical_prices[-1]['date']

        predictions = self._predict_future(df, last_date, days_ahead)

        metrics_dict = {}
        if self.metrics and 'model' in self.metrics:
            m = self.metrics['model']
            metrics_dict = {
                'mae': m.get('mae_price', 0),
                'rmse': m.get('rmse_price', 0),
                'mape': m.get('mape_price_pct', 0),
                'directional_accuracy': m.get('directional_accuracy_pct', 0),
            }

        return PredictionResponse(
            symbol='Custom',
            predictions=predictions,
            model_version='2.0.0',
            generated_at=datetime.now().isoformat(),
            metrics=metrics_dict
        )
