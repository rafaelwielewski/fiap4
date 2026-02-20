import json
import os
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd

from api.domain.models.prediction import PredictedPrice, PredictionResponse
from api.domain.repositories.stock_repository import StockRepository
from api.utils.logger import logger

FORECAST_HORIZON = 5
FEATURE_COLS = [
    'close', 'high', 'low', 'open', 'volume',
    'ret_1', 'log_ret_1',
    'sma_7', 'sma_21', 'ema_12', 'ema_26',
    'macd', 'macd_signal',
    'rsi_14',
    'vol_7', 'vol_21',
]


class PredictStockPriceUseCase:

    def __init__(self, stock_repository: StockRepository):
        self.repository = stock_repository
        self.model_sess = None
        self.scaler_X = None
        self.scaler_y = None
        self.metadata = None
        self.metrics = None
        self.lookback = 60
        self._load_model()

    def _load_model(self):
        artifacts_dir = 'artifacts'
        try:
            import onnxruntime as ort
            self.model_sess = ort.InferenceSession(os.path.join(artifacts_dir, 'final_model.onnx'))
            self.input_name = self.model_sess.get_inputs()[0].name
            self.scaler_X = joblib.load(os.path.join(artifacts_dir, 'scaler_X.joblib'))
            self.scaler_y = joblib.load(os.path.join(artifacts_dir, 'scaler_y.joblib'))
            with open(os.path.join(artifacts_dir, 'metadata.json')) as f:
                self.metadata = json.load(f)
            self.lookback = self.metadata.get('lookback', 60)
            metrics_path = os.path.join(artifacts_dir, 'metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    self.metrics = json.load(f)
            logger.info('Modelo LSTM (ONNX) carregado com sucesso')
        except Exception as e:
            logger.error(f'Erro ao carregar modelo: {e}')
            self.model_sess = None

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        up, down = delta.clip(lower=0), -delta.clip(upper=0)
        rs = up.rolling(period).mean() / (down.rolling(period).mean() + 1e-9)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _build_features(df: pd.DataFrame) -> pd.DataFrame:
        f = df[['close', 'high', 'low', 'open', 'volume']].astype(float).copy()
        f['ret_1'] = f['close'].pct_change()
        f['log_ret_1'] = np.log(f['close']).diff()
        f['sma_7'] = f['close'].rolling(7).mean()
        f['sma_21'] = f['close'].rolling(21).mean()
        f['ema_12'] = f['close'].ewm(span=12, adjust=False).mean()
        f['ema_26'] = f['close'].ewm(span=26, adjust=False).mean()
        f['macd'] = f['ema_12'] - f['ema_26']
        f['macd_signal'] = f['macd'].ewm(span=9, adjust=False).mean()
        f['rsi_14'] = PredictStockPriceUseCase._rsi(f['close'])
        f['vol_7'] = f['ret_1'].rolling(7).std()
        f['vol_21'] = f['ret_1'].rolling(21).std()
        return f.dropna()

    def _predict(self, df: pd.DataFrame, last_date: str) -> PredictedPrice:
        feats = self._build_features(df)
        if len(feats) < self.lookback:
            raise ValueError(f'Dados insuficientes: necessário {self.lookback}, disponível {len(feats)}')

        raw_df = df[['close', 'high', 'low', 'open', 'volume']].reset_index(drop=True).copy()
        current_date = datetime.strptime(last_date, '%Y-%m-%d')

        for step in range(FORECAST_HORIZON):
            feats = self._build_features(raw_df)
            X = self.scaler_X.transform(feats[FEATURE_COLS].values)
            seq = X[-self.lookback:].reshape(1, self.lookback, len(FEATURE_COLS)).astype(np.float32)
            delta = self.scaler_y.inverse_transform(self.model_sess.run(None, {self.input_name: seq})[0])[0][0]
            pred_price = float(feats['close'].iloc[-1]) + float(delta)

            current_date += timedelta(days=1)
            while current_date.weekday() >= 5:
                current_date += timedelta(days=1)

            if step < FORECAST_HORIZON - 1:
                raw_df = pd.concat([raw_df, pd.DataFrame([{
                    'close': pred_price,
                    'high': pred_price + float((raw_df['high'] - raw_df['close']).tail(20).mean()),
                    'low': pred_price - float((raw_df['close'] - raw_df['low']).tail(20).mean()),
                    'open': pred_price + float((raw_df['open'] - raw_df['close']).tail(20).mean()),
                    'volume': float(raw_df['volume'].tail(20).mean()),
                }])], ignore_index=True)

        return PredictedPrice(date=current_date.strftime('%Y-%m-%d'), predicted_close=round(pred_price, 2))

    def _build_metrics(self) -> dict:
        if self.metrics and 'model' in self.metrics:
            m = self.metrics['model']
            return {
                'mae': m.get('mae_price', 0),
                'rmse': m.get('rmse_price', 0),
                'mape': m.get('mape_price_pct', 0),
                'directional_accuracy': m.get('directional_accuracy_pct', 0),
            }
        return {}

    def execute(self, days: int = 90) -> PredictionResponse:
        if self.model_sess is None:
            raise ValueError('Modelo não carregado.')

        latest_data = self.repository.get_latest_data(days)

        if len(latest_data) < self.lookback:
            raise ValueError(f'Dados insuficientes: necessário {self.lookback} pregões, obtido {len(latest_data)}. Aumente o parâmetro days.')

        df = pd.DataFrame([{
            'date': d.date, 'open': d.open, 'high': d.high,
            'low': d.low, 'close': d.close, 'volume': d.volume,
        } for d in latest_data])

        return PredictionResponse(
            symbol=self.metadata.get('symbol', 'AAPL') if self.metadata else 'AAPL',
            prediction=self._predict(df, latest_data[-1].date),
            model_version='2.0.0',
            generated_at=datetime.now().isoformat(),
            metrics=self._build_metrics(),
        )
