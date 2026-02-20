import os

import pandas as pd
import yfinance as yf
from fastapi import HTTPException

from api.domain.models.stock import StockData
from api.domain.repositories.stock_repository import StockRepository

SYMBOL = 'AAPL'


class StockRepositoryImpl(StockRepository):
    """Repository implementation for stock data operations."""

    def __init__(self):
        self.csv_path = os.path.join('data', 'stock_data.csv')

    def _to_model(self, row: pd.Series) -> StockData:
        return StockData(
            date=str(row.get('Date', '')),
            open=float(row.get('Open', 0.0)),
            high=float(row.get('High', 0.0)),
            low=float(row.get('Low', 0.0)),
            close=float(row.get('Close', 0.0)),
            volume=int(row.get('Volume', 0))
        )

    def _get_dataframe(self) -> pd.DataFrame:
        """Load stock data from CSV file."""
        if not os.path.exists(self.csv_path):
            raise HTTPException(status_code=500, detail='Arquivo de dados não encontrado')

        try:
            df = pd.read_csv(self.csv_path)
            return df
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f'Erro ao carregar CSV: {str(e)}'
            ) from e

    def get_stock_history(self) -> list[StockData]:
        """Get full stock history data."""
        df = self._get_dataframe()
        data = []

        for _, row in df.iterrows():
            data.append(self._to_model(row))

        return data

    def get_latest_data(self, n: int = 60) -> list[StockData]:
        """Get the latest N trading days from yfinance."""
        try:
            raw = yf.download(SYMBOL, period=f'{n + 10}d', auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.rename(columns=str.title)
            raw = raw[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().tail(n).reset_index()
            return [
                StockData(
                    date=str(row['Date'].date()),
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=int(row['Volume']),
                )
                for _, row in raw.iterrows()
            ]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Erro ao buscar dados do yfinance: {str(e)}') from e
