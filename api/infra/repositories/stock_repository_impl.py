import os

import pandas as pd
from fastapi import HTTPException

from api.domain.models.stock import StockData
from api.domain.repositories.stock_repository import StockRepository


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
        """Get the latest N records of stock data."""
        df = self._get_dataframe()
        df = df.tail(n)
        data = []

        for _, row in df.iterrows():
            data.append(self._to_model(row))

        return data
