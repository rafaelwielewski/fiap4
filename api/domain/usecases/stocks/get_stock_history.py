from typing import List

from api.domain.models.stock import StockData, StockHistory
from api.domain.repositories.stock_repository import StockRepository


class GetStockHistoryUseCase:
    """Use case for getting stock history data."""

    def __init__(self, stock_repository: StockRepository):
        self.repository = stock_repository

    def execute(self, limit: int = None) -> StockHistory:
        """Execute the use case to get stock history."""
        data = self.repository.get_stock_history()

        if limit and limit > 0:
            data = data[-limit:]

        return StockHistory(
            symbol='AAPL',
            data=data,
            start_date=data[0].date if data else '',
            end_date=data[-1].date if data else '',
            total_records=len(data)
        )
