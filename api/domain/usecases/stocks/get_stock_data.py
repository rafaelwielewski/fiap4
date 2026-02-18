from typing import List

from api.domain.models.stock import StockData
from api.domain.repositories.stock_repository import StockRepository


class GetStockDataUseCase:
    """Use case for getting latest stock data."""

    def __init__(self, stock_repository: StockRepository):
        self.repository = stock_repository

    def execute(self, n: int = 30) -> List[StockData]:
        """Execute the use case to get latest stock data."""
        return self.repository.get_latest_data(n)
