from abc import ABC, abstractmethod
from typing import List

from api.domain.models.stock import StockData


class StockRepository(ABC):
    """Interface for stock data repository operations."""

    @abstractmethod
    def get_stock_history(self) -> List[StockData]:
        """Get full stock history data."""
        pass

    @abstractmethod
    def get_latest_data(self, n: int = 60) -> List[StockData]:
        """Get the latest N records of stock data."""
        pass
