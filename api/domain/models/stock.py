from typing import List, Optional
from datetime import datetime

from pydantic import BaseModel


class StockData(BaseModel):
    """Dados de uma ação em uma data específica."""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class StockHistory(BaseModel):
    """Histórico de dados de uma ação."""
    symbol: str
    data: List[StockData]
    start_date: str
    end_date: str
    total_records: int
