from api.infra.repositories.stock_repository_impl import StockRepositoryImpl
from api.domain.repositories.stock_repository import StockRepository


def build_stock_repository() -> StockRepository:
    """Factory function to create a StockRepository instance."""
    return StockRepositoryImpl()
