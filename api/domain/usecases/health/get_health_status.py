from api.domain.repositories.stock_repository import StockRepository


class GetHealthStatusUseCase:
    """Use case for checking application health status."""

    def __init__(self, stock_repository: StockRepository):
        self.repository = stock_repository

    def execute(self):
        """Execute the use case to check application health status."""
        try:
            data = self.repository.get_stock_history()
            return {
                'status': 'healthy',
                'message': 'API funcionando corretamente',
                'data': {
                    'data_loaded': True,
                    'total_records': len(data)
                }
            }
        except FileNotFoundError:
            return {
                'status': 'unhealthy',
                'message': 'Arquivo de dados não encontrado',
                'data': {
                    'data_loaded': False,
                    'total_records': 0
                }
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Erro ao verificar dados: {str(e)}',
                'data': {
                    'data_loaded': False,
                    'total_records': 0
                }
            }
