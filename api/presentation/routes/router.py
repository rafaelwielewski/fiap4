
from fastapi.routing import APIRoute
from fastapi import Request, Response
from typing import Callable
from api.utils.logger import logger

class DefaultRouter(APIRoute):
    def get_route_handler(self) -> Callable:  # noqa: C901
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            try:
                try:
                    body = await request.json()
                    logger.info(f'Received Body: {body}')
                except Exception:
                    pass
                query = request.query_params
                if query:
                    logger.info(f'Received Query: {query}')
                response: Response = await original_route_handler(request)
                logger.info(f'Response Body: {response.body}')
            except BaseException as error:
                logger.error(f'Error: {error}')
                raise error

            return response

        return custom_route_handler
