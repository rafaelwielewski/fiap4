from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError


async def validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, ValidationError):
        return JSONResponse(
            status_code=422,
            content={
                'detail': 'Erro de validação dos dados',
                'errors': [
                    {
                        'loc': err['loc'],
                        'msg': err['msg'],
                        'type': err['type']
                    } for err in exc.errors()
                ]
            }
        )
    return JSONResponse(
        status_code=500,
        content={'detail': 'Internal server error'}
    )


async def request_validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                'detail': 'Erro de validação da requisição',
                'errors': [
                    {
                        'loc': err['loc'],
                        'msg': err['msg'],
                        'type': err['type']
                    } for err in exc.errors()
                ]
            }
        )
    return JSONResponse(
        status_code=500,
        content={'detail': 'Internal server error'}
    )
