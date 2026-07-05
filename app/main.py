import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import dataset, health, model, predict
from app.services.model import model_service
from app.schemas.churn import ErrorResponse
from app.core.exceptions import ServiceException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    loaded = model_service.load()
    if loaded:
        logger.info("Модель загружена из файла успешно.")
    else:
        logger.info("Не найдена сохранённая модель. Обучите модель через POST /model/train.")
    yield


app = FastAPI(lifespan=lifespan)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            code=f"HTTP_{exc.status_code}",
            message=exc.detail,
        ).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    details = "; ".join(
        f"{' -> '.join(str(loc) for loc in e['loc'])}: {e['msg']}"
        for e in errors
    )
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            code="VALIDATION_ERROR",
            message="Ошибка валидации входных данных",
            details=details,
        ).model_dump(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc, ServiceException):
        code = exc.code
        details = exc.error_details
    else:
        code = f"HTTP_{exc.status_code}"
        details = None

    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            code=code,
            message=exc.detail,
            details=details,
        ).model_dump(),
    )


app.include_router(health.router)
app.include_router(dataset.router)
app.include_router(model.router)
app.include_router(predict.router)