import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.api.routes import dataset, health, model, predict
from app.services.model import model_service

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

app.include_router(health.router)
app.include_router(dataset.router)
app.include_router(model.router)
app.include_router(predict.router)