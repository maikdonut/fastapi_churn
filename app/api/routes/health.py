import logging
from fastapi import APIRouter
from app.services.model import model_service
from app.services.dataset import dataset_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/")
def root():
    return {"message": "ml churn service is running"}


@router.get("/health")
def health():
    dataset_loaded = dataset_service._df is not None
    model_available = model_service.is_trained()

    status = "ok" if model_available and dataset_loaded else "degraded"

    logger.info(f"Health check: status={status}, model_available={model_available}, dataset_loaded={dataset_loaded}")

    return {
        "status": status,
        "model_available": model_available,
        "dataset_loaded": dataset_loaded,
    }