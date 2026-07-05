from fastapi import APIRouter

from app.schemas.churn import DatasetRowChurn
from app.services.dataset import dataset_service
from app.services.preprocessing import preprocessing_service

router = APIRouter(prefix="/dataset", tags=["dataset"])


@router.get("/preview", response_model=list[DatasetRowChurn])
def preview(n: int = 5):
    return dataset_service.preview(n)


@router.get("/info")
def info():
    return dataset_service.info()


@router.get("/split-info")
def split_info():
    return preprocessing_service.split_info()
