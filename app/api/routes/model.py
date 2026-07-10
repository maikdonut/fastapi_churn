from fastapi import APIRouter
from app.schemas.churn import TrainingConfigChurn
from app.services.model import model_service
from app.services.preprocessing import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from app.schemas.churn import FeatureVectorChurn, FeatureSchemaResponse, FeatureInfo, ErrorResponse
from app.core.exceptions import DatasetNotFoundException, TrainingFailedException
from app.services.history import history_service
from app.schemas.churn import MetricsResponse, TrainingRecord
from typing import Literal

router = APIRouter(prefix="/model", tags=["model"])

@router.post(
    "/train",
    responses={
        503: {"model": ErrorResponse, "description": "Датасет не найден или пуст"},
        500: {"model": ErrorResponse, "description": "Ошибка обучения модели"},
    }
)


@router.post("/train")
def train(parameters: TrainingConfigChurn = TrainingConfigChurn()):
    try:
        return model_service.train(parameters)
    except ValueError as e:
        raise DatasetNotFoundException(details=str(e))
    except Exception as e:
        raise TrainingFailedException(details=str(e))

@router.get("/status")
def status():
    return model_service.status()

@router.get("/schema", response_model=FeatureSchemaResponse)
def schema():
    fields = FeatureVectorChurn.model_fields
    features = []
    for name in NUMERICAL_FEATURES:
        features.append(FeatureInfo(
            name=name,
            type=fields[name].annotation.__name__,
            kind="numerical"
        ))
    for name in CATEGORICAL_FEATURES:
        features.append(FeatureInfo(
            name=name,
            type=fields[name].annotation.__name__,
            kind="categorical"
        ))
    return FeatureSchemaResponse(features=features)


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    responses={
        404: {"model": ErrorResponse, "description": "История обучений пуста"},
    }
)
def metrics(model_type: Literal["logreg", "random_forest"] | None = None):
    records = history_service.get_all(model_type=model_type)
    
    return MetricsResponse(
        latest=TrainingRecord(**records[-1]) if records else None,
        history=[TrainingRecord(**r) for r in records],
        total_runs=len(records),
    )