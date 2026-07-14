import pandas as pd
from fastapi import APIRouter

from app.core.exceptions import ModelNotTrainedException, PredictionFailedException
from app.schemas.churn import ErrorResponse, FeatureVectorChurn, PredictionResponseChurn
from app.services.model import model_service
from app.services.preprocessing import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post(
    "/",
    response_model=list[PredictionResponseChurn],
    operation_id="predict_churn",
    responses={
        503: {"model": ErrorResponse, "description": "Модель не обучена"},
        500: {"model": ErrorResponse, "description": "Ошибка предсказания"},
        422: {"model": ErrorResponse, "description": "Ошибка валидации входных данных"},
    },
)
def predict(features: FeatureVectorChurn | list[FeatureVectorChurn]):
    if isinstance(features, FeatureVectorChurn):
        features = [features]

    if not features:
        return []

    if not model_service.is_trained():
        raise ModelNotTrainedException()

    try:
        df = pd.DataFrame([obj.model_dump() for obj in features])
        df = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        return model_service.predict(df)
    except Exception as e:
        raise PredictionFailedException(details=str(e))
