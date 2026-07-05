import pandas as pd
from fastapi import APIRouter, HTTPException
from app.schemas.churn import FeatureVectorChurn, PredictionResponseChurn
from app.services.model import model_service
from app.services.preprocessing import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("/", response_model=list[PredictionResponseChurn])
def predict(features: list[FeatureVectorChurn]):
    if not features:
        return []

    if not model_service.is_trained():
        raise HTTPException(
            status_code=503,
            detail="Модель не обучена. Вызовите сначала POST /model/train.",
        )

    try:
        df = pd.DataFrame([obj.model_dump() for obj in features])
        df = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        return model_service.predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {e}")