from fastapi import APIRouter

from app.schemas.churn import FeatureVectorChurn

router = APIRouter(tags=["predict"])


@router.post("/predict")
def predict(features: FeatureVectorChurn):
    return features
