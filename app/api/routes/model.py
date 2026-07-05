from fastapi import APIRouter, HTTPException
from app.schemas.churn import TrainingConfigChurn
from app.services.model import model_service

router = APIRouter(prefix="/model", tags=["model"])


@router.post("/train")
def train(parameters: TrainingConfigChurn = TrainingConfigChurn()):
    try:
        return model_service.train(parameters)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обучения: {e}")

@router.get("/status")
def status():
    return model_service.status()