from fastapi import APIRouter, HTTPException

from app.services.model import model_service

router = APIRouter(prefix="/model", tags=["model"])


@router.post("/train")
def train():
    try:
        return model_service.train()
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обучения: {e}")

@router.get("/status")
def status():
    return model_service.status()