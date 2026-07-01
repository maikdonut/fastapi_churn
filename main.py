from fastapi import FastAPI, HTTPException
from schemas import FeatureVectorChurn, DatasetRowChurn
from dataset import dataset_service
from preprocessing import preprocessing_service
from model import model_service


app = FastAPI()


@app.get("/")
def root():
    return {"message": "ml churn service is running"}

@app.post("/predict")
def predict(features: FeatureVectorChurn):
    return features

@app.get("/dataset/preview", response_model=list[DatasetRowChurn])
def preview(n: int = 5):
    return dataset_service.preview(n)

@app.get("/dataset/info")
def info():
    return dataset_service.info()

@app.get("/dataset/split-info")
def split_info():
    return preprocessing_service.split_info()

@app.get("/model/train")
def train():
    try:
        return model_service.train()
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обучения: {e}")